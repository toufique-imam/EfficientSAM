import CoreML
import UIKit

/// Runs the **interactive** EfficientSAM path: tap a point, get one mask back.
///
/// This is a different pair of exported models from `Segmenter`, and they are
/// not interchangeable. Two contracts differ:
///
/// - **Resize.** These models were exported to match `EfficientSam.preprocess`,
///   which bilinearly *stretches* the input to 1024x1024 without preserving
///   aspect ratio. `Segmenter` letterboxes. Feeding a letterboxed image here
///   puts every prompt coordinate in the wrong place.
/// - **Prompt shape.** One query of two points, rather than a batch of
///   single-point queries. A box is *one* prompt whose two points are labelled
///   2 (top-left) and 3 (bottom-right) — not two separate prompts.
///
/// The split still matters, and more here than in segment-everything: the
/// encoder runs once when the image is loaded, so each subsequent tap costs
/// only a decoder call (~5ms measured) instead of a full re-encode (~150ms).
enum PromptSegmenterError: Error, LocalizedError {
    case modelsMissing(String)
    case badImage
    case allocationFailed
    case notEncoded

    var errorDescription: String? {
        switch self {
        case .modelsMissing(let name):
            return """
            Core ML model \(name) not found in the app bundle. Run \
            notebooks/EfficientSAM_prompt_coreml_export.ipynb, then \
            generate_project.py.
            """
        case .badImage: return "Could not read that image."
        case .allocationFailed: return "Out of memory preparing the model input."
        case .notEncoded: return "Image has not been encoded yet."
        }
    }
}

/// Which exported checkpoint to run. vits is ~3x slower to encode but scores
/// higher predicted IoU; both are validated against PyTorch in the notebook.
enum PromptModelVariant: String, CaseIterable, Identifiable, Sendable {
    case vitt, vits

    var id: String { rawValue }
    var displayName: String { self == .vitt ? "ViT-Tiny" : "ViT-Small" }
    var encoderResource: String { "efficient_sam_\(rawValue)_prompt_encoder" }
    var decoderResource: String { "efficient_sam_\(rawValue)_prompt_decoder" }
}

struct PromptResult {
    /// The original image with the mask tinted over it.
    let overlay: UIImage
    /// Model-predicted IoU for the returned mask, in [0, 1].
    let iou: Float
    /// Fraction of the image the mask covers.
    let coverage: Float
    let decodeSeconds: Double
}

actor PromptSegmenter {

    // Must match the values baked into the exported models. Changing any of
    // these requires re-exporting from EfficientSAM_prompt_coreml_export.ipynb.
    private static let encoderSize = 1024
    private static let outputSize = 256
    private static let numPoints = 2

    private var variant: PromptModelVariant = .vitt
    private var encoder: MLModel?
    private var decoder: MLModel?

    /// Cached per-image state, so taps only pay for the decoder.
    private var embedding: MLMultiArray?
    private var sourceImage: UIImage?
    private var origW = 0
    private var origH = 0

    private(set) var encodeSeconds: Double = 0

    // MARK: - Model loading

    func loadModels(variant: PromptModelVariant) throws {
        // Switching variants invalidates the cached embedding: it was produced
        // by a different encoder and means nothing to this decoder.
        if variant != self.variant {
            self.variant = variant
            encoder = nil
            decoder = nil
            embedding = nil
        }
        guard encoder == nil || decoder == nil else { return }

        let config = MLModelConfiguration()
        config.computeUnits = .all

        func load(_ name: String) throws -> MLModel {
            // .mlpackage in the bundle appears as a compiled .mlmodelc.
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                         ?? Bundle.main.url(forResource: name, withExtension: "mlpackage")
            else { throw PromptSegmenterError.modelsMissing(name) }
            return try MLModel(contentsOf: url, configuration: config)
        }

        encoder = try load(variant.encoderResource)
        decoder = try load(variant.decoderResource)
    }

    // MARK: - Encoding

    /// Runs the encoder once and caches the embedding for subsequent taps.
    func encode(image: UIImage, variant: PromptModelVariant) async throws {
        try loadModels(variant: variant)
        guard let encoder else { throw PromptSegmenterError.modelsMissing(variant.encoderResource) }
        guard let cg = image.cgImage else { throw PromptSegmenterError.badImage }

        sourceImage = image
        origW = cg.width
        origH = cg.height

        let input = try makeImageInput(cg)

        let start = CFAbsoluteTimeGetCurrent()
        // prediction(from:options:) is async-only on iOS 17+; the synchronous
        // prediction(fromFeatures:) is unavailable on iOS.
        let out = try await encoder.prediction(
            from: try MLDictionaryFeatureProvider(dictionary: ["image": input]),
            options: MLPredictionOptions()
        )
        guard let embed = out.featureValue(for: "image_embeddings")?.multiArrayValue
        else { throw PromptSegmenterError.badImage }

        encodeSeconds = CFAbsoluteTimeGetCurrent() - start
        embedding = embed
    }

    /// True once an image has been encoded and taps can be served.
    var isReady: Bool { embedding != nil && decoder != nil }

    // MARK: - Prompting

    /// Segments whatever is under `point`, given in **original image pixels**.
    ///
    /// The second prompt slot is marked unused (label -1) rather than being
    /// filled with a duplicate: the model pads to 6 points with -1 internally,
    /// so an ignored slot is exactly what the graph expects.
    func segment(atPoint point: CGPoint) async throws -> PromptResult {
        try await segment(
            points: [point, .zero],
            labels: [1, -1]
        )
    }

    /// Segments using two foreground points, both in original image pixels.
    ///
    /// Two points on the same object usually beat one: the model gets a hint
    /// about extent, not just location. This is the prompt shape
    /// `EfficientSAM_example.ipynb` uses for its point example.
    func segmentPair(a: CGPoint, b: CGPoint) async throws -> PromptResult {
        try await segment(points: [a, b], labels: [1, 1])
    }

    /// Segments the contents of a box given by two opposite corners, in
    /// original image pixels. Labels 2/3 are the box top-left / bottom-right
    /// embeddings the prompt encoder was trained with.
    func segment(box: CGRect) async throws -> PromptResult {
        try await segment(
            points: [CGPoint(x: box.minX, y: box.minY), CGPoint(x: box.maxX, y: box.maxY)],
            labels: [2, 3]
        )
    }

    private func segment(points: [CGPoint], labels: [Float]) async throws -> PromptResult {
        guard let decoder, let embedding, let sourceImage else {
            throw PromptSegmenterError.notEncoded
        }
        precondition(points.count == Self.numPoints, "exported model takes \(Self.numPoints) points")

        let size = Double(Self.encoderSize)
        let coords = try MLMultiArray(
            shape: [1, 1, NSNumber(value: Self.numPoints), 2], dataType: .float32
        )
        let labelArray = try MLMultiArray(
            shape: [1, 1, NSNumber(value: Self.numPoints)], dataType: .float32
        )

        let cptr = coords.dataPointer.bindMemory(to: Float.self, capacity: Self.numPoints * 2)
        let lptr = labelArray.dataPointer.bindMemory(to: Float.self, capacity: Self.numPoints)
        for (i, p) in points.enumerated() {
            // Per-axis scale. The encoder input is a stretch, not a uniform
            // resize, so a single scale factor would only be correct for
            // square images.
            cptr[i * 2]     = Float(Double(p.x) * size / Double(origW))
            cptr[i * 2 + 1] = Float(Double(p.y) * size / Double(origH))
            lptr[i] = labels[i]
        }

        let start = CFAbsoluteTimeGetCurrent()
        let out = try await decoder.prediction(
            from: try MLDictionaryFeatureProvider(dictionary: [
                "image_embeddings": embedding,
                "point_coords": coords,
                "point_labels": labelArray,
            ]),
            options: MLPredictionOptions()
        )
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - start

        guard
            let masks = out.featureValue(for: "masks")?.multiArrayValue,
            let ious  = out.featureValue(for: "iou_predictions")?.multiArrayValue
        else { throw PromptSegmenterError.badImage }

        // Candidate 0 is the best: the export sorts by predicted IoU inside the
        // graph, so there is no argmax to do here.
        let mptr = masks.dataPointer.bindMemory(to: Float.self, capacity: masks.count)
        let iptr = ious.dataPointer.bindMemory(to: Float.self, capacity: ious.count)
        let iou = iptr[0]

        let out2 = Self.outputSize
        var mask = [Bool](repeating: false, count: out2 * out2)
        var area = 0
        for i in 0..<(out2 * out2) where mptr[i] >= 0 {
            mask[i] = true
            area += 1
        }

        let overlay = Self.renderOverlay(
            base: sourceImage, mask: mask, maskW: out2, maskH: out2, points: points, labels: labels
        )

        return PromptResult(
            overlay: overlay,
            iou: iou,
            coverage: Float(area) / Float(out2 * out2),
            decodeSeconds: decodeSeconds
        )
    }

    // MARK: - Input preparation

    /// Stretches the image to fill 1024x1024 and returns it as a CHW float
    /// array in [0, 1].
    ///
    /// Deliberately **not** aspect-preserving — see the type doc. ImageNet
    /// mean/std normalization is folded into the exported graph, so the values
    /// here stay in [0, 1] and must not be normalized again.
    private func makeImageInput(_ cg: CGImage) throws -> MLMultiArray {
        let size = Self.encoderSize
        var rgba = [UInt8](repeating: 0, count: size * size * 4)

        guard let ctx = CGContext(
            data: &rgba,
            width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw PromptSegmenterError.allocationFailed }

        ctx.interpolationQuality = .high
        // Filling the whole canvas means there is no padding offset, so unlike
        // the letterboxed path the rows land where the model expects them and
        // are copied in order below.
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: size, height: size))

        let array = try MLMultiArray(
            shape: [1, 3, NSNumber(value: size), NSNumber(value: size)], dataType: .float32
        )
        let ptr = array.dataPointer.bindMemory(to: Float.self, capacity: 3 * size * size)
        let plane = size * size
        for y in 0..<size {
            let srcRow = y * size * 4
            let dstRow = y * size
            for x in 0..<size {
                let s = srcRow + x * 4
                let d = dstRow + x
                ptr[d]             = Float(rgba[s])     / 255.0
                ptr[plane + d]     = Float(rgba[s + 1]) / 255.0
                ptr[plane * 2 + d] = Float(rgba[s + 2]) / 255.0
            }
        }
        return array
    }

    // MARK: - Rendering

    private static func renderOverlay(
        base: UIImage, mask: [Bool], maskW: Int, maskH: Int,
        points: [CGPoint], labels: [Float]
    ) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: base.size, format: format)

        return renderer.image { ctx in
            base.draw(in: CGRect(origin: .zero, size: base.size))

            let sx = base.size.width  / CGFloat(maskW)
            let sy = base.size.height / CGFloat(maskH)

            UIColor(hue: 0.33, saturation: 0.9, brightness: 0.95, alpha: 0.45).setFill()

            // One rect per horizontal run. Rects are snapped to the next row
            // boundary rather than given a rounded-up height, otherwise a
            // fractional sy leaves 1px gaps and the mask renders as stripes.
            for y in 0..<maskH {
                let top = (CGFloat(y) * sy).rounded(.down)
                let bottom = (CGFloat(y + 1) * sy).rounded(.down)
                var x = 0
                while x < maskW {
                    guard mask[y * maskW + x] else { x += 1; continue }
                    var end = x
                    while end < maskW, mask[y * maskW + end] { end += 1 }
                    let left = (CGFloat(x) * sx).rounded(.down)
                    let right = (CGFloat(end) * sx).rounded(.down)
                    ctx.cgContext.fill(CGRect(
                        x: left, y: top,
                        width: max(right - left, 1), height: max(bottom - top, 1)
                    ))
                    x = end
                }
            }

            // Mark the prompt so it is obvious what was asked for.
            let r: CGFloat = max(base.size.width, base.size.height) / 90
            for (i, p) in points.enumerated() where labels[i] >= 0 {
                let dot = CGRect(x: p.x - r, y: p.y - r, width: r * 2, height: r * 2)
                UIColor.white.setFill()
                ctx.cgContext.fillEllipse(in: dot)
                UIColor.systemGreen.setFill()
                ctx.cgContext.fillEllipse(in: dot.insetBy(dx: r * 0.35, dy: r * 0.35))
            }
        }
    }
}
