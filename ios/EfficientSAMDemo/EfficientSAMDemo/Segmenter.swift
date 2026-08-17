import CoreML
import UIKit

/// Runs EfficientSAM as a split encoder/decoder pair.
///
/// The split is what makes this viable on device. Measured on the exported
/// models: the encoder costs ~0.15s and runs **once per image**, while the
/// decoder costs ~5ms per point. Caching the embedding across all prompts is
/// the difference between an interactive feature and a hang.
///
/// Peak memory is `decoderBatch * 3 * 256 * 256` floats per call and does not
/// scale with the number of grid points — a 32x32 grid costs the same peak as
/// 8x8, just more iterations.
enum SegmenterError: Error, LocalizedError {
    case modelsMissing
    case badImage
    case allocationFailed

    var errorDescription: String? {
        switch self {
        case .modelsMissing:
            return """
            Core ML models not found in the app bundle. Add \
            efficient_sam_vitt_encoder.mlpackage and \
            efficient_sam_vitt_decoder.mlpackage to the target.
            """
        case .badImage:   return "Could not read that image."
        case .allocationFailed: return "Out of memory preparing the model input."
        }
    }
}

struct SegmentationResult {
    let overlay: UIImage
    let maskCount: Int
    let encodeSeconds: Double
    let decodeSeconds: Double
}

actor Segmenter {

    // Must match the values baked into the exported models. Changing any of
    // these requires re-exporting from EfficientSAM_coreml_export.ipynb.
    private static let encoderSize = 1024
    private static let outputSize  = 256
    private static let decoderBatch = 8

    private var encoder: MLModel?
    private var decoder: MLModel?

    /// Loads both models. Compiling is slow enough to be worth doing once,
    /// off the main actor, before the user picks an image.
    func loadModels() throws {
        guard encoder == nil, decoder == nil else { return }

        let config = MLModelConfiguration()
        // .all lets Core ML schedule across ANE/GPU/CPU. Profile against
        // .cpuAndNeuralEngine on real hardware before assuming ANE residency.
        config.computeUnits = .all

        func load(_ name: String) throws -> MLModel {
            // .mlpackage in the bundle appears as a compiled .mlmodelc.
            guard let url = Bundle.main.url(forResource: name, withExtension: "mlmodelc")
                         ?? Bundle.main.url(forResource: name, withExtension: "mlpackage")
            else { throw SegmenterError.modelsMissing }
            return try MLModel(contentsOf: url, configuration: config)
        }

        encoder = try load("efficient_sam_vitt_encoder")
        decoder = try load("efficient_sam_vitt_decoder")
    }

    func segmentEverything(
        image: UIImage,
        gridSize: Int = 16,
        iouThreshold: Float = 0.7,
        stabilityThreshold: Float = 0.9,
        progress: @Sendable (Double) -> Void = { _ in }
    ) async throws -> SegmentationResult {
        try loadModels()
        guard let encoder, let decoder else { throw SegmenterError.modelsMissing }
        guard let cg = image.cgImage else { throw SegmenterError.badImage }

        let size = Self.encoderSize
        let origW = cg.width, origH = cg.height

        // --- Letterbox to 1024x1024, preserving aspect ratio ---
        let scale = Double(size) / Double(max(origW, origH))
        let fitW = Int((Double(origW) * scale).rounded())
        let fitH = Int((Double(origH) * scale).rounded())

        guard let input = try? makeImageInput(cg, fitW: fitW, fitH: fitH) else {
            throw SegmenterError.allocationFailed
        }

        // --- Encoder: ONCE ---
        let encStart = CFAbsoluteTimeGetCurrent()
        // prediction(from:options:) is async-only on iOS 17+; the synchronous
        // prediction(fromFeatures:) is unavailable on iOS.
        let encOut = try await encoder.prediction(
            from: try MLDictionaryFeatureProvider(dictionary: ["image": input]),
            options: MLPredictionOptions()
        )
        guard let embedding = encOut.featureValue(for: "image_embeddings")?.multiArrayValue
        else { throw SegmenterError.badImage }
        let encodeSeconds = CFAbsoluteTimeGetCurrent() - encStart

        // --- Grid over the valid (non-padded) region, in 1024-space ---
        // Passing raw image pixel coordinates here yields plausible-looking but
        // wrong masks; the model was exported expecting encoder-space points.
        var points: [(Float, Float)] = []
        points.reserveCapacity(gridSize * gridSize)
        for row in 0..<gridSize {
            let y = (Double(row) + 0.5) / Double(gridSize) * Double(fitH)
            for col in 0..<gridSize {
                let x = (Double(col) + 0.5) / Double(gridSize) * Double(fitW)
                points.append((Float(x), Float(y)))
            }
        }

        let batch = Self.decoderBatch
        let out = Self.outputSize
        // Mask resolution covering the unpadded region only.
        let validH = Int((Double(fitH) * Double(out) / Double(size)).rounded())
        let validW = Int((Double(fitW) * Double(out) / Double(size)).rounded())

        var kept: [(mask: [Bool], score: Float)] = []

        let coords = try MLMultiArray(shape: [1, NSNumber(value: batch), 1, 2], dataType: .float32)
        let labels = try MLMultiArray(shape: [1, NSNumber(value: batch), 1], dataType: .float32)

        let decStart = CFAbsoluteTimeGetCurrent()
        var index = 0
        while index < points.count {
            let count = min(batch, points.count - index)

            let cptr = coords.dataPointer.bindMemory(to: Float.self, capacity: batch * 2)
            let lptr = labels.dataPointer.bindMemory(to: Float.self, capacity: batch)
            for slot in 0..<batch {
                // Pad the final partial batch — the decoder shape is static.
                let p = slot < count ? points[index + slot] : (Float(0), Float(0))
                cptr[slot * 2]     = p.0
                cptr[slot * 2 + 1] = p.1
                lptr[slot] = slot < count ? 1 : -1   // 1 = foreground, -1 = ignore
            }

            let decOut = try await decoder.prediction(
                from: try MLDictionaryFeatureProvider(dictionary: [
                    "image_embeddings": embedding,
                    "point_coords": coords,
                    "point_labels": labels,
                ]),
                options: MLPredictionOptions()
            )
            guard
                let masks = decOut.featureValue(for: "masks")?.multiArrayValue,
                let ious  = decOut.featureValue(for: "iou_predictions")?.multiArrayValue
            else { throw SegmenterError.badImage }

            let mptr = masks.dataPointer.bindMemory(to: Float.self, capacity: masks.count)
            let iptr = ious.dataPointer.bindMemory(to: Float.self, capacity: ious.count)

            for slot in 0..<count {
                // Pick the highest-IoU of the 3 predicted masks.
                var best = 0
                for k in 1..<3 where iptr[slot * 3 + k] > iptr[slot * 3 + best] { best = k }
                let iou = iptr[slot * 3 + best]
                guard iou > iouThreshold else { continue }

                let base = (slot * 3 + best) * out * out

                // Stability: agreement between a lower and higher logit cut.
                // Filtering here — before anything is upsampled — is what keeps
                // the accumulator small.
                var inter = 0, union = 0
                for i in 0..<(out * out) {
                    let v = mptr[base + i]
                    if v > 1.0 { inter += 1 }
                    if v > -1.0 { union += 1 }
                }
                guard union > 0,
                      Float(inter) / Float(union) > stabilityThreshold else { continue }

                // Crop padding, keep as a compact bool mask at decoder resolution.
                var mask = [Bool](repeating: false, count: validH * validW)
                var area = 0
                for y in 0..<validH {
                    let src = base + y * out
                    let dst = y * validW
                    for x in 0..<validW where mptr[src + x] >= 0 {
                        mask[dst + x] = true
                        area += 1
                    }
                }

                // Drop degenerate masks: near-empty ones carry no information,
                // and a mask covering almost the whole frame is the background
                // swallowing the image rather than an object.
                let coverage = Float(area) / Float(validH * validW)
                guard coverage > 0.0005, coverage < 0.85 else { continue }

                kept.append((mask, iou))
            }

            index += count
            progress(Double(index) / Double(points.count))
        }
        let decodeSeconds = CFAbsoluteTimeGetCurrent() - decStart

        // Neighbouring grid points land on the same object; drop duplicates.
        let deduped = Self.nms(kept, iouThreshold: 0.7)
        let overlay = Self.renderOverlay(
            base: image, masks: deduped, maskW: validW, maskH: validH
        )

        return SegmentationResult(
            overlay: overlay,
            maskCount: deduped.count,
            encodeSeconds: encodeSeconds,
            decodeSeconds: decodeSeconds
        )
    }

    // MARK: - Input preparation

    /// Draws the image into the top-left of a 1024x1024 black canvas and
    /// returns it as a normalized CHW float array.
    private func makeImageInput(_ cg: CGImage, fitW: Int, fitH: Int) throws -> MLMultiArray {
        let size = Self.encoderSize
        var rgba = [UInt8](repeating: 0, count: size * size * 4)

        guard let ctx = CGContext(
            data: &rgba,
            width: size, height: size,
            bitsPerComponent: 8, bytesPerRow: size * 4,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        ) else { throw SegmenterError.allocationFailed }

        ctx.interpolationQuality = .high
        // Core Graphics has a bottom-left origin, so drawing at y = size - fitH
        // puts the image in the *top* rows of the buffer — matching the export,
        // which letterboxes to the top-left and pads the bottom/right.
        //
        // The buffer is therefore already in the layout the model expects and
        // must be copied row-for-row. Reversing rows here shifts the image down
        // by the padding height: the model still returns masks, but every grid
        // point addresses the wrong part of the image.
        ctx.draw(cg, in: CGRect(x: 0, y: size - fitH, width: fitW, height: fitH))

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

    // MARK: - Post-processing

    private static func nms(
        _ items: [(mask: [Bool], score: Float)], iouThreshold: Float
    ) -> [[Bool]] {
        // Areas are computed once up front; recomputing them inside the
        // comparison loop makes this O(n^2 * pixels) and dominates runtime.
        let areas = items.map { $0.mask.reduce(0) { $0 + ($1 ? 1 : 0) } }
        let order = items.indices
            .filter { areas[$0] > 0 }
            .sorted { items[$0].score > items[$1].score }

        var keptMasks: [[Bool]] = []
        var keptAreas: [Int] = []

        for i in order {
            let m = items[i].mask
            let area = areas[i]

            var duplicate = false
            for (k, other) in keptMasks.enumerated() {
                // Cheap bound first: if the areas are too dissimilar, IoU cannot
                // clear the threshold, so skip the pixel walk entirely.
                let minA = min(area, keptAreas[k]), maxA = max(area, keptAreas[k])
                if Float(minA) / Float(maxA) <= iouThreshold { continue }

                var inter = 0
                for j in 0..<m.count where m[j] && other[j] { inter += 1 }
                let union = area + keptAreas[k] - inter
                if union > 0, Float(inter) / Float(union) > iouThreshold {
                    duplicate = true
                    break
                }
            }
            if !duplicate {
                keptMasks.append(m)
                keptAreas.append(area)
            }
        }
        return keptMasks
    }

    private static func renderOverlay(
        base: UIImage, masks: [[Bool]], maskW: Int, maskH: Int
    ) -> UIImage {
        let format = UIGraphicsImageRendererFormat.default()
        format.scale = 1
        let renderer = UIGraphicsImageRenderer(size: base.size, format: format)

        // Largest first, so small objects stay visible on top. Decorate-sort so
        // each area is counted once rather than on every comparison.
        let ordered = masks
            .map { (mask: $0, area: $0.reduce(0) { a, b in a + (b ? 1 : 0) }) }
            .sorted { $0.area > $1.area }
            .map(\.mask)

        return renderer.image { ctx in
            base.draw(in: CGRect(origin: .zero, size: base.size))

            let sx = base.size.width  / CGFloat(maskW)
            let sy = base.size.height / CGFloat(maskH)

            for (i, mask) in ordered.enumerated() {
                // Golden-ratio hue stepping keeps adjacent masks distinguishable.
                let hue = (CGFloat(i) * 0.618034).truncatingRemainder(dividingBy: 1.0)
                UIColor(hue: hue, saturation: 0.85, brightness: 0.95, alpha: 0.5).setFill()

                // Emit one rect per horizontal run rather than per pixel. Rects
                // are snapped to the next row boundary instead of being given a
                // rounded-up height, otherwise fractional sy leaves 1px gaps
                // between rows and the overlay renders as stripes.
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
            }
        }
    }
}
