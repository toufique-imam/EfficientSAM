import PhotosUI
import SwiftUI

/// Which of the two exported model families to run.
///
/// They are separate exports with different input contracts, not two settings
/// of one model — see `PromptSegmenter` for why they are not interchangeable.
enum SegmentMode: String, CaseIterable, Identifiable {
    case everything
    case tap

    var id: String { rawValue }
    var title: String { self == .everything ? "Everything" : "Tap to Segment" }
    var blurb: String {
        switch self {
        case .everything:
            return "Find every object in the photo at once."
        case .tap:
            return "Tap anything in the photo to segment just that object."
        }
    }
}

struct ContentView: View {
    @State private var mode: SegmentMode = .everything
    @State private var pickerItem: PhotosPickerItem?
    @State private var sourceImage: UIImage?
    @State private var errorMessage: String?

    // Everything mode
    @State private var result: SegmentationResult?
    @State private var isRunning = false
    @State private var progress: Double = 0
    @State private var gridSize = 16

    // Tap mode
    @State private var variant: PromptModelVariant = .vitt
    @State private var isEncoding = false
    @State private var isDecoding = false
    @State private var promptResult: PromptResult?
    @State private var promptEncodeSeconds: Double = 0

    private let segmenter = Segmenter()
    private let promptSegmenter = PromptSegmenter()

    var body: some View {
        NavigationStack {
            Group {
                switch mode {
                case .everything:
                    if let result {
                        everythingResultView(result)
                    } else if isRunning {
                        runningView
                    } else {
                        landingView
                    }
                case .tap:
                    if sourceImage != nil {
                        tapView
                    } else {
                        landingView
                    }
                }
            }
            .navigationTitle("EfficientSAM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if result != nil || sourceImage != nil {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("New Image") { reset() }
                    }
                }
            }
        }
        .task {
            // Compile up front so the first run isn't penalized. Only the
            // everything models are preloaded; the prompt models load on first
            // use so switching modes doesn't cost memory you never spend.
            try? await segmenter.loadModels()
        }
        .onChange(of: pickerItem) { _, item in
            guard let item else { return }
            Task { await load(item) }
        }
        .alert("Something went wrong", isPresented: .init(
            get: { errorMessage != nil },
            set: { if !$0 { errorMessage = nil } }
        )) {
            Button("OK", role: .cancel) { errorMessage = nil }
        } message: {
            Text(errorMessage ?? "")
        }
    }

    // MARK: - Landing

    private var landingView: some View {
        VStack(spacing: 24) {
            Spacer()

            Image(systemName: mode == .everything
                  ? "square.on.square.dashed"
                  : "hand.tap")
                .font(.system(size: 64, weight: .light))
                .foregroundStyle(.tint)

            VStack(spacing: 8) {
                Text(mode.title)
                    .font(.title2.weight(.semibold))
                Text(mode.blurb)
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }

            Picker("Mode", selection: $mode) {
                ForEach(SegmentMode.allCases) { m in
                    Text(m.title).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 40)

            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Pick an Image", systemImage: "photo.on.rectangle.angled")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 40)

            if mode == .everything {
                VStack(spacing: 6) {
                    Text("Detail").font(.caption).foregroundStyle(.secondary)
                    Picker("Detail", selection: $gridSize) {
                        Text("Fast").tag(8)
                        Text("Balanced").tag(16)
                        Text("Thorough").tag(32)
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal, 40)
                    Text("\(gridSize * gridSize) point prompts")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            } else {
                VStack(spacing: 6) {
                    Text("Model").font(.caption).foregroundStyle(.secondary)
                    Picker("Model", selection: $variant) {
                        ForEach(PromptModelVariant.allCases) { v in
                            Text(v.displayName).tag(v)
                        }
                    }
                    .pickerStyle(.segmented)
                    .padding(.horizontal, 40)
                    Text(variant == .vitt
                         ? "10M params · faster to encode"
                         : "26M params · higher predicted IoU")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }

            Spacer()
        }
    }

    // MARK: - Everything mode

    private var runningView: some View {
        VStack(spacing: 20) {
            Spacer()
            if let sourceImage {
                Image(uiImage: sourceImage)
                    .resizable()
                    .scaledToFit()
                    .frame(maxHeight: 260)
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .overlay(RoundedRectangle(cornerRadius: 12).strokeBorder(.quaternary))
                    .opacity(0.5)
                    .padding(.horizontal, 32)
            }
            ProgressView(value: progress) {
                Text("Segmenting…").font(.subheadline)
            }
            .progressViewStyle(.linear)
            .padding(.horizontal, 40)

            Text("\(Int(progress * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
            Spacer()
        }
    }

    private func everythingResultView(_ result: SegmentationResult) -> some View {
        ScrollView {
            VStack(spacing: 16) {
                Image(uiImage: result.overlay)
                    .resizable()
                    .scaledToFit()
                    .clipShape(RoundedRectangle(cornerRadius: 12))
                    .padding(.horizontal)

                HStack(spacing: 12) {
                    stat("\(result.maskCount)", "masks")
                    stat(String(format: "%.2fs", result.encodeSeconds), "encode")
                    stat(String(format: "%.2fs", result.decodeSeconds), "decode")
                }
                .padding(.horizontal)

                if let sourceImage {
                    DisclosureGroup("Original") {
                        Image(uiImage: sourceImage)
                            .resizable()
                            .scaledToFit()
                            .clipShape(RoundedRectangle(cornerRadius: 10))
                    }
                    .padding(.horizontal)
                }
            }
            .padding(.vertical)
        }
    }

    // MARK: - Tap mode

    private var tapView: some View {
        VStack(spacing: 16) {
            if let image = sourceImage {
                // The overlay replaces the source image in place, so the tap
                // target geometry stays identical between states.
                let shown = promptResult?.overlay ?? image

                GeometryReader { geo in
                    let frame = Self.fittedRect(
                        imageSize: image.size, in: geo.size
                    )
                    ZStack(alignment: .topLeading) {
                        Image(uiImage: shown)
                            .resizable()
                            .scaledToFit()
                            .frame(width: geo.size.width, height: geo.size.height)

                        if isEncoding || isDecoding {
                            // Loading sits over the image rather than replacing
                            // it: the photo stays visible so it is clear which
                            // image is being worked on.
                            Color.black.opacity(0.35)
                                .frame(width: frame.width, height: frame.height)
                                .offset(x: frame.minX, y: frame.minY)
                            VStack(spacing: 10) {
                                ProgressView()
                                    .controlSize(.large)
                                    .tint(.white)
                                Text(isEncoding ? "Preparing image…" : "Segmenting…")
                                    .font(.caption)
                                    .foregroundStyle(.white)
                            }
                            .frame(width: frame.width, height: frame.height)
                            .offset(x: frame.minX, y: frame.minY)
                        }
                    }
                    .contentShape(Rectangle())
                    .onTapGesture { location in
                        // Ignore taps in the letterbox bars around the image,
                        // and while a run is in flight.
                        guard !isEncoding, !isDecoding, frame.contains(location) else { return }
                        let px = (location.x - frame.minX) / frame.width * image.size.width
                        let py = (location.y - frame.minY) / frame.height * image.size.height
                        Task { await runPrompt(at: CGPoint(x: px, y: py)) }
                    }
                }
                .padding(.horizontal)
            }

            if let promptResult {
                HStack(spacing: 12) {
                    stat(String(format: "%.2f", promptResult.iou), "IoU")
                    stat(String(format: "%.0f%%", promptResult.coverage * 100), "coverage")
                    stat(String(format: "%.0fms", promptResult.decodeSeconds * 1000), "decode")
                }
                .padding(.horizontal)
            } else if !isEncoding {
                Text("Tap anything in the photo")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            Text(String(format: "%@ · encoded once in %.2fs",
                        variant.displayName, promptEncodeSeconds))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .padding(.bottom, 8)
        }
    }

    /// Where a `scaledToFit` image actually lands inside its frame. Needed to
    /// convert a tap into image pixels — using the frame directly would skew
    /// every coordinate by the letterbox bars.
    private static func fittedRect(imageSize: CGSize, in container: CGSize) -> CGRect {
        guard imageSize.width > 0, imageSize.height > 0 else { return .zero }
        let scale = min(container.width / imageSize.width,
                        container.height / imageSize.height)
        let w = imageSize.width * scale
        let h = imageSize.height * scale
        return CGRect(x: (container.width - w) / 2,
                      y: (container.height - h) / 2,
                      width: w, height: h)
    }

    private func stat(_ value: String, _ label: String) -> some View {
        VStack(spacing: 2) {
            Text(value).font(.headline.monospacedDigit())
            Text(label).font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 10)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }

    // MARK: - Actions

    private func load(_ item: PhotosPickerItem) async {
        do {
            guard
                let data = try await item.loadTransferable(type: Data.self),
                let image = UIImage(data: data)
            else {
                errorMessage = "Could not read that image."
                return
            }
            sourceImage = image

            switch mode {
            case .everything:
                await run(image)
            case .tap:
                await encodeForTapping(image)
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func run(_ image: UIImage) async {
        isRunning = true
        progress = 0
        defer { isRunning = false }

        do {
            let grid = gridSize
            let output = try await segmenter.segmentEverything(
                image: image,
                gridSize: grid
            ) { fraction in
                Task { @MainActor in progress = fraction }
            }
            result = output
        } catch {
            errorMessage = error.localizedDescription
            sourceImage = nil
        }
    }

    /// Runs the encoder once so taps only pay for the decoder.
    private func encodeForTapping(_ image: UIImage) async {
        isEncoding = true
        defer { isEncoding = false }
        do {
            try await promptSegmenter.encode(image: image, variant: variant)
            promptEncodeSeconds = await promptSegmenter.encodeSeconds
        } catch {
            errorMessage = error.localizedDescription
            sourceImage = nil
        }
    }

    private func runPrompt(at point: CGPoint) async {
        isDecoding = true
        defer { isDecoding = false }
        do {
            promptResult = try await promptSegmenter.segment(atPoint: point)
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func reset() {
        result = nil
        promptResult = nil
        sourceImage = nil
        pickerItem = nil
        progress = 0
        promptEncodeSeconds = 0
    }
}

#Preview {
    ContentView()
}
