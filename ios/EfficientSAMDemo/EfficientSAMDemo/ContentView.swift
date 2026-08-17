import PhotosUI
import SwiftUI

struct ContentView: View {
    @State private var pickerItem: PhotosPickerItem?
    @State private var sourceImage: UIImage?
    @State private var result: SegmentationResult?
    @State private var isRunning = false
    @State private var progress: Double = 0
    @State private var errorMessage: String?
    @State private var gridSize = 16

    private let segmenter = Segmenter()

    var body: some View {
        NavigationStack {
            Group {
                if let result {
                    resultView(result)
                } else if isRunning {
                    runningView
                } else {
                    landingView
                }
            }
            .navigationTitle("EfficientSAM")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                if result != nil {
                    ToolbarItem(placement: .topBarTrailing) {
                        Button("New Image") { reset() }
                    }
                }
            }
        }
        .task {
            // Compile the models up front so the first run isn't penalized.
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

    // MARK: - Screens

    private var landingView: some View {
        VStack(spacing: 28) {
            Spacer()

            Image(systemName: "square.on.square.dashed")
                .font(.system(size: 64, weight: .light))
                .foregroundStyle(.tint)

            VStack(spacing: 8) {
                Text("Segment Everything")
                    .font(.title2.weight(.semibold))
                Text("Pick a photo and EfficientSAM will find every object in it, on device.")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 40)
            }

            PhotosPicker(selection: $pickerItem, matching: .images) {
                Label("Pick an Image", systemImage: "photo.on.rectangle.angled")
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
            }
            .buttonStyle(.borderedProminent)
            .padding(.horizontal, 40)

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

            Spacer()
        }
    }

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

    private func resultView(_ result: SegmentationResult) -> some View {
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
            await run(image)
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

    private func reset() {
        result = nil
        sourceImage = nil
        pickerItem = nil
        progress = 0
    }
}

#Preview {
    ContentView()
}
