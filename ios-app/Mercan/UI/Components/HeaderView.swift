import SwiftUI

struct HeaderView: View {
    let currentModel: String
    let models: [Model]
    var isLoadingModel = false
    var isDownloading = false
    var isGenerating = false
    var downloadProgress: Double = 0

    let onMenuTap: () -> Void
    let onModelSelect: (Model) -> Void
    let onNewChat: () -> Void
    let onManageModels: () -> Void

    @State private var showModelPicker = false

    private var modelLabel: String {
        if isDownloading {
            return downloadProgress > 0 ? "Downloading \(Int(downloadProgress * 100))%" : "Connecting…"
        }
        if isLoadingModel { return "Loading model…" }
        if currentModel.isEmpty { return models.isEmpty ? "No model" : "Select model" }
        return Self.parseModelDisplayName(currentModel)
    }

    static func parseModelDisplayName(_ filename: String) -> String {
        var name = filename
            .replacingOccurrences(of: ".mercan", with: "")
            .replacingOccurrences(of: ".gguf", with: "")

        let suffixes = ["-Q4_K_M", "-Q8_0", "-F16", "-F32", "-Instruct", "-instruct", "-SFT"]
        for suffix in suffixes {
            name = name.replacingOccurrences(of: suffix, with: "")
        }
        name = name.replacingOccurrences(of: "-", with: " ")
        while name.contains("  ") { name = name.replacingOccurrences(of: "  ", with: " ") }
        return name.trimmingCharacters(in: .whitespaces)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Button(action: onMenuTap) {
                    Image(systemName: "line.3.horizontal")
                        .font(.system(size: 18, weight: .semibold))
                        .frame(width: 40, height: 40)
                }

                MercanMark(size: 25)

                Button {
                    showModelPicker = true
                } label: {
                    HStack(spacing: 5) {
                        VStack(alignment: .leading, spacing: 1) {
                            Text("Mercan")
                                .font(.system(size: 17, weight: .semibold, design: .serif))
                            Text(modelLabel)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        Image(systemName: "chevron.down")
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                }
                .buttonStyle(.plain)
                .disabled(isGenerating || isLoadingModel || isDownloading)

                Spacer()

                Text("LOCAL")
                    .font(.caption2.weight(.bold))
                    .foregroundStyle(MercanTheme.coral)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(MercanTheme.coral.opacity(0.10))
                    .clipShape(Capsule())

                Button(action: onNewChat) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 17, weight: .semibold))
                        .frame(width: 40, height: 40)
                }
            }
            .padding(.horizontal, 8)
            .frame(height: 56)

            if isDownloading {
                ProgressView(value: downloadProgress)
                    .tint(MercanTheme.coral)
                    .padding(.horizontal, 16)
            }
        }
        .background(.bar)
        .sheet(isPresented: $showModelPicker) {
            ModelPickerSheet(
                models: models,
                currentModel: currentModel,
                onSelect: { model in
                    showModelPicker = false
                    onModelSelect(model)
                },
                onManageModels: {
                    showModelPicker = false
                    onManageModels()
                }
            )
            .presentationDetents([.medium])
        }
    }
}

private struct ModelPickerSheet: View {
    let models: [Model]
    let currentModel: String
    let onSelect: (Model) -> Void
    let onManageModels: () -> Void
    @Environment(\.dismiss) private var dismiss

    private func isActive(_ model: Model) -> Bool {
        let stem = URL(fileURLWithPath: model.filename).deletingPathExtension().lastPathComponent
        return stem == currentModel || model.name == currentModel
    }

    var body: some View {
        NavigationStack {
            List {
                if models.isEmpty {
                    ContentUnavailableView("No models", systemImage: "cube.box", description: Text("Download or import a Mercan model first."))
                } else {
                    ForEach(models) { model in
                        Button {
                            onSelect(model)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(model.name)
                                        .foregroundStyle(.primary)
                                    Text(model.filename)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if isActive(model) {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(MercanTheme.coral)
                                }
                            }
                        }
                    }
                }

                Button(action: onManageModels) {
                    Label("Manage Models", systemImage: "cube.box")
                }
            }
            .navigationTitle("Models")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
