import SwiftUI

struct ManageModelsView: View {
    @ObservedObject var llamaState: LlamaState
    @Environment(\.dismiss) private var dismiss

    private var activeFilenameStem: String {
        llamaState.currentModelName
    }

    private func isActive(_ model: Model) -> Bool {
        URL(fileURLWithPath: model.filename).deletingPathExtension().lastPathComponent == activeFilenameStem
            || model.name == llamaState.currentModelName
    }

    private func deleteModel(_ model: Model) {
        guard !isActive(model) else { return }
        let fileURL = llamaState.getDocumentsDirectory().appendingPathComponent(model.filename)
        do {
            try FileManager.default.removeItem(at: fileURL)
            llamaState.downloadedModels.removeAll { $0.filename == model.filename }
            llamaState.restoreToUndownloaded(filename: model.filename)
        } catch {
            llamaState.modelLoadError = "Could not delete \(model.filename): \(error.localizedDescription)"
        }
    }

    var body: some View {
        NavigationStack {
            List {
                Section {
                    HStack {
                        VStack(alignment: .leading, spacing: 3) {
                            Text("This iPhone")
                                .fontWeight(.semibold)
                            Text("\(String(format: "%.1f", llamaState.getTotalRAMInGiB())) GiB memory • recommended context \(llamaState.recommendedContextSize())")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Image(systemName: "iphone")
                            .foregroundStyle(MercanTheme.coral)
                    }
                }

                if !llamaState.downloadedModels.isEmpty {
                    Section("On device") {
                        ForEach(llamaState.downloadedModels) { model in
                            DownloadedModelRow(
                                model: model,
                                fileSize: llamaState.localModelSize(filename: model.filename),
                                isActive: isActive(model),
                                onLoad: {
                                    let url = llamaState.getDocumentsDirectory().appendingPathComponent(model.filename)
                                    Task {
                                        try? await llamaState.loadModel(modelUrl: url)
                                    }
                                },
                                onDelete: { deleteModel(model) }
                            )
                        }
                    }
                }

                Section("Mercan Catalog") {
                    let available = llamaState.undownloadedModels
                    if available.isEmpty {
                        Label("All catalog models are on device", systemImage: "checkmark.circle.fill")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(available) { model in
                            VStack(alignment: .leading, spacing: 8) {
                                HStack {
                                    VStack(alignment: .leading, spacing: 3) {
                                        Text(model.name)
                                            .fontWeight(.semibold)
                                        Text(model.filename)
                                            .font(.caption)
                                            .foregroundStyle(.secondary)
                                    }
                                    Spacer()
                                    if model.rec == true {
                                        Text("Recommended")
                                            .font(.caption2.weight(.semibold))
                                            .foregroundStyle(MercanTheme.coral)
                                    }
                                }

                                DownloadButton(
                                    llamaState: llamaState,
                                    modelName: model.name,
                                    modelUrl: model.url,
                                    filename: model.filename
                                )
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }

                Section {
                    LoadCustomButton(llamaState: llamaState)
                    InputButton(llamaState: llamaState)
                } header: {
                    Text("Import")
                } footer: {
                    Text("Imports must use the Mercan model format. libmercan validates architecture, runtime ABI and tokenizer compatibility when a model is loaded.")
                }

                Section {
                    NavigationLink {
                        DiagnosticsView(llamaState: llamaState)
                    } label: {
                        Label("Runtime diagnostics", systemImage: "waveform.path.ecg")
                    }
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

private struct DownloadedModelRow: View {
    let model: Model
    let fileSize: Int64
    let isActive: Bool
    let onLoad: () -> Void
    let onDelete: () -> Void

    private var sizeLabel: String {
        ByteCountFormatter.string(fromByteCount: fileSize, countStyle: .file)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(model.name)
                            .fontWeight(.semibold)
                        if isActive {
                            Text("ACTIVE")
                                .font(.caption2.weight(.bold))
                                .foregroundStyle(MercanTheme.coral)
                        }
                    }
                    Text("\(sizeLabel) • Mercan")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                Spacer()
            }

            HStack {
                Button(isActive ? "Loaded" : "Load", action: onLoad)
                    .buttonStyle(.borderedProminent)
                    .tint(MercanTheme.coral)
                    .disabled(isActive)

                Spacer()

                if !isActive {
                    Button(role: .destructive, action: onDelete) {
                        Label("Delete", systemImage: "trash")
                    }
                    .buttonStyle(.borderless)
                }
            }
        }
        .padding(.vertical, 4)
    }
}
