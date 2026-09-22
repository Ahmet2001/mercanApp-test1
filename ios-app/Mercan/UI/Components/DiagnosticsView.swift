import SwiftUI
import MercanRuntime

struct DiagnosticsView: View {
    @ObservedObject var llamaState: LlamaState

    private var runtimeVersion: String {
        guard let value = mercan_version() else { return "unknown" }
        return String(cString: value)
    }

    var body: some View {
        List {
            Section("Runtime") {
                DiagnosticRow(label: "libmercan", value: runtimeVersion)
                DiagnosticRow(label: "Model", value: llamaState.currentModelName.isEmpty ? "Not loaded" : llamaState.currentModelName)
                DiagnosticRow(label: "Context", value: "\(llamaState.contextSize) tokens")
                DiagnosticRow(label: "Prompt tokens", value: "\(llamaState.lastPromptTokenCount)")
                DiagnosticRow(label: "Generated tokens", value: "\(llamaState.generatedTokenCount)")
            }

            Section("Device") {
                DiagnosticRow(label: "Physical memory", value: String(format: "%.1f GiB", llamaState.getTotalRAMInGiB()))
                DiagnosticRow(label: "Recommended context", value: "\(llamaState.recommendedContextSize())")
                DiagnosticRow(label: "iOS", value: ProcessInfo.processInfo.operatingSystemVersionString)
            }

            Section("Last generation") {
                DiagnosticRow(label: "Duration", value: String(format: "%.2f s", llamaState.lastGenerationDuration))
                DiagnosticRow(label: "Speed", value: String(format: "%.1f tok/s", llamaState.lastGenerationTokensPerSecond))
                DiagnosticRow(label: "Confidence proxy", value: String(format: "%.0f%%", llamaState.modelConfidence * 100))
            }

            if let document = llamaState.attachedDocument {
                Section("Document context") {
                    DiagnosticRow(label: "File", value: document.name)
                    DiagnosticRow(label: "Characters", value: "\(document.characterCount)")
                }
            }
        }
        .navigationTitle("Diagnostics")
    }
}

private struct DiagnosticRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.trailing)
        }
    }
}
