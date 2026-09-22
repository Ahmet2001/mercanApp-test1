import SwiftUI

struct BenchmarkView: View {
    @ObservedObject var llamaState: LlamaState

    var body: some View {
        List {
            Section {
                Text("Runs a short deterministic prompt locally and measures generation throughput. It does not add messages to your conversation.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)

                Button {
                    Task { await llamaState.runBenchmark() }
                } label: {
                    HStack {
                        Label(llamaState.isBenchmarking ? "Running…" : "Run benchmark", systemImage: "speedometer")
                        Spacer()
                        if llamaState.isBenchmarking {
                            ProgressView()
                        }
                    }
                }
                .disabled(llamaState.isBenchmarking || llamaState.isGenerating)
            }

            if let result = llamaState.benchmarkResult {
                Section("Latest result") {
                    DiagnosticBenchmarkRow(label: "Model", value: result.modelName)
                    DiagnosticBenchmarkRow(label: "Tokens", value: "\(result.generatedTokens)")
                    DiagnosticBenchmarkRow(label: "Duration", value: String(format: "%.2f s", result.duration))
                    DiagnosticBenchmarkRow(label: "Throughput", value: String(format: "%.1f tok/s", result.tokensPerSecond))
                }
            }
        }
        .navigationTitle("Benchmark")
    }
}

private struct DiagnosticBenchmarkRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
            Spacer()
            Text(value).foregroundStyle(.secondary)
        }
    }
}
