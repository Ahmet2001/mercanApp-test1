import SwiftUI

struct ContextStatusBar: View {
    @ObservedObject var llamaState: LlamaState

    var body: some View {
        HStack(spacing: 10) {
            Label("On device", systemImage: "iphone")
                .font(.caption)
                .foregroundStyle(.secondary)

            Divider().frame(height: 14)

            HStack(spacing: 6) {
                ProgressView(value: llamaState.contextUsageFraction)
                    .progressViewStyle(.linear)
                    .frame(width: 56)
                    .tint(llamaState.contextUsageFraction > 0.85 ? .orange : MercanTheme.coral)

                Text("\(llamaState.contextTokenCount)/\(llamaState.contextSize)")
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }

            Spacer()

            if llamaState.lastGenerationTokensPerSecond > 0 {
                Text(String(format: "%.1f tok/s", llamaState.lastGenerationTokensPerSecond))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 7)
        .background(Color(.secondarySystemBackground))
    }
}
