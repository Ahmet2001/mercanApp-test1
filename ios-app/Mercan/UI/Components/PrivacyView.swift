import SwiftUI

struct PrivacyView: View {
    var body: some View {
        List {
            Section {
                PrivacyRow(icon: "cpu", title: "Inference", detail: "Language-model inference runs locally through libmercan.")
                PrivacyRow(icon: "bubble.left.and.bubble.right", title: "Conversations", detail: "Chat history is stored in the app's local Documents directory.")
                PrivacyRow(icon: "doc.text", title: "Attached documents", detail: "Text extracted from an attached document is held for the current chat session and cleared with a new chat.")
                PrivacyRow(icon: "arrow.down.circle", title: "Model downloads", detail: "Network access is used when you explicitly download a model from its configured source.")
            } header: {
                Text("Local-first")
            }

            Section {
                Text("Mercan does not require a remote inference API for chat generation. A model can still produce incorrect or inappropriate output; local execution does not make model output inherently accurate.")
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .navigationTitle("Privacy")
    }
}

private struct PrivacyRow: View {
    let icon: String
    let title: String
    let detail: String

    var body: some View {
        HStack(alignment: .top, spacing: 12) {
            Image(systemName: icon)
                .frame(width: 24)
                .foregroundStyle(MercanTheme.coral)
            VStack(alignment: .leading, spacing: 3) {
                Text(title).fontWeight(.semibold)
                Text(detail)
                    .font(.footnote)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 4)
    }
}
