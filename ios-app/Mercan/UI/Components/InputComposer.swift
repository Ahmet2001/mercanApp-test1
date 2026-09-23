import SwiftUI

struct InputComposer: View {
    @Binding var text: String
    let isGenerating: Bool
    var inputsDisabled = false
    let onSend: () -> Void
    let onStop: () -> Void
    let onDocumentImport: () -> Void
    var focusState: FocusState<Bool>.Binding

    var body: some View {
        HStack(alignment: .bottom, spacing: 9) {
            Button(action: onDocumentImport) {
                Image(systemName: "paperclip")
                    .font(.system(size: 18, weight: .medium))
                    .foregroundStyle(.primary)
                    .frame(width: 34, height: 34)
                    .background(Color(.tertiarySystemFill))
                    .clipShape(Circle())
            }
            .disabled(inputsDisabled || isGenerating)

            TextField(
                inputsDisabled ? "Waiting for model…" : "Message Mercan",
                text: $text,
                axis: .vertical
            )
            .lineLimit(1...7)
            .focused(focusState)
            .disabled(inputsDisabled)
            .submitLabel(.send)
            .onSubmit {
                if !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating {
                    onSend()
                }
            }

            Button {
                isGenerating ? onStop() : onSend()
            } label: {
                Image(systemName: isGenerating ? "square.fill" : "arrow.up")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 34, height: 34)
                    .background(
                        (text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating)
                        ? Color.gray
                        : MercanTheme.coral
                    )
                    .clipShape(Circle())
            }
            .disabled(text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && !isGenerating)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color(.secondarySystemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20))
        .padding(.horizontal, 12)
        .padding(.bottom, 8)
    }
}
