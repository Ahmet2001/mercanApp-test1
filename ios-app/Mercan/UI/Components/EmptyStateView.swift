import SwiftUI

struct EmptyStateView: View {
    let onSuggestion: (String) -> Void

    private let suggestions = [
        "Explain this concept simply",
        "Help me write a clear plan",
        "Review an idea critically",
        "Summarize the key points"
    ]

    var body: some View {
        VStack(spacing: 22) {
            MercanMark(size: 92)

            VStack(spacing: 7) {
                Text("Mercan")
                    .font(.system(size: 34, weight: .semibold, design: .serif))
                Text("Private, local AI on your iPhone")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
            }

            LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
                ForEach(suggestions, id: \.self) { suggestion in
                    Button {
                        onSuggestion(suggestion)
                    } label: {
                        Text(suggestion)
                            .font(.subheadline)
                            .foregroundStyle(.primary)
                            .frame(maxWidth: .infinity, minHeight: 58, alignment: .leading)
                            .padding(12)
                            .background(Color(.secondarySystemBackground))
                            .clipShape(RoundedRectangle(cornerRadius: 14))
                    }
                    .buttonStyle(.plain)
                }
            }
            .frame(maxWidth: 520)
        }
        .padding(.horizontal, 22)
    }
}
