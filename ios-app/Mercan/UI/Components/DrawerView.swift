import SwiftUI

struct DrawerView: View {
    @ObservedObject var llamaState: LlamaState
    @ObservedObject var conversationManager: ConversationManager

    let onClose: () -> Void
    let onNewChat: () -> Void
    let onSelectConversation: (ConversationSummary) -> Void
    let onDeleteConversation: (ConversationSummary) -> Void
    let onManageModels: () -> Void
    let onSettings: () -> Void
    let onBenchmark: () -> Void
    let onDiagnostics: () -> Void
    let onPrivacy: () -> Void

    @State private var searchText = ""

    private var visibleConversations: [ConversationSummary] {
        conversationManager.filteredConversations(searchText: searchText)
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                MercanWordmark(compact: true)
                Spacer()
                Button(action: onNewChat) {
                    Image(systemName: "square.and.pencil")
                        .font(.system(size: 17, weight: .semibold))
                        .frame(width: 36, height: 36)
                        .background(Color(.secondarySystemBackground))
                        .clipShape(Circle())
                }
                .buttonStyle(.plain)
            }
            .padding(.horizontal, 16)
            .frame(height: 58)

            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search chats", text: $searchText)
                    .textInputAutocapitalization(.never)
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.tertiary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 12)
            .frame(height: 38)
            .background(Color(.secondarySystemBackground))
            .clipShape(RoundedRectangle(cornerRadius: 12))
            .padding(.horizontal, 12)
            .padding(.bottom, 8)

            if visibleConversations.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: searchText.isEmpty ? "bubble.left" : "magnifyingglass")
                        .foregroundStyle(.tertiary)
                    Text(searchText.isEmpty ? "No conversations yet" : "No matching chats")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                ScrollView {
                    LazyVStack(spacing: 3) {
                        ForEach(visibleConversations) { conversation in
                            ConversationRow(
                                title: conversation.displayTitle,
                                date: conversation.relativeDate,
                                messageCount: conversation.messageCount,
                                isPinned: conversationManager.isPinned(conversation.id),
                                isSelected: conversation.id == conversationManager.currentConversationId,
                                action: { onSelectConversation(conversation) },
                                onPin: { conversationManager.togglePin(conversation.id) },
                                onDelete: { onDeleteConversation(conversation) }
                            )
                        }
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 6)
                }
            }

            Divider()

            VStack(spacing: 2) {
                MenuRow(icon: "cube.box", title: "Models", action: onManageModels)
                MenuRow(icon: "speedometer", title: "Benchmark", action: onBenchmark)
                MenuRow(icon: "waveform.path.ecg", title: "Diagnostics", action: onDiagnostics)
                MenuRow(icon: "hand.raised", title: "Privacy", action: onPrivacy)
                MenuRow(icon: "gearshape", title: "Settings", action: onSettings)
            }
            .padding(.vertical, 6)
        }
        .background(Color(.systemBackground))
        .overlay(alignment: .trailing) {
            Rectangle()
                .fill(Color(.separator))
                .frame(width: 0.5)
        }
    }
}

private struct ConversationRow: View {
    let title: String
    let date: String
    let messageCount: Int
    let isPinned: Bool
    let isSelected: Bool
    let action: () -> Void
    let onPin: () -> Void
    let onDelete: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 10) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 5) {
                        if isPinned {
                            Image(systemName: "pin.fill")
                                .font(.caption2)
                                .foregroundStyle(MercanTheme.coral)
                        }
                        Text(title)
                            .font(.subheadline.weight(.medium))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                    }

                    Text("\(date) • \(messageCount) messages")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Image(systemName: "chevron.right")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 10)
            .background(isSelected ? MercanTheme.coral.opacity(0.10) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 10))
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .contextMenu {
            Button(action: onPin) {
                Label(isPinned ? "Unpin" : "Pin", systemImage: isPinned ? "pin.slash" : "pin")
            }
            Button(role: .destructive, action: onDelete) {
                Label("Delete", systemImage: "trash")
            }
        }
    }
}

struct MenuRow: View {
    let icon: String
    let title: String
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 12) {
                Image(systemName: icon)
                    .frame(width: 24)
                    .foregroundStyle(MercanTheme.coral)
                Text(title)
                    .foregroundStyle(.primary)
                Spacer()
            }
            .padding(.horizontal, 16)
            .frame(height: 42)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}
