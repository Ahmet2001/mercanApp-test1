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
    @State private var selectedProject: String?
    @State private var unfiledOnly = false
    @State private var showCreateProject = false
    @State private var newProjectName = ""
    @State private var renameTarget: ConversationSummary?
    @State private var renameText = ""

    private var visibleConversations: [ConversationSummary] {
        conversationManager.filteredConversations(
            searchText: searchText,
            project: selectedProject,
            unfiledOnly: unfiledOnly
        )
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

            ScrollView(.horizontal, showsIndicators: false) {
                HStack(spacing: 7) {
                    ProjectFilterChip(
                        title: "All",
                        selected: selectedProject == nil && !unfiledOnly
                    ) {
                        selectedProject = nil
                        unfiledOnly = false
                    }

                    ProjectFilterChip(
                        title: "Unfiled",
                        selected: unfiledOnly
                    ) {
                        selectedProject = nil
                        unfiledOnly = true
                    }

                    ForEach(conversationManager.projectNames, id: \.self) { project in
                        ProjectFilterChip(title: project, selected: selectedProject == project) {
                            selectedProject = project
                            unfiledOnly = false
                        }
                    }

                    Button {
                        newProjectName = ""
                        showCreateProject = true
                    } label: {
                        Image(systemName: "plus")
                            .font(.caption.weight(.bold))
                            .frame(width: 28, height: 28)
                            .background(Color(.tertiarySystemFill))
                            .clipShape(Circle())
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }

            if visibleConversations.isEmpty {
                VStack(spacing: 8) {
                    Image(systemName: searchText.isEmpty ? "bubble.left" : "magnifyingglass")
                        .foregroundStyle(.tertiary)
                    Text(searchText.isEmpty ? "No conversations here" : "No matching chats")
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
                                project: conversationManager.project(for: conversation.id),
                                projects: conversationManager.projectNames,
                                isPinned: conversationManager.isPinned(conversation.id),
                                isSelected: conversation.id == conversationManager.currentConversationId,
                                action: { onSelectConversation(conversation) },
                                onPin: { conversationManager.togglePin(conversation.id) },
                                onRename: {
                                    renameTarget = conversation
                                    renameText = conversation.displayTitle
                                },
                                onMove: { project in
                                    conversationManager.assign(conversation.id, to: project)
                                },
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
        .alert("New Project", isPresented: $showCreateProject) {
            TextField("Project name", text: $newProjectName)
            Button("Create") {
                conversationManager.createProject(named: newProjectName)
            }
            Button("Cancel", role: .cancel) {}
        }
        .alert("Rename Conversation", isPresented: Binding(
            get: { renameTarget != nil },
            set: { if !$0 { renameTarget = nil } }
        )) {
            TextField("Conversation title", text: $renameText)
            Button("Rename") {
                if let target = renameTarget {
                    conversationManager.rename(target.id, to: renameText)
                }
                renameTarget = nil
            }
            Button("Cancel", role: .cancel) {
                renameTarget = nil
            }
        }
    }
}

private struct ProjectFilterChip: View {
    let title: String
    let selected: Bool
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            Text(title)
                .font(.caption.weight(selected ? .semibold : .regular))
                .foregroundStyle(selected ? .white : .primary)
                .padding(.horizontal, 10)
                .frame(height: 28)
                .background(selected ? MercanTheme.coral : Color(.tertiarySystemFill))
                .clipShape(Capsule())
        }
        .buttonStyle(.plain)
    }
}

private struct ConversationRow: View {
    let title: String
    let date: String
    let messageCount: Int
    let project: String?
    let projects: [String]
    let isPinned: Bool
    let isSelected: Bool
    let action: () -> Void
    let onPin: () -> Void
    let onRename: () -> Void
    let onMove: (String?) -> Void
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

                    HStack(spacing: 4) {
                        if let project {
                            Text(project)
                                .foregroundStyle(MercanTheme.coral)
                            Text("•")
                        }
                        Text("\(date) • \(messageCount) messages")
                    }
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

            Button(action: onRename) {
                Label("Rename", systemImage: "pencil")
            }

            Menu("Move to Project") {
                Button("Unfiled") { onMove(nil) }
                ForEach(projects, id: \.self) { projectName in
                    Button(projectName) { onMove(projectName) }
                }
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
