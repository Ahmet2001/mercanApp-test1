import Foundation

/// Lightweight metadata for the conversation list — no messages loaded into RAM.
struct ConversationSummary: Identifiable {
    let id: UUID
    var title: String
    let createdAt: Date
    var updatedAt: Date
    var messageCount: Int

    var displayTitle: String {
        if title != "New Chat" && !title.isEmpty {
            return title
        }
        return title
    }

    var relativeDate: String {
        let calendar = Calendar.current
        if calendar.isDateInToday(updatedAt) { return String(localized: "Today") }
        if calendar.isDateInYesterday(updatedAt) { return String(localized: "Yesterday") }
        let formatter = DateFormatter()
        formatter.dateFormat = "MMM d"
        return formatter.string(from: updatedAt)
    }
}

@MainActor
class ConversationManager: ObservableObject {
    @Published var conversations: [ConversationSummary] = []
    @Published var currentConversationId: UUID?
    @Published private(set) var pinnedConversationIds: Set<UUID> = []

    private let fileManager = FileManager.default
    private let pinnedKey = "mercan.pinnedConversationIds"

    private var conversationsDirectory: URL {
        let documentsURL = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        return documentsURL.appendingPathComponent("conversations")
    }

    init() {
        let rawPinned = UserDefaults.standard.stringArray(forKey: pinnedKey) ?? []
        pinnedConversationIds = Set(rawPinned.compactMap(UUID.init(uuidString:)))
        createConversationsDirectoryIfNeeded()
        loadConversations()
    }

    private func createConversationsDirectoryIfNeeded() {
        if !fileManager.fileExists(atPath: conversationsDirectory.path) {
            try? fileManager.createDirectory(at: conversationsDirectory, withIntermediateDirectories: true)
        }
    }

    /// Only loads metadata (id, title, dates, messageCount) — NOT full message content.
    func loadConversations() {
        guard let files = try? fileManager.contentsOfDirectory(at: conversationsDirectory, includingPropertiesForKeys: nil) else {
            return
        }

        let decoder = JSONDecoder()
        var loaded: [ConversationSummary] = []

        for file in files where file.pathExtension == "json" {
            if let data = try? Data(contentsOf: file),
               let conversation = try? decoder.decode(Conversation.self, from: data) {
                loaded.append(ConversationSummary(
                    id: conversation.id,
                    title: conversation.displayTitle,
                    createdAt: conversation.createdAt,
                    updatedAt: conversation.updatedAt,
                    messageCount: conversation.messages.count
                ))
            }
        }

        conversations = sortedSummaries(loaded)
    }

    func save(_ conversation: Conversation) {
        let encoder = JSONEncoder()
        encoder.outputFormatting = .prettyPrinted

        let fileURL = conversationsDirectory.appendingPathComponent("\(conversation.id.uuidString).json")

        if let data = try? encoder.encode(conversation) {
            try? data.write(to: fileURL)
        }

        let summary = ConversationSummary(
            id: conversation.id,
            title: conversation.displayTitle,
            createdAt: conversation.createdAt,
            updatedAt: conversation.updatedAt,
            messageCount: conversation.messages.count
        )

        // Update in-memory summary list
        if let index = conversations.firstIndex(where: { $0.id == conversation.id }) {
            conversations[index] = summary
        } else {
            conversations.insert(summary, at: 0)
        }

        conversations = sortedSummaries(conversations)
    }

    func isPinned(_ conversationId: UUID) -> Bool {
        pinnedConversationIds.contains(conversationId)
    }

    func togglePin(_ conversationId: UUID) {
        if pinnedConversationIds.contains(conversationId) {
            pinnedConversationIds.remove(conversationId)
        } else {
            pinnedConversationIds.insert(conversationId)
        }
        persistPins()
        conversations = sortedSummaries(conversations)
    }

    func filteredConversations(searchText: String) -> [ConversationSummary] {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return sortedSummaries(conversations) }
        return sortedSummaries(conversations.filter {
            $0.displayTitle.localizedCaseInsensitiveContains(query)
        })
    }

    private func sortedSummaries(_ values: [ConversationSummary]) -> [ConversationSummary] {
        values.sorted { lhs, rhs in
            let lp = pinnedConversationIds.contains(lhs.id)
            let rp = pinnedConversationIds.contains(rhs.id)
            if lp != rp { return lp && !rp }
            return lhs.updatedAt > rhs.updatedAt
        }
    }

    private func persistPins() {
        UserDefaults.standard.set(pinnedConversationIds.map(\.uuidString), forKey: pinnedKey)
    }

    func delete(_ conversationId: UUID) {
        let fileURL = conversationsDirectory.appendingPathComponent("\(conversationId.uuidString).json")
        try? fileManager.removeItem(at: fileURL)
        conversations.removeAll { $0.id == conversationId }
        pinnedConversationIds.remove(conversationId)
        persistPins()

        if currentConversationId == conversationId {
            currentConversationId = nil
        }
    }

    func createNew() -> Conversation {
        let conversation = Conversation()
        currentConversationId = conversation.id
        return conversation
    }

    /// Loads the full Conversation (with messages) from disk on demand.
    func loadFullConversation(id: UUID) -> Conversation? {
        let fileURL = conversationsDirectory.appendingPathComponent("\(id.uuidString).json")
        guard let data = try? Data(contentsOf: fileURL) else { return nil }
        return try? JSONDecoder().decode(Conversation.self, from: data)
    }
}
