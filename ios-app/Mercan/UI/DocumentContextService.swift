import Foundation
import PDFKit
import UniformTypeIdentifiers

enum DocumentContextError: LocalizedError {
    case unsupportedType(String)
    case emptyDocument

    var errorDescription: String? {
        switch self {
        case .unsupportedType(let ext):
            return "Unsupported document type: .\(ext)"
        case .emptyDocument:
            return "The selected document does not contain readable text."
        }
    }
}

enum DocumentContextService {
    static let maxCharacters = 120_000
    private static let groundingLimit = 42_000
    private static let chunkTarget = 3_500

    static func groundingText(for document: AttachedDocument, query: String) -> String {
        let terms = Set(
            query.lowercased()
                .components(separatedBy: CharacterSet.alphanumerics.inverted)
                .filter { $0.count >= 3 }
        )

        let paragraphs = document.text.components(separatedBy: "\n\n")
        var chunks: [(index: Int, text: String)] = []
        var buffer = ""
        var chunkIndex = 0

        func flush() {
            let cleaned = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !cleaned.isEmpty else { return }
            chunks.append((chunkIndex, cleaned))
            chunkIndex += 1
            buffer = ""
        }

        for paragraph in paragraphs {
            if buffer.count + paragraph.count + 2 > chunkTarget, !buffer.isEmpty {
                flush()
            }
            if paragraph.count > chunkTarget {
                var remaining = paragraph[...]
                while !remaining.isEmpty {
                    let end = remaining.index(remaining.startIndex, offsetBy: min(chunkTarget, remaining.count))
                    chunks.append((chunkIndex, String(remaining[..<end])))
                    chunkIndex += 1
                    remaining = remaining[end...]
                }
            } else {
                if !buffer.isEmpty { buffer += "\n\n" }
                buffer += paragraph
            }
        }
        flush()

        guard !chunks.isEmpty else { return String(document.text.prefix(groundingLimit)) }
        guard !terms.isEmpty else { return String(document.text.prefix(groundingLimit)) }

        let scored = chunks.map { chunk -> (index: Int, text: String, score: Int) in
            let lower = chunk.text.lowercased()
            let score = terms.reduce(0) { partial, term in
                partial + max(0, lower.components(separatedBy: term).count - 1)
            }
            return (chunk.index, chunk.text, score)
        }

        var selected = scored
            .filter { $0.score > 0 }
            .sorted {
                if $0.score == $1.score { return $0.index < $1.index }
                return $0.score > $1.score
            }

        if selected.isEmpty {
            selected = scored.sorted { $0.index < $1.index }
        }

        var total = 0
        var kept: [(index: Int, text: String)] = []
        for item in selected {
            let remaining = groundingLimit - total
            guard remaining > 0 else { break }
            let text = item.text.count > remaining ? String(item.text.prefix(remaining)) : item.text
            kept.append((item.index, text))
            total += text.count
        }

        return kept
            .sorted { $0.index < $1.index }
            .map(\.text)
            .joined(separator: "\n\n[…]\n\n")
    }

    static func load(from url: URL) throws -> AttachedDocument {
        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed { url.stopAccessingSecurityScopedResource() }
        }

        let ext = url.pathExtension.lowercased()
        let text: String

        switch ext {
        case "pdf":
            guard let document = PDFDocument(url: url) else {
                throw DocumentContextError.emptyDocument
            }
            var parts: [String] = []
            var total = 0
            for index in 0..<document.pageCount {
                guard let pageText = document.page(at: index)?.string, !pageText.isEmpty else { continue }
                let remaining = maxCharacters - total
                guard remaining > 0 else { break }
                let chunk = pageText.count > remaining ? String(pageText.prefix(remaining)) : pageText
                parts.append(chunk)
                total += chunk.count
            }
            text = parts.joined(separator: "\n\n")

        case "txt", "md", "markdown", "json", "csv", "log":
            let raw = try String(contentsOf: url, encoding: .utf8)
            text = raw.count > maxCharacters ? String(raw.prefix(maxCharacters)) : raw

        default:
            throw DocumentContextError.unsupportedType(ext.isEmpty ? "unknown" : ext)
        }

        let cleaned = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty else { throw DocumentContextError.emptyDocument }

        return AttachedDocument(
            name: url.lastPathComponent,
            kind: ext.uppercased(),
            text: cleaned
        )
    }
}
