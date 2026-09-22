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
