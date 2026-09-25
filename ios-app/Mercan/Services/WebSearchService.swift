import Foundation

struct WebSearchResult: Sendable {
    let title: String
    let url: String
    let snippet: String
}

struct WebSearchToolArguments: Decodable, Sendable {
    let query: String
}

struct WebSearchToolCall: Decodable, Sendable {
    let id: String?
    let name: String
    let arguments: WebSearchToolArguments
}

private struct WebSearchToolEnvelope: Decodable {
    let toolCalls: [WebSearchToolCall]

    enum CodingKeys: String, CodingKey {
        case toolCalls = "tool_calls"
    }
}

enum WebSearchServiceError: Error, LocalizedError {
    case invalidQuery
    case invalidResponse
    case httpStatus(Int)

    var errorDescription: String? {
        switch self {
        case .invalidQuery:
            return "Web search query is empty."
        case .invalidResponse:
            return "Web search returned an invalid response."
        case .httpStatus(let code):
            return "Web search failed with HTTP status \(code)."
        }
    }
}

/// Keyless web-search adapter used by the local Mercan tool loop.
///
/// DuckDuckGo HTML is used for ranked web results. If its HTML endpoint is
/// unavailable, the Instant Answer JSON endpoint provides a smaller fallback.
/// The returned context is formatted separately to match Mercan's training
/// tool-result contract.
actor WebSearchService {
    static let toolSchemaJSON = #"[{"type":"function","function":{"name":"web_search","description":"Web'de arama yapar; sıralanmış sonuçları sayfa URL'leri ve metin parçacıklarıyla döndürür.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"Arama sorgusu."}},"required":["query"]}}}]"#

    static func augmentedSystemPrompt(base: String) -> String {
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let foundation = trimmed.isEmpty
            ? "Sen yardımsever bir Türkçe yapay zeka asistanısın. Kullanıcının sorularına doğru, net ve faydalı yanıtlar veriyorsun."
            : trimmed

        return foundation + """

        Emin olmadığın ya da yakın zamanda değişmiş olabilecek bilgiler için tahmin etmek yerine web_search aracını kullan. Gerekirse sorguyu netleştirerek aracı birden çok kez çağırabilirsin. Yeterli kanıtın olduğunda arama sonuçlarına dayanarak doğrudan yanıt ver.

        Kullanılabilir araçlar:
        """ + toolSchemaJSON
    }

    static func parseToolCalls(from text: String) -> [WebSearchToolCall]? {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        let candidates: [String]
        if let first = trimmed.firstIndex(of: "{"),
           let last = trimmed.lastIndex(of: "}"),
           first <= last {
            let object = String(trimmed[first...last])
            candidates = object == trimmed ? [trimmed] : [trimmed, object]
        } else {
            candidates = [trimmed]
        }

        for candidate in candidates {
            guard let data = candidate.data(using: .utf8),
                  let envelope = try? JSONDecoder().decode(WebSearchToolEnvelope.self, from: data)
            else { continue }

            let calls = envelope.toolCalls.filter {
                $0.name == "web_search" &&
                !$0.arguments.query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            }
            if !calls.isEmpty {
                return Array(calls.prefix(4))
            }
        }
        return nil
    }

    /// While streaming the first few characters, keep canonical tool JSON hidden
    /// from the chat UI until it can be classified as a tool call or normal text.
    static func couldStillBeToolCallPrefix(_ text: String) -> Bool {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if value.isEmpty { return true }
        let canonicalPrefix = #"{"tool_calls":"#
        return canonicalPrefix.hasPrefix(value) || value.hasPrefix(canonicalPrefix)
    }

    func search(query: String, maxResults: Int = 5) async throws -> [WebSearchResult] {
        let query = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { throw WebSearchServiceError.invalidQuery }

        do {
            let ranked = try await searchHTML(query: query, maxResults: maxResults)
            if !ranked.isEmpty { return ranked }
        } catch {
            // Fall through to the keyless JSON endpoint.
        }

        return try await searchInstantAnswer(query: query, maxResults: maxResults)
    }

    static func trainingStyleToolResult(
        query: String,
        callID: String,
        results: [WebSearchResult]
    ) -> String {
        let quote = String(UnicodeScalar(34)!)
        let escapedCallID = callID
            .replacingOccurrences(of: "\\", with: "\\\\")
            .replacingOccurrences(of: quote, with: "\\\"")
        var lines = [
            #"<tool_result {"name":"web_search","tool_call_id":""# + escapedCallID + #"">"#
        ]

        if results.isEmpty {
            lines.append("'\(query)' için arama sonucu bulunamadı.")
        } else {
            for (index, result) in results.enumerated() {
                lines.append("[\(index + 1)] \(result.url)")
                if !result.title.isEmpty {
                    lines.append(result.title)
                }
                if !result.snippet.isEmpty {
                    lines.append(result.snippet)
                }
                if index + 1 < results.count {
                    lines.append("")
                }
            }
        }

        lines.append("</tool_result>")
        return lines.joined(separator: "\n")
    }

    private func searchHTML(query: String, maxResults: Int) async throws -> [WebSearchResult] {
        var components = URLComponents(string: "https://html.duckduckgo.com/html/")!
        components.queryItems = [URLQueryItem(name: "q", value: query)]
        guard let url = components.url else { throw WebSearchServiceError.invalidQuery }

        var request = URLRequest(url: url)
        request.timeoutInterval = 15
        request.setValue(
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Mobile/15E148",
            forHTTPHeaderField: "User-Agent"
        )
        request.setValue("text/html,application/xhtml+xml", forHTTPHeaderField: "Accept")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw WebSearchServiceError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw WebSearchServiceError.httpStatus(http.statusCode)
        }
        guard let html = String(data: data, encoding: .utf8) else {
            throw WebSearchServiceError.invalidResponse
        }

        let linkMatches = regexMatches(
            pattern: #"<a[^>]*class=["'][^"']*result__a[^"']*["'][^>]*href=["']([^"']+)["'][^>]*>(.*?)</a>"#,
            in: html
        )
        let snippetMatches = regexMatches(
            pattern: #"<(?:a|div)[^>]*class=["'][^"']*result__snippet[^"']*["'][^>]*>(.*?)</(?:a|div)>"#,
            in: html
        )

        var results: [WebSearchResult] = []
        var seen = Set<String>()

        for (index, match) in linkMatches.enumerated() {
            guard match.count >= 3,
                  let normalizedURL = normalizeResultURL(match[1]),
                  !seen.contains(normalizedURL)
            else { continue }

            seen.insert(normalizedURL)
            let title = cleanHTML(match[2])
            let snippet = index < snippetMatches.count && snippetMatches[index].count >= 2
                ? cleanHTML(snippetMatches[index][1])
                : ""

            results.append(WebSearchResult(title: title, url: normalizedURL, snippet: snippet))
            if results.count >= maxResults { break }
        }

        return results
    }

    private func searchInstantAnswer(query: String, maxResults: Int) async throws -> [WebSearchResult] {
        var components = URLComponents(string: "https://api.duckduckgo.com/")!
        components.queryItems = [
            URLQueryItem(name: "q", value: query),
            URLQueryItem(name: "format", value: "json"),
            URLQueryItem(name: "no_html", value: "1"),
            URLQueryItem(name: "no_redirect", value: "1"),
            URLQueryItem(name: "skip_disambig", value: "1")
        ]
        guard let url = components.url else { throw WebSearchServiceError.invalidQuery }

        var request = URLRequest(url: url)
        request.timeoutInterval = 12
        request.setValue("Mercan/0.1 iOS", forHTTPHeaderField: "User-Agent")

        let (data, response) = try await URLSession.shared.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw WebSearchServiceError.invalidResponse
        }
        guard (200..<300).contains(http.statusCode) else {
            throw WebSearchServiceError.httpStatus(http.statusCode)
        }
        guard let root = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw WebSearchServiceError.invalidResponse
        }

        var results: [WebSearchResult] = []
        var seen = Set<String>()

        func append(title: String, url: String, snippet: String) {
            guard results.count < maxResults,
                  let parsed = URL(string: url),
                  let scheme = parsed.scheme?.lowercased(),
                  scheme == "http" || scheme == "https",
                  !seen.contains(url)
            else { return }
            seen.insert(url)
            results.append(WebSearchResult(title: title, url: url, snippet: snippet))
        }

        if let abstractURL = root["AbstractURL"] as? String, !abstractURL.isEmpty {
            append(
                title: root["Heading"] as? String ?? "",
                url: abstractURL,
                snippet: root["AbstractText"] as? String ?? ""
            )
        }

        func walkTopics(_ items: [Any]) {
            for item in items where results.count < maxResults {
                guard let object = item as? [String: Any] else { continue }
                if let nested = object["Topics"] as? [Any] {
                    walkTopics(nested)
                    continue
                }
                if let url = object["FirstURL"] as? String {
                    append(
                        title: object["Text"] as? String ?? "",
                        url: url,
                        snippet: object["Text"] as? String ?? ""
                    )
                }
            }
        }

        if let topics = root["RelatedTopics"] as? [Any] {
            walkTopics(topics)
        }

        return results
    }

    private func regexMatches(pattern: String, in text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(
            pattern: pattern,
            options: [.caseInsensitive, .dotMatchesLineSeparators]
        ) else { return [] }

        let range = NSRange(text.startIndex..<text.endIndex, in: text)
        return regex.matches(in: text, range: range).map { match in
            (0..<match.numberOfRanges).map { index in
                let nsRange = match.range(at: index)
                guard nsRange.location != NSNotFound,
                      let range = Range(nsRange, in: text)
                else { return "" }
                return String(text[range])
            }
        }
    }

    private func normalizeResultURL(_ raw: String) -> String? {
        var value = decodeEntities(raw)
        if value.hasPrefix("//") {
            value = "https:" + value
        }

        guard let url = URL(string: value) else { return nil }

        if let host = url.host?.lowercased(),
           host.contains("duckduckgo.com"),
           let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
           let redirected = components.queryItems?.first(where: { $0.name == "uddg" })?.value,
           let redirectedURL = URL(string: redirected),
           let scheme = redirectedURL.scheme?.lowercased(),
           scheme == "http" || scheme == "https" {
            return redirected
        }

        guard let scheme = url.scheme?.lowercased(),
              scheme == "http" || scheme == "https"
        else { return nil }

        return url.absoluteString
    }

    private func cleanHTML(_ raw: String) -> String {
        let withoutTags = raw.replacingOccurrences(
            of: #"<[^>]+>"#,
            with: " ",
            options: .regularExpression
        )
        let decoded = decodeEntities(withoutTags)
        return decoded
            .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private func decodeEntities(_ raw: String) -> String {
        let quote = String(UnicodeScalar(34)!)
        return raw
            .replacingOccurrences(of: "&amp;", with: "&")
            .replacingOccurrences(of: "&quot;", with: quote)
            .replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&#x27;", with: "'")
            .replacingOccurrences(of: "&lt;", with: "<")
            .replacingOccurrences(of: "&gt;", with: ">")
            .replacingOccurrences(of: "&nbsp;", with: " ")
    }
}
