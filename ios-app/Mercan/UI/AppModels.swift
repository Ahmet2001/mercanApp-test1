import Foundation

enum GenerationPreset: String, CaseIterable, Identifiable, Hashable {
    case precise
    case balanced
    case creative

    var id: String { rawValue }

    var title: String {
        switch self {
        case .precise: return "Precise"
        case .balanced: return "Balanced"
        case .creative: return "Creative"
        }
    }

    var subtitle: String {
        switch self {
        case .precise: return "More deterministic and focused"
        case .balanced: return "Good default for everyday use"
        case .creative: return "More varied responses"
        }
    }

    var configuration: SamplingConfiguration {
        switch self {
        case .precise:
            return SamplingConfiguration(
                temperature: 0.20,
                topK: 24,
                topP: 0.85,
                minP: 0.08,
                repeatPenalty: 1.12,
                repeatLastN: 64
            )
        case .balanced:
            return .standard
        case .creative:
            return SamplingConfiguration(
                temperature: 1.00,
                topK: 64,
                topP: 0.97,
                minP: 0.03,
                repeatPenalty: 1.08,
                repeatLastN: 64
            )
        }
    }
}

struct AttachedDocument: Identifiable, Equatable {
    let id: UUID
    let name: String
    let kind: String
    let text: String

    init(id: UUID = UUID(), name: String, kind: String, text: String) {
        self.id = id
        self.name = name
        self.kind = kind
        self.text = text
    }

    var characterCount: Int { text.count }
}

struct BenchmarkResult: Equatable {
    let modelName: String
    let generatedTokens: Int
    let duration: TimeInterval
    let tokensPerSecond: Double
    let date: Date
}
