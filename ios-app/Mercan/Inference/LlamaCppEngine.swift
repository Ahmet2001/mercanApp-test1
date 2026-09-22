import Foundation

actor MercanRuntimeEngine: InferenceEngine {
    private var runtimeContext: MercanRuntimeContext?
    var isComplete: Bool = true

    init() {}

    func initialize(
        modelPath: String,
        contextSize: UInt32,
        onProgress: (@Sendable (Float) -> Void)? = nil
    ) async throws {
        let context = try await Task.detached(priority: .userInitiated) {
            try MercanRuntimeContext.create_context(
                path: modelPath,
                contextSize: contextSize,
                onProgress: onProgress
            )
        }.value
        runtimeContext = context
    }

    func generateNext(messages: [(role: String, content: String)]) async throws {
        guard let context = runtimeContext else {
            throw NSError(
                domain: "MercanRuntimeEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "libmercan runtime is not initialized"]
            )
        }

        let finalPrompt = await context.apply_chat_template(messages: messages)
        try await context.completion_init_with_cache(text: finalPrompt)
        isComplete = false
    }

    func encodePrompt(messages: [(role: String, content: String)]) async throws {
        guard let context = runtimeContext else {
            throw NSError(
                domain: "MercanRuntimeEngine",
                code: 1,
                userInfo: [NSLocalizedDescriptionKey: "libmercan runtime is not initialized"]
            )
        }
        let finalPrompt = await context.apply_chat_template(messages: messages)
        try await context.completion_init(text: finalPrompt)
        isComplete = true
    }

    func streamToken() async throws -> String? {
        guard let context = runtimeContext else { return nil }

        if await context.is_done {
            isComplete = true
            return nil
        }

        let token = try await context.completion_loop()
        if await context.is_done {
            isComplete = true
        }
        return token
    }

    func stop() async {
        isComplete = true
        if let context = runtimeContext {
            await context.clearGenerationState()
        }
    }

    func resume() async {
        isComplete = false
    }

    func clear() async {
        if let context = runtimeContext {
            await context.clear()
        }
        isComplete = true
    }

    func modelInfo() async -> String {
        guard let runtimeContext else { return "Mercan Runtime" }
        return await runtimeContext.model_info()
    }

    func clearGenerationState() async {
        if let context = runtimeContext {
            await context.clearGenerationState()
        }
    }

    var currentEntropy: Float {
        get async { await runtimeContext?.currentEntropy ?? 0 }
    }

    var averageEntropy: Float {
        get async { await runtimeContext?.averageEntropy ?? 0 }
    }

    func countTokens(for messages: [(role: String, content: String)]) async -> Int {
        guard let context = runtimeContext else { return 0 }
        let prompt = await context.apply_chat_template(messages: messages)
        return await context.countTokens(text: prompt)
    }

    func deinitialize() async {
        runtimeContext = nil
        isComplete = true
    }
}
