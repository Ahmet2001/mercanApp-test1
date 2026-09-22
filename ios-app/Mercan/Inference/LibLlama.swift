import Foundation
import MercanRuntime

enum LlamaError: Error, LocalizedError {
    case couldNotInitializeContext(path: String, detail: String)
    case decodeFailed(String)
    case tokenizationFailed(String)

    var errorDescription: String? {
        switch self {
        case .couldNotInitializeContext(let path, let detail):
            let name = (path as NSString).lastPathComponent
            return "Could not load \(name) with libmercan. \(detail)"
        case .decodeFailed(let detail):
            return "Mercan inference failed while decoding. \(detail)"
        case .tokenizationFailed(let detail):
            return "Mercan tokenizer failed. \(detail)"
        }
    }
}

/// Swift actor adapter over libmercan's stable C runtime API.
///
/// Model loading, Mercan format/ABI validation, architecture dispatch,
/// tokenizer selection, tokenization, decoding and logits all go through
/// libmercan. Swift only owns chat formatting and sampling policy.
actor MercanRuntimeContext {
    private var model: OpaquePointer
    private var context: OpaquePointer?
    private let requestedContextSize: UInt32
    private let architectureName: String
    private let tokenizerName: String

    var is_done = false
    private var nCur: Int32 = 0
    private var nLen: Int32 = 0
    private var temporaryInvalidBytes: [CChar] = []
    private var generatedStopTail = ""

    private var entropyWindow: [Float] = []
    private let entropyWindowSize = 10
    var currentEntropy: Float = 0
    var averageEntropy: Float = 0

    private init(
        model: OpaquePointer,
        context: OpaquePointer,
        contextSize: UInt32,
        architectureName: String,
        tokenizerName: String
    ) {
        self.model = model
        self.context = context
        self.requestedContextSize = contextSize
        self.architectureName = architectureName
        self.tokenizerName = tokenizerName
        self.nLen = Int32(mercan_context_size(context))
    }

    deinit {
        if let context {
            mercan_context_free(context)
        }
        mercan_model_free(model)
        mercan_backend_free()
    }

    static func create_context(
        path: String,
        contextSize: UInt32 = 4096,
        onProgress: (@Sendable (Float) -> Void)? = nil
    ) throws -> MercanRuntimeContext {
        mercan_backend_init()
        onProgress?(0)

        var modelParams = mercan_model_default_params()
        // iOS builds include Metal; -1 asks the compiled backend to offload all
        // supported layers rather than forcing CPU-only execution.
        modelParams.n_gpu_layers = -1
        modelParams.use_mmap = true
        modelParams.check_tensors = false

        guard let model = mercan_model_load(path, modelParams) else {
            let detail = lastRuntimeError()
            mercan_backend_free()
            throw LlamaError.couldNotInitializeContext(path: path, detail: detail)
        }

        var contextParams = mercan_context_default_params()
        contextParams.n_ctx = contextSize
        contextParams.n_batch = min(UInt32(512), contextSize)

        #if targetEnvironment(simulator)
        let threadCount = max(1, min(4, ProcessInfo.processInfo.processorCount - 2))
        #else
        let threadCount = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        #endif
        contextParams.n_threads = Int32(threadCount)
        contextParams.n_threads_batch = Int32(threadCount)

        guard let context = mercan_context_create(model, contextParams) else {
            let detail = lastRuntimeError()
            mercan_model_free(model)
            mercan_backend_free()
            throw LlamaError.couldNotInitializeContext(path: path, detail: detail)
        }

        let architecture = mercan_model_architecture(model).map { String(cString: $0) } ?? "unknown"
        let tokenizer = mercan_model_tokenizer(model).map { String(cString: $0) } ?? "unknown"
        onProgress?(1)

        return MercanRuntimeContext(
            model: model,
            context: context,
            contextSize: contextSize,
            architectureName: architecture,
            tokenizerName: tokenizer
        )
    }

    private static func lastRuntimeError() -> String {
        guard let error = mercan_last_error() else { return "Unknown libmercan error." }
        let text = String(cString: error)
        return text.isEmpty ? "Unknown libmercan error." : text
    }

    private func runtimeError() -> String {
        Self.lastRuntimeError()
    }

    private func recreateContext() throws {
        if let context {
            mercan_context_free(context)
            self.context = nil
        }

        var params = mercan_context_default_params()
        params.n_ctx = requestedContextSize
        params.n_batch = min(UInt32(512), requestedContextSize)

        #if targetEnvironment(simulator)
        let threadCount = max(1, min(4, ProcessInfo.processInfo.processorCount - 2))
        #else
        let threadCount = max(1, min(8, ProcessInfo.processInfo.processorCount - 2))
        #endif
        params.n_threads = Int32(threadCount)
        params.n_threads_batch = Int32(threadCount)

        guard let newContext = mercan_context_create(model, params) else {
            throw LlamaError.decodeFailed(runtimeError())
        }
        context = newContext
        nLen = Int32(mercan_context_size(newContext))
        nCur = 0
    }

    func model_info() -> String {
        "Mercan \(architectureName) / \(tokenizerName)"
    }

    func completion_init(text: String) throws {
        try recreateContext()
        is_done = false
        temporaryInvalidBytes.removeAll(keepingCapacity: true)
        generatedStopTail = ""
        entropyWindow.removeAll(keepingCapacity: true)
        currentEntropy = 0
        averageEntropy = 0

        let tokens = try tokenize(text: text, addSpecial: false, parseSpecial: true)
        guard !tokens.isEmpty else {
            throw LlamaError.tokenizationFailed("Prompt produced no tokens.")
        }
        guard tokens.count < Int(nLen) else {
            throw LlamaError.decodeFailed("Prompt is larger than the selected context window.")
        }

        guard let context else {
            throw LlamaError.decodeFailed("Mercan context is unavailable.")
        }

        let chunkSize = 512
        var start = 0
        while start < tokens.count {
            let end = min(start + chunkSize, tokens.count)
            let rc = tokens[start..<end].withContiguousStorageIfAvailable { buffer -> Int32 in
                guard let base = buffer.baseAddress else { return -1 }
                return mercan_decode(context, base, Int32(buffer.count))
            } ?? Array(tokens[start..<end]).withUnsafeBufferPointer { buffer in
                guard let base = buffer.baseAddress else { return -1 }
                return mercan_decode(context, base, Int32(buffer.count))
            }
            if rc != 0 {
                is_done = true
                throw LlamaError.decodeFailed(runtimeError())
            }
            start = end
        }

        nCur = Int32(tokens.count)
    }

    /// libmercan owns position/KV state but v0.1.2 does not expose a public
    /// cache reset/rewind API. Rebuild the context for each full chat prompt.
    func completion_init_with_cache(text: String) throws {
        try completion_init(text: text)
    }

    func completion_loop() throws -> String {
        guard let context else {
            is_done = true
            throw LlamaError.decodeFailed("Mercan context is unavailable.")
        }

        guard let logits = mercan_logits(context) else {
            is_done = true
            throw LlamaError.decodeFailed("libmercan returned no logits.")
        }

        let vocabSize = Int(mercan_vocab_size(model))
        guard vocabSize > 0 else {
            is_done = true
            throw LlamaError.decodeFailed("Model vocabulary is unavailable.")
        }

        // Entropy is used only for the existing UI confidence indicator.
        var maxLogit = -Float.infinity
        for i in 0..<vocabSize {
            maxLogit = max(maxLogit, logits[i])
        }
        var sumExp: Float = 0
        for i in 0..<vocabSize {
            sumExp += exp(logits[i] - maxLogit)
        }
        let logSumExp = log(sumExp) + maxLogit
        var entropy: Float = 0
        if sumExp.isFinite && sumExp > 0 {
            for i in 0..<vocabSize {
                let p = exp(logits[i] - logSumExp)
                if p > 0 { entropy -= p * log2(p) }
            }
        }
        entropyWindow.append(entropy)
        if entropyWindow.count > entropyWindowSize {
            entropyWindow.removeFirst()
        }
        currentEntropy = entropy
        averageEntropy = entropyWindow.isEmpty ? 0 : entropyWindow.reduce(0, +) / Float(entropyWindow.count)

        // Preserve the app's current deterministic behavior. Sampling controls
        // can later move into libmercan without changing this engine boundary.
        var nextToken: Int32 = 0
        var best = logits[0]
        if vocabSize > 1 {
            for i in 1..<vocabSize where logits[i] > best {
                best = logits[i]
                nextToken = Int32(i)
            }
        }

        let eos = mercan_eos_token(model)
        if nextToken == eos || nCur >= nLen {
            is_done = true
            let tail = String(cString: temporaryInvalidBytes + [0])
            temporaryInvalidBytes.removeAll(keepingCapacity: true)
            return tail
        }

        let pieceBytes = tokenToPiece(nextToken)
        temporaryInvalidBytes.append(contentsOf: pieceBytes)

        let piece: String
        if let valid = String(validatingUTF8: temporaryInvalidBytes + [0]) {
            piece = valid
            temporaryInvalidBytes.removeAll(keepingCapacity: true)
        } else {
            piece = ""
        }

        generatedStopTail += piece
        if generatedStopTail.count > 64 {
            generatedStopTail = String(generatedStopTail.suffix(64))
        }

        if generatedStopTail.contains("<|im_end|>") || generatedStopTail.contains("<|endoftext|>") {
            is_done = true
            return piece
        }

        var tokenToDecode = nextToken
        let rc = withUnsafePointer(to: &tokenToDecode) { ptr in
            mercan_decode(context, ptr, 1)
        }
        if rc != 0 {
            is_done = true
            throw LlamaError.decodeFailed(runtimeError())
        }

        nCur += 1
        return piece
    }

    func apply_chat_template(
        messages: [(role: String, content: String)],
        enableThinking: Bool = false
    ) -> String {
        // Mercan Format v1 NedoLM uses chatml_tr. libmercan deliberately owns
        // model execution, while chat-template metadata is not yet exposed by
        // the public C API. Keep formatting in this thin UI adapter for now.
        var result = ""
        for message in messages {
            result += "<|im_start|>\(message.role)\n\(message.content)<|im_end|>\n"
        }
        result += "<|im_start|>assistant\n"
        return result
    }

    func countTokens(text: String) -> Int {
        (try? tokenize(text: text, addSpecial: false, parseSpecial: true).count) ?? 0
    }

    func clear() {
        if let context {
            mercan_context_free(context)
            self.context = nil
        }
        is_done = true
        nCur = 0
        temporaryInvalidBytes.removeAll()
        generatedStopTail = ""
        entropyWindow.removeAll()
    }

    func clearGenerationState() {
        clear()
    }

    private func tokenize(text: String, addSpecial: Bool, parseSpecial: Bool) throws -> [Int32] {
        let byteCount = text.utf8.count
        let required = text.withCString { ptr in
            mercan_tokenize(model, ptr, byteCount, addSpecial, parseSpecial, nil, 0)
        }

        if required == 0 {
            if text.isEmpty { return [] }
            throw LlamaError.tokenizationFailed(runtimeError())
        }

        let capacity = Int(required < 0 ? -required : required)
        guard capacity > 0 else { return [] }

        var tokens = [Int32](repeating: 0, count: capacity)
        let got = tokens.withUnsafeMutableBufferPointer { buffer in
            text.withCString { ptr in
                mercan_tokenize(
                    model,
                    ptr,
                    byteCount,
                    addSpecial,
                    parseSpecial,
                    buffer.baseAddress,
                    Int32(buffer.count)
                )
            }
        }

        if got < 0 {
            let needed = Int(-got)
            tokens = [Int32](repeating: 0, count: needed)
            let retry = tokens.withUnsafeMutableBufferPointer { buffer in
                text.withCString { ptr in
                    mercan_tokenize(
                        model,
                        ptr,
                        byteCount,
                        addSpecial,
                        parseSpecial,
                        buffer.baseAddress,
                        Int32(buffer.count)
                    )
                }
            }
            guard retry >= 0 else {
                throw LlamaError.tokenizationFailed(runtimeError())
            }
            tokens.removeSubrange(Int(retry)..<tokens.count)
            return tokens
        }

        if Int(got) < tokens.count {
            tokens.removeSubrange(Int(got)..<tokens.count)
        }
        return tokens
    }

    private func tokenToPiece(_ token: Int32) -> [CChar] {
        var small = [CChar](repeating: 0, count: 64)
        var n = small.withUnsafeMutableBufferPointer { buffer in
            mercan_token_to_piece(model, token, buffer.baseAddress, Int32(buffer.count), false)
        }

        if n >= 0 {
            return Array(small.prefix(Int(n)))
        }

        let needed = Int(-n)
        var large = [CChar](repeating: 0, count: needed)
        n = large.withUnsafeMutableBufferPointer { buffer in
            mercan_token_to_piece(model, token, buffer.baseAddress, Int32(buffer.count), false)
        }
        guard n >= 0 else { return [] }
        return Array(large.prefix(Int(n)))
    }
}
