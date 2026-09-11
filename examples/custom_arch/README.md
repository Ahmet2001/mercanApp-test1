# Custom Architecture Example

`hello_arch.cpp` is the smallest Mercan Architecture SDK v1 provider template.

It demonstrates:

- matching `general.architecture=hello`,
- reading architecture-specific metadata (`hello.block_count`),
- validating a `.mercan` model before backend loading, and
- returning a versioned `mercan_architecture_v1` descriptor.

`graph_ops.cpp` demonstrates Mercan Graph ABI v1. It performs a tensor operation through `mercan_graph_builder_v1` and `mercan_tensor_handle_v1` without including llama.cpp or ggml headers.

Graph ABI v1 is currently experimental and incremental. NedoLM's MorphFFN-specific primitive path already uses it in real inference, while attention, RoPE, KV-cache management, model tensor ownership and several higher-level transformer helpers are still backend-managed. A completely new architecture may therefore still need backend support for operations that the Graph ABI does not expose yet.

For a custom tokenizer, implement `mercan_tokenizer_v1`. If `encode` and `decode_piece` are provided, libmercan dispatches tokenization through those callbacks; otherwise `MERCAN_TOKENIZER_BACKEND_MANAGED` uses the compiled backend tokenizer.

See `docs/ARCHITECTURE_SDK.md` for the complete v1 contract and current Graph ABI boundary.
