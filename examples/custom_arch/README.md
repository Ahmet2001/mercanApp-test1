# Custom Architecture Example

`hello_arch.cpp` is the smallest Mercan Architecture SDK v1 provider template.

It demonstrates:

- matching `general.architecture=hello`,
- reading architecture-specific metadata (`hello.block_count`),
- validating a `.mercan` model before backend loading, and
- returning a versioned `mercan_architecture_v1` descriptor.

Architecture SDK v1 deliberately leaves graph execution to the compiled backend. A real new architecture therefore also needs tensor/graph support in the backend until the future Mercan graph ABI is introduced.

For a custom tokenizer, implement `mercan_tokenizer_v1`. If `encode` and `decode_piece` are provided, libmercan dispatches tokenization through those callbacks; otherwise `MERCAN_TOKENIZER_BACKEND_MANAGED` uses the compiled backend tokenizer.

See `docs/ARCHITECTURE_SDK.md` for the complete v1 contract.
