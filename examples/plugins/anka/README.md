# Anka external architecture plugin

`mercan_arch_anka` is a real loadable Mercan Plugin ABI v1 module. It registers both the
`anka` architecture and the external `anka-byte` tokenizer. The architecture provides a
Graph Callback ABI v1 implementation using only public opaque handles:

`token ids -> token_embd.weight -> get_rows -> output.weight -> matmul -> logits/finalize`

Mercan's generic backend adapter can load a `.mercan` file whose
`general.architecture=anka` directly from GGUF tensor metadata/data, invoke this plugin
callback, and execute the resulting graph without a compiled llama.cpp model class. The
regression suite runs this path on CPU and on the ggml CUDA backend with H100.

Anka remains intentionally tiny: it proves independent third-party architecture +
tokenizer execution. It does not yet exercise generic runtime-owned attention/KV-cache,
which is the next boundary for full transformer-class external plugins.
