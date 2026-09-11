# Anka external architecture plugin

`mercan_arch_anka` is a real loadable Mercan Plugin ABI v1 module. It registers both the
`anka` architecture and the external `anka-byte` tokenizer. The architecture builds a
complete tiny pre-norm causal transformer block using only public opaque Mercan handles:

`token+position embedding -> RMSNorm -> Q/K/V -> runtime self_attention -> residual -> RMSNorm -> SwiGLU FFN -> residual -> output norm -> logits`

Mercan's generic backend adapter loads the `.mercan` tensors directly, owns the causal
mask and persistent per-layer K/V cache, invokes the plugin graph callback, and executes
the graph without a compiled llama.cpp model class for `anka`. The plugin can inspect the
runtime-owned cache through `mercan_kv_resolver_v1` but never receives cache storage or a
`ggml_tensor *`.

The permanent regression verifies multi-token prefill and a subsequent cached decode. A
fresh context decoding the same final token produces different logits, proving that the
second decode actually consumes the retained K/V history. The same test passes on CPU and
the ggml CUDA backend with H100.
