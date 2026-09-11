# Mercan Tensor ABI v1

Mercan Tensor ABI gives architecture providers stable, backend-independent names for model weights and state. Provider code receives opaque `mercan_tensor_handle_v1` values and must not include or dereference `ggml_tensor`, `llama_model`, CUDA, Metal or another backend's internal tensor types.

## Resolver

A graph builder may expose a runtime-owned `mercan_tensor_resolver_v1` through the append-only `mercan_graph_builder_v1::tensors` field. Providers must first check `MERCAN_GRAPH_BUILDER_HAS_V1(graph, tensors)` and `mercan_tensor_resolver_valid_v1()`.

The v1 base API contains:

- `declare_tensor(resolver, name, flags)` — declare an architecture-facing tensor requirement.
- `tensor_by_name(resolver, name)` — optional lookup; missing tensors return `MERCAN_TENSOR_NONE_V1`.
- `require_tensor(resolver, name)` — required lookup; missing tensors return `NONE` and the backend stores a diagnostic.

The current append-only tail also exposes `has_tensor`, `last_error` and declaration introspection helpers.

## Declaration flags

`MERCAN_TENSOR_REQUIRED_V1` and `MERCAN_TENSOR_OPTIONAL_V1` describe presence requirements. `MERCAN_TENSOR_WEIGHT_V1` and `MERCAN_TENSOR_STATE_V1` describe intended use. Flags are architecture-facing metadata; the backend remains responsible for physical storage and device placement.

Architectures that rely on this resolver advertise `MERCAN_ARCH_TENSOR_ABI_V1` in their architecture flags. The built-in NedoLM provider sets this flag, and `mercan arch list` reports `tensor-abi-v1`.

## NedoLM integration

The built-in NedoLM provider currently binds the canonical `.mercan` tensor names used by the converter, including:

```text
token_embd.weight
output_norm.weight
output.weight
blk.N.attn_norm.weight
blk.N.attn_q.weight
blk.N.attn_k.weight
blk.N.attn_v.weight
blk.N.attn_output.weight
blk.N.ffn_norm.weight
blk.N.ffn_gate.weight
blk.N.ffn_up.weight
blk.N.ffn_down.weight
blk.0.ffn_gate_tid2eid.weight
```

Graph code resolves embedding, norms, attention output, MorphFFN/FFN weights and output projection through Tensor ABI v1. Q/K/V are also declared and resolved/validated by name; construction still passes through llama's LoRA-aware `build_qkv` helper until Mercan has a stable LoRA-aware projection primitive.

## ABI rule

Tensor API v1 is append-only. Consumers must use `struct_size`/`MERCAN_TENSOR_API_HAS_V1()` before calling tail functions. Tensor handles are opaque and only valid for the runtime/model/graph lifetime documented by the provider.

This layer deliberately separates architecture code from the current ggml implementation. The ggml adapter may change without changing third-party architecture source code that only uses the public Tensor and Graph ABIs.
