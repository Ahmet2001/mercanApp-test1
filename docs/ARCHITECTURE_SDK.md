# Mercan Architecture SDK v1

Mercan Architecture SDK separates model-family discovery and validation from the Mercan core runtime. A `.mercan` model declares its family with `general.architecture`. Mercan reads metadata first, resolves a registered `mercan_architecture_v1`, validates the model, resolves a tokenizer provider, and only then hands tensor execution to the inference backend.

## Stable ABI surfaces

```c
#define MERCAN_ARCH_ABI_VERSION 1
#define MERCAN_TOKENIZER_ABI_VERSION 1
#define MERCAN_GRAPH_ABI_VERSION 1
```

Public headers installed by Mercan:

- `mercan.h`
- `mercan_arch.h`
- `mercan_tokenizer.h`
- `mercan_graph.h`

The public ABI intentionally exposes no `llama_model`, `llama_context`, `ggml_tensor *`, or other llama.cpp/ggml internal C++ types.

## Architecture registry

An architecture provider exports a `mercan_architecture_v1` descriptor and registers it with `mercan_arch_register_v1()`.

```cpp
static const mercan_architecture_v1 anka_arch = {
    MERCAN_ARCH_ABI_VERSION,
    sizeof(mercan_architecture_v1),
    "anka",
    "AnkaLM",
    "ankatok",
    MERCAN_ARCH_BACKEND_MANAGED_GRAPH,
    anka_probe,
    anka_validate,
};

mercan_arch_register_v1(&anka_arch);
```

A `.mercan` file can then declare:

```text
general.architecture = anka
mercan.tokenizer.type = ankatok
```

Mercan core does not need an `if (arch == "anka")` branch.

## Metadata view

Providers receive `mercan_metadata_v1`, a stable read-only view with `has_key()`, `get_string()`, `get_i64()`, and `get_f64()`. Architecture-specific metadata should use an architecture-owned namespace such as:

```text
anka.block_count
anka.attention_type
anka.expert_count
```

Core Mercan metadata stays under `mercan.*`.

## Tokenizer registry

A tokenizer provider uses `mercan_tokenizer_v1`. The descriptor may either set `MERCAN_TOKENIZER_BACKEND_MANAGED` and let the compiled backend tokenize, or provide `create`, `destroy`, `encode`, and `decode_piece` callbacks.

NedoLM currently registers `ndsurf004` as a built-in backend-managed tokenizer because the exact NDSRF004 bridge is compiled into the current backend.

## Mercan Graph ABI v1

Graph ABI v1 is the first backend-independent tensor-operation boundary. Architecture code receives/uses opaque `mercan_tensor_handle_v1` values rather than raw backend tensor pointers.

The first primitive set contains:

- tensor dimension and byte-stride inspection
- `get_rows`
- `cast_f32`
- `swiglu_split`
- `view_2d`
- `mul`
- `add`
- `concat`
- plain `matmul`
- `rms_norm`
- `rope_ext` for NORMAL/NEOX RoPE

The operation table is carried by `mercan_graph_builder_v1`:

```c
const mercan_graph_api_v1 * api = builder->api;
mercan_tensor_handle_v1 z = api->swiglu_split(builder, gate, up);
```

A provider never receives a `ggml_tensor *`. The current ggml adapter converts opaque handles to backend tensors internally. If llama.cpp/ggml changes, the adapter can change without changing the public Graph ABI.

### Current experimental scope

Graph ABI v1 is intentionally incremental. NedoLM's MorphFFN-specific primitive sequence runs through this ABI, and the real NedoLM path now also uses Graph ABI for RMSNorm+weight application, Q/K RoPE, final-token row selection and residual adds. Attention construction, KV-cache management, LoRA-aware matrix multiplication, and backend model/tensor ownership are still backend-managed.

This gives Mercan a real regression target for the abstraction before the API is opened to fully external graph plugins.

## NedoLM today

```text
general.architecture = nedolm
                 ↓
Mercan architecture registry
                 ↓
nedolm provider
                 ↓
NDSRF004 tokenizer provider
                 ↓
backend graph
        ├── attention / KV cache: backend-managed
        └── MorphFFN + RMSNorm + RoPE + residual primitives: Mercan Graph ABI v1
                 ↓
current ggml adapter
                 ↓
CPU / CUDA / Metal-capable backend
```

NedoLM advertises both `MERCAN_ARCH_BACKEND_MANAGED_GRAPH` and `MERCAN_ARCH_GRAPH_ABI_V1_PRIMITIVES` while this migration is partial.

## Why opaque handles

Third-party architecture code must not depend on the representation behind a tensor. A `mercan_tensor_handle_v1` may currently map to a ggml tensor, but the ABI does not promise that representation. This leaves room for future CUDA-native, Metal-native, or other backend adapters without changing architecture source code.

## Developer flow today

1. Pick a unique architecture name such as `anka`.
2. Implement `mercan_architecture_v1`.
3. Implement `mercan_tokenizer_v1` if needed.
4. Use Mercan Graph ABI primitives where the current primitive set is sufficient.
5. For operations not yet represented by Graph ABI v1, backend graph support is still required.
6. Write a converter that emits `.mercan` metadata and tensors.
7. Verify registration with `mercan arch list` and `mercan tokenizer list`.
8. Run the model with `mercan run ./anka-1b.mercan`.

See `examples/custom_arch/` for a minimal provider template.

## Next compatibility milestone

Before declaring external `.so`/`.dylib` graph plugins stable, Mercan will move enough of NedoLM through Graph ABI to cover the reusable transformer building blocks needed by independent architectures. The planned additions now focus on stable tensor lookup/declaration, LoRA-aware linear helpers, attention/masking, KV-cache operations, reshape/permute/contiguous helpers and graph finalization. Only after the built-in NedoLM regression passes entirely through that boundary will the external dynamic plugin loader be treated as stable.
