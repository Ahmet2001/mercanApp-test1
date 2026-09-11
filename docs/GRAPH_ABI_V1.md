# Mercan Graph ABI v1

Mercan Graph ABI v1 is the backend-independent boundary between architecture graph code and the tensor engine used by Mercan.

## Goals

1. Architecture code must not include or expose llama.cpp/ggml internal types.
2. Tensor values cross the ABI as opaque `mercan_tensor_handle_v1` handles.
3. Backend changes are absorbed by an adapter behind the operation table.
4. ABI structs are versioned and carry `struct_size` for forward-compatible extension.
5. The ABI is validated first on Mercan's own NedoLM path before external graph plugins are declared stable.

## Public types

```c
#define MERCAN_GRAPH_ABI_VERSION 1

typedef uint64_t mercan_tensor_handle_v1;
typedef struct mercan_graph_builder_v1 mercan_graph_builder_v1;
typedef struct mercan_graph_api_v1 mercan_graph_api_v1;
```

`mercan_tensor_handle_v1` is opaque. Architecture code MUST NOT cast it to `ggml_tensor *`, a CUDA pointer, a Metal object or any backend-specific representation.

`MERCAN_TENSOR_NONE_V1` is the null/invalid handle.

## Builder lifetime

A `mercan_graph_builder_v1 *` and the tensor handles produced from it are valid only while the backend is constructing that graph. Providers must not retain the builder or tensor handles after the graph-build callback/phase ends.

The backend owns all graph memory. v1 does not transfer tensor ownership to architecture code.

## Primitive set

The first operation table exposes:

- `dim`
- `stride_bytes`
- `get_rows`
- `cast_f32`
- `swiglu_split`
- `view_2d`
- `mul`
- `add`
- `concat`
- `matmul`
- `rms_norm`
- `rope_ext` (NORMAL/NEOX modes)

Operations return `MERCAN_TENSOR_NONE_V1` on adapter-level failure where a handle is expected.


## Append-only compatibility

Graph ABI v1 uses an append-only operation table. `MERCAN_GRAPH_API_V1_BASE_SIZE` describes the original v1 prefix and `MERCAN_GRAPH_API_HAS_V1(api, member)` must be used before calling operations appended later in v1. This lets a newer provider degrade cleanly when loaded against an older v1 runtime instead of assuming every v1 runtime has the newest tail fields.

## Current ggml adapter

The current implementation lives in Mercan's NedoLM backend integration. It maps an opaque handle to the current backend tensor representation only inside the adapter. That representation is not part of the public contract.

Conceptually:

```text
architecture code
       |
       | mercan_tensor_handle_v1
       v
mercan_graph_api_v1
       |
       v
current ggml adapter
       |
       v
ggml / CPU / CUDA / Metal backend
```

A future adapter may use another representation without requiring architecture-source changes.

## NedoLM regression target

NedoLM is the first real Graph ABI consumer. The following MorphFFN-specific operations are routed through Graph ABI v1:

- token-role table `get_rows`
- role-mask cast to F32
- SwiGLU split
- shared/root/suffix 2D views
- root/suffix role-mask views
- role gating with `mul`
- branch reconstruction with `concat`

The migration now also routes NedoLM RMSNorm+weight application, Q/K RoPE, final-token row selection and residual adds through Graph ABI v1. LoRA-aware matrix multiplication and the attention/KV-cache helper remain backend-managed. This staged migration is intentional.

## Not yet guaranteed by v1

The initial v1 primitive table does not yet provide a complete independent transformer runtime. The following areas still need Mercan-owned abstractions before a completely unknown architecture can be loaded without backend support:

- model tensor declaration and lookup by stable names
- LoRA-aware matrix multiplication (plain `matmul` is available)
- normalization families beyond RMSNorm
- reshape/permute/contiguous helpers
- multi-dimensional RoPE variants beyond NORMAL/NEOX
- attention and masking
- KV-cache read/write and sequence-position semantics
- output/logit declaration
- graph finalization

These will be added while preserving the versioned ABI strategy; new fields are appended and guarded by `struct_size`/capability checks.

## External plugins

Dynamic `.so`/`.dylib` graph plugins are intentionally not declared stable yet. The loader comes after the built-in NedoLM path exercises enough of Graph ABI to prove the abstraction. This avoids publishing a plugin ABI that merely leaks llama.cpp internals under Mercan names.
