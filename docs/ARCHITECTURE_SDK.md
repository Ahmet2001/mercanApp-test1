# Mercan Architecture SDK v1

Mercan Architecture SDK separates model-family discovery and validation from the Mercan core runtime.
A `.mercan` model declares its family with `general.architecture`. Mercan reads metadata first, resolves a registered `mercan_architecture_v1`, validates the model, resolves a tokenizer provider, and only then hands tensor execution to the compiled inference backend.

## ABI

```c
#define MERCAN_ARCH_ABI_VERSION 1
#define MERCAN_TOKENIZER_ABI_VERSION 1
```

Public headers installed by Mercan:

- `mercan.h`
- `mercan_arch.h`
- `mercan_tokenizer.h`

ABI v1 intentionally does **not** expose llama.cpp or ggml internal C++ classes. This keeps the public Mercan ABI from being tied to one llama.cpp commit.

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

Plugins receive `mercan_metadata_v1`, a stable read-only view with:

- `has_key()`
- `get_string()`
- `get_i64()`
- `get_f64()`

Architecture-specific metadata should use a namespace owned by that architecture, for example:

```text
anka.block_count
anka.attention_type
anka.expert_count
```

Core Mercan metadata should stay under the `mercan.*` namespace.

## Tokenizer registry

A tokenizer provider uses `mercan_tokenizer_v1`. The descriptor may either:

1. set `MERCAN_TOKENIZER_BACKEND_MANAGED` and let the compiled backend tokenize, or
2. provide `create`, `destroy`, `encode`, and `decode_piece` callbacks.

NedoLM currently registers `ndsurf004` as a built-in backend-managed tokenizer because the exact NDSRF004 bridge is compiled into the patched llama backend.

## NedoLM

NedoLM is the first built-in Architecture SDK provider:

```text
general.architecture = nedolm
                 ↓
Mercan architecture registry
                 ↓
nedolm provider
                 ↓
NDSRF004 tokenizer provider
                 ↓
patched llama.cpp/ggml execution backend
```

The Mercan core no longer needs NedoLM-specific model-selection logic.

## Current v1 boundary

Architecture SDK v1 makes discovery, validation, tokenizer dispatch, metadata access, and registration extensible. Tensor graph execution is still backend-managed. Therefore a completely new neural architecture must currently provide both:

- a Mercan architecture provider, and
- support for that architecture in the compiled inference backend (today this is normally a llama.cpp patch/module).

This is deliberate. Exposing raw llama.cpp graph internals in ABI v1 would make third-party plugins break whenever llama.cpp changes.

A future graph ABI will introduce Mercan-owned tensor/graph abstractions before external `.so` architecture plugins are considered stable.

## Developer flow

1. Pick a unique architecture name, e.g. `anka`.
2. Implement `mercan_architecture_v1`.
3. Implement `mercan_tokenizer_v1` if the tokenizer is custom.
4. Add backend graph/tensor support without modifying Mercan model dispatch.
5. Write a converter that emits `.mercan` metadata and tensors.
6. Verify registration:

```bash
mercan arch list
mercan tokenizer list
```

7. Run the model:

```bash
mercan run ./anka-1b.mercan
```

See `examples/custom_arch/` for a minimal provider template.
