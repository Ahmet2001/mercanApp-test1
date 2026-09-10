# Mercan Model Format v1

Mercan v1 is the portable deployment format used by the Mercan runtime and CLI.

## Container

A `.mercan` file is a single self-contained model artifact. Mercan v1 uses GGUF v3 as its physical tensor container so the runtime can reuse ggml/llama.cpp tensor loading while exposing a Mercan-specific compatibility contract.

The file MUST start with the GGUF magic and MUST use GGUF version 3.

A `.mercan` file MUST NOT contain executable model-bundled code. Runtime behavior is selected by metadata and implemented by libmercan.

## Required Mercan metadata

- `mercan.format = "mercan"`
- `mercan.format_version = 1`
- `mercan.runtime_abi = 1`
- `mercan.model_family`
- `mercan.tokenizer.spec`
- `mercan.tokenizer.surface_vocab_sha256`
- `mercan.tokenizer.surface_vocab` as `ARRAY<UINT8>`

For NedoLM/Mercan 0.8B v1 the tokenizer spec is `NDSRF004`.

## NedoLM architecture metadata

- `general.architecture = "nedolm"`
- `nedolm.vocab_size`
- `nedolm.context_length`
- `nedolm.embedding_length`
- `nedolm.block_count`
- `nedolm.feed_forward_length`
- `nedolm.attention.head_count`
- `nedolm.attention.head_count_kv`
- `nedolm.attention.layer_norm_rms_epsilon`
- `nedolm.attention.sliding_window`
- `nedolm.rope.dimension_count`
- `nedolm.rope.freq_base`
- `nedolm.morph.layer_count`
- `nedolm.morph.shared_width`
- `nedolm.morph.root_width`
- `nedolm.morph.suffix_width`
- `nedolm.vocab_sha256`

The MorphFFN widths MUST sum to `nedolm.feed_forward_length` and the MorphFFN layer count MUST NOT exceed the transformer block count.

## Tokenizer

The tokenizer is data, not executable code. `mercan.tokenizer.surface_vocab` contains the exact serialized surface vocabulary bytes required by the tokenizer implementation selected by `mercan.tokenizer.spec`.

The runtime MUST verify `mercan.tokenizer.surface_vocab_sha256` before using the tokenizer asset.

Mercan v1 NDSRF004 models use:

- PAD = 0
- BOS = 1
- EOS = 2

## Chat template

Mercan 0.8B SFT models use `chatml_tr`:

```
<|im_start|>{role}\n{content}<|im_end|>\n
```

The training corpus adds EOS once at the end of a conversation. The runtime/CLI may construct prompts from this template but the model file remains the source of tokenizer and architecture truth.

## Tensor names

Mercan v1 NedoLM tensor names follow the llama.cpp-compatible NedoLM mapping, including:

- `token_embd.weight`
- `blk.N.attn_norm.weight`
- `blk.N.attn_q.weight`
- `blk.N.attn_k.weight`
- `blk.N.attn_v.weight`
- `blk.N.attn_output.weight`
- `blk.N.ffn_norm.weight`
- MorphFFN or standard FFN weights for each block
- `output_norm.weight`
- `blk.0.ffn_gate_tid2eid.weight` for the token-role table

## Compatibility

`mercan.runtime_abi` is the native runtime ABI generation. A runtime MUST reject an unsupported ABI or unsupported format version instead of guessing.

GGUF is an implementation detail of Mercan v1. Future `.mercan` versions may change the physical container while preserving the higher-level Mercan runtime contract.

## Distribution

The intended user flow is:

```bash
mercan run model.mercan
mercan run owner/model-repo
```

For Hugging Face repositories, the default artifact name is `model.mercan`. A repository can also be addressed as `owner/repo:filename.mercan`.

Server functionality is out of scope for v1. The CLI and `libmercan` native runtime are in scope.
