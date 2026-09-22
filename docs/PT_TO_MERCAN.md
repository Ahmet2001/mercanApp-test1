# Converting a PyTorch `.pt` Checkpoint to Mercan

This guide explains how to convert a compatible PyTorch checkpoint into a Mercan v1 model artifact (`.mercan`) using the tools in this repository.

> **Important:** a `.pt` filename only tells us that the file is a PyTorch checkpoint. It does **not** tell us the model architecture. The current converter is built for the Mercan/NedoLM architecture and NDSRF004 tokenizer. An arbitrary Llama, Mistral, Gemma, vision model, classifier, or unrelated PyTorch checkpoint cannot be converted correctly just by changing its file extension.

## What the converter produces

Mercan v1 uses a single self-contained `.mercan` model file.

The physical tensor container is GGUF v3, but the file also carries Mercan-specific metadata used by `libmercan`:

- `mercan.format = mercan`
- `mercan.format_version = 1`
- `mercan.runtime_abi = 1`
- `general.architecture = nedolm`
- NedoLM architecture metadata
- NDSRF004 tokenizer identity and embedded surface vocabulary
- the Mercan chat template
- model provenance
- the NedoLM tensor mapping required by the runtime

The converter is:

```text
scripts/mercan_convert.py
```

The format contract is documented in [`spec/MERCAN_FORMAT_V1.md`](../spec/MERCAN_FORMAT_V1.md).

---

## 1. Check whether the `.pt` file is compatible

Before converting, inspect the checkpoint.

Set the path:

```bash
export MODEL=/path/to/checkpoint.pt
```

Then run:

```bash
python3 - "$MODEL" <<'PY'
import sys
import torch

path = sys.argv[1]
obj = torch.load(path, map_location="cpu", weights_only=False)

print("top-level type:", type(obj).__name__)

if isinstance(obj, dict):
    print("top-level keys:", list(obj.keys())[:40])

    if isinstance(obj.get("model"), dict):
        state = obj["model"]
        print("using obj['model']")
    elif isinstance(obj.get("state_dict"), dict):
        state = obj["state_dict"]
        print("using obj['state_dict']")
    elif obj and all(torch.is_tensor(v) for v in obj.values()):
        state = obj
        print("using checkpoint itself as state_dict")
    else:
        state = None

    if state is not None:
        keys = list(state.keys())
        print("tensor count:", len(keys))
        print("\nfirst tensor keys:")
        for key in keys[:80]:
            value = state[key]
            shape = tuple(value.shape) if torch.is_tensor(value) else "-"
            print(f"  {key}: {shape}")
PY
```

The converter accepts these checkpoint layouts:

1. a dictionary containing `model: {tensor_name: tensor, ...}`
2. a dictionary containing `state_dict: {tensor_name: tensor, ...}`
3. a plain tensor state dictionary

It automatically strips common prefixes when the **entire** state dictionary uses them:

- `_orig_mod.`
- `module.`
- `model.`

### Quick NedoLM compatibility check

A compatible checkpoint should contain NedoLM/Llama-like transformer tensors such as:

```text
embed_tokens.weight
layers.0.attn_norm.weight
layers.0.attn.q_proj.weight
layers.0.attn.k_proj.weight
layers.0.attn.v_proj.weight
layers.0.attn.o_proj.weight
layers.0.ffn_norm.weight
...
norm.weight
```

Hugging Face-style aliases are also accepted for several tensors, for example:

```text
model.embed_tokens.weight
model.layers.0.input_layernorm.weight
model.layers.0.self_attn.q_proj.weight
model.layers.0.self_attn.k_proj.weight
model.layers.0.self_attn.v_proj.weight
model.layers.0.self_attn.o_proj.weight
model.layers.0.post_attention_layernorm.weight
model.norm.weight
```

The first `morph_layer_count` blocks are expected to use the NedoLM MorphFFN tensors:

```text
layers.N.ffn.gate_weight
layers.N.ffn.up_weight
layers.N.ffn.down_weight
```

Later non-MorphFFN blocks use the normal projected FFN form:

```text
layers.N.ffn.gate_proj.weight
layers.N.ffn.up_proj.weight
layers.N.ffn.down_proj.weight
```

If the checkpoint does not follow this architecture, do **not** force it through this converter. A new Mercan Architecture SDK provider/tensor mapping is required instead.

---

## 2. Current default NedoLM architecture

Unless `--arch-json` overrides it, `mercan_convert.py` assumes:

| Field | Value |
| --- | ---: |
| vocab_size | 32000 |
| context_length | 4096 |
| embedding_length | 1536 |
| block_count | 24 |
| feed_forward_length | 5632 |
| head_count | 12 |
| head_count_kv | 4 |
| head_dim | 128 |
| rms_epsilon | 1e-6 |
| sliding_window | 2048 |
| rope_freq_base | 1000000 |
| morph_layer_count | 18 |
| morph_shared_width | 4224 |
| morph_root_width | 704 |
| morph_suffix_width | 704 |
| tie_word_embeddings | true |

The MorphFFN widths must satisfy:

```text
morph_shared_width + morph_root_width + morph_suffix_width
= feed_forward_length
```

For the default model:

```text
4224 + 704 + 704 = 5632
```

The current exporter assumes tied word embeddings. A checkpoint with a separate untied output/`lm_head` needs exporter/runtime work before it should be treated as compatible.

---

## 3. Required tokenizer assets

NedoLM v1 uses the exact NDSRF004 tokenizer.

The conversion requires:

### `surface-vocab.bin`

This repository contains:

```text
runtime/NedoTokenizer/assets/surface-vocab.bin
```

The current converter requires its SHA-256 to be exactly:

```text
72412d981dac65a29d1767bc98821fc2bcffc2de53c534e7c719598515bfb600
```

Verify it:

```bash
sha256sum runtime/NedoTokenizer/assets/surface-vocab.bin
```

### `surface-vocab.txt`

The converter also needs the **matching textual vocabulary export** containing every token ID from `0` through `31999`.

This file is not currently committed to this repository. Use the `surface-vocab.txt` generated from the **same NDSRF004 tokenizer/vocabulary version** used for training.

Do not substitute a vocabulary from another tokenizer.

### Token-role table

MorphFFN requires one role entry per vocabulary item.

The converter first looks for either of these tensors inside the checkpoint:

```text
layers.0.ffn.token_roles
model.layers.0.ffn.token_roles
```

If one exists, it is used directly.

If it does not exist, provide the original role table with:

```text
--role-table /path/to/token_role_table.pt
```

The role table must contain exactly `vocab_size` entries.

---

## 4. Install the Python dependencies

A simple conversion requires Python and PyTorch.

Example:

```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch numpy
```

If converting a Hugging Face directory containing `.safetensors`, also install:

```bash
python -m pip install safetensors
```

For a normal `.pt` checkpoint, `safetensors` is not required.

---

## 5. Convert the `.pt` checkpoint to F16 Mercan

From the repository root:

```bash
export MODEL=/path/to/checkpoint.pt
export VOCAB_BIN="$PWD/runtime/NedoTokenizer/assets/surface-vocab.bin"
export VOCAB_TXT=/path/to/matching/surface-vocab.txt
export OUT="$PWD/mercan_export"

mkdir -p "$OUT"

python3 scripts/mercan_convert.py "$MODEL" \
  --output "$OUT/model-f16.mercan" \
  --vocab-bin "$VOCAB_BIN" \
  --vocab-txt "$VOCAB_TXT" \
  --model-name "My-Mercan-Model"
```

If the checkpoint does **not** embed `token_roles`, add:

```bash
  --role-table /path/to/token_role_table.pt
```

A successful conversion ends with:

```text
MERCAN_EXPORT_OK /.../model-f16.mercan
```

It also creates:

```text
model-f16.mercan
model-f16.mercan.json
SHA256SUMS
```

The JSON manifest records the model architecture, source checkpoint, output size, tensor count and SHA-256.

---

## 6. Override architecture values when necessary

If the checkpoint is still NedoLM but uses different dimensions, create an architecture JSON file.

Example `arch.json`:

```json
{
  "vocab_size": 32000,
  "context_length": 4096,
  "embedding_length": 1536,
  "block_count": 24,
  "feed_forward_length": 5632,
  "head_count": 12,
  "head_count_kv": 4,
  "head_dim": 128,
  "rms_epsilon": 0.000001,
  "sliding_window": 2048,
  "rope_freq_base": 1000000.0,
  "morph_layer_count": 18,
  "morph_shared_width": 4224,
  "morph_root_width": 704,
  "morph_suffix_width": 704,
  "tie_word_embeddings": true
}
```

Then:

```bash
python3 scripts/mercan_convert.py "$MODEL" \
  --output "$OUT/model-f16.mercan" \
  --vocab-bin "$VOCAB_BIN" \
  --vocab-txt "$VOCAB_TXT" \
  --arch-json ./arch.json \
  --model-name "My-Mercan-Model"
```

Changing these numbers does not magically make another architecture compatible. The tensor names, tensor shapes, graph implementation and tokenizer still need to match the registered NedoLM provider.

---

## 7. Create the smaller Q4_K_M deployment model

The direct converter writes an F16 deployment container for matrix weights (with 1-D tensors kept as F32).

For phone/desktop deployment, the repository's normal release format is Q4_K_M.

First prepare and build the pinned quantizer:

```bash
chmod +x scripts/prepare_llama.sh
./scripts/prepare_llama.sh

LLAMA_DIR="$PWD/build/_deps/llama-src"
TOOLS_BUILD="$PWD/build/llama-tools"

cmake -S "$LLAMA_DIR" -B "$TOOLS_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_NATIVE=OFF

cmake --build "$TOOLS_BUILD" --target llama-quantize -j "$(nproc)"
```

Quantize:

```bash
"$TOOLS_BUILD/bin/llama-quantize" \
  "$OUT/model-f16.mercan" \
  "$OUT/model.mercan" \
  Q4_K_M
```

Create a checksum:

```bash
sha256sum "$OUT/model.mercan" > "$OUT/model.mercan.sha256"
```

For Hugging Face and the Mercan CLI, `model.mercan` is the recommended final filename.

> `scripts/export_model.sh` performs the F16 + Q4_K_M pipeline, but its current default tokenizer/role-table paths point to the original training environment. On another machine, use the explicit commands in this guide unless those paths have been configured for your environment.

---

## 8. Validate the output

### Basic container check

```bash
xxd -l 8 "$OUT/model.mercan"
```

The file should begin with the GGUF magic:

```text
GGUF
```

That is expected: Mercan v1 uses GGUF v3 as its physical container.

### Verify the checksum

```bash
sha256sum "$OUT/model.mercan"
cat "$OUT/model.mercan.sha256"
```

### Build Mercan and load the model

```bash
chmod +x scripts/build_linux.sh
./scripts/build_linux.sh
```

Then smoke-test:

```bash
./dist/mercan-linux-x86_64/bin/mercan run "$OUT/model.mercan" \
  -p "Merhaba, kendini kısaca tanıt." \
  -n 64 \
  --temperature 0
```

Using `--temperature 0` makes this first test deterministic.

If model loading reaches generation successfully, the file has passed the most useful end-to-end runtime check.

---

## 9. Publish to Hugging Face

Install the client:

```bash
python -m pip install -U huggingface_hub
```

Log in using your normal Hugging Face authentication method, then:

```bash
python scripts/upload_hf.py "$OUT/model.mercan" \
  --repo OWNER/MODEL-REPO
```

The repository should expose the deployment artifact as:

```text
model.mercan
```

Then Mercan can resolve it as:

```bash
mercan run OWNER/MODEL-REPO
```

or:

```bash
mercan run OWNER/MODEL-REPO:model.mercan
```

---

## Common errors

### `unsupported PyTorch checkpoint: expected model/state_dict tensor mapping`

The `.pt` file is not stored in one of the layouts understood by `load_state_dict()`.

Inspect the top-level object and extract the actual model state dictionary before conversion.

### `none of the tensor aliases exist`

A required NedoLM tensor could not be found.

Possible causes:

- the checkpoint is not NedoLM
- the checkpoint uses different tensor names
- a model wrapper prefix was not stripped
- the architecture changed after the converter was written

Do not rename tensors blindly. Confirm the model graph and shapes first.

### `surface-vocab.txt IDs are not exactly 0..31999`

The text vocabulary is incomplete, has duplicated/missing IDs, or does not match the expected vocabulary size.

### `unexpected NDSRF004 SHA256`

The binary tokenizer vocabulary is not the exact tokenizer identity accepted by the current NedoLM runtime.

Do not bypass this validation unless the runtime/tokenizer provider is intentionally being upgraded too.

### `role table has ... entries; expected 32000`

The token-role table and model vocabulary do not belong to the same tokenizer/model configuration.

### Shape or graph errors after conversion

A file can be structurally writable while still being architecturally incompatible.

Check:

- hidden size
- layer count
- attention heads
- KV heads
- FFN dimensions
- MorphFFN layer count and widths
- tokenizer/vocabulary identity
- tensor orientation/shapes

If these do not match the NedoLM runtime implementation, the correct fix is to update the exporter and/or Architecture SDK provider—not to force the file through validation.

---

## What if the `.pt` model is not NedoLM?

The conversion path is then:

```text
.pt checkpoint
      |
      v
identify architecture + tokenizer
      |
      v
define Mercan architecture metadata
      |
      v
define tensor-name mapping
      |
      v
register mercan_architecture_v1 provider
      |
      v
register tokenizer provider if needed
      |
      v
implement/extend graph operations if required
      |
      v
write .mercan exporter
      |
      v
libmercan validation + inference
```

See:

- [Architecture SDK](ARCHITECTURE_SDK.md)
- [Graph ABI v1](GRAPH_ABI_V1.md)
- [Tensor ABI v1](TENSOR_ABI_V1.md)
- [Mercan Model Format v1](../spec/MERCAN_FORMAT_V1.md)

In other words, **`.pt → .mercan` is a model export process, not a file-extension conversion**. The checkpoint architecture and tokenizer must have a corresponding Mercan runtime implementation.
