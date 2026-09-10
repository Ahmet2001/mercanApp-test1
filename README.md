# Mercan

Mercan is a native local inference stack for Mercan/NedoLM models. The user-facing workflow is intentionally simple:

```bash
mercan run model.mercan
```

or directly from Hugging Face:

```bash
mercan run MercanAI/Mercan-0.8B-SFT
```

The Hugging Face form downloads `model.mercan` once into `~/.cache/mercan` and reuses the cached copy on later launches. On Linux, the installer automatically selects the CUDA build when a working NVIDIA GPU/driver is detected with `nvidia-smi`; otherwise it installs the CPU build.

## Install on Linux x86_64

Once the GitHub release is available:

```bash
curl -fsSL https://raw.githubusercontent.com/Ahmet2001/mercanApp-test1/main/install.sh | sh
```

The installer automatically chooses CUDA on NVIDIA systems and CPU elsewhere. Force CPU with `MERCAN_FORCE_CPU=1`.

Then:

```bash
mercan --version
mercan run MercanAI/Mercan-0.8B-SFT
```

Without root access:

```bash
MERCAN_PREFIX="$HOME/.local" ./install.sh
export PATH="$HOME/.local/bin:$PATH"
```

## CLI

```text
mercan run <model.mercan|owner/repo[:file.mercan]>
mercan pull <owner/repo[:file.mercan]>
mercan --version
```

Examples:

```bash
mercan run ./model.mercan
mercan run MercanAI/Mercan-0.8B-SFT
mercan run MercanAI/Mercan-0.8B-SFT:model.mercan -p "Merhaba"
```

Useful run options:

```text
-p, --prompt TEXT
-n, --max-tokens N
--temperature F
--top-k N
--top-p F
-t, --threads N
--gpu-layers N   # -1 = all; CUDA package defaults to all layers
```

## Architecture

```text
PyTorch / Hugging Face checkpoint
              |
              v
       mercan_convert.py
              |
              v
         model.mercan
              |
              v
           mercan CLI
              |
              v
          libmercan C ABI
              |
              v
 patched llama.cpp / ggml
              |
              v
 NedoLM + exact NDSRF004 tokenizer
```

`.mercan` v1 is a self-contained deployment artifact. It uses GGUF v3 as its physical tensor container but adds a Mercan compatibility contract and embeds the exact tokenizer surface-vocabulary bytes. No executable model-bundled code is stored in the model file.

See `spec/MERCAN_FORMAT_V1.md`.

## Build from source

Required tools:

- CMake >= 3.16
- C/C++ compiler with C++17 support
- Rust/Cargo
- Git
- curl

Build:

```bash
git clone https://github.com/Ahmet2001/mercanApp-test1.git
cd mercanApp-test1
chmod +x scripts/*.sh install.sh
./scripts/build_linux.sh
./dist/mercan-linux-x86_64/bin/mercan --version

# CUDA build (requires CUDA toolkit)
MERCAN_CUDA=1 MERCAN_DIST_NAME=mercan-linux-x86_64-cuda ./scripts/build_linux.sh
```

The build script checks out the pinned llama.cpp commit, builds the NDSRF004 Rust static bridge, applies `runtime/nedolm/nedolm-llama.patch`, copies the NedoLM model implementation and builds `libmercan` plus the `mercan` CLI. The release CLI links the llama/ggml backend statically so users do not need separate backend shared libraries.

## Repository layout

```text
cli/main.cpp                         user-facing CLI
runtime/libmercan/                   stable native C ABI
runtime/nedolm/nedolm.cpp            NedoLM graph implementation
runtime/nedolm/nedolm-llama.patch    llama.cpp integration
runtime/nedo004-ffi/                 exact tokenizer bridge
runtime/NedoTokenizer/               tokenizer implementation/assets
scripts/mercan_convert.py            PyTorch/HF -> .mercan converter
scripts/prepare_llama.sh             pinned backend preparation
scripts/build_linux.sh               Linux build/package
scripts/export_model.sh              F16 export + Q4_K_M deployment artifact
scripts/upload_hf.py                 Hugging Face publisher
spec/MERCAN_FORMAT_V1.md             model format contract
```

## Export the final Mercan model

On a machine with the final checkpoint and PyTorch:

```bash
./scripts/export_model.sh /path/to/step_00023334.pt /path/to/output
```

This produces an F16 `.mercan` intermediate and a Q4_K_M deployment artifact named `model.mercan`.

## Publish to Hugging Face

With a valid Hugging Face login/token available to `huggingface_hub`:

```bash
python -m pip install -U huggingface_hub
python scripts/upload_hf.py /path/to/model.mercan --repo MercanAI/Mercan-0.8B-SFT
```

The CLI expects `model.mercan` as the default artifact when a Hugging Face repository name is passed to `mercan run`.

## Current v1 scope

Mercan v1 includes the portable model format, converter, exact tokenizer bridge, native runtime ABI, local/Hugging Face CLI workflow and Linux packaging. An HTTP server/API layer is intentionally left for a later stage.
