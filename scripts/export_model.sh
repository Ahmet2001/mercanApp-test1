#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CHECKPOINT="${1:-/arf/scratch/egitimg16/nedolm_0p8b/runs/nedolm-0p8b-sft-full-v1/checkpoints/step_00023334.pt}"
OUT_DIR="${2:-/arf/scratch/egitimg16/nedolm_0p8b/mercan_export}"
F16="$OUT_DIR/Mercan-0.8B-SFT-F16.mercan"
Q4="$OUT_DIR/model.mercan"

mkdir -p "$OUT_DIR"
python3 "$ROOT/scripts/mercan_convert.py" "$CHECKPOINT" --output "$F16" --model-name "Mercan-0.8B-SFT"

"$ROOT/scripts/prepare_llama.sh"
LLAMA_DIR="$ROOT/build/_deps/llama-src"
TOOLS_BUILD="$ROOT/build/llama-tools"
cmake -S "$LLAMA_DIR" -B "$TOOLS_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DGGML_NATIVE=OFF
cmake --build "$TOOLS_BUILD" --target llama-quantize -j "${MERCAN_JOBS:-$(nproc)}"

"$TOOLS_BUILD/bin/llama-quantize" "$F16" "$Q4" Q4_K_M
sha256sum "$Q4" > "$OUT_DIR/model.mercan.sha256"

printf 'F16=%s\nQ4=%s\n' "$F16" "$Q4"
