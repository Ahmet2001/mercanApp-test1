#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST_NAME="${MERCAN_DIST_NAME:-mercan-linux-x86_64}"
BUILD_DIR="${MERCAN_BUILD_DIR:-$ROOT/build/release}"
DIST_DIR="${MERCAN_DIST_DIR:-$ROOT/dist/$DIST_NAME}"
JOBS="${MERCAN_JOBS:-$(nproc)}"
CUDA_ENABLED="${MERCAN_CUDA:-0}"

"$ROOT/scripts/prepare_llama.sh"

CMAKE_ARGS=(
  -S "$ROOT"
  -B "$BUILD_DIR"
  -DCMAKE_BUILD_TYPE=Release
  -DMERCAN_LLAMA_DIR="$ROOT/build/_deps/llama-src"
  -DMERCAN_BUILD_SHARED=OFF
)

if [[ "$CUDA_ENABLED" != "0" ]]; then
  CMAKE_ARGS+=(
    -DGGML_CUDA=ON
    -DGGML_STATIC=ON
    -DGGML_CUDA_NCCL=OFF
    -DGGML_CUDA_NO_VMM=ON
  )
  if [[ -n "${MERCAN_CUDA_ARCHITECTURES:-}" ]]; then
    CMAKE_ARGS+=("-DCMAKE_CUDA_ARCHITECTURES=${MERCAN_CUDA_ARCHITECTURES}")
  fi
fi

cmake "${CMAKE_ARGS[@]}"
cmake --build "$BUILD_DIR" --target mercan -j "$JOBS"

rm -rf "$DIST_DIR"
cmake --install "$BUILD_DIR" --prefix "$DIST_DIR"

mkdir -p "$ROOT/dist"
PACKAGE="$ROOT/dist/$DIST_NAME.tar.gz"
tar -C "$(dirname "$DIST_DIR")" -czf "$PACKAGE" "$(basename "$DIST_DIR")"
sha256sum "$PACKAGE" > "$PACKAGE.sha256"

echo "Built: $DIST_DIR/bin/mercan"
echo "Package: $PACKAGE"
if [[ "$CUDA_ENABLED" != "0" ]]; then
  echo "Backend package: CUDA"
else
  echo "Backend package: CPU"
fi
