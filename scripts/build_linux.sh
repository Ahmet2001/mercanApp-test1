#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${MERCAN_BUILD_DIR:-$ROOT/build/release}"
DIST_DIR="${MERCAN_DIST_DIR:-$ROOT/dist/mercan-linux-x86_64}"
JOBS="${MERCAN_JOBS:-$(nproc)}"

"$ROOT/scripts/prepare_llama.sh"

cmake -S "$ROOT" -B "$BUILD_DIR" \
  -DCMAKE_BUILD_TYPE=Release \
  -DMERCAN_LLAMA_DIR="$ROOT/build/_deps/llama-src" \
  -DMERCAN_BUILD_SHARED=OFF
cmake --build "$BUILD_DIR" --target mercan -j "$JOBS"

rm -rf "$DIST_DIR"
cmake --install "$BUILD_DIR" --prefix "$DIST_DIR"

mkdir -p "$ROOT/dist"
tar -C "$(dirname "$DIST_DIR")" -czf "$ROOT/dist/mercan-linux-x86_64.tar.gz" "$(basename "$DIST_DIR")"
sha256sum "$ROOT/dist/mercan-linux-x86_64.tar.gz" > "$ROOT/dist/mercan-linux-x86_64.tar.gz.sha256"

echo "Built: $DIST_DIR/bin/mercan"
echo "Package: $ROOT/dist/mercan-linux-x86_64.tar.gz"
