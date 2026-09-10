#!/usr/bin/env bash
set -euo pipefail

REPO="${MERCAN_GITHUB_REPO:-Ahmet2001/mercanApp-test1}"
VERSION="${MERCAN_VERSION:-latest}"
PREFIX="${MERCAN_PREFIX:-/usr/local}"
ASSET="mercan-linux-x86_64.tar.gz"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

if [[ "$VERSION" == "latest" ]]; then
  URL="https://github.com/$REPO/releases/latest/download/$ASSET"
else
  URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
fi

echo "Downloading Mercan CLI from $REPO ($VERSION)..."
curl -fL --retry 3 "$URL" -o "$TMP/$ASSET"
tar -xzf "$TMP/$ASSET" -C "$TMP"
SRC="$TMP/mercan-linux-x86_64"

install_to_prefix() {
  mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/include"
  install -m 0755 "$SRC/bin/mercan" "$PREFIX/bin/mercan"
  if [[ -f "$SRC/lib/libmercan.a" ]]; then install -m 0644 "$SRC/lib/libmercan.a" "$PREFIX/lib/libmercan.a"; fi
  if [[ -f "$SRC/include/mercan.h" ]]; then install -m 0644 "$SRC/include/mercan.h" "$PREFIX/include/mercan.h"; fi
}

if [[ -w "$PREFIX" || ( ! -e "$PREFIX" && -w "$(dirname "$PREFIX")" ) ]]; then
  install_to_prefix
else
  sudo mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/include"
  sudo install -m 0755 "$SRC/bin/mercan" "$PREFIX/bin/mercan"
  [[ ! -f "$SRC/lib/libmercan.a" ]] || sudo install -m 0644 "$SRC/lib/libmercan.a" "$PREFIX/lib/libmercan.a"
  [[ ! -f "$SRC/include/mercan.h" ]] || sudo install -m 0644 "$SRC/include/mercan.h" "$PREFIX/include/mercan.h"
fi

echo "Installed Mercan CLI: $PREFIX/bin/mercan"
echo "Try: mercan --version"
