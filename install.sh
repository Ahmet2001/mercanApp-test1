#!/bin/sh
set -eu

REPO="${MERCAN_GITHUB_REPO:-Ahmet2001/mercanApp-test1}"
VERSION="${MERCAN_VERSION:-latest}"
PREFIX="${MERCAN_PREFIX:-/usr/local}"
ASSET="mercan-linux-x86_64.tar.gz"

command -v curl >/dev/null 2>&1 || { echo "Mercan installer requires curl." >&2; exit 1; }
command -v tar >/dev/null 2>&1 || { echo "Mercan installer requires tar." >&2; exit 1; }
command -v install >/dev/null 2>&1 || { echo "Mercan installer requires install (coreutils)." >&2; exit 1; }

ARCH="$(uname -m)"
case "$ARCH" in
  x86_64|amd64) ;;
  *) echo "Mercan v0.1.0 currently supports Linux x86_64 only (detected: $ARCH)." >&2; exit 1 ;;
esac

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT HUP INT TERM

if [ "$VERSION" = "latest" ]; then
  URL="https://github.com/$REPO/releases/latest/download/$ASSET"
else
  URL="https://github.com/$REPO/releases/download/$VERSION/$ASSET"
fi

echo "Downloading Mercan CLI from $REPO ($VERSION)..."
curl -fL --retry 3 "$URL" -o "$TMP/$ASSET"
tar -xzf "$TMP/$ASSET" -C "$TMP"
SRC="$TMP/mercan-linux-x86_64"

if [ ! -x "$SRC/bin/mercan" ]; then
  echo "Mercan release archive is invalid: bin/mercan not found." >&2
  exit 1
fi

SUDO=""
if [ -d "$PREFIX" ]; then
  if [ ! -w "$PREFIX" ]; then SUDO="sudo"; fi
else
  PARENT="$(dirname "$PREFIX")"
  if [ ! -w "$PARENT" ]; then SUDO="sudo"; fi
fi

if [ -n "$SUDO" ]; then
  command -v sudo >/dev/null 2>&1 || {
    echo "No permission to install into $PREFIX and sudo is unavailable." >&2
    echo "Try a user-writable prefix, for example MERCAN_PREFIX=$HOME/.local" >&2
    exit 1
  }
fi

$SUDO mkdir -p "$PREFIX/bin" "$PREFIX/lib" "$PREFIX/include"
$SUDO install -m 0755 "$SRC/bin/mercan" "$PREFIX/bin/mercan"
if [ -f "$SRC/lib/libmercan.a" ]; then
  $SUDO install -m 0644 "$SRC/lib/libmercan.a" "$PREFIX/lib/libmercan.a"
fi
if [ -f "$SRC/include/mercan.h" ]; then
  $SUDO install -m 0644 "$SRC/include/mercan.h" "$PREFIX/include/mercan.h"
fi

echo "Installed Mercan CLI: $PREFIX/bin/mercan"
if command -v mercan >/dev/null 2>&1; then
  mercan --version
else
  echo "If mercan is not found, add $PREFIX/bin to PATH."
fi
echo "Run: mercan run MercanAI/Mercan-0.8B-SFT"
