#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="${1:?llama.cpp dizini gerekli}"
NEDO_FFI="${2:?libnedo004_ffi.a gerekli}"
OUT_XC="${3:?çıktı xcframework yolu gerekli}"

IOS_MIN="${IOS_MIN:-18.2}"
BUILD="$LLAMA_DIR/build-nedolm-ios-device"
FRAMEWORK="$BUILD/framework/llama.framework"

test -s "$NEDO_FFI"

cmake -S "$LLAMA_DIR" -B "$BUILD" -G Xcode \
  -DCMAKE_SYSTEM_NAME=iOS \
  -DCMAKE_OSX_SYSROOT=iphoneos \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="$IOS_MIN" \
  -DCMAKE_XCODE_ATTRIBUTE_SUPPORTED_PLATFORMS=iphoneos \
  -DCMAKE_XCODE_ATTRIBUTE_CODE_SIGNING_ALLOWED=NO \
  -DBUILD_SHARED_LIBS=OFF \
  -DGGML_METAL=ON \
  -DGGML_METAL_EMBED_LIBRARY=ON \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_TOOLS=OFF \
  -DLLAMA_BUILD_SERVER=OFF \
  -DLLAMA_CURL=OFF

# Build only the library target and its dependencies. ALL_BUILD also includes
# llama-app/common targets that are not part of the embedded iOS framework.
cmake --build "$BUILD" --config Release --target llama \
  -j "$(sysctl -n hw.logicalcpu)" -- CODE_SIGNING_ALLOWED=NO

# macOS ships Bash 3.2, which has no mapfile/readarray.
LLAMA_LIBS=()
while IFS= read -r lib; do
  LLAMA_LIBS+=("$lib")
done < <(find "$BUILD" -type f -name '*.a' -path '*Release*' -print | sort)

if [ "${#LLAMA_LIBS[@]}" -eq 0 ]; then
  while IFS= read -r lib; do
    LLAMA_LIBS+=("$lib")
  done < <(find "$BUILD" -type f -name '*.a' -print | sort)
fi

echo "Static libraries:"
printf '  %s\n' "${LLAMA_LIBS[@]}"
test "${#LLAMA_LIBS[@]}" -gt 0

TMP="$BUILD/nedolm-combine"
rm -rf "$TMP" "$FRAMEWORK" "$OUT_XC"
mkdir -p "$TMP" "$FRAMEWORK/Headers" "$FRAMEWORK/Modules"

# Merge llama/ggml objects + exact NDSRF004 Rust staticlib.
xcrun libtool -static -o "$TMP/combined.a" "${LLAMA_LIBS[@]}" "$NEDO_FFI"

cp "$LLAMA_DIR/include/llama.h" "$FRAMEWORK/Headers/"
for h in ggml.h ggml-alloc.h ggml-backend.h ggml-cpu.h ggml-metal.h ggml-blas.h gguf.h; do
  if [ -f "$LLAMA_DIR/ggml/include/$h" ]; then
    cp "$LLAMA_DIR/ggml/include/$h" "$FRAMEWORK/Headers/"
  fi
done

# Match the Clang module contract used by Silo's pinned llama.xcframework.
cat > "$FRAMEWORK/Modules/module.modulemap" <<'EOF'
framework module llama {
    header "llama.h"
    header "ggml.h"
    header "ggml-alloc.h"
    header "ggml-backend.h"
    header "ggml-metal.h"
    header "ggml-cpu.h"
    header "ggml-blas.h"
    header "gguf.h"

    link "c++"
    link framework "Accelerate"
    link framework "Metal"
    link framework "Foundation"

    export *
}
EOF

for required_header in llama.h ggml.h ggml-alloc.h ggml-backend.h ggml-metal.h ggml-cpu.h ggml-blas.h gguf.h; do
  test -s "$FRAMEWORK/Headers/$required_header" || {
    echo "Missing framework header: $required_header" >&2
    exit 10
  }
done

cat > "$FRAMEWORK/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>llama</string>
  <key>CFBundleIdentifier</key><string>org.mercan.nedolm.llama</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>llama</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>MinimumOSVersion</key><string>${IOS_MIN}</string>
  <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
  <key>UIDeviceFamily</key><array><integer>1</integer><integer>2</integer></array>
</dict>
</plist>
EOF

SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
xcrun --sdk iphoneos clang++ -dynamiclib \
  -isysroot "$SDK" \
  -arch arm64 \
  -mios-version-min="$IOS_MIN" \
  -Wl,-force_load,"$TMP/combined.a" \
  -framework Foundation \
  -framework Metal \
  -framework Accelerate \
  -lc++ \
  -install_name '@rpath/llama.framework/llama' \
  -o "$FRAMEWORK/llama"

xcrun vtool \
  -set-build-version ios "$IOS_MIN" "$IOS_MIN" \
  -replace \
  -output "$FRAMEWORK/llama.vtool" \
  "$FRAMEWORK/llama"
mv "$FRAMEWORK/llama.vtool" "$FRAMEWORK/llama"

mkdir -p "$(dirname "$OUT_XC")"
xcodebuild -create-xcframework \
  -framework "$FRAMEWORK" \
  -output "$OUT_XC"

# Fail early if Swift cannot import the same module Silo imports.
DEVICE_FRAMEWORK="$(find "$OUT_XC" -type d -name 'llama.framework' -print -quit)"
test -n "$DEVICE_FRAMEWORK"
cat > "$TMP/import-llama.swift" <<'EOF'
import llama
EOF
xcrun swiftc \
  -target "arm64-apple-ios${IOS_MIN}" \
  -sdk "$SDK" \
  -F "$(dirname "$DEVICE_FRAMEWORK")" \
  -typecheck "$TMP/import-llama.swift"

echo "Created and Swift-import verified: $OUT_XC"
