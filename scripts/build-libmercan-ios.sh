#!/usr/bin/env bash
set -euo pipefail

LLAMA_DIR="${1:?patched llama.cpp directory required}"
NEDO_FFI="${2:?libnedo004_ffi.a required}"
OUT_XC="${3:?output xcframework path required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_MIN="${IOS_MIN:-18.2}"
BUILD="$ROOT/build/ios/libmercan-device"
FRAMEWORK="$BUILD/framework/MercanRuntime.framework"

test -s "$NEDO_FFI"
test -s "$ROOT/runtime/libmercan/include/mercan.h"
test -s "$LLAMA_DIR/CMakeLists.txt"

rm -rf "$BUILD" "$OUT_XC"
mkdir -p "$BUILD"

cmake -S "$ROOT" -B "$BUILD" -G Xcode \
  -DMERCAN_LLAMA_DIR="$LLAMA_DIR" \
  -DMERCAN_BUILD_SHARED=OFF \
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

cmake --build "$BUILD" --config Release --target libmercan \
  -j "$(sysctl -n hw.logicalcpu)" -- CODE_SIGNING_ALLOWED=NO

MERCAN_LIB="$(find "$BUILD" -type f -name 'libmercan.a' -print -quit)"
test -s "$MERCAN_LIB"

STATIC_LIBS=("$MERCAN_LIB")
while IFS= read -r lib; do
  [ "$lib" = "$MERCAN_LIB" ] && continue
  STATIC_LIBS+=("$lib")
done < <(find "$BUILD/llama" -type f -name '*.a' -print | sort)
STATIC_LIBS+=("$NEDO_FFI")

echo "MercanRuntime static inputs:"
printf '  %s\n' "${STATIC_LIBS[@]}"

TMP="$BUILD/combined"
rm -rf "$TMP" "$FRAMEWORK"
mkdir -p "$TMP" "$FRAMEWORK/Headers" "$FRAMEWORK/Modules"

xcrun libtool -static -o "$TMP/libMercanRuntimeCombined.a" "${STATIC_LIBS[@]}"

cp "$ROOT"/runtime/libmercan/include/*.h "$FRAMEWORK/Headers/"

cat > "$FRAMEWORK/Modules/module.modulemap" <<'EOF'
framework module MercanRuntime {
    umbrella header "mercan.h"
    export *
    module * { export * }
}
EOF

cat > "$FRAMEWORK/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleDevelopmentRegion</key><string>en</string>
  <key>CFBundleExecutable</key><string>MercanRuntime</string>
  <key>CFBundleIdentifier</key><string>org.mercan.runtime</string>
  <key>CFBundleInfoDictionaryVersion</key><string>6.0</string>
  <key>CFBundleName</key><string>MercanRuntime</string>
  <key>CFBundlePackageType</key><string>FMWK</string>
  <key>CFBundleShortVersionString</key><string>0.1.2</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>MinimumOSVersion</key><string>$IOS_MIN</string>
  <key>CFBundleSupportedPlatforms</key><array><string>iPhoneOS</string></array>
  <key>UIDeviceFamily</key><array><integer>1</integer></array>
</dict>
</plist>
EOF

SDK="$(xcrun --sdk iphoneos --show-sdk-path)"
xcrun --sdk iphoneos clang++ -dynamiclib \
  -isysroot "$SDK" \
  -arch arm64 \
  -mios-version-min="$IOS_MIN" \
  -Wl,-force_load,"$TMP/libMercanRuntimeCombined.a" \
  -framework Foundation \
  -framework Metal \
  -framework MetalKit \
  -framework Accelerate \
  -lc++ \
  -install_name '@rpath/MercanRuntime.framework/MercanRuntime' \
  -o "$FRAMEWORK/MercanRuntime"

xcrun vtool \
  -set-build-version ios "$IOS_MIN" "$IOS_MIN" \
  -replace \
  -output "$FRAMEWORK/MercanRuntime.vtool" \
  "$FRAMEWORK/MercanRuntime"
mv "$FRAMEWORK/MercanRuntime.vtool" "$FRAMEWORK/MercanRuntime"

mkdir -p "$(dirname "$OUT_XC")"
xcodebuild -create-xcframework \
  -framework "$FRAMEWORK" \
  -output "$OUT_XC"

test -s "$OUT_XC/ios-arm64/MercanRuntime.framework/MercanRuntime"
file "$OUT_XC/ios-arm64/MercanRuntime.framework/MercanRuntime"
lipo -info "$OUT_XC/ios-arm64/MercanRuntime.framework/MercanRuntime"

echo "Created libmercan iPhone framework: $OUT_XC"
