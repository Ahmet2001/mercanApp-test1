#!/usr/bin/env bash
set -euo pipefail
: "${IOS_BUNDLE_ID:?IOS_BUNDLE_ID is required}"
: "${APPLE_TEAM_ID:?APPLE_TEAM_ID is required}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
APP_DIR="$ROOT/ios-app"
WORK="$ROOT/build/ios"
ARCHIVE="$WORK/Mercan.xcarchive"
EXPORT_DIR="$WORK/export"
EXPORT_PLIST="$WORK/ExportOptions.plist"
BUILD_NUMBER="${IOS_BUILD_NUMBER:-1}"
MARKETING_VERSION="${IOS_MARKETING_VERSION:-0.1.0}"

# The provisioning-profile download action installs profiles using their UUID
# as the filename. Do not assume the profile's display name. Resolve the
# freshly installed profile by its application identifier and team.
PROFILE_DIR="$HOME/Library/MobileDevice/Provisioning Profiles"
PROFILE_NAME="${IOS_PROFILE_NAME:-}"
PROFILE_UUID=""

if [[ -z "$PROFILE_NAME" ]]; then
  shopt -s nullglob
  for profile in "$PROFILE_DIR"/*.mobileprovision; do
    plist="$(mktemp)"
    if ! security cms -D -i "$profile" > "$plist" 2>/dev/null; then
      rm -f "$plist"
      continue
    fi

    application_identifier="$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:application-identifier' "$plist" 2>/dev/null || true)"
    profile_team="$(/usr/libexec/PlistBuddy -c 'Print :Entitlements:com.apple.developer.team-identifier' "$plist" 2>/dev/null || true)"

    if [[ "$application_identifier" == "$APPLE_TEAM_ID.$IOS_BUNDLE_ID" && "$profile_team" == "$APPLE_TEAM_ID" ]]; then
      PROFILE_NAME="$(/usr/libexec/PlistBuddy -c 'Print :Name' "$plist")"
      PROFILE_UUID="$(/usr/libexec/PlistBuddy -c 'Print :UUID' "$plist")"
      rm -f "$plist"
      break
    fi

    rm -f "$plist"
  done
fi

if [[ -z "$PROFILE_NAME" ]]; then
  echo "No installed App Store provisioning profile matches $APPLE_TEAM_ID/$IOS_BUNDLE_ID" >&2
  exit 1
fi

echo "Using provisioning profile: $PROFILE_NAME${PROFILE_UUID:+ ($PROFILE_UUID)}"

rm -rf "$ARCHIVE" "$EXPORT_DIR"
mkdir -p "$WORK"

xcodebuild \
  -project "$APP_DIR/Mercan.xcodeproj" \
  -scheme Mercan \
  -configuration Release \
  -sdk iphoneos \
  -destination 'generic/platform=iOS' \
  -archivePath "$ARCHIVE" \
  DEVELOPMENT_TEAM="$APPLE_TEAM_ID" \
  PRODUCT_BUNDLE_IDENTIFIER="$IOS_BUNDLE_ID" \
  CODE_SIGN_STYLE=Manual \
  CODE_SIGN_IDENTITY="Apple Distribution" \
  PROVISIONING_PROFILE_SPECIFIER="$PROFILE_NAME" \
  CURRENT_PROJECT_VERSION="$BUILD_NUMBER" \
  MARKETING_VERSION="$MARKETING_VERSION" \
  archive

cat > "$EXPORT_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>method</key>
  <string>app-store-connect</string>
  <key>provisioningProfiles</key>
  <dict>
    <key>$IOS_BUNDLE_ID</key>
    <string>$PROFILE_NAME</string>
  </dict>
  <key>signingStyle</key>
  <string>manual</string>
  <key>teamID</key>
  <string>$APPLE_TEAM_ID</string>
</dict>
</plist>
EOF

xcodebuild -exportArchive \
  -archivePath "$ARCHIVE" \
  -exportPath "$EXPORT_DIR" \
  -exportOptionsPlist "$EXPORT_PLIST"

IPA="$(find "$EXPORT_DIR" -maxdepth 1 -name '*.ipa' -print -quit)"
test -n "$IPA"
cp "$IPA" "$WORK/Mercan-AppStore.ipa"
shasum -a 256 "$WORK/Mercan-AppStore.ipa"
