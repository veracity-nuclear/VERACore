#!/bin/bash
# ============================================================
# VeraCore — one-shot macOS sign / notarize / staple / package
# ============================================================
# Run from the repo root, AFTER `pyinstaller VeraCore.spec` has produced
# dist/VeraCore.app.
#
# Usage:
#   ./build_sign_mac.sh "Developer ID Application: YOUR NAME (TEAM ID)" veracore-notary
#
#   arg 1 = your signing identity (exact string from: security find-identity -v -p codesigning)
#   arg 2 = the notarytool keychain profile you stored earlier
# ============================================================

set -euo pipefail

# ---- arguments ----
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 \"Developer ID Application: NAME (TEAMID)\" <notary-profile>"
  exit 1
fi
IDENTITY="$1"
PROFILE="$2"
APP="dist/VeraCore.app"
ZIP="VeraCore.zip"
DMG="VeraCore.dmg"

# ---- sanity checks ----
if [ ! -d "$APP" ]; then
  echo "ERROR: $APP not found. Build it first: pyinstaller VeraCore.spec --noconfirm --clean"
  exit 1
fi

submit_and_check () {
  # $1 = file to submit. Streams result; aborts the script if not Accepted.
  local target="$1"
  echo ">> Submitting $target to Apple notary service (this can take a few minutes)..."
  local out
  out="$(xcrun notarytool submit "$target" --keychain-profile "$PROFILE" --wait 2>&1)"
  echo "$out"
  if ! echo "$out" | grep -q "status: Accepted"; then
    local subid
    subid="$(echo "$out" | awk '/id:/{print $2; exit}')"
    echo ""
    echo "!! Notarization was NOT accepted for $target."
    echo "!! Read the detailed log with:"
    echo "     xcrun notarytool log $subid --keychain-profile \"$PROFILE\""
    echo "!! Fix the flagged binaries, then re-run this script."
    exit 1
  fi
  echo ">> $target: Accepted."
}

echo "=== STEP 1/2: signing nested binaries, then the .app ==="
find "$APP" \( -name "*.dylib" -o -name "*.so" \) -type f -print0 \
  | while IFS= read -r -d '' f; do
      codesign --force --options runtime --timestamp --sign "$IDENTITY" "$f"
    done
codesign --force --options runtime --timestamp --sign "$IDENTITY" "$APP"

echo "=== STEP 3: verifying .app signature ==="
codesign --verify --deep --strict --verbose=2 "$APP"

echo "=== STEP 4: zipping .app for notarization ==="
rm -f "$ZIP"
ditto -c -k --keepParent "$APP" "$ZIP"

echo "=== STEP 5: notarizing the .app ==="
submit_and_check "$ZIP"

echo "=== STEP 6: stapling the .app ==="
xcrun stapler staple "$APP"
xcrun stapler validate "$APP"

echo "=== STEP 7: building the .dmg ==="
rm -f "$DMG"
hdiutil create -volname "VeraCore" -srcfolder "$APP" -ov -format UDZO "$DMG"

echo "=== STEP 8: signing + notarizing + stapling the .dmg ==="
codesign --force --timestamp --sign "$IDENTITY" "$DMG"
submit_and_check "$DMG"
xcrun stapler staple "$DMG"

echo "=== STEP 9: final verification ==="
xcrun stapler validate "$DMG"
spctl --assess --type open --context context:primary-signature -v "$DMG" || true

echo ""
echo "DONE. Distributable: $DMG"
echo "Test it on a CLEAN Mac: download via a browser, open, drag VeraCore.app to"
echo "Applications, and launch. It should open with no warning and no terminal."
