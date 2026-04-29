#!/usr/bin/env bash
# ============================================================
#  build_macos.sh — Build iColoriT LoRA macOS .app + DMG
#  Run from the repo root with the venv activated.
#
#  Prerequisites (run once):
#    pip install pyinstaller
#    pip install -r requirements.txt
#    brew install create-dmg   # optional, for DMG packaging
#
#  Usage:
#    chmod +x build_macos.sh
#    ./build_macos.sh
#
#  Output:
#    dist/iColoriT LoRA.app
#    dist/iColoriT_LoRA_mac.dmg  (if create-dmg is available)
# ============================================================

set -euo pipefail

APP_NAME="iColoriT LoRA"
BUNDLE_ID="com.icolorit.lora"
DIST_DIR="dist"
APP_BUNDLE="$DIST_DIR/$APP_NAME.app"
DMG_OUT="$DIST_DIR/iColoriT_LoRA_mac.dmg"

echo "============================================================"
echo "  iColoriT LoRA — macOS build"
echo "============================================================"
echo

# ── 1. Verify Python & PyInstaller ───────────────────────────────────────────
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Activate your virtual environment first."
    exit 1
fi

if ! python3 -c "import PyInstaller" &>/dev/null; then
    echo "[INFO] PyInstaller not found — installing..."
    pip install pyinstaller
fi

echo "[INFO] Python: $(python3 --version)"
echo "[INFO] PyInstaller: $(pyinstaller --version)"
echo

# ── 2. Convert icon (PNG → ICNS) ─────────────────────────────────────────────
if [ -f "gui/icon.png" ] && [ ! -f "gui/icon.icns" ]; then
    echo "[INFO] Converting icon.png → icon.icns..."
    ICONSET="gui/icon.iconset"
    mkdir -p "$ICONSET"
    for size in 16 32 64 128 256 512; do
        sips -z $size $size gui/icon.png --out "$ICONSET/icon_${size}x${size}.png" &>/dev/null
        double=$((size * 2))
        sips -z $double $double gui/icon.png --out "$ICONSET/icon_${size}x${size}@2x.png" &>/dev/null
    done
    iconutil -c icns "$ICONSET" -o "gui/icon.icns" && echo "[INFO] Icon created: gui/icon.icns" \
        || echo "[WARN] iconutil failed — building without custom icon."
    rm -rf "$ICONSET"

    # Patch spec to use .icns
    if [ -f "gui/icon.icns" ]; then
        sed -i '' 's|# icon="gui/icon.icns"|icon="gui/icon.icns"|g' icolorit_macos.spec
    fi
fi

# ── 3. Clean previous build ───────────────────────────────────────────────────
echo "[INFO] Cleaning previous build artifacts..."
rm -rf "build/iColoriT_LoRA"
rm -rf "$APP_BUNDLE"

# ── 4. Run PyInstaller ────────────────────────────────────────────────────────
echo "[INFO] Running PyInstaller..."
pyinstaller icolorit_macos.spec --noconfirm --clean

if [ ! -d "$APP_BUNDLE" ]; then
    echo "[ERROR] Build failed — .app bundle not found."
    exit 1
fi

# ── 5. Copy checkpoints + samples into the .app Resources ────────────────────
RES="$APP_BUNDLE/Contents/Resources"

if [ -d "checks" ]; then
    echo "[INFO] Copying checks/ into app bundle..."
    cp -r checks "$RES/"
fi

if [ -d "samples" ]; then
    echo "[INFO] Copying samples/ into app bundle..."
    cp -r samples "$RES/"
fi

# ── 6. Remove quarantine attribute (allows running without Gatekeeper prompt) ─
echo "[INFO] Removing quarantine attribute..."
xattr -cr "$APP_BUNDLE" 2>/dev/null || true

# ── 7. Optional: code-sign ────────────────────────────────────────────────────
#  Uncomment and set DEVELOPER_ID to sign the app:
#
# DEVELOPER_ID="Developer ID Application: Your Name (TEAMID)"
# echo "[INFO] Signing app..."
# codesign --deep --force --options runtime \
#          --sign "$DEVELOPER_ID" \
#          "$APP_BUNDLE"
# codesign --verify --verbose "$APP_BUNDLE"

# ── 8. Create DMG ─────────────────────────────────────────────────────────────
if command -v create-dmg &>/dev/null; then
    echo "[INFO] Creating DMG..."
    create-dmg \
        --volname "iColoriT LoRA" \
        --volicon "gui/icon.icns" \
        --window-pos 200 120 \
        --window-size 700 400 \
        --icon-size 120 \
        --icon "$APP_NAME.app" 175 190 \
        --hide-extension "$APP_NAME.app" \
        --app-drop-link 525 190 \
        "$DMG_OUT" \
        "$DIST_DIR/" \
        && echo "[INFO] DMG created: $DMG_OUT" \
        || echo "[WARN] create-dmg failed — app bundle still available."
else
    echo "[INFO] create-dmg not found — skipping DMG. Install with: brew install create-dmg"
    echo "[INFO] You can manually drag $APP_BUNDLE to /Applications."
fi

# ── 9. Done ───────────────────────────────────────────────────────────────────
echo
echo "============================================================"
echo "  BUILD COMPLETE"
echo "  App bundle : $APP_BUNDLE"
[ -f "$DMG_OUT" ] && echo "  DMG        : $DMG_OUT"
echo "============================================================"
echo
echo "  To run directly:"
echo "    open \"$APP_BUNDLE\""
echo
echo "  To distribute without signing, users must:"
echo "    Right-click → Open (first launch only, to bypass Gatekeeper)"
