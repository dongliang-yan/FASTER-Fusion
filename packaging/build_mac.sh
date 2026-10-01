#!/bin/bash
# Builds dist/FASTER Fusion.app and dist/FASTER-Fusion-<version>-macOS.dmg.
# Run from the FASTER_Fusion_App folder, with the build environment set up:
#   python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
VER=$($PY -c "import faster_fusion; print(faster_fusion.__version__)")
$PY packaging/make_icon.py
rm -rf build dist
$PY -m PyInstaller --noconfirm --clean --distpath dist --workpath build packaging/faster_fusion.spec
# a disk image with the app and a link to /Applications, for sharing
STAGE=$(mktemp -d)
cp -R "dist/FASTER Fusion.app" "$STAGE/"
ln -s /Applications "$STAGE/Applications"
hdiutil create -volname "FASTER Fusion" -srcfolder "$STAGE" -ov -format UDZO \
    "dist/FASTER-Fusion-$VER-macOS.dmg"
rm -rf "$STAGE"
echo "Built dist/FASTER Fusion.app and dist/FASTER-Fusion-$VER-macOS.dmg"
