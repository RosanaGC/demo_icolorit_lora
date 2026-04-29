# icolorit_macos.spec — PyInstaller spec for iColoriT LoRA (macOS .app)
#
# Usage (from repo root, inside the activated venv):
#   pyinstaller icolorit_macos.spec
#
# Output: dist/iColoriT LoRA.app
#         dist/iColoriT_LoRA_mac.dmg  (created by build_macos.sh)
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None
SRC = os.path.abspath(".")

# ── Data files ────────────────────────────────────────────────────────────────
added_datas = [
    (os.path.join(SRC, "gui", "icon.png"),   "gui"),
    (os.path.join(SRC, "samples"),           "samples"),
    *collect_data_files("timm"),
    *collect_data_files("skimage"),
    *collect_data_files("cv2"),
    *collect_data_files("einops"),
]

# ── Hidden imports ────────────────────────────────────────────────────────────
hidden = [
    "torch", "torch.nn", "torch.nn.functional",
    "timm", "timm.models", "timm.models.registry", "timm.models.layers",
    "skimage", "skimage.color",
    "cv2",
    "PIL", "PIL.Image",
    "einops",
    "loralib",
    "PyQt5", "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
    "scipy", "scipy.ndimage", "scipy.spatial",
    "modeling", "utils",
    "gui", "gui.gui_draw", "gui.gui_gamut", "gui.gui_palette",
    "gui.gui_vis", "gui.gui_main_v2", "gui.ui_control", "gui.lab_gamut",
]
hidden += collect_submodules("timm")
hidden += collect_submodules("skimage")

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(SRC, "icolorit_ui_v2.py")],
    pathex=[SRC],
    binaries=[],
    datas=added_datas,
    hiddenimports=hidden,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        "matplotlib", "notebook", "ipython", "IPython",
        "tensorboard", "tensorboardX", "pandas", "pytest",
    ],
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="iColoriT_LoRA",
    debug=False,
    strip=False,
    upx=True,
    console=False,
    # icon="gui/icon.icns",  # uncomment after converting gui/icon.png → gui/icon.icns
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    name="iColoriT_LoRA",
)

# ── macOS .app bundle ─────────────────────────────────────────────────────────
app = BUNDLE(
    coll,
    name="iColoriT LoRA.app",
    # icon="gui/icon.icns",  # uncomment after converting
    bundle_identifier="com.icolorit.lora",
    info_plist={
        "NSPrincipalClass":             "NSApplication",
        "NSHighResolutionCapable":      True,
        "CFBundleShortVersionString":   "1.0.0",
        "CFBundleVersion":              "1",
        "LSMinimumSystemVersion":       "10.14",
        "NSHumanReadableCopyright":     "© 2026 iColoriT LoRA",
        # Allow file drag-and-drop onto the app icon
        "CFBundleDocumentTypes": [
            {
                "CFBundleTypeName":       "Image",
                "CFBundleTypeRole":       "Viewer",
                "CFBundleTypeExtensions": ["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
            }
        ],
    },
)
