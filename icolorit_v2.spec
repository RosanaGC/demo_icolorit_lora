# icolorit_v2.spec — PyInstaller spec for iColoriT LoRA (Windows .exe)
#
# Usage (from repo root, inside the activated venv):
#   pyinstaller icolorit_v2.spec
#
# Output: dist/iColoriT_LoRA/iColoriT_LoRA.exe
# ──────────────────────────────────────────────────────────────────────────────

import sys
import os
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, collect_all

block_cipher = None

# ── Source root ───────────────────────────────────────────────────────────────
SRC = os.path.abspath(".")

# ── Collect numpy + torch binaries (fixes DLL load errors on Windows) ────────
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all("numpy")
torch_datas,  torch_binaries,  torch_hiddenimports  = collect_all("torch")

# ── Data files to bundle ──────────────────────────────────────────────────────
# Format: (src_glob_or_dir, dest_dir_inside_bundle)
added_datas = [
    # GUI assets
    (os.path.join(SRC, "gui", "icon.png"),        "gui"),
    # Sample images (optional — user can load their own)
    (os.path.join(SRC, "samples"),                "samples"),
    # timm model configs (required by create_model)
    *collect_data_files("timm"),
    # skimage data
    *collect_data_files("skimage"),
    # cv2 (sometimes needs haarcascades etc.)
    *collect_data_files("cv2"),
    # einops
    *collect_data_files("einops"),
    # numpy/torch: datos (los binarios van aparte en Analysis)
    *numpy_datas,
    *torch_datas,
]

# ── Hidden imports ────────────────────────────────────────────────────────────
# Libraries that PyInstaller static analysis misses.
hidden = [
    # torch
    "torch",
    "torch.nn",
    "torch.nn.functional",
    "torchvision",
    # timm
    "timm",
    "timm.models",
    "timm.models.registry",
    "timm.models.layers",
    # image / color
    "skimage",
    "skimage.color",
    "cv2",
    "PIL",
    "PIL.Image",
    # einops
    "einops",
    # loralib
    "loralib",
    # PyQt5
    "PyQt5",
    "PyQt5.QtCore",
    "PyQt5.QtGui",
    "PyQt5.QtWidgets",
    # numpy C extensions (fixes "imported numpy-c extensions failed" on Windows)
    "numpy",
    "numpy.core",
    "numpy.core._multiarray_umath",
    "numpy.core.multiarray",
    "numpy.core._multiarray_tests",
    "numpy.core._operand_flag_tests",
    "numpy.core._rational_tests",
    "numpy.core._struct_ufunc_tests",
    "numpy.core._umath_tests",
    "numpy.random",
    "numpy.random._common",
    "numpy.random._bounded_integers",
    "numpy.random._mt19937",
    "numpy.random.mtrand",
    "numpy.linalg",
    "numpy.linalg._umath_linalg",
    # scipy / sklearn (skimage deps)
    "scipy",
    "scipy.ndimage",
    "scipy.spatial",
    # project modules
    "modeling",
    "utils",
    "gui",
    "gui.gui_draw",
    "gui.gui_gamut",
    "gui.gui_palette",
    "gui.gui_vis",
    "gui.gui_main_v2",
    "gui.ui_control",
    "gui.lab_gamut",
]


# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(SRC, "icolorit_ui_v2.py")],
    pathex=[SRC],
    binaries=numpy_binaries + torch_binaries,
    datas=added_datas,
    hiddenimports=hidden + numpy_hiddenimports + torch_hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["runtime_hook_numpy.py"],
    excludes=[
        # Exclude heavy/unused packages to reduce bundle size
        "matplotlib",
        "notebook",
        "ipython",
        "IPython",
        "tensorboard",
        "tensorboardX",
        "pandas",
        "pytest",
        "sphinx",
        "docutils",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
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
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,           # True para ver errores; cambiar a False en release final
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # icon="gui/icon.ico",  # uncomment after converting icon.png → icon.ico
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="iColoriT_LoRA",
)
