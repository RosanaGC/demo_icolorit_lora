# Runtime hook: fix numpy DLL loading on Windows with Python 3.8+
#
# Python 3.8+ changed DLL search semantics on Windows — directories in PATH
# are no longer trusted for DLL resolution. PyInstaller bundles the files but
# they sit inside sys._MEIPASS subdirectories that Windows won't find
# automatically. os.add_dll_directory() registers them as trusted.
import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _base = sys._MEIPASS
    # Always trust the bundle root
    os.add_dll_directory(_base)
    # numpy stores OpenBLAS and other C-library DLLs in .libs and core
    for _sub in ("numpy/.libs", "numpy\\libs", "numpy/core", "numpy\\core"):
        _p = os.path.join(_base, _sub)
        if os.path.isdir(_p):
            os.add_dll_directory(_p)
