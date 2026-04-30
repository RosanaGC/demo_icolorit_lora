# Runtime hook: fix numpy DLL loading on Windows with Python 3.8+
#
# Python 3.8+ requires explicit os.add_dll_directory() calls — PATH is not
# searched for DLLs anymore. We scan the entire bundle and register every
# subdirectory that contains .dll or .pyd files.
import os
import sys

if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    os.add_dll_directory(sys._MEIPASS)
    for _root, _dirs, _files in os.walk(sys._MEIPASS):
        if any(f.lower().endswith((".dll", ".pyd")) for f in _files):
            try:
                os.add_dll_directory(_root)
            except OSError:
                pass
