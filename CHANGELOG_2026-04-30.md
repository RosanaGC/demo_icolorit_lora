# Changelog 2026-04-30 — Windows exe debugging

## Contexto

Se intentó distribuir la app como `.exe` de Windows via GitHub Actions + PyInstaller.
Cada error surgió al correr el artifact en la máquina de un usuario externo (Windows limpio).

---

## Error 1: `imported numpy C-extensions failed`

**Síntoma**
```
ImportError: Importing the numpy C-extensions failed.
numpy version: 1.22.4
```

**Causa**  
`collect_data_files("numpy")` solo copia archivos de datos, no las DLLs binarias de numpy (`.pyd`, OpenBLAS, etc.).

**Fix aplicado**  
Reemplazar `collect_data_files("numpy")` por `collect_all("numpy")` en el spec, que devuelve `(datas, binaries, hiddenimports)`. Los binarios se pasan a `Analysis(binaries=...)`.

---

## Error 2: `DLL load failed while importing _multiarray_umath`

**Síntoma**
```
ImportError: DLL load failed while importing _multiarray_umath:
No se puede encontrar el módulo especificado.
numpy version: 1.22.4
```

**Causa**  
Python 3.8+ cambió la política de búsqueda de DLLs en Windows: ya no busca en `PATH` automáticamente. Las DLLs de numpy estaban en el bundle pero Windows no las encontraba. Además, numpy 1.22.4 tiene incompatibilidad conocida con PyInstaller en este escenario.

**Fixes aplicados**
1. Crear `runtime_hook_numpy.py`: registra todos los subdirectorios del bundle que contienen `.dll`/`.pyd` via `os.add_dll_directory()`, ejecutándose antes de cualquier import.
2. Actualizar `numpy==1.22.4` → `numpy==1.26.4` en `requirements.txt`. numpy 1.26 maneja el DLL loading de Windows correctamente.

---

## Error 3: `[WinError 1114] Error loading c10.dll` (primera aparición)

**Síntoma**
```
OSError: [WinError 1114] Error loading "..._internal\torch\lib\c10.dll"
or one of its dependencies.
```

**Causa sospechada inicialmente**  
Se pensó que faltaban DLLs de torch en el bundle (`libiomp5md.dll`, `fbgemm.dll`, etc.).

**Fixes aplicados (resultaron insuficientes)**
1. Agregar `pyinstaller-hooks-contrib` al workflow (hooks específicos para torch).
2. Agregar `collect_all("torch")` al spec para incluir todos los binarios de torch explícitamente.
3. Instalar Visual C++ Redistributable 2022 (`vc_redist.x64.exe`) — incluido en el artifact y auto-instalado por el launcher `.bat`.

**Resultado**: el error persistió exactamente igual, lo que descartó DLLs faltantes como causa.

---

## Error 3: `[WinError 1114] Error loading c10.dll` (causa real)

**Síntoma**  
Idéntico al anterior, persiste aunque `c10.dll` y todas sus dependencias están presentes en el bundle.

**Causa real**  
UPX estaba activado (`upx=True`) en el spec. UPX comprime las DLLs del bundle para reducir tamaño, pero corrompe DLLs complejas de PyTorch (`c10.dll`, `torch_cpu.dll`, etc.) que usan secciones especiales en su formato PE. El resultado es `WinError 1114` (DLL_INIT_FAILED) al intentar cargar la DLL corrupta.

**Fix aplicado**  
Deshabilitar UPX en el spec: `upx=False` en ambos `EXE(...)` y `COLLECT(...)`.

**Estado**: pendiente de confirmar con el próximo build.

---

## Otros cambios al workflow durante el proceso

| Cambio | Motivo |
|--------|--------|
| `console=True` en el exe | Ver el traceback completo en lugar de fallar silenciosamente |
| Launcher `.bat` captura stdout/stderr a `icolorit_log.txt` | Poder leer el error aunque la ventana se cierre |
| `.bat` instala `vc_redist.x64.exe` silenciosamente antes de lanzar | Evitar que el usuario tenga que instalarlo manualmente |
| YAML syntax fix: reemplazar here-string `@"..."@` por one-liner | El delimitador `"@` confundía al parser YAML de GitHub Actions |

---

## Estado final del spec (cambios acumulados)

```python
# collect_all en lugar de collect_data_files
numpy_datas, numpy_binaries, numpy_hiddenimports = collect_all("numpy")
torch_datas,  torch_binaries,  torch_hiddenimports  = collect_all("torch")

# runtime hook para DLL directories
runtime_hooks=["runtime_hook_numpy.py"]

# UPX deshabilitado
upx=False  # en EXE y COLLECT
```

```txt
# requirements.txt
numpy==1.26.4   # era 1.22.4
```

```yaml
# build_windows.yml
pip install pyinstaller pyinstaller-hooks-contrib
```
