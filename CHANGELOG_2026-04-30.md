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

## Error 4: `[WinError 1114] c10.dll` — causa real #2 (VC++ Runtime faltante)

**Síntoma**  
Idéntico al Error 3 (`WinError 1114` en `c10.dll`), persiste incluso con `upx=False`.

**Causa real**  
El runner de GitHub Actions tiene el VC++ Runtime instalado, pero la máquina del usuario no necesariamente. Si el usuario ejecuta `iColoriT_LoRA.exe` directamente (sin pasar por `run_icolorit.bat`), `vc_redist.x64.exe` nunca se instala. `c10.dll` depende de `MSVCP140.dll`, `VCRUNTIME140.dll` y `VCRUNTIME140_1.dll`; si alguna falta, su `DllMain` falla con error 1114.

**Fixes aplicados**
1. Agregar paso `Bundle VC++ Runtime DLLs into _internal` en el workflow: copia `MSVCP140.dll`, `VCRUNTIME140.dll` y `VCRUNTIME140_1.dll` desde `%SystemRoot%\System32\` del runner directamente a `dist\iColoriT_LoRA\_internal\` después del build. El exe queda autónomo — no requiere instalación previa en la máquina del usuario.
2. Fix del regex en `Patch spec to use icon.ico`: cambiado de comillas simples a dobles para que matchee exactamente el spec (`# icon="gui/icon.ico",`).

**Estado**: fix incompleto — los DLLs ya estaban presentes en `_internal/` pero el error persistió.

---

## Error 5: `[WinError 1114] c10.dll` — causa real #3 (PyTorch demasiado nuevo para el hardware)

**Síntoma**  
Exactamente el mismo `WinError 1114` en `c10.dll`. Los VC++ DLLs (`msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`) están en `_internal/`. `torch_cpu.dll` pesa **265MB**, lo que indica PyTorch 2.5+ instalado.

**Causa real**  
El workflow instalaba la versión más reciente de torch (sin pinear). PyTorch 2.3+ cambió la estructura interna de `c10.dll` en Windows e introdujo nuevas dependencias de inicialización que fallan en CPUs AMD antiguos (AMD A10-9620P — arquitectura Excavator 2016). La inicialización de `c10.dll` llama código que requiere features de CPU no disponibles en esa microarquitectura.

**Fix aplicado**  
Pinear torch a `torch==2.1.2 torchvision==0.16.2` en el workflow. Es la última versión estable anterior a los cambios de 2.3+, tiene soporte confirmado en Windows CPU + AMD, y es compatible con el modelo iColoriT + LoRA.

**Estado**: pendiente de confirmar con el próximo build.

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
