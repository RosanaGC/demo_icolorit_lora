# Changelog — 2026-04-29

## Resumen de cambios

---

### Limpieza de archivos viejos

- Eliminados `gui/gui_main.py`, `gui/gui_main_gt.py`, `icolorit_ui_gt.py`, `icolorit_ui_2.py` — todos reemplazados por la nueva GUI (`gui_main_v2.py` / `icolorit_ui_v2.py`)
- Clases `MagnifierOverlay` y `HoverZoomFilter` migradas dentro de `gui_main_v2.py` para eliminar la dependencia con los archivos borrados

---

### Fix: sesión save/load (`gui_main_v2.py`)

- **Bug PNG 224×224**: el PNG guardado quedaba cuadrado porque se construía desde `im_lab` (224×224) en lugar de `im_full` (resolución original)
- **Bug L no guardada**: al guardar solo se aplicaba el ajuste de BC al PNG; la calibración de L (histogram matching) se perdía completamente
- **Fix aplicado**: el save ahora deriva el mapeo completo de L comparando el canal L original 224×224 (recomputado desde `im_full`) con el L actual en `dw.im_lab` (que acumula BC + calibración en cualquier orden). Ese mapeo se aplica vía `np.interp` al full-res antes de guardar el PNG
- Al cargar el PNG bundled, los sliders se resetean a neutro automáticamente (`_clear_L_cal_originals`); si se carga desde `original_path`, se re-aplica solo el BC desde el JSON

---

### Build para Windows (sin CUDA)

- `icolorit_v2.spec` — spec de PyInstaller para generar `dist/iColoriT_LoRA/iColoriT_LoRA.exe`
- `build_windows.bat` — script automatizado para buildear en una máquina Windows local
- `.github/workflows/build_windows.yml` — workflow de GitHub Actions (`workflow_dispatch`) que buildea en un runner `windows-latest` sin necesitar una máquina Windows física. Usa torch CPU-only para reducir tamaño
- **Fix workflow**: eliminados `collect_submodules("timm")` y `collect_submodules("skimage")` del spec, que causaban un cuelgue de +2 horas durante el análisis de PyInstaller. Agregados timeouts por paso (20 min torch, 15 min deps, 30 min PyInstaller, 60 min total)

---

### Git

- Branch `bu-main` creado como backup del `main` original antes del merge
- Todo el código del branch `new` mergeado a `main`
- `main` queda como branch activo y por defecto

---

## Commits del día

| Hash | Descripción |
|------|-------------|
| `d194fdf` | Merge branch new into main |
| `99d3e5a` | Fix build: remove collect_submodules that caused infinite hang |
| `cb4dcab` | Add GitHub Actions workflow for Windows CPU build |
| `2c8f318` | Remove legacy GUI files; fix session save/load L channel handling |
| `ca2eff8` | Revert "Remove legacy GUI files; move MagnifierOverlay/HoverZoomFilter into gui_main_v2" |
| `72dd0cb` | Remove legacy GUI files; move MagnifierOverlay/HoverZoomFilter into gui_main_v2 |
| `b9f0129` | feat: new GUI v2 with GT mode, session save/load, reference tools |

---

## Estado del repo al cierre

| Branch | Estado |
|--------|--------|
| `main` | Código completo y actualizado — listo para buildear |
| `new` | Branch de desarrollo (mismo estado que main) |
| `bu-main` | Backup del main original antes del merge |
