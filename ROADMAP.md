# iColoriT LoRA — Development Roadmap

> Estado: **activo**  
> Repo: `demo_icolorit_lora`  
> Rama base: `dev` → merge a `main` en cada fase completada

---

## Leyenda

| Símbolo | Significado |
|---------|-------------|
| ✅ | Completado |
| 🔧 | En progreso |
| 📋 | Pendiente |
| ⏸ | En pausa / depende de otra fase |

---

## Phase 1 — Core GUI v2 + Fixes críticos

**Objetivo:** reemplazar la UI original por una ventana unificada moderna y corregir los bugs bloqueantes.

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 1.1 | Nueva ventana `IColoriTUIv2` (dark theme, single window) | ✅ | `gui/gui_main_v2.py` |
| 1.2 | Entry point `icolorit_ui_v2.py` con splash screen | ✅ | |
| 1.3 | Toolbar con Load / Save / Undo / Redo / Reset | ✅ | |
| 1.4 | Status bar: hints, zoom, device | ✅ | |
| 1.5 | Drag & drop de imágenes | ✅ | |
| 1.6 | Sliders de zoom y brush en toolbar | ✅ | |
| 1.7 | Panel izquierdo: Gamut + Paleta + Referencia integrados | ✅ | |
| 1.8 | **Fix: ventanas Gamut → close = hide** (reapertura siempre disponible) | ✅ | `gui/gui_main.py` |
| 1.9 | Menú View en ambas versiones (show/hide paneles) | ✅ | |
| 1.10 | Color Swatch visual animado | ✅ | |
| 1.11 | **Fix: session save/load** — PNG full-res + canal L (BC + calibración) guardados correctamente | ✅ | `gui/gui_main_v2.py` |

---

## Phase 2 — Distribución: Windows & macOS

**Objetivo:** generar ejecutables autocontenidos para ambas plataformas sin necesidad de instalar Python.

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 2.1 | `icolorit_v2.spec` — PyInstaller spec Windows | ✅ | |
| 2.2 | `build_windows.bat` — script de build + lanzador | ✅ | |
| 2.3 | `icolorit_macos.spec` — PyInstaller spec macOS | ✅ | `.app` bundle |
| 2.4 | `build_macos.sh` — script de build + DMG | ✅ | |
| 2.5 | Workflow GitHub Actions para build Windows CPU-only (`workflow_dispatch`) | ✅ | `.github/workflows/build_windows.yml` |
| 2.6 | Icono `.ico` (Windows) y `.icns` (macOS) | 📋 | Convertir `gui/icon.png` |
| 2.7 | Firma de código macOS (`codesign`) | 📋 | Requiere Apple Developer ID |
| 2.8 | Testar ejecutable Windows (máquina limpia) | 📋 | |
| 2.9 | Testar `.app` macOS (máquina limpia) | 📋 | |

---

## Phase 3 — UX Polish

**Objetivo:** pulir la interacción del usuario; hacer la herramienta más intuitiva en uso diario.

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 3.1 | Cursor personalizado al dibujar hints (círculo del tamaño del brush) | 📋 | |
| 3.2 | Animación de carga durante `compute_result` (spinner / progress) | 📋 | |
| 3.3 | Vista "Before / After" con slider central deslizable | 📋 | |
| 3.4 | Zoom centrado en cursor (no en esquina) | 📋 | |
| 3.5 | Picker de color completo al hacer doble-click en un hint | 📋 | |
| 3.6 | Tooltip con coordenadas y valor Lab al hover sobre el pad | 📋 | |
| 3.7 | Paleta de colores ampliada (favoritos, recientes, presets) | 📋 | |
| 3.8 | Mensaje de bienvenida al abrir sin imagen ("Drop an image here") | 📋 | |
| 3.9 | Historial de hints visible como lista (panel lateral colapsable) | 📋 | |

---

## Phase 4 — Características nuevas

**Objetivo:** ampliar las capacidades del tool más allá del uso básico.

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 4.1 | **Modo Batch**: colorizar una carpeta completa de imágenes | 📋 | Con hints desde JSON |
| 4.2 | Export side-by-side (B&N + Colorizada en una sola imagen) | 📋 | |
| 4.3 | Export de video (secuencia de frames) | ⏸ | Depende de 4.1 |
| 4.4 | Presets de hints (paletas predefinidas por tipo de foto) | 📋 | |
| 4.5 | Modo "Auto-hint": sugerir colores automáticamente por región | 📋 | Requiere modelo auxiliar |
| 4.6 | Comparación multi-checkpoint (abrir 2 resultados en split view) | 📋 | |

---

## Phase 5 — Gestión de modelos

**Objetivo:** hacer más fácil el uso de distintos checkpoints sin editar argumentos en CLI.

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 5.1 | Diálogo de bienvenida con selector de modelo + imagen | 📋 | Al iniciar sin args |
| 5.2 | Soporte de múltiples formatos: `.pth`, directorio HuggingFace, `safetensors` | 📋 | |
| 5.3 | Detección automática de GPU y selección de device | 📋 | |
| 5.4 | Panel "Model Info": nombre, rank LoRA, profundidad, parámetros | 📋 | |
| 5.5 | Hot-swap de modelo sin reiniciar la app | 📋 | |

---

## Phase 6 — Calidad y distribución final

**Objetivo:** estabilizar, testear y publicar una versión 1.0 distribuible.

| # | Tarea | Estado | Notas |
|---|-------|--------|-------|
| 6.1 | Suite de tests unitarios (rendering, hints, señales) | 📋 | `pytest` + `pytest-qt` |
| 6.2 | CI/CD con GitHub Actions (lint + test en push) | 📋 | |
| 6.3 | Installer Windows con InnoSetup (`.exe` autoinstalable) | 📋 | |
| 6.4 | Installer macOS con `.pkgbuild` o Packages.app | 📋 | |
| 6.5 | Panel de ayuda integrado en la app (F1) | 📋 | |
| 6.6 | Internacionalización (ES / EN) | 📋 | |
| 6.7 | Página de GitHub releases con binarios adjuntos | 📋 | |
| 6.8 | CHANGELOG.md y versionado semántico | 📋 | |

---

## Orden de implementación sugerido

```
Phase 1  →  Phase 2  →  Phase 3  →  Phase 5  →  Phase 4  →  Phase 6
(core)     (deploy)    (polish)   (modelos)    (features)  (QA/distrib)
```

Phases 3 y 5 pueden avanzar en paralelo. Phase 4 necesita Phase 3 completa para UX consistente.

---

## Stack técnico

| Componente | Librería |
|-----------|----------|
| UI | PyQt5 |
| Modelo | PyTorch + timm (ViT) |
| LoRA | loralib |
| Color | scikit-image (LAB), OpenCV |
| Build | PyInstaller |
| Tests | pytest, pytest-qt |

---

*Última actualización: 2026-04-29*
