# gui_main_v2.py — Unified modern UI for iColoriT LoRA
# Single window: no floating dialogs.
# Dark theme, toolbar, left panel (gamut + palette + reference), status bar.

import os
import time

import numpy as np
from PIL import Image
from skimage import color as ski_color

from PyQt5.QtCore import Qt, QSize, QPoint, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPixmap, QIcon, QPainter, QTransform, QPen
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QScrollArea,
    QVBoxLayout, QHBoxLayout, QGridLayout,
    QToolBar, QAction, QStatusBar, QLabel,
    QSlider, QPushButton, QCheckBox, QGroupBox,
    QTabWidget, QFrame, QFileDialog, QSizePolicy,
    QGraphicsView, QGraphicsScene, QApplication,
    QDockWidget, QSpacerItem, QMenuBar, QMenu,
)
from PyQt5.QtCore import QRect, QObject, QEvent

from .gui_draw import GUIDraw
from .gui_draw_gt import GUIDrawGTHints
from .gui_gamut import GUIGamut
from .gui_palette import GUIPalette
from .gui_vis import GUI_VIS


# ──────────────────────────────────────────────────────────────────────────────
# Dark theme stylesheet (Catppuccin Mocha-inspired)
# ──────────────────────────────────────────────────────────────────────────────
DARK_QSS = """
QMainWindow, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
}
QMainWindow::separator { background: #313244; width: 1px; height: 1px; }

/* ---- Toolbar ---- */
QToolBar {
    background-color: #181825;
    border-bottom: 1px solid #313244;
    spacing: 4px;
    padding: 4px 8px;
}
QToolBar QToolButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 10px;
    color: #cdd6f4;
    font-size: 13px;
}
QToolBar QToolButton:hover  { background-color: #45475a; border-color: #89b4fa; }
QToolBar QToolButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
QToolBar QToolButton:checked { background-color: #89b4fa; color: #1e1e2e; border-color: #89b4fa; }
QToolBar::separator { background: #45475a; width: 1px; margin: 4px 6px; }
QToolBar QLabel { color: #a6adc8; font-size: 12px; margin-left: 4px; }

/* ---- Buttons ---- */
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 5px 14px;
    color: #cdd6f4;
    font-weight: 500;
}
QPushButton:hover   { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
QPushButton:disabled { color: #6c7086; border-color: #313244; }

/* ---- Sliders ---- */
QSlider::groove:horizontal {
    height: 4px;
    background: #45475a;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #89b4fa;
    border: none;
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal { background: #89b4fa; border-radius: 2px; }

/* ---- GroupBox ---- */
QGroupBox {
    border: 1px solid #313244;
    border-radius: 8px;
    margin-top: 10px;
    padding: 6px 6px 6px 6px;
    color: #a6adc8;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 6px;
    left: 10px;
}

/* ---- Tabs ---- */
QTabWidget::pane   { border: 1px solid #313244; border-radius: 6px; }
QTabBar::tab {
    background: #181825;
    color: #a6adc8;
    border: 1px solid #313244;
    padding: 6px 14px;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}
QTabBar::tab:selected  { background: #1e1e2e; color: #cdd6f4; border-color: #89b4fa; }
QTabBar::tab:hover     { color: #cdd6f4; }

/* ---- Status bar ---- */
QStatusBar {
    background: #181825;
    border-top: 1px solid #313244;
    color: #a6adc8;
    font-size: 12px;
    padding: 2px 8px;
}
QStatusBar::item { border: none; }

/* ---- ScrollArea / frame ---- */
QScrollArea, QAbstractScrollArea { background: transparent; border: none; }
QScrollBar:vertical {
    background: #181825; width: 10px; border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #45475a; border-radius: 5px; min-height: 20px;
}
QScrollBar::handle:vertical:hover { background: #89b4fa; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal {
    background: #181825; height: 10px; border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #45475a; border-radius: 5px; min-width: 20px;
}
QScrollBar::handle:horizontal:hover { background: #89b4fa; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }

/* ---- CheckBox ---- */
QCheckBox { color: #cdd6f4; spacing: 6px; }
QCheckBox::indicator {
    width: 16px; height: 16px;
    border: 1px solid #45475a; border-radius: 4px;
    background: #313244;
}
QCheckBox::indicator:checked {
    background: #89b4fa; border-color: #89b4fa;
    image: none;
}

/* ---- Dock ---- */
QDockWidget {
    color: #cdd6f4;
    titlebar-close-icon: none;
}
QDockWidget::title {
    background: #181825;
    border-bottom: 1px solid #313244;
    padding: 4px 8px;
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    color: #a6adc8;
}

/* ---- Graphics view ---- */
QGraphicsView { background: #181825; border: 1px solid #313244; border-radius: 6px; }

/* ---- Labels ---- */
QLabel { color: #cdd6f4; }
QLabel#section_title {
    color: #a6adc8; font-size: 11px; font-weight: 600;
    letter-spacing: 0.5px; text-transform: uppercase;
}
"""


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers
# ──────────────────────────────────────────────────────────────────────────────
def _np_to_qpix(arr: np.ndarray) -> QPixmap:
    h, w = arr.shape[:2]
    qimg = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ──────────────────────────────────────────────────────────────────────────────
# Gamut magnifier (hover zoom on gamut widgets)
# ──────────────────────────────────────────────────────────────────────────────
class MagnifierOverlay(QWidget):
    """Floating loupe that zooms into the gamut widget under the cursor."""
    def __init__(self, parent=None, size=420, zoom=16.0):
        super().__init__(parent)
        self.size = int(size)
        self.zoom = float(zoom)
        self.min_zoom = 2.0
        self.max_zoom = 200.0
        self.pixel_zoom_threshold = 8.0
        self.min_size = 160
        self.max_size = 900
        self.show_grid = True
        self.cross_len = 14
        self.cell_fill    = QColor(255, 255, 255, 40)
        self.cell_border1 = QPen(QColor(255, 255, 255, 220), 2)
        self.cell_border2 = QPen(QColor(0, 0, 0, 220), 1)
        self.setWindowFlags(Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.resize(self.size, self.size)
        self._pm = None
        self._piece_rect = QRect()
        self._cell_rect  = QRect()
        self._step_x = 1.0
        self._step_y = 1.0

    def set_zoom(self, z: float):
        self.zoom = max(self.min_zoom, min(self.max_zoom, float(z)))

    def set_size(self, s: int):
        self.size = max(self.min_size, min(self.max_size, int(s)))
        self.resize(self.size, self.size)
        self.update()

    def update_from_widget(self, src_widget: QWidget, local_pos: QPoint):
        if not src_widget:
            return
        w = max(1, int(self.size / self.zoom))
        h = max(1, int(self.size / self.zoom))
        x = max(0, local_pos.x() - w // 2)
        y = max(0, local_pos.y() - h // 2)
        rect = QRect(x, y, w, h).intersected(src_widget.rect())
        if rect.isEmpty():
            return
        piece = src_widget.grab(rect)
        mode = Qt.FastTransformation if self.zoom >= self.pixel_zoom_threshold else Qt.SmoothTransformation
        pm_scaled = piece.scaled(self.size, self.size, Qt.KeepAspectRatio, mode)
        off_x = (self.width()  - pm_scaled.width())  // 2
        off_y = (self.height() - pm_scaled.height()) // 2
        self._pm = pm_scaled
        self._piece_rect = QRect(off_x, off_y, pm_scaled.width(), pm_scaled.height())
        self._step_x = pm_scaled.width()  / max(1, rect.width())
        self._step_y = pm_scaled.height() / max(1, rect.height())
        dx = float(local_pos.x() - rect.x())
        dy = float(local_pos.y() - rect.y())
        cell_x = int(np.floor(dx * self._step_x))
        cell_y = int(np.floor(dy * self._step_y))
        self._cell_rect = QRect(
            self._piece_rect.left() + cell_x, self._piece_rect.top() + cell_y,
            max(1, int(round(self._step_x))), max(1, int(round(self._step_y)))
        )
        self.move(src_widget.mapToGlobal(local_pos) + QPoint(16, 16))
        self.show()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        if self._pm is not None and not self._piece_rect.isEmpty():
            p.drawPixmap(self._piece_rect, self._pm)
            if self.show_grid and self.zoom >= self.pixel_zoom_threshold:
                p.setPen(QPen(QColor(0, 0, 0, 40), 1))
                x0, y0 = self._piece_rect.left(), self._piece_rect.top()
                for i in range(int(round(self._pm.width() / self._step_x)) + 1):
                    gx = int(round(x0 + i * self._step_x))
                    p.drawLine(gx, y0, gx, y0 + self._piece_rect.height())
                for j in range(int(round(self._pm.height() / self._step_y)) + 1):
                    gy = int(round(y0 + j * self._step_y))
                    p.drawLine(x0, gy, x0 + self._piece_rect.width(), gy)
            if not self._cell_rect.isEmpty():
                p.fillRect(self._cell_rect, self.cell_fill)
                p.setPen(self.cell_border1)
                p.drawRect(self._cell_rect.adjusted(0, 0, -1, -1))
                p.setPen(self.cell_border2)
                p.drawRect(self._cell_rect.adjusted(1, 1, -2, -2))
        cx, cy = self.width() // 2, self.height() // 2
        L = max(self.cross_len, int(self.size * 0.035))
        p.setPen(QPen(QColor(255, 255, 255, 230), 3))
        p.drawLine(cx - L, cy, cx + L, cy)
        p.drawLine(cx, cy - L, cx, cy + L)
        p.setPen(QPen(QColor(0, 0, 0, 220), 1))
        p.drawLine(cx - L, cy, cx + L, cy)
        p.drawLine(cx, cy - L, cx, cy + L)
        p.end()


class HoverZoomFilter(QObject):
    """Event filter that shows MagnifierOverlay when hovering over a widget."""
    def __init__(self, owner_widget: QWidget, overlay: MagnifierOverlay):
        super().__init__(owner_widget)
        self.owner = owner_widget
        self.overlay = overlay
        self.owner.setMouseTracking(True)

    def eventFilter(self, obj, ev):
        t = ev.type()
        if t == QEvent.Leave:
            self.overlay.hide()
            return False
        if t == QEvent.MouseMove:
            self.overlay.update_from_widget(self.owner, ev.pos())
            return False
        if t == QEvent.Wheel:
            dy = ev.angleDelta().y() if hasattr(ev, "angleDelta") else 0
            if dy == 0 and hasattr(ev, "pixelDelta"):
                dy = ev.pixelDelta().y()
            if dy == 0:
                return False
            if ev.modifiers() & Qt.ShiftModifier:
                self.overlay.set_size(self.overlay.size + (40 if dy > 0 else -40))
            else:
                step = 1.45 if (ev.modifiers() & Qt.ControlModifier) else 1.20
                self.overlay.set_zoom(self.overlay.zoom * (step if dy > 0 else 1 / step))
            self.overlay.update_from_widget(self.owner, ev.pos() if hasattr(ev, "pos") else QPoint(0, 0))
            return True
        return False


class ColorSwatch(QWidget):
    """Small square showing the current hint color."""
    def __init__(self, size=40, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self._color = QColor(128, 128, 128)

    def set_from_stylesheet(self, stylesheet: str):
        # stylesheet is "background-color: #rrggbb"
        try:
            hex_str = stylesheet.split(":")[-1].strip()
            self._color = QColor(hex_str)
        except Exception:
            pass
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setBrush(self._color)
        p.setPen(QColor(69, 71, 90))
        p.drawRoundedRect(2, 2, self.width() - 4, self.height() - 4, 6, 6)
        p.end()


class ZoomableImageView(QGraphicsView):
    """Pan/zoom preview widget (reference image). Zoom via slider controls only."""

    zoom_changed = pyqtSignal(float)   # emitted whenever scale changes

    MIN_SCALE = 0.1
    MAX_SCALE = 16.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None
        self.setTransformationAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self._scale = 1.0

    def set_image_np(self, arr: np.ndarray):
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(_np_to_qpix(arr))
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self.set_zoom(1.0)
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def update_image_np(self, arr: np.ndarray):
        """Replace the displayed image without resetting zoom or pan."""
        if self.pixmap_item is None:
            self.set_image_np(arr)
            return
        self.pixmap_item.setPixmap(_np_to_qpix(arr))

    def set_zoom(self, scale: float):
        self._scale = max(self.MIN_SCALE, min(self.MAX_SCALE, scale))
        t = QTransform()
        t.scale(self._scale, self._scale)
        self.setTransform(t)
        self.zoom_changed.emit(self._scale)

    def zoom_in(self):
        self.set_zoom(self._scale * 1.25)

    def zoom_out(self):
        self.set_zoom(self._scale / 1.25)

    def zoom_reset(self):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._scale = 1.0
            self.zoom_changed.emit(self._scale)

    # Disable wheel zoom — use the slider controls instead
    def wheelEvent(self, event):
        event.ignore()

    def mouseDoubleClickEvent(self, event):
        self.zoom_reset()
        super().mouseDoubleClickEvent(event)


# ──────────────────────────────────────────────────────────────────────────────
# Dominant-color extraction
# ──────────────────────────────────────────────────────────────────────────────
def _extract_dominant_colors(pil_image, n: int = 16, bins: int = 24,
                              L_target=None, delta_L: float = 2.0):
    """
    Return the N most-represented colors from pil_image as a list of dicts:
        {'rgb': (r, g, b),  'lab': (L, a, b)}
    Strategy: 2-D histogram over the ab-plane; pick the top-N occupied bins;
    represent each bin by its median L from the original image.
    If L_target is given, only pixels within |L - L_target| <= delta_L are used.
    """
    img = np.array(pil_image.convert("RGB"), dtype=np.uint8)
    lab = ski_color.rgb2lab(img.astype(np.float32) / 255.0)

    a_flat = lab[..., 1].ravel()
    b_flat = lab[..., 2].ravel()
    l_flat = lab[..., 0].ravel()

    if L_target is not None:
        keep = np.abs(l_flat - L_target) <= delta_L
        if not np.any(keep):
            return []
        a_flat = a_flat[keep]
        b_flat = b_flat[keep]
        l_flat = l_flat[keep]

    hist, a_edges, b_edges = np.histogram2d(
        a_flat, b_flat, bins=bins, range=[[-110, 110], [-110, 110]]
    )

    order = np.argsort(hist.ravel())[::-1]
    colors = []
    for fi in order:
        if len(colors) >= n:
            break
        row, col = np.unravel_index(fi, hist.shape)
        if hist[row, col] < 5:
            break
        a_c = float((a_edges[row] + a_edges[row + 1]) / 2)
        b_c = float((b_edges[col] + b_edges[col + 1]) / 2)
        mask = (
            (a_flat >= a_edges[row]) & (a_flat < a_edges[row + 1]) &
            (b_flat >= b_edges[col]) & (b_flat < b_edges[col + 1])
        )
        L_med = float(np.median(l_flat[mask])) if np.any(mask) else 50.0
        lab_px = np.array([[[L_med, a_c, b_c]]], dtype=np.float32)
        rgb_px = (np.clip(ski_color.lab2rgb(lab_px), 0, 1) * 255).astype(np.uint8)
        colors.append({
            "rgb": (int(rgb_px[0, 0, 0]), int(rgb_px[0, 0, 1]), int(rgb_px[0, 0, 2])),
            "lab": (L_med, a_c, b_c),
        })
    return colors


# ──────────────────────────────────────────────────────────────────────────────
# Top-colors swatch widget
# ──────────────────────────────────────────────────────────────────────────────
class TopColorsWidget(QWidget):
    """
    Grid of clickable color swatches showing the dominant colors of the
    reference image (extracted via Lab histogram).
    Emits color_selected({'rgb': (r,g,b), 'lab': (L,a,b)}) on click.
    """
    color_selected = pyqtSignal(object)
    color_hovered  = pyqtSignal(object)   # emits dict or None

    SWATCH = 28
    GAP    = 5

    def __init__(self, cols: int = 8, rows: int = 2, parent=None):
        super().__init__(parent)
        self.cols = cols
        self.rows = rows
        self._colors = []
        self._hover  = -1
        w = cols * (self.SWATCH + self.GAP) + self.GAP
        h = rows * (self.SWATCH + self.GAP) + self.GAP
        self.setFixedSize(w, h)
        self.setMouseTracking(True)

    # ── public API ────────────────────────────────────────────────────────────
    def set_colors(self, colors: list):
        self._colors = colors[: self.cols * self.rows]
        self.update()

    def clear(self):
        self._colors = []
        self.update()

    # ── internals ─────────────────────────────────────────────────────────────
    def _idx_at(self, pos) -> int:
        g, s = self.GAP, self.SWATCH
        col = (pos.x() - g) // (s + g)
        row = (pos.y() - g) // (s + g)
        if 0 <= col < self.cols and 0 <= row < self.rows:
            idx = row * self.cols + col
            if idx < len(self._colors):
                return idx
        return -1

    def mouseMoveEvent(self, ev):
        idx = self._idx_at(ev.pos())
        if idx != self._hover:
            self._hover = idx
            self.update()
            self.color_hovered.emit(self._colors[idx] if idx >= 0 else None)

    def leaveEvent(self, _):
        if self._hover != -1:
            self._hover = -1
            self.update()
            self.color_hovered.emit(None)

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            idx = self._idx_at(ev.pos())
            if idx >= 0:
                self.color_selected.emit(self._colors[idx])

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(30, 30, 46))

        g, s = self.GAP, self.SWATCH
        for i, c in enumerate(self._colors):
            row = i // self.cols
            col = i % self.cols
            x = g + col * (s + g)
            y = g + row * (s + g)
            r, gr, b = c["rgb"]
            color = QColor(r, gr, b)

            if i == self._hover:
                # highlight border
                p.setPen(QPen(QColor(137, 180, 250), 2))
                p.setBrush(color)
                p.drawRoundedRect(x - 1, y - 1, s + 2, s + 2, 5, 5)
                # tooltip-style count (bottom-right tiny dot)
                p.setPen(Qt.NoPen)
                p.setBrush(QColor(255, 255, 255, 180))
                p.drawEllipse(x + s - 7, y + s - 7, 6, 6)
            else:
                p.setPen(QPen(QColor(69, 71, 90), 1))
                p.setBrush(color)
                p.drawRoundedRect(x, y, s, s, 5, 5)
        p.end()


# ──────────────────────────────────────────────────────────────────────────────
# Main window
# ──────────────────────────────────────────────────────────────────────────────
class IColoriTUIv2(QMainWindow):
    """
    Modern single-window UI for iColoriT LoRA.

    Improvements over gui_main.py:
    - Single window (no floating gamut dialogs).
    - Dark Catppuccin Mocha theme.
    - Toolbar with Load / Save / Undo / Redo / Reset.
    - Left panel: tabbed Color (gamut + palette) & Reference.
    - Drawing Pad (center) + Colorized Result (right) in a resizable splitter.
    - Status bar: hint count, zoom level, device.
    - Drag-and-drop image loading.
    - Brush size slider.
    - Zoom +/- buttons + slider in toolbar.
    """

    def __init__(self, color_model, img_file=None, load_size=224,
                 win_size=512, device="cpu", switch_model_fn=None):
        super().__init__()
        self.setWindowTitle("iColoriT LoRA")
        self.setAcceptDrops(True)
        self.setStyleSheet(DARK_QSS)

        self._start_t = time.time()
        self._device = device
        self._ref_img_pil  = None
        self._ref_img_np   = None
        self._ref_lab_cache = None
        self._switch_model_fn = switch_model_fn
        # L calibration originals
        self._orig_L = None
        self._orig_l_win = None
        self._orig_gray_win = None
        # Brightness/contrast originals (stored on first slider touch)
        self._bc_orig_L = None
        self._bc_orig_l_win = None
        self._bc_orig_gray_win = None

        # ── Core widgets ──────────────────────────────────────────────────────
        self.drawWidget = GUIDrawGTHints(
            color_model, load_size=load_size, win_size=win_size, device=device
        )
        self.drawWidget.gt_mode = False
        self.visWidget = GUI_VIS(
            win_size=win_size, scale=win_size / float(load_size)
        )
        self.visWidget.setFixedSize(win_size, win_size)
        self.visWidget.set_follow_zoom(True)

        self.gamutFree   = GUIGamut(gamut_size=110)
        self.gamutRef    = GUIGamut(gamut_size=110)
        self.palette     = GUIPalette(grid_sz=(10, 2))
        self.palette2    = GUIPalette(grid_sz=(10, 2))   # mirror in Reference tab
        self.colorSwatch = ColorSwatch(size=36)
        self.refView     = ZoomableImageView()
        self.refView.setMinimumSize(200, 160)
        self.topColors   = TopColorsWidget(cols=8, rows=2)

        # ── Zoom magnifier on the gamut widgets ───────────────────────────────
        self._gamutMag = MagnifierOverlay(self, size=300, zoom=10.0)
        self.gamutFree.installEventFilter(HoverZoomFilter(self.gamutFree, self._gamutMag))
        self.gamutRef.installEventFilter(HoverZoomFilter(self.gamutRef,  self._gamutMag))

        # ── Build UI ──────────────────────────────────────────────────────────
        self._build_menu_bar()
        self._build_toolbar()
        self._build_mask_toolbar()
        self._build_status_bar()
        self._build_central()
        self._build_left_dock()
        self._connect_signals()

        # ── Initial image ─────────────────────────────────────────────────────
        if img_file:
            self.drawWidget.init_result(img_file)

        # ── Window sizing ──────────────────────────────────────────────────────
        scr = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(1100, 680)
        self.resize(min(1600, int(scr.width() * 0.92)),
                    min(960,  int(scr.height() * 0.92)))
        geo = self.frameGeometry()
        geo.moveCenter(scr.center())
        self.move(geo.topLeft())

    # ── Menu bar ──────────────────────────────────────────────────────────────
    def _build_menu_bar(self):
        mb = self.menuBar()

        # File
        file_menu = mb.addMenu("&File")
        file_menu.addAction("📂  Open image…",     self._load,           "L")
        file_menu.addAction("💾  Save result",      self._save,           "S")
        file_menu.addAction("Save result as…",      self._save_as)
        file_menu.addSeparator()
        file_menu.addAction("📦  Save session…",    self._save_session,   "Ctrl+Shift+S")
        file_menu.addAction("📂  Load session…",    self._load_session,   "Ctrl+Shift+O")
        file_menu.addSeparator()
        file_menu.addAction("🔄  Switch Model…",    self._switch_model)
        file_menu.addSeparator()
        file_menu.addAction("📁  Load Reference…",  self._load_reference)
        file_menu.addAction("📥  Load Hints…",       self._load_hints)
        file_menu.addSeparator()
        file_menu.addAction("Quit",                 self.close,           "Ctrl+Q")

        # Edit
        edit_menu = mb.addMenu("&Edit")
        edit_menu.addAction("↩  Undo",  self.drawWidget.undo, "Ctrl+Z")
        edit_menu.addAction("↪  Redo",  self.drawWidget.redo, "Ctrl+Y")
        edit_menu.addSeparator()
        edit_menu.addAction("⟳  Reset all hints", self._reset, "R")

        # View — dock & panel toggles (populated after dock is built)
        self._view_menu = mb.addMenu("&View")
        # Gray toggle
        self._actGrayMenu = QAction("◑  Show grayscale", self)
        self._actGrayMenu.setCheckable(True)
        self._actGrayMenu.setChecked(True)
        self._actGrayMenu.setShortcut("G")
        self._actGrayMenu.triggered.connect(
            lambda checked: (
                setattr(self._actGray, "_syncing", True),
                self._actGray.setChecked(checked),
                self._toggle_gray(checked),
                setattr(self._actGray, "_syncing", False),
            )
        )
        self._view_menu.addAction(self._actGrayMenu)
        self._view_menu.addSeparator()
        # Dock actions are added after _build_left_dock() via _finalize_view_menu()

        # Help
        help_menu = mb.addMenu("&Help")
        help_menu.addAction("Keyboard shortcuts", self._show_shortcuts)
        help_menu.addAction("About iColoriT LoRA", self._show_about)

    def _finalize_view_menu(self, dock):
        """Called after the dock is created to add its toggle action to the View menu."""
        self._view_menu.addAction(dock.toggleViewAction())

    def _show_shortcuts(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.information(self, "Keyboard Shortcuts",
            "L — Load image\n"
            "S — Save result\n"
            "R — Reset all hints\n"
            "G — Toggle grayscale\n"
            "T — Toggle GT Mode (take colors from target image)\n"
            "H — Load hints from file\n"
            "Ctrl+Z — Undo\n"
            "Ctrl+Y — Redo\n"
            "+  /  − — Zoom in / out\n"
            "0 — Reset zoom\n"
            "Q — Save + Quit"
        )

    def _show_about(self):
        from PyQt5.QtWidgets import QMessageBox
        QMessageBox.about(self, "About iColoriT LoRA",
            "<b>iColoriT LoRA</b><br>"
            "Interactive image colorization using a Vision Transformer fine-tuned with LoRA.<br><br>"
            "Based on <a href='https://github.com/junyanz/interactive-deep-colorization'>"
            "interactive-deep-colorization</a>."
        )

    # ── Toolbar ───────────────────────────────────────────────────────────────
    def _build_toolbar(self):
        tb = QToolBar("Main", self)
        tb.setMovable(False)
        tb.setIconSize(QSize(20, 20))
        self.addToolBar(tb)

        def _act(text, tip, slot, checkable=False, shortcut=None):
            a = QAction(text, self)
            a.setToolTip(tip)
            if checkable:
                a.setCheckable(True)
            if shortcut:
                a.setShortcut(shortcut)
            a.triggered.connect(slot)
            tb.addAction(a)
            return a

        _act("📂 Load",    "Load image  (L)",      self._load,      shortcut="L")
        _act("💾 Save",    "Save result  (S)",     self._save,      shortcut="S")
        _act("Save As…",   "Save to custom path",  self._save_as)
        tb.addSeparator()
        _act("↩ Undo",    "Undo  (Ctrl+Z)",        self.drawWidget.undo, shortcut="Ctrl+Z")
        _act("↪ Redo",    "Redo  (Ctrl+Y)",        self.drawWidget.redo, shortcut="Ctrl+Y")
        _act("⟳ Reset",   "Reset all hints  (R)",  self._reset,     shortcut="R")
        tb.addSeparator()
        self._actGray = _act("◑ Gray", "Toggle grayscale (G)", self._toggle_gray,
                             checkable=True, shortcut="G")
        self._actGray.setChecked(True)
        self._actGTMode = _act("🎯 GT Mode", "Use ground-truth colors on click (T)",
                               self._toggle_gt_mode, checkable=True, shortcut="T")
        tb.addSeparator()

        # Zoom controls
        tb.addWidget(QLabel("Zoom:"))
        self._btnZoomOut = QPushButton("−")
        self._btnZoomOut.setFixedWidth(28)
        self._btnZoomOut.setToolTip("Zoom out  (−)")
        self._btnZoomOut.clicked.connect(self.drawWidget.zoom_out)
        tb.addWidget(self._btnZoomOut)

        self._zoomSlider = QSlider(Qt.Horizontal)
        self._zoomSlider.setRange(10, 160)   # 0.1x … 16x (×10)
        self._zoomSlider.setValue(10)
        self._zoomSlider.setFixedWidth(110)
        self._zoomSlider.setToolTip("Zoom level")
        self._zoomSlider.valueChanged.connect(self._zoom_slider_changed)
        tb.addWidget(self._zoomSlider)

        self._btnZoomIn = QPushButton("+")
        self._btnZoomIn.setFixedWidth(28)
        self._btnZoomIn.setToolTip("Zoom in  (+)")
        self._btnZoomIn.clicked.connect(self.drawWidget.zoom_in)
        tb.addWidget(self._btnZoomIn)

        self._zoomLabel = QLabel("1.0×")
        self._zoomLabel.setFixedWidth(40)
        tb.addWidget(self._zoomLabel)
        tb.addSeparator()

        # Brush size
        tb.addWidget(QLabel("Brush:"))
        self._brushSlider = QSlider(Qt.Horizontal)
        self._brushSlider.setRange(1, 30)
        self._brushSlider.setValue(8)
        self._brushSlider.setFixedWidth(90)
        self._brushSlider.setToolTip("Brush size for hints")
        self._brushSlider.valueChanged.connect(self._brush_changed)
        tb.addWidget(self._brushSlider)
        self._brushLabel = QLabel("8")
        self._brushLabel.setFixedWidth(24)
        tb.addWidget(self._brushLabel)

    # ── Mask toolbar (second row, always visible) ─────────────────────────────
    def _build_mask_toolbar(self):
        self.addToolBarBreak()
        tb = QToolBar("Mask", self)
        tb.setMovable(False)
        tb.setObjectName("maskToolbar")
        self.addToolBar(tb)

        tb.addWidget(QLabel("  Mask: "))

        self._actMaskBrush = QAction("✏  Brush", self)
        self._actMaskBrush.setToolTip("Pintar máscara  ·  Izq=pintar  Der=borrar")
        self._actMaskBrush.setCheckable(True)
        self._actMaskBrush.triggered.connect(
            lambda checked: self._set_mask_tool('brush' if checked else None))
        tb.addAction(self._actMaskBrush)

        self._actMaskRect = QAction("⬜  Rect", self)
        self._actMaskRect.setToolTip("Máscara rectangular  ·  arrastrar")
        self._actMaskRect.setCheckable(True)
        self._actMaskRect.triggered.connect(
            lambda checked: self._set_mask_tool('rect' if checked else None))
        tb.addAction(self._actMaskRect)

        self._actMaskLasso = QAction("🔷  Lasso", self)
        self._actMaskLasso.setToolTip("Lasso geométrico  ·  Izq=punto  Der=cerrar y rellenar")
        self._actMaskLasso.setCheckable(True)
        self._actMaskLasso.triggered.connect(
            lambda checked: self._set_mask_tool('lasso' if checked else None))
        tb.addAction(self._actMaskLasso)

        tb.addSeparator()

        act_cancel = QAction("⟳  Cancel", self)
        act_cancel.setToolTip("Cancelar lasso en curso")
        act_cancel.triggered.connect(self.drawWidget.cancel_lasso)
        tb.addAction(act_cancel)

        act_undo = QAction("↩  Undo shape", self)
        act_undo.setToolTip("Deshacer último shape de máscara")
        act_undo.triggered.connect(self.drawWidget.undo_mask_shape)
        tb.addAction(act_undo)

        act_clear = QAction("✕  Clear mask", self)
        act_clear.setToolTip("Borrar máscara completa (canvas e hints se conservan)")
        act_clear.triggered.connect(self._clear_mask_action)
        tb.addAction(act_clear)

        tb.addSeparator()
        tb.addWidget(QLabel("Brush sz: "))
        self._maskSzSlider = QSlider(Qt.Horizontal)
        self._maskSzSlider.setRange(3, 120)
        self._maskSzSlider.setValue(20)
        self._maskSzSlider.setFixedWidth(90)
        self._maskSzSlider.setToolTip("Tamaño del brush de máscara")
        self._maskSzSlider.valueChanged.connect(
            lambda v: setattr(self.drawWidget, 'mask_brush_size', v))
        tb.addWidget(self._maskSzSlider)

    # ── Status bar ────────────────────────────────────────────────────────────
    def _build_status_bar(self):
        sb = QStatusBar(self)
        sb.setFixedHeight(28)
        self.setStatusBar(sb)

        self._statusMsg  = QLabel("Ready")
        self._hintCount  = QLabel("Hints: 0")
        self._zoomInfo   = QLabel("Zoom: 1.0×")
        self._deviceInfo = QLabel(f"Device: {self._device}")

        for lbl in (self._statusMsg, self._hintCount, self._zoomInfo, self._deviceInfo):
            lbl.setStyleSheet("color: #a6adc8; font-size: 12px; padding: 0 12px;")

        sep = lambda: QLabel("|") if True else None
        sb.addWidget(self._statusMsg)
        sb.addPermanentWidget(QLabel("|"))
        sb.addPermanentWidget(self._hintCount)
        sb.addPermanentWidget(QLabel("|"))
        sb.addPermanentWidget(self._zoomInfo)
        sb.addPermanentWidget(QLabel("|"))
        sb.addPermanentWidget(self._deviceInfo)

    # ── Central area ──────────────────────────────────────────────────────────
    def _build_central(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(6)

        # ── Drawing Pad panel ──────────────────────────────────────────────
        padPanel = QWidget()
        padLayout = QVBoxLayout(padPanel)
        padLayout.setContentsMargins(8, 8, 8, 8)
        padLayout.setSpacing(6)

        padTitle = QLabel("Drawing Pad")
        padTitle.setObjectName("section_title")
        padLayout.addWidget(padTitle)

        hint_lbl = QLabel("Left-click: add hint  ·  Right-click: erase  ·  Scroll: zoom")
        hint_lbl.setStyleSheet("color: #6c7086; font-size: 11px;")
        padLayout.addWidget(hint_lbl)

        self.drawScroll = QScrollArea()
        self.drawScroll.setWidget(self.drawWidget)
        self.drawScroll.setWidgetResizable(False)
        self.drawScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.drawScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.drawScroll.setFrameShape(QFrame.NoFrame)
        self.drawScroll.setStyleSheet("background: #181825; border-radius: 6px;")
        padLayout.addWidget(self.drawScroll, stretch=1)

        splitter.addWidget(padPanel)

        # ── Result panel ────────────────────────────────────────────────────
        resPanel = QWidget()
        resLayout = QVBoxLayout(resPanel)
        resLayout.setContentsMargins(8, 8, 8, 8)
        resLayout.setSpacing(6)

        resTitle = QLabel("Colorized Result")
        resTitle.setObjectName("section_title")
        resLayout.addWidget(resTitle)

        res_hint = QLabel("Auto-updates when you place / move hints")
        res_hint.setStyleSheet("color: #6c7086; font-size: 11px;")
        resLayout.addWidget(res_hint)

        resScroll = QScrollArea()
        resScroll.setWidget(self.visWidget)
        resScroll.setWidgetResizable(False)
        resScroll.setFrameShape(QFrame.NoFrame)
        resScroll.setStyleSheet("background: #181825; border-radius: 6px;")
        resLayout.addWidget(resScroll, stretch=1)

        splitter.addWidget(resPanel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)

        self.setCentralWidget(splitter)

    # ── Left dock panel ───────────────────────────────────────────────────────
    def _build_left_dock(self):
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea | Qt.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)

        # Wrap everything in a scroll area so controls below the tabs are reachable
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setMinimumWidth(270)
        scroll.setMaximumWidth(330)

        container = QWidget()
        container.setMinimumWidth(260)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # ── Color swatch + label ──────────────────────────────────────────
        colorRow = QHBoxLayout()
        colorRow.addWidget(QLabel("Current color:"))
        colorRow.addStretch()
        colorRow.addWidget(self.colorSwatch)
        layout.addLayout(colorRow)

        # ── Tabs: Color / Reference ────────────────────────────────────────
        tabs = QTabWidget()

        # Tab 1: Color (gamut + palette)
        colorTab = QWidget()
        colorTabLayout = QVBoxLayout(colorTab)
        colorTabLayout.setContentsMargins(4, 4, 4, 4)
        colorTabLayout.setSpacing(8)

        gamutBox = QGroupBox("Color Gamut  (ab-space)")
        gamutBoxLayout = QVBoxLayout(gamutBox)
        gamutBoxLayout.setContentsMargins(4, 12, 4, 4)
        gamutBoxLayout.addWidget(self.gamutFree, 0, Qt.AlignCenter)
        colorTabLayout.addWidget(gamutBox)

        paletteBox = QGroupBox("Recently Used Colors")
        paletteBoxLayout = QVBoxLayout(paletteBox)
        paletteBoxLayout.setContentsMargins(4, 12, 4, 4)
        paletteBoxLayout.addWidget(self.palette, 0, Qt.AlignCenter)
        colorTabLayout.addWidget(paletteBox)
        colorTabLayout.addStretch()
        tabs.addTab(colorTab, "🎨 Color")

        # Tab 2: Reference
        refTab = QWidget()
        refTabLayout = QVBoxLayout(refTab)
        refTabLayout.setContentsMargins(4, 4, 4, 4)
        refTabLayout.setSpacing(8)

        refImgBox = QGroupBox("Reference Image")
        refImgBoxLayout = QVBoxLayout(refImgBox)
        refImgBoxLayout.setContentsMargins(4, 12, 4, 4)
        refImgBoxLayout.setSpacing(4)
        self.refView.setMinimumHeight(180)
        refImgBoxLayout.addWidget(self.refView)

        # Zoom controls for reference image
        zoomRow = QHBoxLayout()
        self._refZoomOut = QPushButton("−")
        self._refZoomOut.setFixedWidth(26)
        self._refZoomOut.clicked.connect(self.refView.zoom_out)
        zoomRow.addWidget(self._refZoomOut)
        self._refZoomSlider = QSlider(Qt.Horizontal)
        self._refZoomSlider.setRange(10, 160)
        self._refZoomSlider.setValue(10)
        self._refZoomSlider.valueChanged.connect(
            lambda v: self.refView.set_zoom(v / 10.0)
        )
        zoomRow.addWidget(self._refZoomSlider)
        self._refZoomIn = QPushButton("+")
        self._refZoomIn.setFixedWidth(26)
        self._refZoomIn.clicked.connect(self.refView.zoom_in)
        zoomRow.addWidget(self._refZoomIn)
        self._refZoomLabel = QLabel("1.0×")
        self._refZoomLabel.setFixedWidth(34)
        self._refZoomLabel.setStyleSheet("color:#a6adc8; font-size:11px;")
        zoomRow.addWidget(self._refZoomLabel)
        refImgBoxLayout.addLayout(zoomRow)

        self.refView.zoom_changed.connect(self._on_ref_zoom_changed)

        self._btnLoadRef = QPushButton("📁  Load Reference…")
        self._btnLoadRef.clicked.connect(self._load_reference)
        refImgBoxLayout.addWidget(self._btnLoadRef)
        refTabLayout.addWidget(refImgBox)

        refGamutBox = QGroupBox("Reference Color Gamut  (hover to zoom)")
        refGamutBoxLayout = QVBoxLayout(refGamutBox)
        refGamutBoxLayout.setContentsMargins(4, 12, 4, 4)
        refGamutBoxLayout.addWidget(self.gamutRef, 0, Qt.AlignCenter)
        # Hint palette mirrored in Reference tab
        refGamutBoxLayout.addWidget(self.palette2, 0, Qt.AlignCenter)
        refTabLayout.addWidget(refGamutBox)

        # ── L Calibration ──────────────────────────────────────────────────
        calBox = QGroupBox("L Calibration")
        calBoxLayout = QVBoxLayout(calBox)
        calBoxLayout.setContentsMargins(4, 12, 4, 6)
        calBoxLayout.setSpacing(6)

        calInfo = QLabel(
            "Maps target L values to reference L values\n"
            "via histogram matching so that the dominant\n"
            "color lookup uses equivalent luminance levels."
        )
        calInfo.setStyleSheet("color: #6c7086; font-size: 11px;")
        calInfo.setWordWrap(True)
        calBoxLayout.addWidget(calInfo)

        btnRow = QHBoxLayout()
        self._btnCalibrate = QPushButton("⚖  Apply L Calibration")
        self._btnCalibrate.clicked.connect(self._compute_L_calibration)
        btnRow.addWidget(self._btnCalibrate)

        self._btnResetL = QPushButton("↺  Reset")
        self._btnResetL.setEnabled(False)
        self._btnResetL.clicked.connect(self._reset_L_calibration)
        btnRow.addWidget(self._btnResetL)
        calBoxLayout.addLayout(btnRow)

        self._calStatus = QLabel("Not applied")
        self._calStatus.setStyleSheet("color: #6c7086; font-size: 11px;")
        calBoxLayout.addWidget(self._calStatus)

        refTabLayout.addWidget(calBox)

        topColorsBox = QGroupBox("Dominant Colors  (click to select)")
        topColorsBoxLayout = QVBoxLayout(topColorsBox)
        topColorsBoxLayout.setContentsMargins(4, 12, 4, 4)
        self._topColorsHint = QLabel("Load a reference image to see dominant colors")
        self._topColorsHint.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._topColorsHint.setWordWrap(True)
        topColorsBoxLayout.addWidget(self._topColorsHint)
        topColorsBoxLayout.addWidget(self.topColors, 0, Qt.AlignCenter)
        refTabLayout.addWidget(topColorsBox)
        refTabLayout.addStretch()
        tabs.addTab(refTab, "🖼 Reference")

        layout.addWidget(tabs, stretch=1)

        # ── Target: Brightness / Contrast ────────────────────────────────
        adjBox = QGroupBox("Target Adjustments")
        adjLayout = QGridLayout(adjBox)
        adjLayout.setContentsMargins(8, 14, 8, 8)
        adjLayout.setSpacing(4)

        # Brightness  −50 … +50
        adjLayout.addWidget(QLabel("Brightness"), 0, 0)
        self._brightnessSlider = QSlider(Qt.Horizontal)
        self._brightnessSlider.setRange(-50, 50)
        self._brightnessSlider.setValue(0)
        self._brightnessSlider.setTickInterval(10)
        adjLayout.addWidget(self._brightnessSlider, 0, 1)
        self._brightnessVal = QLabel("0")
        self._brightnessVal.setFixedWidth(28)
        self._brightnessVal.setStyleSheet("color:#a6adc8; font-size:11px;")
        adjLayout.addWidget(self._brightnessVal, 0, 2)

        # Contrast  50 % … 200 %  (stored ×100)
        adjLayout.addWidget(QLabel("Contrast"), 1, 0)
        self._contrastSlider = QSlider(Qt.Horizontal)
        self._contrastSlider.setRange(50, 200)
        self._contrastSlider.setValue(100)
        self._contrastSlider.setTickInterval(10)
        adjLayout.addWidget(self._contrastSlider, 1, 1)
        self._contrastVal = QLabel("1.0×")
        self._contrastVal.setFixedWidth(34)
        self._contrastVal.setStyleSheet("color:#a6adc8; font-size:11px;")
        adjLayout.addWidget(self._contrastVal, 1, 2)

        self._btnResetBC = QPushButton("↺  Reset")
        self._btnResetBC.setEnabled(False)
        adjLayout.addWidget(self._btnResetBC, 2, 0, 1, 3)

        self._brightnessSlider.valueChanged.connect(self._on_bc_slider_moved)
        self._contrastSlider.valueChanged.connect(self._on_bc_slider_moved)
        self._brightnessSlider.sliderReleased.connect(self._on_bc_slider_released)
        self._contrastSlider.sliderReleased.connect(self._on_bc_slider_released)
        self._btnResetBC.clicked.connect(self._reset_bc_adjustment)

        layout.addWidget(adjBox)
        layout.addStretch()

        scroll.setWidget(container)
        dock.setWidget(scroll)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)
        # Wire dock's show/hide into the View menu
        self._finalize_view_menu(dock)

    # ── Signals ───────────────────────────────────────────────────────────────
    def _connect_signals(self):
        # canvas geometry → result panel (geometry + size sync)
        self.drawWidget.canvas_geom_changed.connect(self.visWidget.on_canvas_geom)
        self.drawWidget.canvas_geom_changed.connect(self._sync_vis_size)
        self.drawScroll.horizontalScrollBar().valueChanged.connect(self.visWidget.on_hscroll)
        self.drawScroll.verticalScrollBar().valueChanged.connect(self.visWidget.on_vscroll)

        # color updates
        self.drawWidget.update_color.connect(self.colorSwatch.set_from_stylesheet)
        self.drawWidget.update_result.connect(self.visWidget.update_result)
        self.drawWidget.update_result.connect(self._on_result_updated)

        # gamut
        self.drawWidget.update_gammut.connect(self.gamutFree.set_gamut)
        self.drawWidget.update_ab.connect(self.gamutFree.set_ab)
        self.drawWidget.update_ab.connect(self.gamutRef.set_ab)
        self.gamutFree.update_lab.connect(self.drawWidget.set_color_from_lab)
        self.gamutRef.update_lab.connect(self.drawWidget.set_color_from_lab)

        # reference gamut update when ab changes
        self.drawWidget.update_ab.connect(
            lambda *_: self._update_ref_gamut(delta_L=2.0)
        )

        # palette (both instances stay in sync)
        self.drawWidget.used_colors.connect(self.palette.set_colors)
        self.drawWidget.used_colors.connect(self.palette2.set_colors)
        self.palette.update_color.connect(self.drawWidget.set_color)
        self.palette.update_color.connect(self.gamutFree.set_ab)
        self.palette.update_color.connect(self.gamutRef.set_ab)
        self.palette2.update_color.connect(self.drawWidget.set_color)
        self.palette2.update_color.connect(self.gamutFree.set_ab)
        self.palette2.update_color.connect(self.gamutRef.set_ab)

        # dominant color swatches
        self.topColors.color_selected.connect(self._on_top_color_selected)
        self.topColors.color_hovered.connect(self._on_top_color_hovered)

        # GT mode: show picked RGB in status bar
        self.drawWidget.update_rgb_text.connect(self._statusMsg.setText)

        # zoom slider sync (drawing widget changes zoom internally, we poll via keyPress)
        # we also connect zoom_in/zoom_out to keep slider in sync
        self._sync_zoom_timer = QTimer(self)
        self._sync_zoom_timer.setInterval(200)
        self._sync_zoom_timer.timeout.connect(self._sync_zoom_display)
        self._sync_zoom_timer.start()

    # ── Slot helpers ─────────────────────────────────────────────────────────
    def _on_ref_zoom_changed(self, scale: float):
        self._refZoomLabel.setText(f"{scale:.1f}×")
        self._refZoomSlider.blockSignals(True)
        self._refZoomSlider.setValue(max(10, min(160, int(round(scale * 10)))))
        self._refZoomSlider.blockSignals(False)

    def _sync_vis_size(self, dw, dh, ww, wh, zoom):
        """Mantiene visWidget del mismo tamaño que el drawing pad."""
        if not hasattr(self.drawWidget, "win_size"):
            return
        side = int(round(self.drawWidget.win_size * zoom))
        self.visWidget.setFixedSize(side, side)

    def _on_result_updated(self, _):
        n = len(self.drawWidget._history)
        self._hintCount.setText(f"Hints: {n}")
        self._statusMsg.setText("Updated")

    def _sync_zoom_display(self):
        if not hasattr(self.drawWidget, "zoom"):
            return
        z = self.drawWidget.zoom
        self._zoomInfo.setText(f"Zoom: {z:.1f}×")
        self._zoomLabel.setText(f"{z:.1f}×")
        slider_val = int(round(z * 10))
        self._zoomSlider.blockSignals(True)
        self._zoomSlider.setValue(max(10, min(160, slider_val)))
        self._zoomSlider.blockSignals(False)

    def _zoom_slider_changed(self, val: int):
        z = val / 10.0
        self.drawWidget.set_zoom(z)

    def _brush_changed(self, val: int):
        self._brushLabel.setText(str(val))
        if hasattr(self.drawWidget, "brushWidth"):
            self.drawWidget.brushWidth = float(val)

    # ── Actions ───────────────────────────────────────────────────────────────
    # ── Session save / load ───────────────────────────────────────────────────
    def _save_session(self):
        import json, cv2 as _cv2
        dw = self.drawWidget
        if not getattr(dw, 'image_loaded', False):
            self._statusMsg.setText("Load a target image before saving a session")
            return
        if not dw._history:
            self._statusMsg.setText("No hints to save — place at least one hint first")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "",
            "iColoriT Session (*.iclr);;All Files (*)"
        )
        if not path:
            return
        if not path.endswith(".iclr"):
            path += ".iclr"

        # ── Save target PNG at original resolution with ALL adjustments baked in ─
        # (covers BC sliders + L calibration in any combination/order)
        png_path = path.replace(".iclr", "_target.png")
        from skimage import color as ski_c

        # Derive the combined L mapping by comparing original 224×224 L
        # (recomputed from im_full) with the current adjusted 224×224 L.
        orig_bgr_224 = _cv2.resize(
            dw.im_full, (dw.load_size, dw.load_size), interpolation=_cv2.INTER_CUBIC
        )
        orig_lab_224 = ski_c.rgb2lab(orig_bgr_224[:, :, ::-1].astype(np.float32) / 255.0)
        orig_L_flat  = orig_lab_224[:, :, 0].ravel()
        curr_L_flat  = dw.im_lab[:, :, 0].ravel()

        # Sort by source L so np.interp gets a monotone lookup table
        idx = np.argsort(orig_L_flat)
        src_sorted = orig_L_flat[idx]
        dst_sorted = curr_L_flat[idx]

        # Apply the derived mapping to the full-resolution image
        im_rgb_full = dw.im_full[:, :, ::-1].astype(np.float32) / 255.0
        lab_full = ski_c.rgb2lab(im_rgb_full)
        lab_full[:, :, 0] = np.interp(lab_full[:, :, 0], src_sorted, dst_sorted).astype(np.float32)
        rgb_full = (np.clip(ski_c.lab2rgb(lab_full), 0, 1) * 255).astype(np.uint8)
        _cv2.imwrite(png_path, rgb_full[:, :, ::-1])

        # ── Build JSON payload ────────────────────────────────────────────────
        hints = []
        for act in dw._history:
            if act.get("type") != "point" or act.get("pos") is None:
                continue
            hints.append({
                "pos":       list(act["pos"]),
                "user_rgb":  list(act["user_rgb"]),
                "snap_rgb":  list(act["snap_rgb"]),
                "exact_ab":  list(act["exact_ab"]) if act.get("exact_ab") else None,
                "width":     act["width"],
            })

        # Unique hint colors (preserving order of first appearance)
        seen, palette_colors = set(), []
        for h in hints:
            key = tuple(h["user_rgb"])
            if key not in seen:
                seen.add(key)
                palette_colors.append(list(h["user_rgb"]))

        # ── Guardar committed_canvas (resultado acumulado de regiones) ────────
        canvas_basename = ""
        if dw.committed_canvas is not None:
            canvas_path = path.replace(".iclr", "_canvas.png")
            _cv2.imwrite(canvas_path, _cv2.cvtColor(dw.committed_canvas, _cv2.COLOR_RGB2BGR))
            canvas_basename = os.path.basename(canvas_path)

        # ── Guardar region_mask activa (si existe) ────────────────────────────
        mask_basename = ""
        if dw.region_mask is not None and np.any(dw.region_mask):
            mask_path = path.replace(".iclr", "_region_mask.png")
            _cv2.imwrite(mask_path, dw.region_mask * 255)
            mask_basename = os.path.basename(mask_path)

        session = {
            "version":          "1.2",
            "original_path":    getattr(dw, 'image_file', ''),
            "target_png":       os.path.basename(png_path),
            "canvas_png":       canvas_basename,
            "region_mask_png":  mask_basename,
            "brightness":       0,    # adjustments are baked into target_png
            "contrast":         100,
            "palette_colors":   palette_colors,
            "hints":            hints,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(session, f, indent=2)

        self._statusMsg.setText(
            f"Session saved — {len(hints)} hints  →  {os.path.basename(path)}"
        )

    def _load_session(self):
        import json, tempfile
        import cv2 as _cv2

        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            "iColoriT Session (*.iclr);;All Files (*)"
        )
        if not path:
            return

        with open(path, "r", encoding="utf-8") as f:
            session = json.load(f)

        session_dir = os.path.dirname(path)

        # ── Find target image ─────────────────────────────────────────────────
        # Prefer bundled PNG, fallback to original_path
        bundled_png = os.path.join(session_dir, session.get("target_png", ""))
        orig_path   = session.get("original_path", "")

        if os.path.exists(bundled_png):
            img_to_load = bundled_png
            apply_adj   = False   # adjustments already baked into the PNG
        elif os.path.exists(orig_path):
            img_to_load = orig_path
            apply_adj   = True
        else:
            from PyQt5.QtWidgets import QMessageBox
            QMessageBox.warning(self, "Session load error",
                f"Could not find target image.\n"
                f"Bundled PNG: {bundled_png}\n"
                f"Original: {orig_path}")
            return

        # ── Load image ────────────────────────────────────────────────────────
        self._clear_L_cal_originals()
        self.drawWidget.zoom = 1.0
        self.drawWidget.pos  = None
        self.drawWidget.init_result(img_to_load)

        # ── Re-apply BC if loading from original path (bundled PNG has it baked in)
        if apply_adj:
            b = session.get("brightness", 0)
            c = session.get("contrast", 100)
            if b != 0 or c != 100:
                self._brightnessSlider.blockSignals(True)
                self._contrastSlider.blockSignals(True)
                self._brightnessSlider.setValue(b)
                self._contrastSlider.setValue(c)
                self._brightnessSlider.blockSignals(False)
                self._contrastSlider.blockSignals(False)
                self._brightnessVal.setText(f"{b:+d}")
                self._contrastVal.setText(f"{c/100:.2f}×")
                self._apply_bc_adjustment(b, c / 100.0, run_model=False)
        # else: bundled PNG already has all adjustments baked in;
        #       sliders were reset to neutral by _clear_L_cal_originals() above.

        # ── Restore hints ─────────────────────────────────────────────────────
        from PyQt5.QtCore import QPoint
        from PyQt5.QtGui import QColor as QC
        history = []
        for h in session.get("hints", []):
            history.append({
                "type":     "point",
                "pos":      tuple(h["pos"]),
                "user_rgb": tuple(h["user_rgb"]),
                "snap_rgb": tuple(h["snap_rgb"]),
                "exact_ab": tuple(h["exact_ab"]) if h.get("exact_ab") else None,
                "width":    h["width"],
            })
        self.drawWidget._history = history
        self.drawWidget._redo.clear()
        self.drawWidget._replay_history()

        dw = self.drawWidget

        # ── Restaurar committed_canvas ────────────────────────────────────────
        canvas_png = session.get("canvas_png", "")
        canvas_path = os.path.join(session_dir, canvas_png)
        if canvas_png and os.path.exists(canvas_path):
            canvas_bgr = _cv2.imdecode(np.fromfile(canvas_path, dtype=np.uint8), _cv2.IMREAD_COLOR)
            if canvas_bgr is not None:
                canvas_rgb = _cv2.cvtColor(canvas_bgr, _cv2.COLOR_BGR2RGB)
                if canvas_rgb.shape[:2] != (dw.win_h, dw.win_w):
                    canvas_rgb = _cv2.resize(canvas_rgb, (dw.win_w, dw.win_h),
                                             interpolation=_cv2.INTER_CUBIC)
                dw.committed_canvas = canvas_rgb
                dw.update_result.emit(canvas_rgb)

        # ── Restaurar region_mask ─────────────────────────────────────────────
        mask_png = session.get("region_mask_png", "")
        mask_path = os.path.join(session_dir, mask_png)
        if mask_png and os.path.exists(mask_path):
            mask_gray = _cv2.imdecode(np.fromfile(mask_path, dtype=np.uint8), _cv2.IMREAD_GRAYSCALE)
            if mask_gray is not None:
                if mask_gray.shape != (dw.win_h, dw.win_w):
                    mask_gray = _cv2.resize(mask_gray, (dw.win_w, dw.win_h),
                                            interpolation=_cv2.INTER_NEAREST)
                dw.region_mask = (mask_gray > 128).astype(np.uint8)
                dw.update()

        # Restore palette explicitly from saved colors
        pal = session.get("palette_colors")
        if pal:
            pal_arr = np.array(pal, dtype=np.float32) / 255.0
            self.palette.set_colors(pal_arr)
            self.palette2.set_colors(pal_arr)

        self._hintCount.setText(f"Hints: {len(history)}")
        self._statusMsg.setText(
            f"Session loaded — {len(history)} hints  ←  {os.path.basename(path)}"
        )

    def _switch_model(self):
        if not self._switch_model_fn:
            self._statusMsg.setText("Model switching not available in this session")
            return
        model, name = self._switch_model_fn()
        if model is None:
            return
        self.drawWidget.model = model
        self.setWindowTitle(f"iColoriT LoRA  —  {name}")
        self._statusMsg.setText(f"Model switched to: {name}")

    # ── Brightness / Contrast ─────────────────────────────────────────────────
    def _on_bc_slider_moved(self):
        b = self._brightnessSlider.value()
        c = self._contrastSlider.value() / 100.0
        self._brightnessVal.setText(f"{b:+d}")
        self._contrastVal.setText(f"{c:.2f}×")
        if not getattr(self.drawWidget, 'image_loaded', False):
            return
        self._apply_bc_adjustment(b, c, run_model=False)

    def _on_bc_slider_released(self):
        if not getattr(self.drawWidget, 'image_loaded', False):
            return
        b = self._brightnessSlider.value()
        c = self._contrastSlider.value() / 100.0
        self._apply_bc_adjustment(b, c, run_model=True)

    def _apply_bc_adjustment(self, brightness: float, contrast: float,
                              run_model: bool = False):
        dw = self.drawWidget
        # Store originals on first touch
        if self._bc_orig_L is None:
            self._bc_orig_L        = dw.im_lab[:, :, 0].copy()
            self._bc_orig_l_win    = dw.l_win.copy()
            self._bc_orig_gray_win = dw.gray_win.copy()
            self._btnResetBC.setEnabled(True)

        def transform(L):
            return np.clip((L - 50.0) * contrast + 50.0 + brightness, 0.0, 100.0)

        new_L_224          = transform(self._bc_orig_L).astype(np.float32)
        dw.im_lab[:, :, 0] = new_L_224
        dw.im_l            = dw.im_lab[:, :, 0]
        dw.l_win           = transform(self._bc_orig_l_win).astype(np.float32)

        import cv2 as _cv2
        gray_u8 = np.clip(new_L_224 / 100.0 * 255.0, 0, 255).astype(np.uint8)
        dw.gray_win = _cv2.resize(
            np.stack([gray_u8, gray_u8, gray_u8], axis=-1),
            (dw.win_w, dw.win_h), interpolation=_cv2.INTER_CUBIC
        )
        dw.update()
        if run_model and dw.result is not None:
            dw.compute_result()

    def _reset_bc_adjustment(self):
        if self._bc_orig_L is None:
            return
        dw = self.drawWidget
        dw.im_lab[:, :, 0] = self._bc_orig_L
        dw.im_l            = dw.im_lab[:, :, 0]
        dw.l_win           = self._bc_orig_l_win
        dw.gray_win        = self._bc_orig_gray_win
        self._bc_orig_L        = None
        self._bc_orig_l_win    = None
        self._bc_orig_gray_win = None
        self._brightnessSlider.blockSignals(True)
        self._contrastSlider.blockSignals(True)
        self._brightnessSlider.setValue(0)
        self._contrastSlider.setValue(100)
        self._brightnessSlider.blockSignals(False)
        self._contrastSlider.blockSignals(False)
        self._brightnessVal.setText("0")
        self._contrastVal.setText("1.00×")
        self._btnResetBC.setEnabled(False)
        dw.update()
        if dw.result is not None:
            dw.compute_result()

    def _clear_L_cal_originals(self):
        self._orig_L = None
        self._orig_l_win = None
        self._orig_gray_win = None
        self._calStatus.setText("Not applied")
        self._calStatus.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._btnResetL.setEnabled(False)
        # also clear BC originals and reset sliders
        self._bc_orig_L = None
        self._bc_orig_l_win = None
        self._bc_orig_gray_win = None
        self._brightnessSlider.blockSignals(True)
        self._contrastSlider.blockSignals(True)
        self._brightnessSlider.setValue(0)
        self._contrastSlider.setValue(100)
        self._brightnessSlider.blockSignals(False)
        self._contrastSlider.blockSignals(False)
        self._brightnessVal.setText("0")
        self._contrastVal.setText("1.00×")
        self._btnResetBC.setEnabled(False)

    def _zoom_draw_to_fit(self):
        """Ajusta el zoom para que el canvas llene el scroll area disponible."""
        vp = self.drawScroll.viewport()
        vw, vh = vp.width(), vp.height()
        dw = self.drawWidget
        if not hasattr(dw, 'win_size') or dw.win_size == 0 or vw == 0 or vh == 0:
            return
        z = min(vw / dw.win_size, vh / dw.win_size)
        z = max(dw.min_zoom, min(dw.max_zoom, z))
        dw.set_zoom(z)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self.drawWidget, 'image_loaded', False):
            from PyQt5.QtCore import QTimer
            QTimer.singleShot(80, self._zoom_draw_to_fit)

    def _load(self):
        self.drawWidget.load_image()
        self._clear_L_cal_originals()
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(50, self._zoom_draw_to_fit)
        self._statusMsg.setText("Image loaded")

    def _save(self):
        self.drawWidget.save_result()
        self._statusMsg.setText(f"Saved  ({time.time() - self._start_t:.1f}s)")

    def _save_as(self):
        self.drawWidget.save_result_as()

    def _set_mask_tool(self, tool):
        self.drawWidget.set_mask_tool(tool)
        self._actMaskBrush.setChecked(tool == 'brush')
        self._actMaskRect.setChecked(tool == 'rect')
        self._actMaskLasso.setChecked(tool == 'lasso')
        msgs = {
            'brush': "Mask Brush activo  ·  Izq=pintar  Der=borrar",
            'rect':  "Mask Rect activo  ·  arrastrar para definir región  ·  Der=cancelar",
            'lasso': "Mask Lasso activo  ·  Izq=agregar punto  ·  Der=cerrar y rellenar",
        }
        self._statusMsg.setText(msgs.get(tool, "Mask tool desactivado"))

    def _clear_mask_action(self):
        self.drawWidget.clear_mask()
        self._statusMsg.setText("Máscara borrada  ·  canvas e hints conservados")

    def _reset(self):
        self.visWidget.reset()
        self.gamutFree.reset()
        self.gamutRef.reset()
        self.gamutRef.reference_mask = None
        self.palette.reset()
        self.drawWidget.reset()
        self._set_mask_tool(None)
        self._hintCount.setText("Hints: 0")
        self._statusMsg.setText("Reset")
        self.colorSwatch.set_from_stylesheet("background-color: grey")
        self.update()

    def _toggle_gray(self, checked: bool):
        if self.drawWidget.use_gray != checked:
            self.drawWidget.enable_gray()

    def _toggle_gt_mode(self, checked: bool):
        self.drawWidget.gt_mode = checked
        if checked:
            self._statusMsg.setText(
                "GT Mode ON — clicks take color from the target image directly"
            )
        else:
            self._statusMsg.setText("GT Mode OFF — manual color selection")

    def _load_hints(self):
        if not getattr(self.drawWidget, "image_loaded", False):
            return
        added = self.drawWidget.load_hints_from_file()
        if added:
            self._hintCount.setText(f"Hints: {len(self.drawWidget._history)}")
            self._statusMsg.setText(f"{added} hints loaded from file")

    # ── Reference image ───────────────────────────────────────────────────────
    def _load_reference(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select reference image", "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return
        img = Image.open(path).convert("RGB")
        img.thumbnail((1600, 1600), Image.LANCZOS)
        self._ref_img_pil = img
        self._ref_img_np  = np.array(img, dtype=np.uint8)
        self.refView.set_image_np(self._ref_img_np)
        self._ref_lab_cache = ski_color.rgb2lab(self._ref_img_np.astype(np.float32) / 255.0)
        self._update_ref_gamut(delta_L=2.0)
        # _update_ref_gamut already calls _update_top_colors(L_target) when a point is
        # selected; only fall back to the unfiltered palette when nothing is clicked yet.
        if self.drawWidget.pos is None or not hasattr(self.drawWidget, "im_lab"):
            self._update_top_colors()
        self._statusMsg.setText(f"Reference: {os.path.basename(path)}")

    def _update_top_colors(self, L_target=None, delta_L=2.0):
        if self._ref_img_pil is None:
            return
        colors = _extract_dominant_colors(self._ref_img_pil, n=16, bins=24,
                                           L_target=L_target, delta_L=delta_L)
        self.topColors.set_colors(colors)
        self._topColorsHint.setVisible(len(colors) == 0)

    def _on_top_color_hovered(self, c):
        if self._ref_img_np is None:
            return
        if c is None:
            # Restore original reference image
            self.refView.update_image_np(self._ref_img_np)
            return
        # Build highlight overlay: matching pixels in color, others dimmed to gray
        lab   = self._ref_lab_cache
        _, a_c, b_c = c["lab"]
        radius = 110.0 / 24 * 1.2   # slightly wider than one histogram bin half-width
        mask = (
            (np.abs(lab[..., 1] - a_c) < radius) &
            (np.abs(lab[..., 2] - b_c) < radius)
        )
        overlay = np.empty_like(self._ref_img_np, dtype=np.float32)
        # Non-matching: desaturate and dim
        gray = np.mean(self._ref_img_np, axis=2, keepdims=True)
        overlay[~mask] = (gray * np.ones((1, 1, 3)))[~mask] * 0.35
        # Matching: original color, slightly brightened
        overlay[mask]  = np.clip(self._ref_img_np[mask].astype(np.float32) * 1.15, 0, 255)
        self.refView.update_image_np(overlay.astype(np.uint8))

    def _on_top_color_selected(self, c: dict):
        """Called when user clicks a swatch in the TopColorsWidget."""
        L, a, b = c["lab"]
        r, g, bv = c["rgb"]
        self.drawWidget.set_color_from_lab({"rgb": [r, g, bv], "lab": [L, a, b]})
        rgb_arr = np.array([r, g, bv], dtype=np.uint8)
        self.gamutFree.set_ab(rgb_arr)
        self.gamutRef.set_ab(rgb_arr)

    # ── L Calibration ─────────────────────────────────────────────────────────
    def _compute_L_calibration(self):
        """Histogram-match target L → reference L and apply directly to the image."""
        if self._ref_img_pil is None:
            self._statusMsg.setText("Load a reference image first")
            return
        if not getattr(self.drawWidget, 'image_loaded', False):
            self._statusMsg.setText("Load a target image first")
            return

        dw = self.drawWidget

        # ── Store originals on first calibration (allows reset) ───────────────
        if self._orig_L is None:
            self._orig_L        = dw.im_lab[:, :, 0].copy()
            self._orig_l_win    = dw.l_win.copy()
            self._orig_gray_win = dw.gray_win.copy()
        else:
            # Re-apply from original so chained calibrations don't compound
            dw.im_lab[:, :, 0] = self._orig_L.copy()
            dw.im_l            = dw.im_lab[:, :, 0]
            dw.l_win           = self._orig_l_win.copy()
            dw.gray_win        = self._orig_gray_win.copy()

        # ── Build histogram-matching LUT ──────────────────────────────────────
        ref_np  = np.array(self._ref_img_pil.convert("RGB"), dtype=np.uint8)
        ref_lab = ski_color.rgb2lab(ref_np.astype(np.float32) / 255.0)
        ref_L   = ref_lab[..., 0].ravel()
        tgt_L   = self._orig_L.ravel()

        bins = 512
        tgt_hist, edges = np.histogram(tgt_L, bins=bins, range=(0.0, 100.0))
        ref_hist, _     = np.histogram(ref_L, bins=bins, range=(0.0, 100.0))

        tgt_cdf = np.cumsum(tgt_hist).astype(np.float64)
        tgt_cdf /= tgt_cdf[-1] if tgt_cdf[-1] > 0 else 1.0
        ref_cdf = np.cumsum(ref_hist).astype(np.float64)
        ref_cdf /= ref_cdf[-1] if ref_cdf[-1] > 0 else 1.0

        centers = (edges[:-1] + edges[1:]) / 2.0
        lut_mapped = np.interp(tgt_cdf, ref_cdf, centers)

        def apply_lut(L_arr):
            return np.interp(L_arr, centers, lut_mapped).astype(np.float32)

        # ── Patch drawWidget ──────────────────────────────────────────────────
        new_L_224           = apply_lut(self._orig_L)
        dw.im_lab[:, :, 0]  = new_L_224
        dw.im_l             = dw.im_lab[:, :, 0]

        dw.l_win            = apply_lut(self._orig_l_win)

        # Rebuild grayscale display: L in [0,100] → uint8 [0,255] → BGR 3ch
        gray_u8  = np.clip(new_L_224 / 100.0 * 255.0, 0, 255).astype(np.uint8)
        import cv2 as _cv2
        gray_win_new = _cv2.resize(
            np.stack([gray_u8, gray_u8, gray_u8], axis=-1),
            (dw.win_w, dw.win_h), interpolation=_cv2.INTER_CUBIC
        )
        dw.gray_win = gray_win_new

        dw.update()
        if dw.result is not None:
            dw.compute_result()

        tgt_mid = float(np.median(tgt_L))
        ref_mid = float(np.median(ref_L))
        self._calStatus.setText(
            f"Applied  ·  target median L {tgt_mid:.1f} → {ref_mid:.1f}"
        )
        self._calStatus.setStyleSheet("color: #a6e3a1; font-size: 11px;")
        self._btnResetL.setEnabled(True)
        self._statusMsg.setText("L calibration applied — target image updated")
        self._update_ref_gamut()

    def _reset_L_calibration(self):
        """Restore the original L channel in the target image."""
        if self._orig_L is None:
            return
        dw = self.drawWidget
        dw.im_lab[:, :, 0] = self._orig_L
        dw.im_l            = dw.im_lab[:, :, 0]
        dw.l_win           = self._orig_l_win
        dw.gray_win        = self._orig_gray_win
        self._orig_L        = None
        self._orig_l_win    = None
        self._orig_gray_win = None
        dw.update()
        if dw.result is not None:
            dw.compute_result()
        self._calStatus.setText("Reset to original L")
        self._calStatus.setStyleSheet("color: #6c7086; font-size: 11px;")
        self._btnResetL.setEnabled(False)
        self._statusMsg.setText("L calibration reset")
        self._update_ref_gamut()

    def _update_ref_gamut(self, delta_L=2.0):
        if self._ref_img_pil is None:
            return
        if self.drawWidget.pos is None or not hasattr(self.drawWidget, "im_lab"):
            return
        x_t, y_t = self.drawWidget.scale_point(self.drawWidget.pos)
        L_target = float(self.drawWidget.im_lab[y_t, x_t, 0])

        mask_bins = self._build_ref_mask(L_target, delta_L)
        self.gamutRef.reference_mask = mask_bins
        self.gamutFree.set_gamut(L_target)
        self.gamutRef.set_gamut(L_target)
        self._update_top_colors(L_target, delta_L)

    def _build_ref_mask(self, L0: float, delta_L: float):
        if self._ref_img_pil is None:
            return None
        grid = self.gamutRef.ab_grid
        H = W = grid.gamut_size * 2 + 1
        ref_rgb = np.array(self._ref_img_pil.convert("RGB"), dtype=np.uint8)
        ref_lab = ski_color.rgb2lab(ref_rgb.astype(np.float32) / 255.0)
        L, A, B = ref_lab[..., 0], ref_lab[..., 1], ref_lab[..., 2]
        m = np.abs(L - L0) <= delta_L
        if not np.any(m):
            return np.zeros((H, W), dtype=bool)
        a_sel, b_sel = A[m].ravel(), B[m].ravel()
        mask = np.zeros((H, W), dtype=bool)
        for a, b in zip(a_sel, b_sel):
            x, y = grid.ab2xy(float(a), float(b))
            xi, yi = int(round(x)), int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                mask[yi, xi] = True
        return mask

    # ── Drag and drop ─────────────────────────────────────────────────────────
    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls and urls[0].toLocalFile().lower().rsplit(".", 1)[-1] in (
                "png", "jpg", "jpeg", "bmp", "tif", "tiff"
            ):
                event.acceptProposedAction()

    def dropEvent(self, event):
        path = event.mimeData().urls()[0].toLocalFile()
        self._clear_L_cal_originals()
        self.drawWidget.zoom = 1.0
        self.drawWidget._history.clear()
        self.drawWidget._redo.clear()
        self.drawWidget.pos = None
        self.drawWidget.ui_mode = "none"
        self.drawWidget.save_dir = None
        self.drawWidget.init_result(path)
        from PyQt5.QtCore import QTimer
        QTimer.singleShot(50, self._zoom_draw_to_fit)
        self._hintCount.setText("Hints: 0")
        self._statusMsg.setText(f"Loaded: {os.path.basename(path)}")

    # ── Keyboard shortcuts ────────────────────────────────────────────────────
    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()
        ctrl = bool(mods & Qt.ControlModifier) or bool(mods & Qt.MetaModifier)

        if key == Qt.Key_R:
            self._reset()
        elif key == Qt.Key_T:
            self._actGTMode.setChecked(not self._actGTMode.isChecked())
            self._toggle_gt_mode(self._actGTMode.isChecked())
        elif key == Qt.Key_Q:
            self._save(); self.close()
        elif key == Qt.Key_S and not (mods & Qt.ShiftModifier):
            self._save()
        elif key == Qt.Key_G:
            self._actGray.setChecked(not self._actGray.isChecked())
            self._toggle_gray(self._actGray.isChecked())
        elif key == Qt.Key_L:
            self._load()
        elif key == Qt.Key_H:
            self._load_hints()
        elif key in (Qt.Key_Plus, Qt.Key_Equal):
            self.drawWidget.zoom_in()
        elif key in (Qt.Key_Minus, Qt.Key_Underscore):
            self.drawWidget.zoom_out()
        elif key == Qt.Key_0:
            self.drawWidget.zoom_reset()
        elif key == Qt.Key_Escape:
            self.drawWidget.cancel_lasso()
        elif ctrl and key == Qt.Key_Z and not (mods & Qt.ShiftModifier):
            self.drawWidget.undo()
        elif ctrl and ((key == Qt.Key_Z and (mods & Qt.ShiftModifier)) or key == Qt.Key_Y):
            self.drawWidget.redo()

    # ── Close ─────────────────────────────────────────────────────────────────
    def closeEvent(self, event):
        self._sync_zoom_timer.stop()
        super().closeEvent(event)
