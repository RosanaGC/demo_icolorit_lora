# gui_main.py — principal con Drawing Pad + Result; gamuts y referencia en ventanas aparte.
# Mantiene el zoom tipo “lupa” original (MagnifierOverlay + HoverZoomFilter).
# NO se modifica gui_gamut.py.

import time
from PIL import Image

from PyQt5.QtGui import QPixmap, QImage, QTransform, QPainter, QPen, QColor
from PyQt5.QtWidgets import (
    QCheckBox, QGroupBox, QHBoxLayout, QPushButton, QVBoxLayout,
    QWidget, QFileDialog, QSizePolicy, QApplication,
    QGraphicsView, QGraphicsScene, QSplitter, QScrollArea
)
from PyQt5.QtCore import (
    Qt, QObject, QPoint, QSize, QRect, QEvent
)

import numpy as np
from skimage import color as ski_color

from .gui_draw import GUIDraw
from .gui_gamut import GUIGamut
from .gui_palette import GUIPalette
from .gui_vis import GUI_VIS


# ---------- Helpers ----------
def np_to_qpix(arr_uint8):
    h, w, _ = arr_uint8.shape
    qimg = QImage(arr_uint8.data, w, h, 3 * w, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg)


class AspectRatioContainer(QWidget):
    """Mantiene al hijo con un aspecto fijo (por defecto 1:1) y lo centra."""
    def __init__(self, child: QWidget, ratio: float = 1.0, parent=None):
        super().__init__(parent)
        self.ratio = float(ratio)
        self.child = child
        self.child.setParent(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def minimumSizeHint(self) -> QSize:
        return QSize(200, 200)

    def resizeEvent(self, e):
        W, H = self.width(), self.height()
        if W <= 0 or H <= 0:
            return
        target_w = W
        target_h = int(round(W / self.ratio))
        if target_h > H:
            target_h = H
            target_w = int(round(H * self.ratio))
        x = (W - target_w) // 2
        y = (H - target_h) // 2
        self.child.setGeometry(x, y, target_w, target_h)


# ---------- Lupa / hover-zoom (misma que tenías) ----------
class MagnifierOverlay(QWidget):
    """
    Lupa con mega-zoom que resalta la celda (píxel) exacta bajo el cursor.
    Rueda = zoom (Ctrl acelera). Shift+rueda = tamaño de la lupa.
    """
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
        self.cell_border1 = QPen(QColor(255, 255, 255, 220), 2)  # blanco
        self.cell_border2 = QPen(QColor(0, 0, 0, 220), 1)        # negro

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
        s = int(s)
        self.size = max(self.min_size, min(self.max_size, s))
        self.resize(self.size, self.size)
        self.update()

    def update_from_widget(self, src_widget: QWidget, local_pos: QPoint):
        if not src_widget:
            return
        # recorte centrado en el cursor
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
        cell_w = max(1, int(round(self._step_x)))
        cell_h = max(1, int(round(self._step_y)))
        self._cell_rect = QRect(self._piece_rect.left() + cell_x,
                                self._piece_rect.top()  + cell_y,
                                cell_w, cell_h)

        gpos = src_widget.mapToGlobal(local_pos)
        self.move(gpos + QPoint(16, 16))
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
                # líneas verticales
                cols = int(round(self._pm.width() / self._step_x)) + 1
                for i in range(cols):
                    gx = int(round(x0 + i * self._step_x))
                    p.drawLine(gx, y0, gx, y0 + self._piece_rect.height())
                # líneas horizontales
                rows = int(round(self._pm.height() / self._step_y)) + 1
                for j in range(rows):
                    gy = int(round(y0 + j * self._step_y))
                    p.drawLine(x0, gy, x0 + self._piece_rect.width(), gy)

            if not self._cell_rect.isEmpty():
                p.fillRect(self._cell_rect, self.cell_fill)
                p.setPen(self.cell_border1); p.drawRect(self._cell_rect.adjusted(0, 0, -1, -1))
                p.setPen(self.cell_border2); p.drawRect(self._cell_rect.adjusted(1, 1, -2, -2))

        # crosshair (centro)
        cx, cy = self.width() // 2, self.height() // 2
        L = max(self.cross_len, int(self.size * 0.035))
        p.setPen(QPen(QColor(255, 255, 255, 230), 3))
        p.drawLine(cx - L, cy, cx + L, cy); p.drawLine(cx, cy - L, cx, cy + L)
        p.setPen(QPen(QColor(0, 0, 0, 220), 1))
        p.drawLine(cx - L, cy, cx + L, cy); p.drawLine(cx, cy - L, cx, cy + L)
        p.end()


class HoverZoomFilter(QObject):
    """Lupa al mover el mouse; rueda = zoom (Ctrl acelera) o tamaño (Shift)."""
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
            # rueda (angleDelta) o trackpad (pixelDelta)
            dy = ev.angleDelta().y() if hasattr(ev, "angleDelta") else 0
            if dy == 0 and hasattr(ev, "pixelDelta"):
                dy = ev.pixelDelta().y()
            if dy == 0:
                return False
            if ev.modifiers() & Qt.ShiftModifier:
                delta = 40 if dy > 0 else -40
                self.overlay.set_size(self.overlay.size + delta)
            else:
                step = 1.45 if (ev.modifiers() & Qt.ControlModifier) else 1.20
                factor = step if dy > 0 else 1/step
                self.overlay.set_zoom(self.overlay.zoom * factor)
            pos = ev.pos() if hasattr(ev, "pos") else QPoint(0, 0)
            self.overlay.update_from_widget(self.owner, pos)
            return True
        return False


# ---------- Reference con zoom/pan (solo preview, sin clicks funcionales) ----------
class ZoomableImageView(QGraphicsView):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.pixmap_item = None
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.NoDrag)
        self._scale = 1.0
        self._min = 0.05
        self._max = 40.0

    def _apply_scale(self):
        t = QTransform()
        t.scale(self._scale, self._scale)
        self.setTransform(t)

    def set_image_np(self, arr_uint8):
        self.set_image_pixmap(np_to_qpix(arr_uint8))

    def set_image_pixmap(self, pixmap):
        self.scene().clear()
        self.pixmap_item = self.scene().addPixmap(pixmap)
        self.scene().setSceneRect(self.pixmap_item.boundingRect())
        self._scale = 1.0
        self._apply_scale()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def wheelEvent(self, event):
        if not self.pixmap_item:
            return
        step = 1.3 if (event.modifiers() & Qt.ControlModifier) else 1.18
        factor = step if event.angleDelta().y() > 0 else 1/step
        self._scale = max(self._min, min(self._max, self._scale * factor))
        self._apply_scale()

    def mousePressEvent(self, event):
        # Solo pan con click derecho/medio
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            super().mousePressEvent(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() in (Qt.RightButton, Qt.MiddleButton):
            self.setDragMode(QGraphicsView.NoDrag)
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
            self._scale = 1.0
            self._apply_scale()
        super().mouseDoubleClickEvent(event)


# ========= Ventanas flotantes (gamut free y referencia) =========
class GamutFreeWindow(QWidget):
    def __init__(self, gamut_widget: GUIGamut, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ab Gamut (free)")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.addLayout(self._boxed(AspectRatioContainer(gamut_widget, 1.0), "ab Gamut (free)"))
        self.resize(320, 340)

    def _boxed(self, widget, title):
        box = QGroupBox(title); box.setFlat(True)
        v = QVBoxLayout(box); v.setContentsMargins(8, 8, 8, 8); v.addWidget(widget)
        out = QVBoxLayout(); out.addWidget(box); return out


class GamutRefWindow(QWidget):
    def __init__(self, gamut_ref: GUIGamut, ref_view: QWidget,
                 ref_btn: QPushButton, used_palette: GUIPalette, color_push: QPushButton, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Reference / Gamut (reference) / Palette")
        lay = QVBoxLayout(self); lay.setContentsMargins(8, 8, 8, 8)
        # referencia
        lay.addWidget(ref_view)
        row = QHBoxLayout(); row.addWidget(ref_btn); row.addStretch(1); lay.addLayout(row)
        # gamut ref
        lay.addLayout(self._boxed(AspectRatioContainer(gamut_ref, 1.0), "ab Gamut (reference)"))
        # paleta + color
        lay.addLayout(self._boxed(used_palette, 'Recently used colors'))
        lay.addLayout(self._boxed(color_push, 'Current Color'))
        self.resize(380, 700)

    def _boxed(self, widget, title):
        box = QGroupBox(title); box.setFlat(True)
        v = QVBoxLayout(box); v.setContentsMargins(8, 8, 8, 8); v.addWidget(widget)
        out = QVBoxLayout(); out.addWidget(box); return out


# ===================== UI principal =====================
class IColoriTUI(QWidget):
    def __init__(self, color_model, img_file=None, load_size=224, win_size=256, device='cpu'):
        super().__init__()

        self.ref_img_pil = None
        self.ref_lab_cache = None

        # ===== Splitter principal (solo centro + derecha) =====
        splitter = QSplitter(Qt.Horizontal, self)

        # ----------------- Panel CENTRAL (Drawing Pad) -----------------
        centerPanel = QWidget(); center = QVBoxLayout(centerPanel)

        self.drawWidget = GUIDraw(color_model, load_size=load_size, win_size=win_size, device=device)
        self.drawWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        # Scroll para que el Drawing Pad pueda crecer con el zoom
        self.drawScroll = QScrollArea()
        self.drawScroll.setWidget(self.drawWidget)
        self.drawScroll.setWidgetResizable(False)
        self.drawScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.drawScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        center.addLayout(self._boxed(self.drawScroll, 'Drawing Pad'))

        drawPadMenu = QHBoxLayout()
        self.bGray = QCheckBox("&Gray"); self.bGray.setToolTip('show gray-scale image')
        self.bLoad = QPushButton('&Load'); self.bLoad.setToolTip('load an input image')
        self.bSave = QPushButton("&Save"); self.bSave.setToolTip('Save the current result.')
        self.bSaveAs = QPushButton("Save &As…"); self.bSaveAs.setToolTip('Choose where to save the result.')
        for w in (self.bGray, self.bLoad, self.bSave, self.bSaveAs):
            drawPadMenu.addWidget(w)
        center.addLayout(drawPadMenu)

        # ----------------- Panel DERECHO (Result) -----------------
        rightPanel = QWidget(); right = QVBoxLayout(rightPanel)

        self.visWidget = GUI_VIS(win_size=win_size, scale=win_size / float(load_size))
        visBox = self._boxed(self.visWidget, 'Colorized Result')
        right.addLayout(visBox)
        visMenu = QHBoxLayout()
        self.bRestart = QPushButton("&Restart"); self.bRestart.setToolTip('Restart the system')
        self.bQuit    = QPushButton("&Quit");    self.bQuit.setToolTip('Quit the system.')
        visMenu.addWidget(self.bRestart); visMenu.addWidget(self.bQuit)
        visBox.addLayout(visMenu)

        # Añadir paneles al splitter
        splitter.addWidget(centerPanel)
        splitter.addWidget(rightPanel)
        splitter.setStretchFactor(0, 5)   # centro
        splitter.setStretchFactor(1, 5)   # derecha
        splitter.setSizes([920, 920])

        # Layout raíz
        root = QHBoxLayout(self)
        root.addWidget(splitter)

        # ======= Widgets que van en ventanas aparte =======
        self.gamutWidgetFree = GUIGamut(gamut_size=110)
        self.gamutWidgetRef  = GUIGamut(gamut_size=110)

        self.ref_view = ZoomableImageView(); self.ref_view.setMinimumSize(220, 180)
        self.ref_img_btn = QPushButton("Reference image")
        self.usedPalette = GUIPalette(grid_sz=(10, 1))
        self.colorPush   = QPushButton(); self.colorPush.setFixedHeight(25)
        self.colorPush.setStyleSheet("background-color: grey")

        # Ventanas flotantes
        self.win_gamut_free = GamutFreeWindow(self.gamutWidgetFree)
        self.win_gamut_ref  = GamutRefWindow(self.gamutWidgetRef, self.ref_view,
                                             self.ref_img_btn, self.usedPalette, self.colorPush)

        self.win_gamut_free.show()
        self.win_gamut_ref.show()

        # ===== Lupa (igual que antes) DIRECTAMENTE en los gamuts =====
        self._magnifier = MagnifierOverlay(self, size=420, zoom=16.0)
        self.gamutWidgetFree.installEventFilter(HoverZoomFilter(self.gamutWidgetFree, self._magnifier))
        self.gamutWidgetRef.installEventFilter(HoverZoomFilter(self.gamutWidgetRef,  self._magnifier))

        # ===== Conexiones =====
        self.drawWidget.update()
        self.visWidget.update()
        self.colorPush.clicked.connect(self.drawWidget.change_color)

        self.drawWidget.canvas_geom_changed.connect(self.visWidget.match_canvas)
        self.visWidget.set_follow_zoom(False)  # fijo (no follow)

        self.drawWidget.update_color.connect(self.colorPush.setStyleSheet)
        self.drawWidget.update_result.connect(self.visWidget.update_result)

        self.drawWidget.update_gammut.connect(self.gamutWidgetFree.set_gamut)
        self.drawWidget.update_ab.connect(self.gamutWidgetFree.set_ab)
        self.drawWidget.update_ab.connect(self.gamutWidgetRef.set_ab)
        self.gamutWidgetFree.update_color.connect(self.drawWidget.set_color)
        self.gamutWidgetRef.update_color.connect(self.drawWidget.set_color)

        # recorte por L para referencia
        self.drawWidget.update_ab.connect(
            lambda *_: self.update_reference_ab_gamut_from_target(2.0, True)
        )

        # paleta
        self.drawWidget.used_colors.connect(self.usedPalette.set_colors)
        self.usedPalette.update_color.connect(self.drawWidget.set_color)
        self.usedPalette.update_color.connect(self.gamutWidgetFree.set_ab)
        self.usedPalette.update_color.connect(self.gamutWidgetRef.set_ab)

        # menús
        self.bGray.setChecked(True)
        self.bRestart.clicked.connect(self.reset)
        self.bQuit.clicked.connect(self.quit)
        self.bGray.toggled.connect(self.enable_gray)
        self.bSave.clicked.connect(self.save)
        self.bSaveAs.clicked.connect(lambda: self.drawWidget.save_result_as())
        self.bLoad.clicked.connect(self.load)
        self.ref_img_btn.clicked.connect(self.load_reference_image)

        # arranque
        self.start_t = time.time()
        if img_file is not None:
            self.drawWidget.init_result(img_file)
        print('UI initialized (principal + ventanas flotantes)')

        # Ventana principal centrada
        scr = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(1100, 680)
        self.resize(int(scr.width() * 0.9), int(scr.height() * 0.9))
        geo = self.frameGeometry(); geo.moveCenter(scr.center()); self.move(geo.topLeft())

    # ---------- Helpers UI ----------
    def _boxed(self, widget, title):
        box = QGroupBox(title)
        box.setFlat(True)
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        v.addWidget(widget)
        out = QVBoxLayout(); out.addWidget(box)
        return out

    # ---------- Acciones ----------
    def nextImage(self): self.drawWidget.nextImage()

    def reset(self):
        print('============================reset all=========================================')
        self.visWidget.reset()
        self.gamutWidgetFree.reset()
        self.gamutWidgetRef.reset()
        self.gamutWidgetRef.reference_mask = None
        self.usedPalette.reset()
        self.drawWidget.reset()
        self.update()
        self.colorPush.setStyleSheet("background-color: grey")

    def enable_gray(self): self.drawWidget.enable_gray()

    def quit(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        self.close()
        # (Opcional) cerrar ventanas flotantes:
        # self.win_gamut_free.close(); self.win_gamut_ref.close()

    def save(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        self.drawWidget.save_result()

    def load(self): self.drawWidget.load_image()

    def change_color(self):
        print('change color')
        self.drawWidget.change_color(use_suggest=True)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R: self.reset()
        if event.key() == Qt.Key_Q: self.save(); self.quit()
        if event.key() == Qt.Key_S and not (event.modifiers() & Qt.ShiftModifier): self.save()
        if event.key() == Qt.Key_G: self.bGray.toggle()
        if event.key() == Qt.Key_L: self.load()

        # Zoom rápido en Drawing Pad
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):      self.drawWidget.zoom_in()
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore): self.drawWidget.zoom_out()
        elif event.key() == Qt.Key_0:                       self.drawWidget.zoom_reset()

        # Undo / Redo
        mods = event.modifiers()
        cmd_or_ctrl = (mods & Qt.ControlModifier) or (mods & Qt.MetaModifier)
        if cmd_or_ctrl and event.key() == Qt.Key_Z and not (mods & Qt.ShiftModifier):
            self.drawWidget.undo(); return
        if cmd_or_ctrl and ((event.key() == Qt.Key_Z and (mods & Qt.ShiftModifier)) or event.key() == Qt.Key_Y):
            self.drawWidget.redo(); return

    # ---------- Reference ----------
    def load_reference_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select reference image", "", "Images (*.png *.jpg *.jpeg)")
        if not path: return
        img = Image.open(path).convert("RGB")
        img.thumbnail((1600, 1600), Image.LANCZOS)
        self.ref_img_pil = img
        ref_np = np.array(img, dtype=np.uint8)
        self.ref_view.set_image_np(ref_np)
        self.ref_lab_cache = ski_color.rgb2lab(ref_np.astype(np.float32) / 255.0)

        # si ya hay un punto elegido en el target, recalcular la máscara
        self.update_reference_ab_gamut_from_target(delta_L=2.0, send_to_gamut=True)

    # ---------- Máscara de referencia ----------
    def _render_reference_ab_gamut_via_grid(self, L0, delta_L):
        """
        Devuelve una máscara (H,W) booleana alineada EXACTA al grid del GUIGamut,
        con True donde la referencia usa (a,b) a L≈L0 (±delta_L).
        """
        if self.ref_img_pil is None:
            return None

        grid = self.gamutWidgetRef.ab_grid  # mismo grid del gamut ref
        H = W = grid.gamut_size * 2 + 1

        # Lab de la referencia
        ref_rgb = np.array(self.ref_img_pil.convert("RGB"), dtype=np.uint8)
        ref_lab = ski_color.rgb2lab(ref_rgb.astype(np.float32) / 255.0)
        L, A, B = ref_lab[..., 0], ref_lab[..., 1], ref_lab[..., 2]

        m = (np.abs(L - float(L0)) <= float(delta_L))
        if not np.any(m):
            return np.zeros((H, W), dtype=bool)

        a_sel = A[m].ravel()
        b_sel = B[m].ravel()

        mask = np.zeros((H, W), dtype=bool)

        # IMPORTANTE: forzar a enteros al indexar
        for a, b in zip(a_sel, b_sel):
            x, y = grid.ab2xy(float(a), float(b))  # puede devolver floats
            xi = int(round(x))
            yi = int(round(y))
            if 0 <= xi < W and 0 <= yi < H:
                mask[yi, xi] = True

        return mask

    def update_reference_ab_gamut_from_target(self, delta_L=2.0, send_to_gamut=True):
        if self.ref_img_pil is None:
            print("[RefGamut] No reference image loaded.")
            return
        if self.drawWidget.pos is None or getattr(self.drawWidget, "im_lab", None) is None:
            print("[RefGamut] Select a point in the target image first.")
            return

        # L objetivo tomado del punto actual de la imagen target
        x_t, y_t = self.drawWidget.scale_point(self.drawWidget.pos)
        L_target = float(self.drawWidget.im_lab[y_t, x_t, 0])

        if send_to_gamut:
            # máscara alineada al grid del gamut de referencia
            mask_bins = self._render_reference_ab_gamut_via_grid(L_target, delta_L)
            self.gamutWidgetRef.reference_mask = mask_bins

            # refrescar ambos gamuts con la misma L
            self.gamutWidgetFree.set_gamut(L_target)  # sin máscara -> gamut completo a esa L
            self.gamutWidgetRef.set_gamut(L_target)   # con máscara -> intersección con la referencia

        print(f"[RefGamut] L_target={L_target:.2f}, ΔL={delta_L}")
