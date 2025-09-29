
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QImage, QColor
from PyQt5.QtCore import QRect, Qt

class GUI_VIS(QWidget):
    def __init__(self, win_size=512, scale=1.0, parent=None):
        super().__init__(parent)
        self.win_size = win_size
        self.scale = scale
        self._result = None

        # Geometría BASE del pad (zoom=1)
        self._pad_dw = self._pad_dh = None
        self._pad_ww = self._pad_wh = None
        self._pad_zoom = 1.0

        # Offsets de scroll del Drawing Pad (si aplica)
        self._h_off = 0
        self._v_off = 0

        # Modo: fijo (no sigue zoom/pan) o seguir
        self._follow_zoom = False

    # --- API ---
    def set_follow_zoom(self, follow: bool):
        self._follow_zoom = bool(follow)
        self.update()

    def on_canvas_geom(self, dw: int, dh: int, ww: int, wh: int, zoom: float):
        # guardamos SIEMPRE la geometría BASE y el zoom actual
        self._pad_dw, self._pad_dh, self._pad_ww, self._pad_wh = int(dw), int(dh), int(ww), int(wh)
        self._pad_zoom = float(zoom)
        self.update()

    def on_hscroll(self, v: int):
        self._h_off = int(v)
        if self._follow_zoom:
            self.update()

    def on_vscroll(self, v: int):
        self._v_off = int(v)
        if self._follow_zoom:
            self.update()

    def update_result(self, rgb_uint8):
        self._result = rgb_uint8
        self.update()

    def match_canvas(self, dw: int, dh: int, ww: int, wh: int):
        self._pad_dw, self._pad_dh, self._pad_ww, self._pad_wh = map(int, (dw, dh, ww, wh))
        self.update()

    def reset(self):
        """
        Limpia el estado visual del widget:
        - Quita el resultado actual.
        - Restablece geometría base, zoom y offsets.
        - Mantiene el modo follow/fijo.
        """
        self._result = None

        # Geometría/zoom
        self._pad_dw = None
        self._pad_dh = None
        self._pad_ww = None
        self._pad_wh = None
        self._pad_zoom = 1.0

        # Offsets de scroll
        self._h_off = 0
        self._v_off = 0

        # Repintar vacío
        self.update()

    # --- helpers internos ---
    def _aspect_fit_rect(self, img_w: int, img_h: int, win_w: int, win_h: int) -> QRect:
        """Calcula un rectángulo de destino que mantiene aspecto y entra completo."""
        if img_w <= 0 or img_h <= 0 or win_w <= 0 or win_h <= 0:
            return QRect(0, 0, 0, 0)
        scale = min(win_w / img_w, win_h / img_h)
        draw_w = int(round(img_w * scale))
        draw_h = int(round(img_h * scale))
        x = (win_w - draw_w) // 2
        y = (win_h - draw_h) // 2
        return QRect(x, y, draw_w, draw_h)

    # --- Pintado ---
    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(ev.rect(), QColor(49, 54, 49))

        if self._result is None:
            p.end(); return

        h, w, _ = self._result.shape
        qimg = QImage(self._result.data, w, h, w * 3, QImage.Format_RGB888)

        if (self._follow_zoom and
            self._pad_dw is not None and self._pad_dh is not None and
            self._pad_ww is not None and self._pad_wh is not None):
            # MODO FOLLOW: reproduce zoom/pan del pad
            zw = int(round(self._pad_ww * self._pad_zoom))
            zh = int(round(self._pad_wh * self._pad_zoom))
            dx = int(round(self._pad_dw * self._pad_zoom)) - self._h_off
            dy = int(round(self._pad_dh * self._pad_zoom)) - self._v_off
            dest = QRect(dx, dy, zw, zh)
        else:
            # MODO NO FOLLOW: aspect‑fit a TODO el widget (sin cortes)
            dest = self._aspect_fit_rect(w, h, self.width(), self.height())

        # Suavizado al escalar
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.setRenderHint(QPainter.Antialiasing, True)
        p.drawImage(dest, qimg, QRect(0, 0, w, h))
        p.end()
