# gui_gamut.py
import cv2
import numpy as np
from PyQt5.QtCore import QPointF, QSize, Qt, pyqtSignal
from PyQt5.QtGui import QImage, QPainter
from PyQt5.QtWidgets import QWidget

from .lab_gamut import abGrid, lab2rgb_1d, rgb2lab_1d


class GUIGamut(QWidget):
    """
    Selector ab con muestreo directo del mapa RGB escalado.
    - Sin cruz persistente ni ejes punteados.
    - Valida con máscara a la misma escala del widget.
    """
    update_color = pyqtSignal(object)  # emite np.uint8[3] (RGB)

    def __init__(self, gamut_size=110):
        super().__init__()
        self.gamut_size = int(gamut_size)
        self.win_size = self.gamut_size * 2
        self.setFixedSize(self.win_size, self.win_size)

        self.ab_grid = abGrid(gamut_size=self.gamut_size, D=1)

        # estado
        self.reference_mask = None  # máscara opcional desde la imagen de referencia (en bins del grid)
        self.l_in = 50

        self.ab_map = None          # mapa en resolución del grid
        self.mask = None            # máscara en resolución del grid
        self.ab_map_up = None       # mapa escalado al tamaño del widget (para pintar y samplear)
        self.mask_up = None         # máscara escalada al tamaño del widget (para validar clic)

        self.pos = None
        self.mouseClicked = False

        # mejoras de interacción
        self.setMouseTracking(True)

        self.reset()

    # ---------------- API ----------------
    def set_gamut(self, l_in=50):
        """Actualiza el mapa ab para una L fija y opcionalmente restringe por reference_mask."""
        self.l_in = float(l_in)
        self.ab_map, self.mask = self.ab_grid.update_gamut(l_in=self.l_in,
                                                           ref_mask=self.reference_mask)
        # Escalamos a tamaño del widget y cacheamos para usar en pintura y sampleo
        self.ab_map_up = cv2.resize(self.ab_map, (self.win_size, self.win_size),
                                    interpolation=cv2.INTER_LINEAR)
        # Asegurar binaria {0,1} antes de escalar con nearest
        if self.mask is not None:
            mask_u8 = (self.mask.astype(np.uint8) > 0).astype(np.uint8)
            self.mask_up = cv2.resize(mask_u8, (self.win_size, self.win_size),
                                      interpolation=cv2.INTER_NEAREST)
        else:
            self.mask_up = None

        self.update()

    def set_ab(self, color_rgb):
        """Recibe un RGB (uint8[3]) externo y mueve el puntero interno (opcional)."""
        # Mantengo compatibilidad por si otro widget llama a esto.
        # Sólo actualizo pos si el color cae dentro del gamut; si no, lo ignoro silenciosamente.
        self.color = np.array(color_rgb, dtype=np.uint8)
        self.lab = rgb2lab_1d(self.color)
        # Convertimos a x,y del grid y lo llevamos a coords del widget
        x_g, y_g = self.ab_grid.ab2xy(self.lab[1], self.lab[2])
        # x_g, y_g están en resolución del grid -> escalar a widget
        if 0 <= x_g < self.gamut_size * 2 + 1 and 0 <= y_g < self.gamut_size * 2 + 1:
            sx = self.win_size / float(self.gamut_size * 2 + 1)
            sy = self.win_size / float(self.gamut_size * 2 + 1)
            self.pos = QPointF(x_g * sx, y_g * sy)
        self.update()

    # ---------------- Interna ----------------
    def _is_valid_point(self, pos):
        if pos is None or self.mask_up is None:
            return False
        x = int(pos.x()); y = int(pos.y())
        if x < 0 or y < 0 or x >= self.win_size or y >= self.win_size:
            return False
        return bool(self.mask_up[y, x])  # misma escala: 1 dentro, 0 fuera

    def _emit_color_from_pos(self, pos):
        """Lee el RGB directamente del mapa escalado (ab_map_up) y lo emite."""
        if self.ab_map_up is None:
            return
        x = int(np.clip(pos.x(), 0, self.win_size - 1))
        y = int(np.clip(pos.y(), 0, self.win_size - 1))
        rgb = self.ab_map_up[y, x].astype(np.uint8)
        self.update_color.emit(rgb)

    # ---------------- Qt Overrides ----------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, False)
        p.fillRect(event.rect(), Qt.white)

        if self.ab_map_up is not None:
            qImg = QImage(self.ab_map_up.data, self.win_size, self.win_size,
                          self.win_size * 3, QImage.Format_RGB888)
            p.drawImage(0, 0, qImg)

        # NO dibujamos ejes ni cruz persistente
        p.end()

    def mousePressEvent(self, event):
        pos = event.pos()
        if event.button() == Qt.LeftButton and self._is_valid_point(pos):
            self.mouseClicked = True
            self.pos = pos
            self._emit_color_from_pos(pos)
            self.update()

    def mouseMoveEvent(self, event):
        pos = event.pos()
        # Sólo mientras está cliqueado arrastramos y actualizamos color
        if self.mouseClicked and self._is_valid_point(pos):
            self.pos = pos
            self._emit_color_from_pos(pos)
            self.update()

    def mouseReleaseEvent(self, event):
        self.mouseClicked = False

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)

    def reset(self):
        self.ab_map = None
        self.mask = None
        self.ab_map_up = None
        self.mask_up = None
        self.color = None
        self.lab = None
        self.pos = None
        self.mouseClicked = False
        self.reference_mask = None
        self.update()

