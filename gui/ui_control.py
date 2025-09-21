import cv2
import numpy as np
from PyQt5.QtCore import Qt, QRect
from PyQt5.QtGui  import QPainter, QPen, QColor


from skimage import color as skcolor

MIN_HINT_PX = 7



class UserEdit(object):
    def __init__(self, mode, win_size, load_size, img_size):
        self.mode = mode
        self.win_size = win_size
        self.img_size = img_size
        self.load_size = load_size
        print('image_size', self.img_size)
        max_width = np.max(self.img_size)
        self.scale = float(max_width) / self.load_size # original image to 224 ration
        self.dw = int((self.win_size - img_size[0]) // 2)
        self.dh = int((self.win_size - img_size[1]) // 2)
        self.img_w = img_size[0]
        self.img_h = img_size[1]
        self.ui_count = 0
        print(self)

    def scale_point(self, in_x, in_y, w):
        x = int((in_x - self.dw) / float(self.img_w) * self.load_size) + w
        y = int((in_y - self.dh) / float(self.img_h) * self.load_size) + w
        return x, y

    def __str__(self):
        return "add (%s) with win_size %3.3f, load_size %3.3f" % (self.mode, self.win_size, self.load_size)


class PointEdit(UserEdit):
    def __init__(self, win_size, load_size, img_size):
        UserEdit.__init__(self, 'point', win_size, load_size, img_size)
        # --- NUEVO
        self._hint_yxs = []  # lista de (y, x) por píxel
        self._hint_ab = []  # lista de (a, b) float por píxel
        self._hw_cache = None  # (H, W) del último get_input

    def add(self, pnt, color, userColor, width, ui_count):
        self.pnt = pnt
        self.color = color
        self.userColor = userColor
        self.width = width
        self.ui_count = ui_count

    def select_old(self, pnt, ui_count):
        self.pnt = pnt
        self.ui_count = ui_count
        return self.userColor, self.width

    def update_color(self, qcolor, userColor):
        self.color = qcolor
        self.userColor = userColor

    def updateInput(self, im, mask, vis_im):
        w = int(self.width / self.scale)
        pnt = self.pnt
        x1, y1 = self.scale_point(pnt.x(), pnt.y(), -w)
        tl = (x1, y1)
        # x2, y2 = self.scale_point(pnt.x(), pnt.y(), w)
        # br = (x2, y2)
        br = (x1+1, y1+1) # hint size fixed to 2
        print(f"[DEBUG][PointEdit.updateInput] PINTANDO rect: tl={tl}, br={br}, w={w}, pnt=({pnt.x()}, {pnt.y()})")

        print(f"[DEBUG] Pintando hint en: {tl} - {br}")

        c = (self.color.red(), self.color.green(), self.color.blue())
        uc = (self.userColor.red(), self.userColor.green(), self.userColor.blue())
        cv2.rectangle(mask, tl, br, 255, -1)
        cv2.rectangle(im, tl, br, c, -1)
        cv2.rectangle(vis_im, tl, br, uc, -1)

        # --- NUEVO: recordar (a,b) del hint en coordenadas (x,y) -> guardamos (y,x) ---
        # mask shape puede ser (H, W, 1) o (H, W); tomamos H, W del propio mask
        H, W = mask.shape[:2]
        self._remember_hint(tl, br, self.userColor, hw=(H, W))

    def is_same(self, pnt):
        dx = abs(self.pnt.x() - pnt.x())
        dy = abs(self.pnt.y() - pnt.y())
        return dx <= self.width + 1 and dy <= self.width + 1

    def update_painter(self, painter):
        c = self.color
        r, g, b = c.red(), c.green(), c.blue()
        ca = QColor(r, g, b, 255)

        # tamaño final
        s = max(MIN_HINT_PX, int(round(self.width)))  # <- antes: max(1, ...)
        if s % 2 == 0:
            s += 1

        x = int(round(self.pnt.x()))
        y = int(round(self.pnt.y()))

        wasAA = painter.testRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.Antialiasing, False)

        if s == 1:
            painter.setPen(Qt.NoPen)
            painter.fillRect(x, y, 1, 1, ca)
        else:
            half = s // 2
            rect = QRect(x - half, y - half, s, s)
            painter.setPen(Qt.NoPen)
            painter.fillRect(rect, ca)

            d_to_black = r * r + g * g + b * b
            d_to_white = (255 - r) ** 2 + (255 - g) ** 2 + (255 - b) ** 2
            edge = Qt.black if d_to_white < d_to_black else Qt.white
            painter.setPen(QPen(edge, 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))

        painter.setRenderHint(QPainter.Antialiasing, wasAA)

    def _rgb_to_ab(self, userColor):
        """Convierte QColor (0..255) a Lab y devuelve (a,b)."""
        r, g, b, _ = userColor.getRgb()
        rgb = np.array([[[r / 255.0, g / 255.0, b / 255.0]]], dtype=np.float32)
        lab = skcolor.rgb2lab(rgb)  # (1,1,3) -> L,a,b
        return float(lab[0, 0, 1]), float(lab[0, 0, 2])

    def _remember_hint(self, tl, br, userQColor, hw):
        """Guarda coordenadas (y,x) y (a,b) para el rectángulo tl..br."""
        H, W = hw
        if self._hw_cache != (H, W):
            self._hint_yxs = []
            self._hint_ab = []
            self._hw_cache = (H, W)

        # QColor -> Lab -> (a,b)
        rgb01 = np.array([userQColor.red(), userQColor.green(), userQColor.blue()], dtype=np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb01.reshape(1, 1, 3)).reshape(3)
        a_val = float(lab[1])
        b_val = float(lab[2])

        x1, y1 = tl
        x2, y2 = br
        for y in range(y1, y2 + 1):
            for x in range(x1, x2 + 1):
                if 0 <= y < H and 0 <= x < W:
                    self._hint_yxs.append((y, x))
                    self._hint_ab.append((a_val, b_val))

    def get_hint_ab_map(self, H, W):
        """Devuelve (2,H,W) con (a,b) en los píxeles de este PointEdit, 0 fuera."""
        if not self._hint_yxs:
            return None
        ab_map = np.zeros((2, H, W), dtype=np.float32)
        for (y, x), (a_val, b_val) in zip(self._hint_yxs, self._hint_ab):
            if 0 <= y < H and 0 <= x < W:
                ab_map[0, y, x] = a_val
                ab_map[1, y, x] = b_val
        return ab_map

    def get_hints(self):
        """Devuelve lista [(y0,y1,x0,x1,a,b), ...]"""
        return list(self._hints)


class UIControl:
    def __init__(self, win_size=256, load_size=224):
        self.win_size = win_size
        self.load_size = load_size
        self.reset()
        self.userEdit = None
        self.userEdits = []
        self.ui_count = 0

        self.last_user_ab = None  # (a,b) del último color elegido por el usuario
        self.last_user_rgb = None  # (r,g,b) 0..255 del último color elegido
        self.mask = None  # copia de la última máscara (H,W,1) para el helper
        self.last_user_ab = None

    def setImageSize(self, img_size):
        self.img_size = img_size

    def addStroke(self, prevPnt, nextPnt, color, userColor, width):
        pass

    def erasePoint(self, pnt):
        isErase = False
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdits.remove(ue)
                print('remove user edit %d\n' % id)
                isErase = True
                break
        return isErase

    def addPoint(self, pnt, color, userColor, width):
        self.ui_count += 1
        print('process add Point')
        self.userEdit = None
        isNew = True
        for id, ue in enumerate(self.userEdits):
            if ue.is_same(pnt):
                self.userEdit = ue
                isNew = False
                print('select user edit %d\n' % id)
                break

        if self.userEdit is None:
            self.userEdit = PointEdit(self.win_size, self.load_size, self.img_size)
            self.userEdits.append(self.userEdit)
            print('add user edit %d\n' % len(self.userEdits))
            self.userEdit.add(pnt, color, userColor, width, self.ui_count)
            return userColor, width, isNew
        else:
            userColor, width = self.userEdit.select_old(pnt, self.ui_count)
            return userColor, width, isNew

    def movePoint(self, pnt, color, userColor, width):
        self.userEdit.add(pnt, color, userColor, width, self.ui_count)

    def update_color(self, qcolor, userColor):
        print(f"[DEBUG] UIControl.update_color llamado con color={qcolor.getRgb()} userColor={userColor.getRgb()}")
        if self.userEdit is not None:
            self.userEdit.update_color(qcolor, userColor)

        # Fallback opcional
        self.last_user_rgb = (userColor.red(), userColor.green(), userColor.blue())
        rgb01 = np.array(self.last_user_rgb, dtype=np.float32) / 255.0
        lab = skcolor.rgb2lab(rgb01.reshape(1, 1, 3)).reshape(3)
        self.last_user_ab = (float(lab[1]), float(lab[2]))
        # ---------------------------------------------------------------------------
        self.userEdit.update_color(qcolor, userColor)

    def update_painter(self, painter):
        for ue in self.userEdits:
            if ue is not None:
                ue.update_painter(painter)

    def get_stroke_image(self, im):
        return im

    def used_colors(self):  # get recently used colors
        if len(self.userEdits) == 0:
            return None
        nEdits = len(self.userEdits)
        ui_counts = np.zeros(nEdits)
        ui_colors = np.zeros((nEdits, 3))
        for n, ue in enumerate(self.userEdits):
            ui_counts[n] = ue.ui_count
            c = ue.userColor
            ui_colors[n, :] = [c.red(), c.green(), c.blue()]

        ui_counts = np.array(ui_counts)
        ids = np.argsort(-ui_counts)
        ui_colors = ui_colors[ids, :]
        unique_colors = []
        for ui_color in ui_colors:
            is_exit = False
            for u_color in unique_colors:
                d = np.sum(np.abs(u_color - ui_color))
                if d < 0.1:
                    is_exit = True
                    break

            if not is_exit:
                unique_colors.append(ui_color)

        unique_colors = np.vstack(unique_colors)
        return unique_colors / 255.0

    def get_input(self):
        h = self.load_size
        w = self.load_size
        im = np.zeros((h, w, 3), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)
        mask = np.zeros((h, w, 1), np.uint8)

        print("[DEBUG][UIControl.get_input] INIT mask únicos:", np.unique(mask))

        vis_im = np.zeros((h, w, 3), np.uint8)

        for ue in self.userEdits:
            ue.updateInput(im, mask, vis_im)

        print("[DEBUG][UIControl.get_input] mask únicos:", np.unique(mask))
        print("[DEBUG][UIControl.get_input] mask suma:", np.sum(mask))
        # --- NUEVO: guardar la máscara para get_hint_ab_map ---
        self.mask = mask.copy()

        return im, mask

    def reset(self):
        self.userEdits = []
        self.userEdit = None
        self.ui_count = 0

    def get_hint_ab_map(self, H, W):
        """Devuelve mapa (2,H,W) con todos los hints (a,b) de todos los PointEdit."""
        ab_total = np.zeros((2, H, W), dtype=np.float32)
        any_hint = False

        for ue in self.userEdits:
            if ue is None:
                continue
            if hasattr(ue, "get_hint_ab_map"):
                ab_local = ue.get_hint_ab_map(H, W)
                if ab_local is None:
                    continue
                m = (ab_local[0] != 0) | (ab_local[1] != 0)
                if np.any(m):
                    ab_total[:, m] = ab_local[:, m]
                    any_hint = True

        if any_hint:
            nz = int(((ab_total[0] != 0) | (ab_total[1] != 0)).sum())
            print(f"[DEBUG][UI] get_hint_ab_map: píxeles hint con (a,b) custom = {nz}")
            return ab_total
        else:
            print("[DEBUG][UI] get_hint_ab_map: sin hints -> None")
            return None


