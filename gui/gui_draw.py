# gui_draw.py
import datetime
import glob
import os
import json
import time
import cv2
import numpy as np
import torch
from einops import rearrange
from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal, QRect
from PyQt5.QtGui import QColor, QFont, QImage, QPainter, QPen, QTransform
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

from skimage import color

from .lab_gamut import lab2rgb_1d, snap_ab
from .ui_control import UIControl


def pixel_mask_to_patch_mask(mask_pix, patch_size=16):
    """
    Convierte una máscara de hints en píxeles (0=bg, 1=hint) a una máscara booleana de parches (0=hint, 1=no-hint).
    Entrada:
        mask_pix: [1, H, W] o [H, W] (torch o np)
    Salida:
        patch_mask: [1, N_patches] (torch.BoolTensor)
    """
    if isinstance(mask_pix, torch.Tensor):
        mask_pix = mask_pix.cpu().numpy()
    if mask_pix.ndim == 3:
        mask_pix = mask_pix[0]
    H, W = mask_pix.shape
    mask_blocks = mask_pix.reshape(H // patch_size, patch_size, W // patch_size, patch_size)
    patch_hint = mask_blocks.max(axis=(1, 3))   # 1 si hay hint en algún pixel
    patch_mask = 1 - patch_hint                 # 0=hint, 1=no-hint
    patch_mask = patch_mask.astype(bool).reshape(1, -1)
    return torch.from_numpy(patch_mask)


class GUIDraw(QWidget):
    # Señales
    update_color = pyqtSignal(str)
    update_gammut = pyqtSignal(object)
    used_colors = pyqtSignal(object)
    update_ab = pyqtSignal(object)
    update_result = pyqtSignal(object)
    update_l_color = pyqtSignal(object)   # emite rgb uint8 del gris L del pixel clickeado

    # NUEVO: para alinear el panel derecho
    canvas_geom_changed = pyqtSignal(int, int, int, int, float) # dw, dh, win_w, win_h, zoom

    def __init__(self, model=None, load_size=224, win_size=512, device='cpu'):
        super().__init__()
        self.image_file = None
        self.pos = None
        self.model = model
        self.win_size = win_size
        self.load_size = load_size
        self.device = device

        # el tamaño visible del widget lo controlamos con el zoom
        self.setFixedSize(win_size, win_size)

        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.movie = True
        self.init_color()
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'   # 'stroke' | 'point' | 'erase' | 'none'
        self.image_loaded = False
        self.use_gray = True

        # ---- Zoom del Drawing Pad ----
        self.zoom = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 16.0

        # ---- Historial Undo/Redo ----
        self._history = []      # acciones aplicadas
        self._redo = []         # acciones deshechas
        self._dragging = False
        self._current_action = None

        # ---- Carpeta de guardado (opcional) ----
        self.save_dir = None

        # ---- Máscara / Region compositing ----
        self.mask_tool        = None   # 'brush' | 'rect' | 'lasso' | None
        self.region_mask      = None   # uint8 ndarray (win_h, win_w) o None
        self.committed_canvas = None   # uint8 ndarray (win_h, win_w, 3) o None
        self.mask_brush_size  = 20     # px en espacio de imagen (win_w × win_h)
        self._mask_shapes     = []     # list[np.ndarray] — una por lasso/rect, para undo
        self._mask_rect_start = None   # QPoint
        self._mask_lasso_pts  = []     # list[QPoint]
        self._mask_is_drawing = False
        self._mask_cursor_pos = None

        # ---- Canvas a resolución completa: se persiste en disco, no en RAM ----
        # Se actualiza solo al cerrar una máscara o guardar (ver
        # _sync_full_res_canvas / _fullres_cache_path) — nunca en cada hint.
        self._fullres_dirty = False  # cambios pendientes de llevar al cache full-res

    # -------------------- Helpers zoom/coords --------------------
    def _zoom_dims(self):
        """(dw_z, dh_z, ww_z, wh_z) del área de imagen con el zoom actual."""
        z = float(getattr(self, "zoom", 1.0))
        return (int(round(self.dw * z)),
                int(round(self.dh * z)),
                int(round(self.win_w * z)),
                int(round(self.win_h * z)))

    def _emit_canvas_geom(self):
        self.canvas_geom_changed.emit(int(self.dw), int(self.dh),
                                      int(self.win_w), int(self.win_h), float(self.zoom))

    def _affine_params(self):
        """
        Devuelve (sx, sy, tx, ty) tal que:
          [x']   [sx  0  tx][x]
          [y'] = [ 0 sy  ty][y]
        donde (x,y) están en coords base (zoom=1) y (x',y') en coords actuales.
        """
        dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
        sx = ww_z / float(self.win_w) if self.win_w else 1.0
        sy = wh_z / float(self.win_h) if self.win_h else 1.0
        tx = dw_z - sx * self.dw
        ty = dh_z - sy * self.dh
        return sx, sy, tx, ty

    def _to_base_coords(self, pos_widget_now: QPoint) -> QPoint:
        """Widget (con zoom) -> coords base (zoom=1) para UIControl."""
        sx, sy, tx, ty = self._affine_params()
        x0 = (pos_widget_now.x() - tx) / sx
        y0 = (pos_widget_now.y() - ty) / sy
        return QPoint(int(round(x0)), int(round(y0)))

    def _base_to_224_xy(self, pos_base: QPoint):
        """Mapea coords base (zoom=1) -> coords 224x224 del modelo."""
        x224 = int((pos_base.x() - self.dw) / float(self.win_w) * self.load_size)
        y224 = int((pos_base.y() - self.dh) / float(self.win_h) * self.load_size)
        x224 = max(0, min(self.load_size - 1, x224))
        y224 = max(0, min(self.load_size - 1, y224))
        return x224, y224

    # -------------------- Init / IO --------------------
    def init_result(self, image_file):
        self.read_image(image_file)
        self.reset()

    def read_image(self, image_file):
        self.image_loaded = True
        self.image_file = image_file
        # cv2.imread fails on Windows paths with non-ASCII characters; use imdecode instead
        raw = np.fromfile(image_file, dtype=np.uint8)
        im_bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if im_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_file}")
        self.im_full = im_bgr.copy()
        lab_full = color.rgb2lab(im_bgr[:, :, ::-1])
        self.l_full = lab_full[:, :, 0]
        self.result_full = None

        # preparar imagen ajustada al lienzo
        h, w, _ = self.im_full.shape
        max_side = max(h, w)
        r = self.win_size / float(max_side)
        self.scale = float(self.win_size) / self.load_size
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh

        self.uiControl.setImageSize((rw, rh))

        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)

        im_bgr_224 = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr_224, cv2.COLOR_BGR2RGB)

        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])
        self.l_win = lab_win[:, :, 0]

        self.im_lab = color.rgb2lab(im_bgr_224[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        #self.brushWidth = 2 * self.scale
        self.brushWidth = 1.0

        # Resetear máscara al cargar nueva imagen
        self.region_mask      = None
        self.committed_canvas = None
        self._mask_shapes     = []
        self._mask_rect_start = None
        self._mask_lasso_pts  = []
        self._mask_is_drawing = False
        self._mask_cursor_pos = None
        self._fullres_dirty = False

        # avisar geometría base (zoom=1) al resto de la UI
        self.canvas_geom_changed.emit(self.dw, self.dh, self.win_w, self.win_h, float(self.zoom))

        # resetear tamaño del widget acorde al zoom actual
        self.set_zoom(self.zoom)


    def get_batches(self, img_dir):
        self.img_list = glob.glob(os.path.join(img_dir, '*.JPEG'))
        self.total_images = len(self.img_list)
        self.image_id = 0
        if self.total_images:
            self.init_result(self.img_list[0])

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id >= len(self.img_list):
            print('you have finished all the results')
            return
        self.init_result(self.img_list[self.image_id])

    # -------------------- Zoom API --------------------
    def set_zoom(self, z: float):
        z = max(self.min_zoom, min(self.max_zoom, float(z)))
        if z != self.zoom:
            self.zoom = z
            # redimensionar el lienzo AQUÍ (no en paintEvent)
            canvas = int(round(self.win_size * self.zoom))
            self.setFixedSize(canvas, canvas)

            # notificar al panel derecho para alinear
            self.canvas_geom_changed.emit(
                int(round(self.dw * self.zoom)),
                int(round(self.dh * self.zoom)),
                int(round(self.win_w * self.zoom)),
                int(round(self.win_h * self.zoom)),
                float(self.zoom),
            )
            self.update()
            self._emit_canvas_geom()

    def zoom_in(self):  self.set_zoom(self.zoom * 1.25)
    def zoom_out(self): self.set_zoom(self.zoom / 1.25)
    def zoom_reset(self): self.set_zoom(1.0)

    # -------------------- UI / Hints --------------------
    def update_ui(self, move_point=True):
        if self.ui_mode == 'none':
            return False
        is_predict = False

        # Color “snap” usando L local (scale_point ya contempla zoom)
        if self._exact_hint_ab is not None:
            x, y = self.scale_point(self.pos)
            lab = np.array([self.im_l[y, x], self._exact_hint_ab[0], self._exact_hint_ab[1]], dtype=np.float32)
            exact_rgb = lab2rgb_1d(lab, clip=True, dtype='uint8')
            snap_qcolor = QColor(int(exact_rgb[0]), int(exact_rgb[1]), int(exact_rgb[2]))
        else:
            snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        self.update_color.emit(str('background-color: %s' % self.color.name()))

        # Posición en coords base (zoom=1) para UIControl
        pos_base = self._to_base_coords(self.pos)

        if self.ui_mode == 'point':
            if move_point:
                self.uiControl.movePoint(pos_base, snap_qcolor, self.user_color, self.brushWidth)
                self.uiControl.update_exact_ab(self._exact_hint_ab)
            else:
                self.user_color, self.brushWidth, exact_ab, isNew = self.uiControl.addPoint(
                    pos_base, snap_qcolor, self.user_color, self.brushWidth
                )
                if exact_ab is not None:
                    self._exact_hint_ab = exact_ab
                self.uiControl.update_exact_ab(self._exact_hint_ab)
                if isNew:
                    is_predict = True

        if self.ui_mode == 'stroke':
            prev_base = self._to_base_coords(self.prev_pos) if getattr(self, "prev_pos", None) else pos_base
            self.uiControl.addStroke(prev_base, pos_base, snap_qcolor, self.user_color, self.brushWidth)

        if self.ui_mode == 'erase':
            isRemoved = self.uiControl.erasePoint(pos_base)
            if isRemoved:
                is_predict = True
        return is_predict

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.result_full = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self._history.clear()
        self._redo.clear()
        self.region_mask      = None
        self.committed_canvas = None
        self._mask_shapes     = []
        self._mask_rect_start = None
        self._mask_lasso_pts  = []
        self._mask_is_drawing = False
        self._mask_cursor_pos = None
        self._fullres_dirty = False
        # Nota: no se sincroniza el canvas full-res acá — correr esto al
        # cargar cada imagen agrega una recomputación bloqueante justo al
        # terminar de cargar. Se hace perezosamente al cerrar una máscara o
        # guardar (ver _sync_full_res_canvas).
        self.compute_result()
        self.update()

    def scale_point(self, pnt):
        """Widget (con zoom) -> coords 224x224 para el modelo."""
        dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
        x = int((pnt.x() - dw_z) / float(ww_z) * self.load_size)
        y = int((pnt.y() - dh_z) / float(wh_z) * self.load_size)
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            return None
        dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
        if (pnt.x() >= dw_z and pnt.y() >= dh_z and
            pnt.x() <  dw_z + ww_z and pnt.y() < dh_z + wh_z):
            return QPoint(int(round(pnt.x())), int(round(pnt.y())))
        return None

    def init_color(self):
        self.user_color = QColor(128, 128, 128)  # gris por defecto
        self.color = self.user_color
        self._exact_hint_ab = None

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            self.update_gammut.emit(L)

            # Emitir el gris equivalente al valor L del píxel
            gray_rgb = (np.clip(color.lab2rgb(np.array([[[L, 0.0, 0.0]]])), 0, 1) * 255
                        ).astype(np.uint8)[0, 0]
            self.update_l_color.emit(gray_rgb)

            used_colors = self.uiControl.used_colors()
            self.used_colors.emit(used_colors)

            snap_color = self.calibrate_color(self.user_color, pos)
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)
            self.update_ab.emit(c)

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)
        color_array = np.array((c.red(), c.green(), c.blue()), dtype='uint8')
        mean_L = self.im_l[y, x]
        snap_color = snap_ab(mean_L, color_array)
        return QColor(int(snap_color[0]), int(snap_color[1]), int(snap_color[2]))

    def set_color(self, c_rgb):
        # llamada desde el gamut: si no hay self.pos, evitamos crash
        c = QColor(int(c_rgb[0]), int(c_rgb[1]), int(c_rgb[2]))
        self.user_color = c
        self._exact_hint_ab = None
        snap_qcolor = c if self.pos is None else self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()
        self.update()

    def set_color_from_lab(self, payload):
        rgb = np.array(payload["rgb"], dtype=np.uint8)
        lab = np.array(payload["lab"], dtype=np.float32)
        self.user_color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self.color = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
        self._exact_hint_ab = (float(lab[1]), float(lab[2]))
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.uiControl.update_color(self.color, self.user_color)
        self.uiControl.update_exact_ab(self._exact_hint_ab)
        self.compute_result()
        self.update()

    # -------------------- Guardado --------------------
    def save_result(self, dir_path: str = None, make_subdir: bool = True):
        """
        Guarda:
          - ours.png (resultado)
          - input_mask.png (máscara binaria)
          - hints_rgba.png (color de hints con alpha)
          - hints.json (lista de hints con colores/posiciones)
          - L_with_hints.png (L* en gris con hints coloreados superpuestos)
        """
        if self.result is None:
            print("[Save] No hay resultado para guardar.")
            return

        if dir_path is None:
            if not self.save_dir:
                base_default = os.path.dirname(os.path.abspath(self.image_file)) if self.image_file else os.getcwd()
                chosen = QFileDialog.getExistingDirectory(self, "carpeta de destino", base_default)
                if not chosen:
                    print("[Save] Cancelado por el usuario.")
                    return
                self.save_dir = chosen
            dir_path = self.save_dir
        else:
            self.save_dir = dir_path

        ts = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        dst_dir = os.path.join(dir_path, f"icolor_{ts}") if make_subdir else dir_path
        os.makedirs(dst_dir, exist_ok=True)

        # Canvas a resolución completa, persistido en disco (ver
        # _sync_full_res_canvas). Se sincroniza siempre acá, para reflejar la
        # máscara/hints activos aunque la región no se haya cerrado todavía.
        self._sync_full_res_canvas()
        _, cache_path = self._fullres_cache_path()
        cached = self._read_png_rgb(cache_path) if cache_path and os.path.exists(cache_path) else None
        if cached is not None:
            out = cached
        elif self.result_full is not None:
            out = self.result_full
        else:
            out = self.result
        result_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        mask_bin = (self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255)  # (H,W,1)
        cv2.imwrite(os.path.join(dst_dir, "ours.png"), result_bgr)
        cv2.imwrite(os.path.join(dst_dir, "input_mask.png"), mask_bin)

        # reconstruimos RGBA hints en 224×224
        im_ui, mask_ui = self.uiControl.get_input()
        H, W = im_ui.shape[:2]
        im_u8 = (np.clip(im_ui * 255.0 + 0.5, 0, 255).astype(np.uint8)
                 if im_ui.dtype != np.uint8 else im_ui)
        mask01 = (mask_ui.squeeze() > 0).astype(np.uint8)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., 0:3][mask01 == 1] = im_u8[..., 0:3][mask01 == 1]
        rgba[..., 3] = (mask01 * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(dst_dir, "hints_rgba.png"),
                    cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        # JSON con metadata
        hints_meta = []
        for act in self._history:
            if act.get("type") == "point" and act.get("pos") is not None:
                xb, yb = act["pos"]
                pos_base = QPoint(int(xb), int(yb))
                x224, y224 = self._base_to_224_xy(pos_base)

                snap_rgb = act["snap_rgb"]
                user_rgb = act["user_rgb"]

                rgb_1 = (np.array(snap_rgb, dtype=np.float32) / 255.0)[None, None, :]
                lab_snap = color.rgb2lab(rgb_1).reshape(-1).tolist()  # [L,a,b]
                L_here = float(self.im_lab[y224, x224, 0])

                hints_meta.append({
                    "pos_base": [int(xb), int(yb)],
                    "pos_224":  [int(x224), int(y224)],
                    "L_at_224": L_here,
                    "user_rgb": [int(user_rgb[0]), int(user_rgb[1]), int(user_rgb[2])],
                    "snap_rgb": [int(snap_rgb[0]), int(snap_rgb[1]), int(snap_rgb[2])],
                    "snap_lab": {"L": float(lab_snap[0]), "a": float(lab_snap[1]), "b": float(lab_snap[2])},
                    "width":    float(act.get("width", 0.0)),
                })

        meta = {
            "image_file": self.image_file,
            "save_time": ts,
            "load_size": int(self.load_size),
            "hints_count": len(hints_meta),
            "hints": hints_meta,
        }
        with open(os.path.join(dst_dir, "hints.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # L* con hints superpuestos
        L224 = self.im_lab[..., 0]
        L8   = np.clip((L224 / 100.0) * 255.0, 0, 255).astype(np.uint8)
        L3   = cv2.cvtColor(L8, cv2.COLOR_GRAY2RGB)

        L_overlay = L3.copy()
        kernel = np.ones((3, 3), np.uint8)
        dil1   = cv2.dilate(mask01, kernel, iterations=1)
        border = (dil1 - mask01).astype(bool)
        dil2   = cv2.dilate(dil1, kernel, iterations=1)
        border2 = (dil2 - dil1).astype(bool)

        L_overlay[border2] = [0, 0, 0]
        L_overlay[border]  = [255, 255, 255]
        L_overlay[mask01 == 1] = im_u8[mask01 == 1, :3]
        cv2.imwrite(os.path.join(dst_dir, "L_with_hints.png"),
                    cv2.cvtColor(L_overlay, cv2.COLOR_RGB2BGR))

        print(f"[Save] Guardado en {dst_dir}")

    def save_result_as(self):
        if self.result is None:
            print("[Save As] No hay resultado para guardar.")
            return
        base_dir = os.path.dirname(os.path.abspath(self.image_file)) if self.image_file else os.getcwd()
        suggested = os.path.join(base_dir, "icolor.png")
        path, _ = QFileDialog.getSaveFileName(
            self, "Save As… (nombre base)",
            suggested,
            "PNG Image (*.png);;All Files (*)"
        )
        if not path:
            print("[Save As] Cancelado por el usuario.")
            return
        base_path, _ = os.path.splitext(path)
        out_dir = os.path.dirname(base_path)
        os.makedirs(out_dir, exist_ok=True)

        # Canvas a resolución completa, persistido en disco (ver
        # _sync_full_res_canvas). Se sincroniza siempre acá, para reflejar la
        # máscara/hints activos aunque la región no se haya cerrado todavía.
        self._sync_full_res_canvas()
        _, cache_path = self._fullres_cache_path()
        cached = self._read_png_rgb(cache_path) if cache_path and os.path.exists(cache_path) else None
        if cached is not None:
            out = cached
        elif self.result_full is not None:
            out = self.result_full
        else:
            out = self.result
        result_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"{base_path}_ours.png", result_bgr)

        mask_bin = (self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255)
        cv2.imwrite(f"{base_path}_mask.png", mask_bin)

        im_ui, mask_ui = self.uiControl.get_input()
        H, W = im_ui.shape[:2]
        im_u8 = (np.clip(im_ui * 255.0 + 0.5, 0, 255).astype(np.uint8)
                 if im_ui.dtype != np.uint8 else im_ui)
        mask01 = (mask_ui.squeeze() > 0).astype(np.uint8)

        rgba = np.zeros((H, W, 4), dtype=np.uint8)
        rgba[..., 0:3][mask01 == 1] = im_u8[..., 0:3][mask01 == 1]
        rgba[..., 3] = (mask01 * 255).astype(np.uint8)
        cv2.imwrite(f"{base_path}_hints_rgba.png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        L224 = self.im_lab[..., 0]
        L8 = np.clip((L224 / 100.0) * 255.0, 0, 255).astype(np.uint8)
        L3 = cv2.cvtColor(L8, cv2.COLOR_GRAY2RGB)

        L_overlay = L3.copy()
        kernel = np.ones((3, 3), np.uint8)
        dil1 = cv2.dilate(mask01, kernel, iterations=1)
        border = (dil1 - mask01).astype(bool)
        dil2 = cv2.dilate(dil1, kernel, iterations=1)
        border2 = (dil2 - dil1).astype(bool)

        L_overlay[border2] = [0, 0, 0]
        L_overlay[border] = [255, 255, 255]
        L_overlay[mask01 == 1] = im_u8[mask01 == 1, :3]
        cv2.imwrite(f"{base_path}_L_with_hints.png", cv2.cvtColor(L_overlay, cv2.COLOR_RGB2BGR))

        hints_meta = []
        for act in self._history:
            if act.get("type") == "point" and act.get("pos") is not None:
                xb, yb = act["pos"]
                pos_base = QPoint(int(xb), int(yb))
                x224, y224 = self._base_to_224_xy(pos_base)
                snap_rgb = act["snap_rgb"]
                user_rgb = act["user_rgb"]

                rgb_1 = (np.array(snap_rgb, dtype=np.float32) / 255.0)[None, None, :]
                lab_snap = color.rgb2lab(rgb_1).reshape(-1).tolist()
                L_here = float(self.im_lab[y224, x224, 0])

                hints_meta.append({
                    "pos_base": [int(xb), int(yb)],
                    "pos_224": [int(x224), int(y224)],
                    "L_at_224": L_here,
                    "user_rgb": [int(user_rgb[0]), int(user_rgb[1]), int(user_rgb[2])],
                    "snap_rgb": [int(snap_rgb[0]), int(snap_rgb[1]), int(snap_rgb[2])],
                    "snap_lab": {"L": float(lab_snap[0]), "a": float(lab_snap[1]), "b": float(lab_snap[2])},
                    "width": float(act.get("width", 0.0)),
                })

        meta = {
            "image_file": self.image_file,
            "save_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "load_size": int(self.load_size),
            "hints_count": len(hints_meta),
            "hints": hints_meta,
        }
        with open(f"{base_path}_hints.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        self.save_dir = out_dir
        print("[Save As] Listo:", out_dir)

    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()


    def load_image(self):
        """Abre un diálogo y carga una nueva imagen target."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select target image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*)"
        )
        if not path:
            return

        # Reset mínimo al cambiar de imagen:
        self.zoom = 1.0
        self._history.clear()
        self._redo.clear()
        self.pos = None
        self.ui_mode = 'none'
        self.save_dir = None

        # Reusar tu flujo normal de carga
        self.init_result(path)  # -> read_image(...) -> compute_result() -> update()

    # -------------------- Modelo --------------------
    def compute_result(self):
        im, mask = self.uiControl.get_input()
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))  # (1,H,W)
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))  # (3,H,W)
        self.im_ab0 = im_lab[1:3, :, :]
        hint_ab = self.uiControl.get_hint_ab_map(self.load_size, self.load_size)
        if hint_ab is not None:
            self.im_ab0 = hint_ab

        _im_lab = self.im_lab.transpose((2, 0, 1))
        _im_lab = np.concatenate(((_im_lab[[0], :, :] - 50) / 100, _im_lab[1:, :, :] / 110), axis=0)
        _im_lab = torch.from_numpy(_im_lab).type(torch.FloatTensor).to(self.device)

        _img_mask = np.concatenate((self.im_ab0 / 110, (255 - self.im_mask0) / 255), axis=0)
        _img_mask = torch.from_numpy(_img_mask).type(torch.FloatTensor).to(self.device)

        ab = self.model(_im_lab.unsqueeze(0), _img_mask.unsqueeze(0))
        ab = rearrange(ab, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                       h=self.load_size // self.model.patch_size,
                       w=self.load_size // self.model.patch_size,
                       p1=self.model.patch_size, p2=self.model.patch_size)[0]
        ab = ab.detach().numpy()
        self._fullres_dirty = True

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb

        # ── Compositing ───────────────────────────────────────────────────────
        # NO CAMBIAR: lógica de referencia acordada en bug_fix branch (2026-06-15).
        # Con máscara activa: copia hard-copy de pred_rgb solo en la región de la
        # máscara sobre el canvas acumulado.  Sin máscara: reemplaza canvas completo.
        # Este comportamiento es la base estable desde la cual se hacen mejoras.
        if self.region_mask is not None and np.any(self.region_mask):
            if self.committed_canvas is None:
                # NO CAMBIAR: inicializar canvas en negro la primera vez
                self.committed_canvas = np.zeros((self.win_h, self.win_w, 3), dtype=np.uint8)
            canvas = self.committed_canvas.copy()
            m = self.region_mask.astype(bool)
            canvas[m] = pred_rgb[m]   # NO CAMBIAR: hard copy de píxeles de la máscara
            self.committed_canvas = canvas
            self.update_result.emit(canvas)
        else:
            # NO CAMBIAR: sin máscara, el canvas es la colorización global completa
            self.committed_canvas = pred_rgb.copy()
            self.update_result.emit(pred_rgb)
        self.update()

    def _fullres_cache_path(self):
        """(carpeta, archivo) del canvas de trabajo a resolución completa,
        persistido en disco junto a la imagen — no en RAM. Devuelve (None, None)
        si todavía no hay imagen cargada."""
        if not self.image_file:
            return None, None
        base_dir = os.path.dirname(os.path.abspath(self.image_file))
        stem = os.path.splitext(os.path.basename(self.image_file))[0]
        out_dir = os.path.join(base_dir, f"carpeta_colorizacion_{stem}")
        return out_dir, os.path.join(out_dir, f"{stem}_fullres.png")

    def _read_png_rgb(self, path):
        """Lee un PNG a RGB. Usa imdecode (no imread) por compat con paths
        no-ASCII en Windows, igual que read_image()."""
        raw = np.fromfile(path, dtype=np.uint8)
        bgr = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _write_png_rgb(self, path, rgb):
        """Escribe un PNG desde RGB. Usa imencode + tofile (no imwrite) por
        compat con paths no-ASCII en Windows, igual que read_image()."""
        ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if ok:
            buf.tofile(path)

    def _sync_full_res_canvas(self):
        """Actualiza el archivo de trabajo a resolución completa en disco sin
        volver a ejecutar el modelo.

        `committed_canvas` es la fuente de verdad de lo que ve el usuario. Se
        lleva su Lab completo a resolución original y se agrega solamente el
        detalle fino de luminancia de `l_full`. Así se preservan exactamente
        los colores y todas las regiones ya compuestas, sin reducir el archivo
        final a un simple upscale borroso de 512 px.

        Se persiste en disco (no como array en RAM) para no perder el trabajo
        si la app se cuelga, y para no mantener un array grande vivo toda la
        sesión.

        Esto es caro para fotos grandes (varios segundos) — se llama solo en
        acciones deliberadas y poco frecuentes: cerrar una máscara
        (clear_mask), guardar sesión (.iclr) y guardar el resultado final
        (save_result/save_result_as). Nunca en mouseMoveEvent, al poner un
        hint, ni al cargar la imagen.
        """
        if (not self._fullres_dirty or
                self.im_full is None or self.committed_canvas is None):
            return
        out_dir, cache_path = self._fullres_cache_path()
        if cache_path is None:
            return

        t0 = time.time()
        h_full, w_full = self.im_full.shape[:2]
        canvas_lab = color.rgb2lab(self.committed_canvas).astype(np.float32)
        canvas_lab_full = cv2.resize(canvas_lab, (w_full, h_full),
                                     interpolation=cv2.INTER_CUBIC)

        # Separar del original sólo la textura que desaparece al llevarlo al
        # tamaño del canvas. El color y la luminosidad de escala grande siguen
        # viniendo de committed_canvas, por lo que el resultado coincide con
        # la vista incluso tras varias regiones congeladas.
        l_full = self.l_full.astype(np.float32, copy=False)
        l_low = cv2.resize(l_full, (self.win_w, self.win_h),
                           interpolation=cv2.INTER_AREA)
        l_low_full = cv2.resize(l_low, (w_full, h_full),
                                interpolation=cv2.INTER_CUBIC)
        canvas_lab_full[..., 0] = np.clip(
            canvas_lab_full[..., 0] + l_full - l_low_full, 0, 100
        )
        out_img = (np.clip(color.lab2rgb(canvas_lab_full), 0, 1) * 255).astype(np.uint8)

        # Mantener negro puro en zonas deliberadamente no comprometidas.
        untouched = np.all(self.committed_canvas == 0, axis=2)
        if np.any(untouched):
            untouched_full = cv2.resize(untouched.astype(np.uint8),
                                         (w_full, h_full),
                                         interpolation=cv2.INTER_NEAREST).astype(bool)
            out_img[untouched_full] = 0

        os.makedirs(out_dir, exist_ok=True)
        self._write_png_rgb(cache_path, out_img)
        self._fullres_dirty = False

        print(f"[PERF] _sync_full_res_canvas: {time.time() - t0:.3f}s "
              f"(size={w_full}x{h_full}) -> {cache_path}")

    # -------------------- Pintado --------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(event.rect(), QColor(30, 30, 46))
        if not self.image_loaded:
            p.setPen(QColor(69, 71, 90))
            p.setFont(QFont("Arial", 14))
            p.drawText(event.rect(), Qt.AlignCenter, "Drop an image here\nor use  📂 Load")
            p.end()
            return
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        im = getattr(self, 'gray_win', None) if (self.use_gray or self.result is None) else self.result
        if im is not None:
            im_c = np.ascontiguousarray(im, dtype=np.uint8)  # asegurar strides
            h, w = im_c.shape[:2]
            qImg = QImage(im_c.data, w, h, w * 3, QImage.Format_RGB888)

            dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
            p.drawImage(QRect(dw_z, dh_z, ww_z, wh_z), qImg, QRect(0, 0, w, h))

        # Overlay semitransparente de la máscara activa
        self._draw_mask_overlay(p)

        # Hints en coords base con la afín de zoom
        sx, sy, tx, ty = self._affine_params()
        p.save()
        p.setWorldTransform(QTransform(sx, 0, 0, sy, tx, ty), combine=True)
        self.uiControl.update_painter(p)
        p.restore()

        # Preview del tool de máscara en curso (rect/lasso)
        self._draw_mask_preview(p)
        p.end()

    def sizeHint(self):
        return QSize(int(self.win_size * self.zoom), int(self.win_size * self.zoom))

    # -------------------- Mouse --------------------
    def is_same_point(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        dx = pos1.x() - pos2.x()
        dy = pos1.y() - pos2.y()
        return (dx * dx + dy * dy) < 25

    def mousePressEvent(self, event):
        pos = self.valid_point(event.pos())
        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)             # fija L, actualiza gamut/palette
                self.update_ui(move_point=False)   # crea/selecciona el hint
                self.compute_result()

                # Iniciar acción de historial
                self._dragging = True
                snap_q = self.calibrate_color(self.user_color, self.pos)
                self._current_action = {
                    "type": "point",
                    "pos": None,  # se setea luego
                    "user_rgb": (self.user_color.red(), self.user_color.green(), self.user_color.blue()),
                    "snap_rgb": (snap_q.red(), snap_q.green(), snap_q.blue()),
                    "width": self.brushWidth,
                }

            if event.button() == Qt.RightButton:
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                self.compute_result()

    def mouseMoveEvent(self, event):
        self.pos = self.valid_point(event.pos())
        if self.pos is not None and self.ui_mode == 'point':
            self.update_ui(move_point=True)
            self.compute_result()
            if self._dragging and self._current_action:
                pos_base = self._to_base_coords(self.pos)
                self._current_action["pos"] = (pos_base.x(), pos_base.y())

    def mouseReleaseEvent(self, event):
        if self.handle_mask_release(event):
            return
        if self._dragging and self._current_action and self.ui_mode == 'point' and self.pos is not None:
            if not self._current_action.get("pos"):
                pos_base = self._to_base_coords(self.pos)
                self._current_action["pos"] = (pos_base.x(), pos_base.y())
            self._history.append(self._current_action)
            self._redo.clear()
        self._dragging = False
        self._current_action = None
        super().mouseReleaseEvent(event)

    # -------------------- Undo / Redo --------------------
    def _replay_history(self):
        """Reconstruye el estado de UIControl a partir de _history."""
        self.uiControl.reset()
        for act in self._history:
            if act.get("type") == "point" and act.get("pos") is not None:
                x, y = act["pos"]
                pos_base = QPoint(int(x), int(y))
                snap = QColor(*act["snap_rgb"])
                user = QColor(*act["user_rgb"])
                bw = act["width"]
                self.uiControl.addPoint(pos_base, snap, user, bw)
        self.compute_result()
        self.update()

    def undo(self):
        if self.mask_tool is not None and self._mask_shapes:
            self.undo_mask_shape()
            return
        if not self._history:
            return
        last = self._history.pop()
        self._redo.append(last)
        self._replay_history()

    def redo(self):
        if not self._redo:
            return
        act = self._redo.pop()
        self._history.append(act)
        self._replay_history()

    # ── Máscara / Region compositing ─────────────────────────────────────────

    def set_mask_tool(self, tool):
        """Activa un tool de máscara: 'brush' | 'rect' | 'lasso' | None."""
        self.mask_tool        = tool
        self._mask_rect_start = None
        self._mask_lasso_pts  = []
        self._mask_is_drawing = False
        self._mask_cursor_pos = None
        self.update()

    def clear_mask(self):
        """Borra solo la máscara activa (canvas e hints se conservan)."""
        # Cerrar la región acá: sincronizar el canvas full-res en disco con su
        # estado final antes de soltarla — costo concentrado en esta acción
        # deliberada, no en cada hint puesto mientras la máscara estaba activa.
        if self.region_mask is not None and np.any(self.region_mask):
            self._sync_full_res_canvas()
        self.region_mask      = None
        self._mask_shapes     = []
        self._mask_rect_start = None
        self._mask_lasso_pts  = []
        self._mask_is_drawing = False
        self._mask_cursor_pos = None
        self.update()

    def _rebuild_region_mask(self):
        if not self._mask_shapes:
            self.region_mask = None
            return
        result = np.zeros((self.win_h, self.win_w), dtype=np.uint8)
        for shape in self._mask_shapes:
            np.bitwise_or(result, shape, out=result)
        self.region_mask = result

    def undo_mask_shape(self):
        if self._mask_shapes:
            self._mask_shapes.pop()
            self._rebuild_region_mask()
            self.update()

    def cancel_lasso(self):
        self._mask_lasso_pts  = []
        self._mask_is_drawing = False
        self._mask_cursor_pos = None
        self.update()

    def _ensure_mask(self):
        if not hasattr(self, 'win_h') or not hasattr(self, 'win_w'):
            return
        if self.region_mask is None:
            self.region_mask = np.zeros((self.win_h, self.win_w), dtype=np.uint8)

    def _widget_to_img(self, pos):
        """Convierte posición en el widget (con zoom) a coords de imagen (win_w × win_h)."""
        dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
        x = int((pos.x() - dw_z) / max(1, ww_z) * self.win_w)
        y = int((pos.y() - dh_z) / max(1, wh_z) * self.win_h)
        return (max(0, min(self.win_w - 1, x)),
                max(0, min(self.win_h - 1, y)))

    def _paint_brush_mask(self, pos, erase=False):
        self._ensure_mask()
        x, y = self._widget_to_img(pos)
        cv2.circle(self.region_mask, (x, y), max(1, self.mask_brush_size),
                   0 if erase else 1, -1)
        self.update()

    def _fill_rect_mask(self, p1, p2):
        self._ensure_mask()
        x1, y1 = self._widget_to_img(p1)
        x2, y2 = self._widget_to_img(p2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        shape = np.zeros((self.win_h, self.win_w), dtype=np.uint8)
        shape[y1:y2 + 1, x1:x2 + 1] = 1
        self._mask_shapes.append(shape)
        np.bitwise_or(self.region_mask, shape, out=self.region_mask)
        self.update()

    def _fill_lasso_mask(self, pts):
        if len(pts) < 3:
            return
        self._ensure_mask()
        shape = np.zeros((self.win_h, self.win_w), dtype=np.uint8)
        img_pts = np.array([self._widget_to_img(p) for p in pts], dtype=np.int32)
        cv2.fillPoly(shape, [img_pts], 1)
        self._mask_shapes.append(shape)
        np.bitwise_or(self.region_mask, shape, out=self.region_mask)
        self.update()

    def _draw_mask_overlay(self, painter):
        if self.region_mask is None or not np.any(self.region_mask):
            return
        if not hasattr(self, 'win_h') or not hasattr(self, 'win_w'):
            return
        dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
        H, W = self.region_mask.shape
        overlay = np.zeros((H, W, 4), dtype=np.uint8)
        overlay[self.region_mask.astype(bool)] = [100, 149, 237, 90]
        overlay_c = np.ascontiguousarray(overlay)
        qimg = QImage(overlay_c.data, W, H, 4 * W, QImage.Format_RGBA8888)
        painter.drawImage(QRect(dw_z, dh_z, ww_z, wh_z), qimg, QRect(0, 0, W, H))

    def _draw_mask_preview(self, painter):
        if not self.image_loaded:
            return
        if (self.mask_tool == 'rect' and self._mask_is_drawing
                and self._mask_rect_start and self._mask_cursor_pos):
            pen = QPen(QColor(100, 149, 237, 200), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            p1, p2 = self._mask_rect_start, self._mask_cursor_pos
            painter.drawRect(QRect(
                min(p1.x(), p2.x()), min(p1.y(), p2.y()),
                abs(p1.x() - p2.x()), abs(p1.y() - p2.y())
            ))
        elif self.mask_tool == 'lasso' and self._mask_lasso_pts:
            pen = QPen(QColor(100, 149, 237, 200), 2, Qt.SolidLine)
            painter.setPen(pen)
            pts = self._mask_lasso_pts
            for i in range(len(pts) - 1):
                painter.drawLine(pts[i], pts[i + 1])
            if self._mask_cursor_pos and self._mask_is_drawing:
                painter.drawLine(pts[-1], self._mask_cursor_pos)
                if len(pts) >= 2:
                    # closing edge: cursor → first point
                    close_pen = QPen(QColor(100, 149, 237, 110), 1, Qt.DashLine)
                    painter.setPen(close_pen)
                    painter.drawLine(self._mask_cursor_pos, pts[0])

    def handle_mask_press(self, event):
        """Devuelve True si el evento fue consumido por el tool de máscara."""
        if self.mask_tool is None or not self.image_loaded:
            return False
        pos = event.pos()
        if self.mask_tool == 'brush':
            if event.button() == Qt.LeftButton:
                self._paint_brush_mask(pos, erase=False)
            elif event.button() == Qt.RightButton:
                self._paint_brush_mask(pos, erase=True)
            return True
        elif self.mask_tool == 'rect':
            if event.button() == Qt.LeftButton:
                self._mask_rect_start = pos
                self._mask_is_drawing = True
            elif event.button() == Qt.RightButton:
                self._mask_rect_start = None
                self._mask_is_drawing = False
                self.update()
            return True
        elif self.mask_tool == 'lasso':
            if event.button() == Qt.LeftButton:
                if not self._mask_is_drawing:
                    self._mask_lasso_pts = [pos]
                    self._mask_is_drawing = True
                else:
                    self._mask_lasso_pts.append(pos)
            elif event.button() == Qt.RightButton:
                self._fill_lasso_mask(self._mask_lasso_pts)
                self._mask_lasso_pts  = []
                self._mask_is_drawing = False
                self.update()
            return True
        return False

    def handle_mask_move(self, event):
        """Devuelve True si el evento fue consumido por el tool de máscara."""
        if self.mask_tool is None:
            return False
        pos = event.pos()
        self._mask_cursor_pos = pos
        if self.mask_tool == 'brush':
            if event.buttons() & Qt.LeftButton:
                self._paint_brush_mask(pos, erase=False)
            elif event.buttons() & Qt.RightButton:
                self._paint_brush_mask(pos, erase=True)
            return True
        elif self.mask_tool in ('rect', 'lasso'):
            self.update()
            return True
        return False

    def handle_mask_release(self, event):
        """Devuelve True si el evento fue consumido por el tool de máscara."""
        if self.mask_tool is None:
            return False
        if (self.mask_tool == 'rect' and self._mask_is_drawing
                and self._mask_rect_start and event.button() == Qt.LeftButton):
            self._fill_rect_mask(self._mask_rect_start, event.pos())
            self._mask_rect_start = None
            self._mask_is_drawing = False
        return True  # cualquier release mientras hay mask_tool activo se consume


