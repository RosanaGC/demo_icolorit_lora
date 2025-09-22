# gui_draw.py
import datetime
import glob
import os
import json
import cv2
import numpy as np
import torch
from einops import rearrange
from PyQt5.QtCore import QPoint, QSize, Qt, pyqtSignal, QRect
from PyQt5.QtGui import QColor, QImage, QPainter, QTransform
from PyQt5.QtWidgets import QApplication, QFileDialog, QWidget

from skimage import color

from .lab_gamut import snap_ab
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

    # NUEVO: para alinear el panel derecho
    canvas_geom_changed = pyqtSignal(int, int, int, int) # dw, dh, win_w, win_h

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
                                      int(self.win_w), int(self.win_h))

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
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()

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

        # avisar geometría base (zoom=1) al resto de la UI
        self.canvas_geom_changed.emit(self.dw, self.dh, self.win_w, self.win_h)

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
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        self.update_color.emit(str('background-color: %s' % self.color.name()))

        # Posición en coords base (zoom=1) para UIControl
        pos_base = self._to_base_coords(self.pos)

        if self.ui_mode == 'point':
            if move_point:
                self.uiControl.movePoint(pos_base, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(
                    pos_base, snap_qcolor, self.user_color, self.brushWidth
                )
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
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self._history.clear()
        self._redo.clear()
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

    def change_color(self, pos=None):
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            self.update_gammut.emit(L)

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
        snap_qcolor = c if self.pos is None else self.calibrate_color(c, self.pos)
        self.color = snap_qcolor
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
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

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
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

        # reutilizamos save_result pero sin subcarpeta: escribimos variantes con sufijo
        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
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

        ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        ab_win = ab_win * 110
        pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        self.result = pred_rgb
        self.update_result.emit(self.result)
        self.update()

    # -------------------- Pintado --------------------
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(event.rect(), QColor(49, 54, 49))
        p.setRenderHint(QPainter.Antialiasing)

        im = self.gray_win if (self.use_gray or self.result is None) else self.result
        if im is not None:
            im_c = np.ascontiguousarray(im, dtype=np.uint8)  # asegurar strides
            h, w = im_c.shape[:2]
            qImg = QImage(im_c.data, w, h, w * 3, QImage.Format_RGB888)

            dw_z, dh_z, ww_z, wh_z = self._zoom_dims()
            p.drawImage(QRect(dw_z, dh_z, ww_z, wh_z), qImg, QRect(0, 0, w, h))

        # Hints en coords base con la afín de zoom
        sx, sy, tx, ty = self._affine_params()
        p.save()
        p.setWorldTransform(QTransform(sx, 0, 0, sy, tx, ty), combine=True)
        self.uiControl.update_painter(p)
        p.restore()
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


