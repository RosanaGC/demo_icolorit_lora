import json
import os

import numpy as np
from PyQt5.QtCore import QPoint, Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import QFileDialog

from .gui_draw import GUIDraw


class GUIDrawGTHints(GUIDraw):
    """Drawing pad que toma hints directamente del GT RGB cargado."""
    update_rgb_text = pyqtSignal(str)

    def _set_gt_hint_from_pos(self, pos):
        x, y = self.scale_point(pos)
        x = max(0, min(self.load_size - 1, x))
        y = max(0, min(self.load_size - 1, y))

        gt_rgb = self.im_rgb[y, x].astype(np.uint8)
        gt_ab = self.im_ab[y, x].astype(np.float32)

        self.user_color = QColor(int(gt_rgb[0]), int(gt_rgb[1]), int(gt_rgb[2]))
        self.color = QColor(int(gt_rgb[0]), int(gt_rgb[1]), int(gt_rgb[2]))
        self._exact_hint_ab = (float(gt_ab[0]), float(gt_ab[1]))
        self.update_color.emit(str('background-color: %s' % self.color.name()))
        self.update_rgb_text.emit(f"RGB: ({int(gt_rgb[0])}, {int(gt_rgb[1])}, {int(gt_rgb[2])})  HEX: {self.color.name()}")

    def change_color(self, pos=None):
        if pos is None:
            return
        self._set_gt_hint_from_pos(pos)

    def _emit_used_colors(self):
        used_colors = self.uiControl.used_colors()
        self.used_colors.emit(used_colors)

    def _pos224_to_base(self, x224, y224):
        x_base = self.dw + (float(x224) + 0.5) * self.win_w / float(self.load_size)
        y_base = self.dh + (float(y224) + 0.5) * self.win_h / float(self.load_size)
        x_base = max(self.dw, min(self.dw + self.win_w - 1, int(round(x_base))))
        y_base = max(self.dh, min(self.dh + self.win_h - 1, int(round(y_base))))
        return QPoint(x_base, y_base)

    def _parse_hint_line(self, line, line_no):
        clean = line.strip()
        if not clean or clean.startswith('#'):
            return None
        if any(ch.isalpha() for ch in clean):
            return None
        for sep in (',', ';', '\t'):
            clean = clean.replace(sep, ' ')
        parts = clean.split()
        if len(parts) < 2:
            raise ValueError(f"Línea {line_no}: se esperaban al menos 2 columnas, llegó '{line.strip()}'")
        try:
            x = int(round(float(parts[0])))
            y = int(round(float(parts[1])))
        except ValueError as exc:
            raise ValueError(f"Línea {line_no}: no pude parsear coordenadas en '{line.strip()}'") from exc
        return x, y

    def _extract_hint_coords(self, payload):
        coords = []

        def append_point(item):
            if isinstance(item, dict):
                if 'pos_224' in item:
                    point = item['pos_224']
                elif 'point' in item:
                    point = item['point']
                elif 'xy' in item:
                    point = item['xy']
                elif 'x' in item and 'y' in item:
                    point = [item['x'], item['y']]
                else:
                    return
                if len(point) < 2:
                    return
                coords.append((int(round(float(point[0]))), int(round(float(point[1])))))
                return

            if isinstance(item, (list, tuple)) and len(item) >= 2:
                coords.append((int(round(float(item[0]))), int(round(float(item[1])))))

        if isinstance(payload, dict):
            if isinstance(payload.get('hints'), list):
                for item in payload['hints']:
                    append_point(item)
            elif isinstance(payload.get('points'), list):
                for item in payload['points']:
                    append_point(item)
            else:
                append_point(payload)
        elif isinstance(payload, list):
            for item in payload:
                append_point(item)

        return coords

    def _load_hint_coords_from_file(self, file_path):
        ext = os.path.splitext(file_path)[1].lower()
        if ext == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                payload = json.load(f)
            coords = self._extract_hint_coords(payload)
            if not coords:
                raise ValueError("El JSON no contiene hints reconocibles. Usá `hints[].pos_224`, `points`, o pares `x,y`.")
            return coords

        coords = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_no, line in enumerate(f, start=1):
                parsed = self._parse_hint_line(line, line_no)
                if parsed is not None:
                    coords.append(parsed)
        if not coords:
            raise ValueError("El archivo no contiene coordenadas válidas.")
        return coords

    def apply_hint_points(self, coords_224, clear_existing=False):
        if not self.image_loaded:
            raise RuntimeError("No hay imagen cargada.")

        if clear_existing:
            self.reset()

        added = 0
        for raw_x, raw_y in coords_224:
            x224 = max(0, min(self.load_size - 1, int(raw_x)))
            y224 = max(0, min(self.load_size - 1, int(raw_y)))
            pos_base = self._pos224_to_base(x224, y224)
            self.pos = pos_base
            self.ui_mode = 'point'
            self.change_color(pos_base)
            self.update_ui(move_point=False)

            action = {
                "type": "point",
                "pos": (pos_base.x(), pos_base.y()),
                "user_rgb": (self.user_color.red(), self.user_color.green(), self.user_color.blue()),
                "snap_rgb": (self.color.red(), self.color.green(), self.color.blue()),
                "exact_ab": tuple(self._exact_hint_ab) if self._exact_hint_ab is not None else None,
                "width": self.brushWidth,
            }
            self._history.append(action)
            added += 1

        self._redo.clear()
        self._emit_used_colors()
        self.compute_result()
        return added

    def load_hints_from_file(self, file_path=None, clear_existing=False):
        if file_path is None:
            start_dir = os.path.dirname(os.path.abspath(self.image_file)) if self.image_file else ""
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select hints file",
                start_dir,
                "Hints (*.json *.txt *.csv *.tsv);;All Files (*)"
            )
            if not file_path:
                return 0

        coords = self._load_hint_coords_from_file(file_path)
        added = self.apply_hint_points(coords, clear_existing=clear_existing)
        print(f"[Hints] Cargados {added} hints desde {file_path}")
        return added

    def mousePressEvent(self, event):
        pos = self.valid_point(event.pos())
        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                self._emit_used_colors()
                self.compute_result()

                self._dragging = True
                self._current_action = {
                    "type": "point",
                    "pos": None,
                    "user_rgb": (self.user_color.red(), self.user_color.green(), self.user_color.blue()),
                    "snap_rgb": (self.color.red(), self.color.green(), self.color.blue()),
                    "exact_ab": tuple(self._exact_hint_ab) if self._exact_hint_ab is not None else None,
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
            self.change_color(self.pos)
            self.update_ui(move_point=True)
            self._emit_used_colors()
            self.compute_result()
            if self._dragging and self._current_action:
                pos_base = self._to_base_coords(self.pos)
                self._current_action["pos"] = (pos_base.x(), pos_base.y())
                self._current_action["user_rgb"] = (
                    self.user_color.red(), self.user_color.green(), self.user_color.blue()
                )
                self._current_action["snap_rgb"] = (
                    self.color.red(), self.color.green(), self.color.blue()
                )
                self._current_action["exact_ab"] = tuple(self._exact_hint_ab) if self._exact_hint_ab is not None else None

    def _replay_history(self):
        self.uiControl.reset()
        for act in self._history:
            if act.get("type") == "point" and act.get("pos") is not None:
                x, y = act["pos"]
                pos_base = QPoint(int(x), int(y))
                snap = QColor(*act["snap_rgb"])
                user = QColor(*act["user_rgb"])
                bw = act["width"]
                self.uiControl.addPoint(pos_base, snap, user, bw)
                self.uiControl.update_exact_ab(act.get("exact_ab"))
        self._emit_used_colors()
        self.compute_result()
        self.update()
