"""
iColoriT LoRA — Modern UI (v2)
Usage:
    python icolorit_ui_v2.py                            # model picker dialog
    python icolorit_ui_v2.py --model_path <path>        # skip picker, load directly
    python icolorit_ui_v2.py --target_image <path>      # open with image pre-loaded
"""
import sys
import os
import argparse
import warnings

from PyQt5.QtGui import QIcon, QColor, QFont
from PyQt5.QtWidgets import (
    QApplication, QSplashScreen, QDialog, QVBoxLayout, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QPushButton, QAbstractItemView,
)
from PyQt5.QtCore import Qt
from timm.models import create_model
import torch

from gui import gui_main_v2
import modeling


# ── Model catalog ──────────────────────────────────────────────────────────────
# Add entries here to extend the picker list.
MODEL_CATALOG = [
    {
        "name":        "Original",
        "description": "iColoriT base — pretrained weights, no LoRA",
        "path":        "checks/icolorit_base_4ch_patch16_224_original",
        "rank":        0,
    },
    {
        "name":        "Torres",
        "description": "LoRA r32 — fine-tuned on Torres dataset (1000 epochs)",
        "path":        "checks/Lora_r32_Dataset_4_epochs_1000_wES_lrf_1e-3",
        "rank":        32,
    },
]

_DIALOG_QSS = """
QDialog, QWidget {
    background-color: #1e1e2e;
    color: #cdd6f4;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}
QLabel#title {
    color: #89b4fa;
    font-size: 18px;
    font-weight: 700;
}
QLabel#subtitle {
    color: #a6adc8;
    font-size: 12px;
}
QListWidget {
    background-color: #181825;
    border: 1px solid #313244;
    border-radius: 8px;
    padding: 4px;
    outline: none;
}
QListWidget::item {
    padding: 10px 14px;
    border-radius: 6px;
    color: #cdd6f4;
}
QListWidget::item:selected {
    background-color: #313244;
    color: #89b4fa;
}
QListWidget::item:hover {
    background-color: #27273a;
}
QPushButton {
    background-color: #313244;
    border: 1px solid #45475a;
    border-radius: 6px;
    padding: 7px 22px;
    color: #cdd6f4;
    font-weight: 600;
}
QPushButton:hover   { background-color: #45475a; border-color: #89b4fa; }
QPushButton:pressed { background-color: #89b4fa; color: #1e1e2e; }
QPushButton#primary {
    background-color: #89b4fa;
    color: #1e1e2e;
    border-color: #89b4fa;
}
QPushButton#primary:hover   { background-color: #b4d0ff; }
QPushButton#primary:pressed { background-color: #6c9fe8; }
QPushButton#primary:disabled { background-color: #45475a; color: #6c7086; border-color: #45475a; }
"""


class ModelPickerDialog(QDialog):
    """Simple modal dialog for choosing a model from MODEL_CATALOG."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("iColoriT LoRA — Select Model")
        self.setStyleSheet(_DIALOG_QSS)
        self.setMinimumWidth(420)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        self._selected = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 20)
        layout.setSpacing(14)

        title = QLabel("iColoriT LoRA")
        title.setObjectName("title")
        layout.addWidget(title)

        subtitle = QLabel("Select a model to load:")
        subtitle.setObjectName("subtitle")
        layout.addWidget(subtitle)

        self._list = QListWidget()
        self._list.setSelectionMode(QAbstractItemView.SingleSelection)
        for entry in MODEL_CATALOG:
            item = QListWidgetItem(f"  {entry['name']}\n  {entry['description']}")
            item.setSizeHint(item.sizeHint().__class__(0, 56))
            self._list.addItem(item)
        self._list.setCurrentRow(0)
        self._list.itemDoubleClicked.connect(self._accept)
        layout.addWidget(self._list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(10)

        self._btnCancel = QPushButton("Cancel")
        self._btnCancel.clicked.connect(self.reject)
        btn_row.addWidget(self._btnCancel)

        btn_row.addStretch()

        self._btnOK = QPushButton("Load Model")
        self._btnOK.setObjectName("primary")
        self._btnOK.setDefault(True)
        self._btnOK.clicked.connect(self._accept)
        btn_row.addWidget(self._btnOK)

        layout.addLayout(btn_row)

    def _accept(self):
        row = self._list.currentRow()
        if row >= 0:
            self._selected = MODEL_CATALOG[row]
            self.accept()

    def selected_entry(self):
        return self._selected


# ── CLI args ───────────────────────────────────────────────────────────────────
def get_args():
    parser = argparse.ArgumentParser("iColoriT LoRA — Modern UI", add_help=False)
    parser.add_argument("--model_path",   type=str, default="",
                        help="Path to checkpoint (skips model picker if provided)")
    parser.add_argument("--target_image", type=str, default="",
                        help="Path to image to colorize")
    parser.add_argument("--device",       type=str, default="cpu")
    parser.add_argument("--input_size",   type=int, default=224)
    parser.add_argument("--model",        type=str, default="icolorit_base_4ch_patch16_224")
    parser.add_argument("--drop_path",    type=float, default=0.0)
    parser.add_argument("--use_rpb",      action="store_true")
    parser.add_argument("--no_use_rpb",   action="store_false", dest="use_rpb")
    parser.set_defaults(use_rpb=True)
    parser.add_argument("--avg_hint",     action="store_true")
    parser.add_argument("--no_avg_hint",  action="store_false", dest="avg_hint")
    parser.set_defaults(avg_hint=True)
    parser.add_argument("--head_mode",    type=str, default="cnn")
    parser.add_argument("--mask_cent",    action="store_true")
    parser.add_argument("--rank",         type=int, default=32)
    parser.add_argument("--win_size",     type=int, default=512)
    return parser.parse_args()


def _make_splash(app: QApplication, model_name: str) -> QSplashScreen:
    from PyQt5.QtGui import QPixmap, QPainter
    pix = QPixmap(520, 120)
    pix.fill(QColor(30, 30, 46))
    p = QPainter(pix)
    p.setPen(QColor(137, 180, 250))
    p.setFont(QFont("Segoe UI", 18, QFont.Bold))
    p.drawText(pix.rect(), Qt.AlignCenter, f'Loading  "{model_name}"…')
    p.end()
    splash = QSplashScreen(pix, Qt.WindowStaysOnTopHint)
    splash.show()
    app.processEvents()
    return splash


def _make_switch_fn(args):
    """Returns a callable that shows the model picker and loads the chosen model."""
    def switch():
        picker = ModelPickerDialog()
        if picker.exec_() != QDialog.Accepted or picker.selected_entry() is None:
            return None, None
        entry = picker.selected_entry()
        path  = _resolve_path(entry["path"])
        rank  = entry["rank"]
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            model = create_model(
                args.model,
                pretrained=False,
                drop_path_rate=args.drop_path,
                use_rpb=args.use_rpb,
                avg_hint=args.avg_hint,
                head_mode=args.head_mode,
                mask_cent=args.mask_cent,
                rank=rank,
            )
            model.to(args.device)
            checkpoint = torch.load(path, map_location="cpu")
            model.load_state_dict(checkpoint["model"], strict=False)
            model.eval()
        finally:
            QApplication.restoreOverrideCursor()
        return model, entry["name"]
    return switch


def _resolve_path(p: str) -> str:
    if not os.path.isabs(p):
        base = getattr(sys, "_MEIPASS", os.path.dirname(os.path.abspath(__file__)))
        p = os.path.join(base, p)
    return p


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = get_args()

    app = QApplication(sys.argv)
    app.setApplicationName("iColoriT LoRA")
    app.setOrganizationName("iColorIT")

    # ── Model selection ───────────────────────────────────────────────────────
    if args.model_path:
        # CLI override: use exactly what was passed
        model_path = _resolve_path(args.model_path)
        model_rank = args.rank
        model_name = os.path.basename(model_path)
    else:
        picker = ModelPickerDialog()
        if picker.exec_() != QDialog.Accepted or picker.selected_entry() is None:
            sys.exit(0)
        entry = picker.selected_entry()
        model_path = _resolve_path(entry["path"])
        model_rank = entry["rank"]
        model_name = entry["name"]
        args.rank   = model_rank

    # ── Load model ────────────────────────────────────────────────────────────
    splash = _make_splash(app, model_name)

    print(f"[main] Creating model: {args.model}  rank={model_rank}")
    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        use_rpb=args.use_rpb,
        avg_hint=args.avg_hint,
        head_mode=args.head_mode,
        mask_cent=args.mask_cent,
        rank=model_rank,
    )
    model.to(args.device)

    if model_path:
        print(f"[main] Loading weights: {model_path}")
        checkpoint = torch.load(model_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"], strict=False)
    else:
        print("[main] WARNING: no model path, running with random weights.")

    model.eval()
    splash.finish(None)

    # ── Launch UI ─────────────────────────────────────────────────────────────
    target = _resolve_path(args.target_image) if args.target_image else None
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gui", "icon.png")

    window = gui_main_v2.IColoriTUIv2(
        color_model=model,
        img_file=target,
        load_size=args.input_size,
        win_size=args.win_size,
        device=args.device,
        switch_model_fn=_make_switch_fn(args),
    )
    window.setWindowTitle(f"iColoriT LoRA  —  {model_name}")
    if os.path.exists(icon_path):
        window.setWindowIcon(QIcon(icon_path))
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
