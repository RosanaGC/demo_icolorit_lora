import time

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QCheckBox, QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QSizePolicy, QSplitter, QVBoxLayout, QWidget, QApplication
)

from .gui_draw_gt import GUIDrawGTHints
from .gui_palette import GUIPalette
from .gui_vis import GUI_VIS


class IColoriTGTUI(QWidget):
    """Demo separada: hints tomados del GT RGB de la imagen cargada."""

    def __init__(self, color_model, img_file=None, hints_file=None, load_size=224, win_size=256, device='cpu'):
        super().__init__()

        splitter = QSplitter(Qt.Horizontal, self)

        center_panel = QWidget()
        center = QVBoxLayout(center_panel)

        self.drawWidget = GUIDrawGTHints(color_model, load_size=load_size, win_size=win_size, device=device)
        self.drawWidget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.drawScroll = QScrollArea()
        self.drawScroll.setWidget(self.drawWidget)
        self.drawScroll.setWidgetResizable(False)
        self.drawScroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.drawScroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.drawScroll.setFixedSize(win_size, win_size)
        self.drawScroll.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.drawScroll.setFrameShape(QFrame.NoFrame)
        center.addLayout(self._boxed(self.drawScroll, 'Drawing Pad', align=Qt.AlignCenter))

        draw_menu = QHBoxLayout()
        self.bGray = QCheckBox("&Gray")
        self.bLoad = QPushButton('&Load')
        self.bLoadHints = QPushButton('Load &Hints')
        self.bSave = QPushButton("&Save")
        self.bSaveAs = QPushButton("Save &As…")
        for w in (self.bGray, self.bLoad, self.bLoadHints, self.bSave, self.bSaveAs):
            draw_menu.addWidget(w)
        center.addLayout(draw_menu)

        right_panel = QWidget()
        right = QVBoxLayout(right_panel)

        self.visWidget = GUI_VIS(win_size=win_size, scale=win_size / float(load_size))
        self.visWidget.setFixedSize(win_size, win_size)
        self.visWidget.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        vis_box = self._boxed(self.visWidget, 'Colorized Result', align=Qt.AlignCenter)
        right.addLayout(vis_box)

        vis_menu = QHBoxLayout()
        self.bRestart = QPushButton("&Restart")
        self.bQuit = QPushButton("&Quit")
        vis_menu.addWidget(self.bRestart)
        vis_menu.addWidget(self.bQuit)
        vis_box.addLayout(vis_menu)

        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([920, 920])

        root = QHBoxLayout(self)
        root.addWidget(splitter)

        self.usedPalette = GUIPalette(grid_sz=(10, 1))
        self.colorPush = QPushButton()
        self.colorPush.setFixedHeight(25)
        self.colorPush.setEnabled(False)
        self.colorPush.setStyleSheet("background-color: grey")
        self.rgbLabel = QLabel("RGB: -")

        self.infoWindow = QWidget()
        self.infoWindow.setWindowTitle('GT Hint Colors')
        info_layout = QVBoxLayout(self.infoWindow)
        info_layout.addLayout(self._boxed(self.usedPalette, 'Recently used colors'))
        info_layout.addLayout(self._boxed(self.colorPush, 'Current Color'))
        info_layout.addLayout(self._boxed(self.rgbLabel, 'Current RGB'))
        self.infoWindow.show()
        self.infoWindow.raise_()

        self.drawWidget.update_result.connect(self.visWidget.update_result)
        self.drawWidget.canvas_geom_changed.connect(self.visWidget.on_canvas_geom)
        self.drawWidget.update_color.connect(self.colorPush.setStyleSheet)
        self.drawWidget.update_rgb_text.connect(self.rgbLabel.setText)
        self.drawWidget.used_colors.connect(self.usedPalette.set_colors)
        self.visWidget.set_follow_zoom(True)
        self.drawScroll.horizontalScrollBar().valueChanged.connect(self.visWidget.on_hscroll)
        self.drawScroll.verticalScrollBar().valueChanged.connect(self.visWidget.on_vscroll)

        self.bGray.setChecked(True)
        self.bRestart.clicked.connect(self.reset)
        self.bQuit.clicked.connect(self.quit)
        self.bGray.toggled.connect(self.enable_gray)
        self.bSave.clicked.connect(self.save)
        self.bSaveAs.clicked.connect(lambda: self.drawWidget.save_result_as())
        self.bLoad.clicked.connect(self.load)
        self.bLoadHints.clicked.connect(self.load_hints)

        self.start_t = time.time()
        if img_file is not None:
            self.drawWidget.init_result(img_file)
            if hints_file is not None:
                self.drawWidget.load_hints_from_file(hints_file)

        scr = QApplication.primaryScreen().availableGeometry()
        self.setMinimumSize(1100, 680)
        self.resize(int(scr.width() * 0.9), int(scr.height() * 0.9))
        geo = self.frameGeometry()
        geo.moveCenter(scr.center())
        self.move(geo.topLeft())

    def _boxed(self, widget, title, align=None):
        box = QGroupBox(title)
        box.setFlat(True)
        v = QVBoxLayout(box)
        v.setContentsMargins(8, 8, 8, 8)
        if align is None:
            v.addWidget(widget)
        else:
            v.addWidget(widget, 0, align)
        out = QVBoxLayout()
        out.addWidget(box)
        return out

    def reset(self):
        self.visWidget.reset()
        self.drawWidget.reset()
        self.usedPalette.reset()
        self.colorPush.setStyleSheet("background-color: grey")
        self.rgbLabel.setText("RGB: -")
        self.update()

    def enable_gray(self):
        self.drawWidget.enable_gray()

    def quit(self):
        print('time spent = %3.3f' % (time.time() - self.start_t))
        if self.infoWindow is not None:
            self.infoWindow.close()
        self.close()

    def save(self):
        self.drawWidget.save_result()

    def load(self):
        self.drawWidget.load_image()

    def load_hints(self):
        self.drawWidget.load_hints_from_file()

    def closeEvent(self, event):
        try:
            if self.infoWindow is not None:
                self.infoWindow.close()
        finally:
            super().closeEvent(event)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_R:
            self.reset()
        if event.key() == Qt.Key_Q:
            self.save()
            self.quit()
        if event.key() == Qt.Key_S and not (event.modifiers() & Qt.ShiftModifier):
            self.save()
        if event.key() == Qt.Key_G:
            self.bGray.toggle()
        if event.key() == Qt.Key_L:
            self.load()
        if event.key() == Qt.Key_H:
            self.load_hints()

        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.drawWidget.zoom_in()
        elif event.key() in (Qt.Key_Minus, Qt.Key_Underscore):
            self.drawWidget.zoom_out()
        elif event.key() == Qt.Key_0:
            self.drawWidget.zoom_reset()

        mods = event.modifiers()
        cmd_or_ctrl = (mods & Qt.ControlModifier) or (mods & Qt.MetaModifier)
        if cmd_or_ctrl and event.key() == Qt.Key_Z and not (mods & Qt.ShiftModifier):
            self.drawWidget.undo()
            return
        if cmd_or_ctrl and ((event.key() == Qt.Key_Z and (mods & Qt.ShiftModifier)) or event.key() == Qt.Key_Y):
            self.drawWidget.redo()
            return
