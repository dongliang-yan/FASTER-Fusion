"""The FASTER Fusion window: a rail on the left chooses the mode, the rest of the
window is that mode's page. Pages are built the first time they are chosen and
then kept, so switching away and back finds each as it was left."""
from __future__ import annotations

import os

from PySide6 import QtCore, QtGui, QtWidgets

from .. import __version__

MODES = [
    ("cal_stepped", "Calibration – stepped",
     "Index tabs on a stepped position set, one DICOM per angle. Saves the calibration for Color Flow and SWE."),
    ("cal_cine", "Calibration – cine",
     "Index tabs on a continuous cine sweep. Saves the calibration for B-Mode."),
    ("bmode", "B-Mode",
     "3-D B-mode from a cine sweep - Siemens DICOM or Verasonics IQ. Needs a cine calibration."),
    ("colorflow", "Color Flow",
     "3-D colour flow, velocity or speed, and the volumetric flow rate by SIVV. Needs a stepped calibration."),
    ("swe", "SWE",
     "3-D shear-wave elastography, velocity or quality. Needs a stepped calibration."),
]

RAIL_CSS = """
QWidget#rail { background: #1a2438; }
QLabel#logo { color: white; font-size: 19px; font-weight: bold; }
QLabel#about { color: #ccd9f2; font-size: 11px; }
QLabel#foot { color: #8c9ebf; font-size: 9px; }
QPushButton.mode { background: #333f5c; color: #d9e0f2; border: none; border-radius: 6px;
                   font-size: 13px; font-weight: bold; padding: 6px; }
QPushButton.mode:checked { background: #296bcc; color: white; }
QPushButton.mode:hover:!checked { background: #3d4a6b; }
QPushButton.kind { background: #333f5c; color: #bfcce6; border: none; border-radius: 4px;
                   font-size: 11px; padding: 4px; }
QPushButton.kind[mine="true"] { background: #54668a; color: white; }
QPushButton.kind:checked { background: #8cb8f5; color: #1a2438; }
QPushButton.tool { background: #333f5c; color: white; border: none; border-radius: 5px;
                   font-size: 12px; padding: 6px; }
QPushButton.tool:hover { background: #3d4a6b; }
"""


def find_data_root() -> str:
    """The Data folder: remembered, or found beside the app / the source tree."""
    s = QtCore.QSettings("FASTER", "FASTER Fusion")
    d = s.value("data_root", "", str)
    if d and os.path.isdir(d):
        return d
    here = os.path.dirname(os.path.abspath(__file__))
    for up in range(1, 7):
        cand = os.path.join(here, *([".."] * up), "Data")
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return ""


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, mode: str = "cal_stepped"):
        super().__init__()
        self.setWindowTitle("FASTER Fusion")
        self.data_root = find_data_root()
        self.pages: dict[str, QtWidgets.QWidget] = {}
        self.key = ""
        self.cal_kind = "cal_stepped"

        central = QtWidgets.QWidget()
        lay = QtWidgets.QHBoxLayout(central)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        self.setCentralWidget(central)

        rail = QtWidgets.QWidget(objectName="rail")
        rail.setStyleSheet(RAIL_CSS)
        rail.setFixedWidth(122)
        rl = QtWidgets.QVBoxLayout(rail)
        rl.setContentsMargins(8, 12, 8, 8)
        rl.setSpacing(8)
        logo = QtWidgets.QLabel("FASTER\nFusion", objectName="logo")
        logo.setAlignment(QtCore.Qt.AlignCenter)
        rl.addWidget(logo)
        rl.addSpacing(10)

        def mode_btn(text, key):
            b = QtWidgets.QPushButton(text)
            b.setProperty("class", "mode")
            b.setCheckable(True)
            b.setMinimumHeight(52)
            b.clicked.connect(lambda _=False, k=key: self.set_mode(k))
            return b

        self.bt_cal = mode_btn("Calibration\nindex tabs", "calibration")
        rl.addWidget(self.bt_cal)
        kinds = QtWidgets.QHBoxLayout(); kinds.setSpacing(3)
        self.bt_step = QtWidgets.QPushButton("Stepped"); self.bt_step.setProperty("class", "kind")
        self.bt_cine = QtWidgets.QPushButton("Cine"); self.bt_cine.setProperty("class", "kind")
        for b, k in ((self.bt_step, "cal_stepped"), (self.bt_cine, "cal_cine")):
            b.setCheckable(True)
            b.setToolTip(dict((m[0], m[2]) for m in MODES)[k])
            b.clicked.connect(lambda _=False, kk=k: self.set_mode(kk))
            kinds.addWidget(b)
        rl.addLayout(kinds)
        self.bt_b = mode_btn("B-Mode\ncine sweep", "bmode")
        self.bt_cf = mode_btn("Color Flow\n+ SIVV", "colorflow")
        self.bt_swe = mode_btn("SWE\nshear wave", "swe")
        for b, k in ((self.bt_b, "bmode"), (self.bt_cf, "colorflow"), (self.bt_swe, "swe")):
            b.setToolTip(dict((m[0], m[2]) for m in MODES)[k])
            rl.addWidget(b)
        rl.addStretch(1)
        self.lb_about = QtWidgets.QLabel("", objectName="about")
        self.lb_about.setWordWrap(True)
        rl.addWidget(self.lb_about)
        bt_data = QtWidgets.QPushButton("Data folder…"); bt_data.setProperty("class", "tool")
        bt_data.setToolTip("Where the file dialogs start. See Data/README.md for what to load where.")
        bt_data.clicked.connect(self._pick_data)
        rl.addWidget(bt_data)
        bt_new = QtWidgets.QPushButton("New window"); bt_new.setProperty("class", "tool")
        bt_new.setToolTip("Open a second, independent FASTER Fusion window on the same mode, for "
                          "two sets side by side.")
        bt_new.clicked.connect(self._new_window)
        rl.addWidget(bt_new)
        foot = QtWidgets.QLabel(f"v{__version__}", objectName="foot")
        foot.setAlignment(QtCore.Qt.AlignCenter)
        rl.addWidget(foot)
        lay.addWidget(rail)

        self.stack = QtWidgets.QStackedWidget()
        lay.addWidget(self.stack, 1)
        self.statusBar().showMessage(f"Data folder: {self.data_root or '(not set)'}")

        # Fit the screen it opens on: a little inside it, never beyond it.
        scr = QtGui.QGuiApplication.primaryScreen().availableGeometry()
        self.resize(min(1600, scr.width() - 16), min(980, scr.height() - 16))
        self.move(scr.left() + (scr.width() - self.width()) // 2, scr.top() + (scr.height() - self.height()) // 2)
        self.set_mode(mode)

    # ------------------------------------------------------------------
    def page(self, key: str):
        """The page for a mode, built the first time."""
        if key in self.pages:
            return self.pages[key]
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            if key == "cal_stepped":
                from .calibration import CalibrationPage
                p = CalibrationPage("stepped", self.data_root)
            elif key == "cal_cine":
                from .calibration import CalibrationPage
                p = CalibrationPage("cine", self.data_root)
            elif key == "bmode":
                from .bmode import BModePage
                p = BModePage(self.data_root)
            elif key == "colorflow":
                from .step_recon import StepReconPage
                p = StepReconPage("cf", self.data_root)
            elif key == "swe":
                from .step_recon import StepReconPage
                p = StepReconPage("swe", self.data_root)
            else:
                raise KeyError(key)
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
        p.status.connect(lambda m: self.statusBar().showMessage(m, 15000))
        # A page that still does not fit a small screen scrolls, rather than
        # pushing the window past the edge of it.
        sa = QtWidgets.QScrollArea()
        sa.setWidgetResizable(True)
        sa.setFrameShape(QtWidgets.QFrame.NoFrame)
        sa.setWidget(p)
        p._scroll = sa
        self.stack.addWidget(sa)
        self.pages[key] = p
        return p

    def set_mode(self, m: str):
        key = {"calibration": self.cal_kind, "stepped": "cal_stepped", "cine": "cal_cine",
               "cf": "colorflow", "color flow": "colorflow", "b-mode": "bmode"}.get(m, m)
        if key not in dict((k[0], k) for k in MODES):
            return
        if key.startswith("cal_"):
            self.cal_kind = key
        self.stack.setCurrentWidget(self.page(key)._scroll)
        self.key = key
        info = next(x for x in MODES if x[0] == key)
        self.setWindowTitle(f"FASTER Fusion   –   {info[1]}")
        self.lb_about.setText(info[2])
        self._paint()

    def _paint(self):
        is_cal = self.key.startswith("cal_")
        self.bt_cal.setChecked(is_cal)
        self.bt_b.setChecked(self.key == "bmode")
        self.bt_cf.setChecked(self.key == "colorflow")
        self.bt_swe.setChecked(self.key == "swe")
        for b, k in ((self.bt_step, "cal_stepped"), (self.bt_cine, "cal_cine")):
            mine = self.cal_kind == k
            b.setChecked(mine and is_cal)
            b.setProperty("mine", "true" if mine else "false")
            b.style().unpolish(b); b.style().polish(b)

    def _pick_data(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Where is the Data folder?",
                                                       self.data_root or os.path.expanduser("~"))
        if d:
            self.data_root = d
            QtCore.QSettings("FASTER", "FASTER Fusion").setValue("data_root", d)
            for p in self.pages.values():
                p.data_root = d
            self.statusBar().showMessage(f"Data folder: {d}")

    def _new_window(self):
        w = MainWindow(self.key)
        w.show()
        MainWindow._others = getattr(MainWindow, "_others", []) + [w]
