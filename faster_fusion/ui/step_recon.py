"""Stepped-set reconstruction: Color Flow (Recon_CF_v2, with SIVV) and SWE (Recon_SWE).

One page class for both, as the two MATLAB GUIs share most of their code. The
mode decides the sources, how a set is decoded, what the clean-up row holds, and
whether the SIVV block is shown:

  cf   sources Velocity (signed, cm/s) and Speed (|v|); Keep |v| rejects on speed;
       Bar +/- is the scale; SIVV flow rate through a constant-range surface
  swe  sources Quality and Velocity (m/s); Keep rejects velocity, with Margin
       grown round the high side
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import ndimage

from ..core import decode, io, recon
from ..core.mlcompat import imresize3, mround
from ..core.sector import SectorGrid
from .widgets import (ElideLabel, BACKGROUNDS, MAGENTA, Busy, CurvePlot, ImagePanel, Spin, Volume3D, alert,
                      float_lut, hbox, label, lut_from, next_free)

DEF_MIRROR = 23.0
DEF_LAMBDA = 1540 / 5.2083e6 * 1000
DEF_KNEE = 0.66


@dataclass
class Source:
    name: str
    rgb: np.ndarray            # nz x nx x 3 x N
    t: np.ndarray              # nz x nx x N, 0..1 of the source's scale, NaN = none
    grey: np.ndarray
    mask: np.ndarray
    cmap: np.ndarray           # 256 x 3, 0..1, LOW to HIGH
    box: list                  # 1-based [r0 r1 c0 c1]
    resid: float
    dz_mm: float
    dx_mm: float
    alpha: float = float("nan")
    flow_frac: Optional[np.ndarray] = None
    extra: dict = field(default_factory=dict)


def bar_map(C: np.ndarray) -> np.ndarray:
    C = np.flipud(np.asarray(C, float)) / 255
    x = np.linspace(0, 1, C.shape[0])
    xi = np.linspace(0, 1, 256)
    return np.clip(np.stack([np.interp(xi, x, C[:, i]) for i in range(3)], 1), 0, 1)


def dilate_ellipsoid(R, rz, rx, rp):
    rz, rx, rp = (max(v, np.finfo(float).eps) for v in (rz, rx, rp))
    zg, xg, pg_ = np.meshgrid(np.arange(-np.floor(rz), np.floor(rz) + 1),
                              np.arange(-np.floor(rx), np.floor(rx) + 1),
                              np.arange(-np.floor(rp), np.floor(rp) + 1), indexing="ij")
    se = (zg / rz) ** 2 + (xg / rx) ** 2 + (pg_ / rp) ** 2 <= 1
    return ndimage.binary_dilation(R, structure=se)


class StepReconPage(QtWidgets.QWidget):
    status = QtCore.Signal(str)

    def __init__(self, mode: str, data_root: str = ""):
        super().__init__()
        assert mode in ("cf", "swe")
        self.mode = mode
        self.data_root = data_root
        cf = mode == "cf"
        self.names = ["Velocity", "Speed"] if cf else ["Quality", "Velocity"]
        self.units = ({"Velocity": "cm/s", "Speed": "cm/s (|v|)"} if cf
                      else {"Quality": "(0 = LO, 1 = HI)", "Velocity": "m/s"})
        self.vmax = 8.0
        self.scale = ({"Velocity": [-8.0, 8.0], "Speed": [0.0, 8.0]} if cf
                      else {"Quality": [0.0, 1.0], "Velocity": [0.5, 8.0]})
        self.range = {k: list(v) for k, v in self.scale.items()}
        self.clean = {"keep": [0.0, 8.0] if cf else [0.5, 8.0], "margin": 1.0, "smooth": 0.0}
        self.cal = None
        self.data_dir = ""
        self.loaded = ""
        self.src: dict[str, Source] = {}
        self.tv: dict[str, np.ndarray] = {}
        self.rej_frac = 0.0
        self.src_name = "Velocity"
        self.N = 0
        self.files, self.times = [], np.zeros(0)
        self.pos = 1
        self.rec = None
        self.vol: dict = {}
        self.volB = None
        self.shown = ""
        self.full = True
        self.z0 = 0.0
        self.dz = float("nan")
        self.dx = float("nan")
        self.mirror = DEF_MIRROR
        self.busy = False
        self.sivv = {"depth": float("nan"), "n": 2, "show": True, "prof": None, "at": None}
        self._build()
        self._refresh()

    # ================================================================ layout
    def _build(self):
        cf = self.mode == "cf"
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ---- setup row
        self.gb_top = QtWidgets.QGroupBox("Setup")
        top = QtWidgets.QGridLayout(self.gb_top)
        top.setContentsMargins(8, 4, 8, 4)
        self.lb_cal = ElideLabel("(none)"); self.lb_cal.setStyleSheet("color:#777")
        self.lb_set = ElideLabel("(none)"); self.lb_set.setStyleSheet("color:#777")
        self.bt_cal = QtWidgets.QPushButton("Browse…"); self.bt_cal.clicked.connect(self._browse_cal)
        self.bt_set = QtWidgets.QPushButton("Browse…"); self.bt_set.clicked.connect(self._browse_set)
        self.bt_load = QtWidgets.QPushButton("Load set"); self.bt_load.clicked.connect(self.load)
        # one line, ended with an ellipsis when long: the whole text is in the
        # tooltip and the status bar. (A wrapping label here made the scroll area
        # think the page needed three times its height.)
        self.lb_status = ElideLabel(
            "Choose a calibration .mat and the folder of " + ("colour-flow" if cf else "SWE") + " captures.", QtCore.Qt.ElideRight)
        top.addWidget(label("Calibration"), 0, 0)
        top.addWidget(self.lb_cal, 0, 1)
        top.addWidget(self.bt_cal, 0, 2)
        top.addWidget(label("CF set" if cf else "SWE set",
                            "The folder holding one single-frame DICOM per reflector position."), 0, 3)
        top.addWidget(self.lb_set, 0, 4)
        top.addWidget(self.bt_set, 0, 5)
        top.addWidget(self.bt_load, 0, 6)
        top.addWidget(self.lb_status, 1, 0, 1, 7)
        top.setColumnStretch(1, 1); top.setColumnStretch(4, 1)
        self.src_btn = {}
        sb = QtWidgets.QHBoxLayout()
        for nm in self.names:
            b = QtWidgets.QPushButton(nm)
            b.setCheckable(True)
            b.setMinimumSize(84, 40)
            f = b.font(); f.setPointSize(15); f.setBold(True); b.setFont(f)
            b.clicked.connect(lambda _=False, n=nm: self.set_source(n))
            b.setToolTip(
                ("Velocity is signed, in the bar's colours, with the 3-D opacity following "
                 "the speed. Speed is |v|, the one for a MIP of a vessel.") if cf else
                "Which half of the dual display to reconstruct: Quality (left) or Velocity (right).")
            self.src_btn[nm] = b
            sb.addWidget(b)
        top.addLayout(sb, 0, 7, 2, 1)
        root.addWidget(self.gb_top)

        # ---- middle
        mid = QtWidgets.QHBoxLayout()
        mid.setSpacing(6)
        root.addLayout(mid, 1)

        left = QtWidgets.QGroupBox("Positions  →  2-D reconstruction")
        ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(6, 6, 6, 6)
        views = QtWidgets.QHBoxLayout()
        self.ax_frame = ImagePanel("Capture", "Lateral line", "Depth (mm)")
        self.ax_frame.setMaximumWidth(190)
        self.ax_frame.clicked.connect(lambda x, y, b: self.set_lat(x))
        self.ax_map = ImagePanel("Load a set", "Position  (click to see its capture)", "Depth (mm)")
        self.ax_map.clicked.connect(lambda x, y, b: self.set_position(x))
        views.addWidget(self.ax_frame)
        views.addWidget(self.ax_map, 1)
        ll.addLayout(views, 9)

        g = QtWidgets.QGridLayout()
        g.setHorizontalSpacing(6); g.setVerticalSpacing(2)
        self.sp_lat = Spin(1, 4096, 100, 1, integer=True,
                           tip="Which lateral line the map and the 2-D frame are cut from. "
                               "Clicking on the capture moves it there.")
        self.sp_lat.valueEdited.connect(self.set_lat)
        self.sp_pos = Spin(1, 4096, 1, 1, integer=True,
                           tip="Which position the capture view shows. Clicking on the map moves it there.")
        self.sp_pos.valueEdited.connect(self.set_position)
        self.sp_shift = Spin(-50, 50, 0, 1, integer=True,
                             tip="Slide the pairing with the calibration by whole positions.")
        self.sp_shift.valueEdited.connect(lambda v: self._settings_changed())
        self.cb_flip = QtWidgets.QCheckBox("Flip sweep")
        self.cb_flip.setToolTip("Reverse the angles, if the volume comes out mirrored")
        self.cb_flip.toggled.connect(lambda v: self._settings_changed())
        self.dd_angle = QtWidgets.QComboBox(); self.dd_angle.addItems(["fraction", "index"])
        self.dd_angle.setToolTip("When the calibration and this set have different numbers of "
                                 "positions: fraction spreads the calibration over this set's "
                                 "length; index pairs position k with position k.")
        self.dd_angle.currentTextChanged.connect(lambda v: self._settings_changed())
        g.addWidget(label("Lat line"), 0, 0); g.addWidget(self.sp_lat, 0, 1)
        g.addWidget(label("Position"), 0, 2); g.addWidget(self.sp_pos, 0, 3)
        g.addWidget(label("Shift"), 0, 4); g.addWidget(self.sp_shift, 0, 5)
        g.addWidget(label("Angle map"), 1, 0); g.addWidget(self.dd_angle, 1, 1)
        g.addWidget(self.cb_flip, 1, 2, 1, 2)
        mtip = ("Depth of the mirror centre, in mm. Everything above it is thrown away and "
                "everything below is swept into the volume. Drawn dotted on the capture and the map.")
        self.sp_mirror = Spin(0, 500, DEF_MIRROR, 0.5, 2, tip=mtip)
        self.sp_mirror.valueEdited.connect(lambda v: self._geom_changed())
        self.sp_z0 = Spin(0, 500, 0, 1, 2, tip="Depth of the FIRST axial sample, in mm. 0 for a DICOM.")
        self.sp_z0.valueEdited.connect(lambda v: self._geom_changed())
        self.sp_dz = Spin(0, 10, 0, 0.005, 4, tip="Axial mm per pixel, as the DICOM states it.")
        self.sp_dx = Spin(0, 10, 0, 0.005, 4, tip="Lateral mm per pixel, as the DICOM states it.")
        self.sp_dz.setReadOnly(True); self.sp_dx.setReadOnly(True)
        g.addWidget(label("Mirror mm", mtip), 1, 4); g.addWidget(self.sp_mirror, 1, 5)
        g.addWidget(label("Starts at"), 2, 0); g.addWidget(self.sp_z0, 2, 1)
        g.addWidget(label("mm/smp"), 2, 2); g.addWidget(self.sp_dz, 2, 3)
        g.addWidget(label("mm/line"), 2, 4); g.addWidget(self.sp_dx, 2, 5)
        g.setColumnStretch(6, 1)

        # clean-up row
        if cf:
            ktip = ("The rejection window, on the SPEED |v| in cm/s. A pixel whose speed is outside "
                    "it is taken OUT before the sweep, the smoothing and the volume. Raise the bottom "
                    "to take out slow flash. Range at the bottom only hides.")
            self.sp_klo = Spin(0, 1e6, 0, 0.25, 2, tip=ktip)
            self.sp_khi = Spin(0, 1e6, 8, 0.25, 2, tip=ktip)
            self.sp_vmax = Spin(0.01, 1000, 8, 1, 2,
                                tip="What the two ends of the flow bar mean, +/- cm/s - the number on "
                                    "the bar's labels. 8 on the phantom sets.")
            self.sp_vmax.valueEdited.connect(self.set_scale)
            row = [label("Keep |v|", ktip), self.sp_klo, label("to", align=QtCore.Qt.AlignCenter),
                   self.sp_khi, label("cm/s", align=QtCore.Qt.AlignLeft)]
            row2 = [label("Bar ±"), self.sp_vmax]
        else:
            ktip = ("The rejection window, in m/s. A velocity outside it is taken OUT before the "
                    "sweep, the smoothing and the volume. Range at the bottom only hides.")
            self.sp_klo = Spin(-1e6, 1e6, 0.5, 0.25, 2, tip=ktip)
            self.sp_khi = Spin(-1e6, 1e6, 8, 0.25, 2, tip=ktip)
            self.sp_margin = Spin(0, 10, 1, 0.25, 2, suffix=" mm",
                                  tip="How far round a velocity ABOVE the Keep window the rejection "
                                      "is grown, in mm.")
            self.sp_margin.valueEdited.connect(lambda v: self._clean_changed())
            row = [label("Keep", ktip), self.sp_klo, label("to", align=QtCore.Qt.AlignCenter),
                   self.sp_khi, label("m/s", align=QtCore.Qt.AlignLeft)]
            row2 = [label("Margin"), self.sp_margin]
        self.sp_klo.valueEdited.connect(lambda v: self._clean_changed())
        self.sp_khi.valueEdited.connect(lambda v: self._clean_changed())
        self.sp_smooth = Spin(0, 5, 0, 0.25, 2, suffix=" mm",
                              tip="Gaussian smoothing, sigma in mm: 3-D on the volume, 2-D on the frame. "
                                  "It ignores pixels with no value. 0 is off.")
        self.sp_smooth.valueEdited.connect(lambda v: self._smooth_changed())
        self.lb_clean = QtWidgets.QLabel(""); self.lb_clean.setStyleSheet("color:#8c4d00")
        row2 += [label("Smooth"), self.sp_smooth, self.lb_clean, None]
        ll.addLayout(g)
        ll.addLayout(hbox(*row, None))
        ll.addLayout(hbox(*row2))

        self.ax_rec = ImagePanel("2-D elevational-axial frame", "Elevational position (mm)",
                                 "Axial depth (mm)", lock_aspect=True)
        self.ax_rec.clicked.connect(lambda x, y, b: self.read_at(x, y))
        ll.addWidget(self.ax_rec, 16)
        mid.addWidget(left, 9)       # the images need less width than the 3-D view

        arrow = QtWidgets.QVBoxLayout()
        arrow.addStretch(1)
        self.bt_go = QtWidgets.QPushButton("▶ ▶\ndisplay3D")
        f = self.bt_go.font(); f.setPointSize(15); f.setBold(True); self.bt_go.setFont(f)
        self.bt_go.setMinimumSize(84, 90)
        self.bt_go.setToolTip("Reconstruct and render the volume: the whole field, B-mode and "
                              "values together, or only the box - see Full volume under the viewer")
        self.bt_go.clicked.connect(lambda: self.display3d())
        arrow.addWidget(self.bt_go)
        arrow.addStretch(1)
        mid.addLayout(arrow)

        right = QtWidgets.QGroupBox("3-D volume")
        rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        self.view = Volume3D("Load a set, then press display3D")
        self.view.setMinimumHeight(200)
        rl.addWidget(self.view, 1)

        # Under the viewer: the display settings, and for colour flow the SIVV
        # block beside them as a second tab - one shown at a time, so the view
        # keeps the height on a laptop screen.
        disp = QtWidgets.QWidget()
        dl = QtWidgets.QVBoxLayout(disp)
        dl.setContentsMargins(4, 4, 4, 2)
        dl.setSpacing(3)
        full = QtWidgets.QHBoxLayout()
        self.cb_full = QtWidgets.QCheckBox("Full volume"); self.cb_full.setChecked(True)
        f = self.cb_full.font(); f.setBold(True); self.cb_full.setFont(f)
        self.cb_full.setToolTip("Reconstruct the whole field - every lateral line, every depth below "
                                "the mirror, B-mode everywhere - rather than only the box.")
        self.cb_full.toggled.connect(self._full_changed)
        self.dd_bstyle = QtWidgets.QComboBox(); self.dd_bstyle.addItems(["MIP", "Volume", "Slices", "Off"])
        self.dd_bstyle.currentTextChanged.connect(lambda v: self._rerender())
        self.sp_blo = Spin(0, 254, 0, 5, integer=True, tip="B-mode grey window, bottom.")
        self.sp_bhi = Spin(1, 255, 255, 5, integer=True, tip="B-mode grey window, top.")
        self.sp_bopa = Spin(0, 1, 0.5, 0.05, 2,
                            tip="Opacity of the brightest B-mode, 0 to 1. Volume needs it far lower "
                                "than MIP, 0.05 or so.")
        for w in (self.sp_blo, self.sp_bhi, self.sp_bopa):
            w.valueEdited.connect(lambda v: self._b_changed())
        self.sp_vox = Spin(0.05, 2, 0.25, 0.05, 2, suffix=" mm",
                           tip="Voxel size of the full view, mm. Changing it takes a display3D.")
        self.sp_vox.valueEdited.connect(lambda v: self._invalidate("The voxel size changed - press display3D"))
        full.addWidget(self.cb_full)
        full.addStretch(1)
        full.addWidget(label("voxel")); full.addWidget(self.sp_vox)
        dl.addLayout(full)
        dl.addLayout(hbox(label("B-mode"), self.dd_bstyle, label("window"), self.sp_blo, self.sp_bhi,
                          label("B opacity"), self.sp_bopa, None))

        opa = QtWidgets.QHBoxLayout()
        lb = label("Opacity", bold=True)
        lb.setToolTip("How opaque the low end of the window is made. Higher lifts low values into view.")
        self.sp_gamma = Spin(0.1, 20, 1, 0.5, 2)
        self.sp_gamma.valueEdited.connect(lambda v: self._apply_alpha())
        opa.addWidget(lb); opa.addWidget(self.sp_gamma)
        self.curve = CurvePlot("value (|v| for Velocity), bottom to top" if cf else
                               "value, bottom to top of window")
        self.curve.setMaximumHeight(84)
        self.curve.setMinimumHeight(64)
        opa.addWidget(self.curve, 1)
        dl.addLayout(opa)
        if cf:
            self.tabs = QtWidgets.QTabWidget()
            self.tabs.setDocumentMode(True)
            self.tabs.addTab(self._build_sivv(), "SIVV flow rate")
            self.tabs.addTab(disp, "3-D display")
            self.tabs.setMaximumHeight(230)
            rl.addWidget(self.tabs)
        else:
            rl.addWidget(disp)
        self._rl = rl
        mid.addWidget(right, 13)

        # ---- bottom row
        bot = QtWidgets.QHBoxLayout()
        rtip = "The range of values SHOWN, in the units of the source. It only hides."
        lo, hi = self.range[self.src_name]
        self.sp_lo = Spin(-1e6, 1e6, lo, 0.25, 2, tip="Bottom of the range shown. " + rtip)
        self.sp_hi = Spin(-1e6, 1e6, hi, 0.25, 2, tip="Top of the range shown. " + rtip)
        self.sp_lo.valueEdited.connect(lambda v: self._display_changed())
        self.sp_hi.valueEdited.connect(lambda v: self._display_changed())
        self.sp_knee = Spin(0, 1, DEF_KNEE, 0.05, 2,
                            tip="How hard the low end is rolled off before the projection.")
        self.sp_knee.valueEdited.connect(lambda v: self._apply_alpha())
        self.dd_style = QtWidgets.QComboBox(); self.dd_style.addItems(["MIP", "Volume", "Isosurface", "Slices"])
        self.dd_style.setToolTip("MIP shows the highest value along each ray; Volume composites "
                                 "through the opacity curve; Isosurface draws round what Range shows; "
                                 "Slices shows three planes.")
        self.dd_style.currentTextChanged.connect(lambda v: self._rerender())
        self.cb_largest = QtWidgets.QCheckBox("Largest blob")
        self.cb_largest.setToolTip("Show only the biggest connected region inside Range.")
        self.cb_largest.toggled.connect(lambda v: self._rerender())
        self.dd_bg = QtWidgets.QComboBox(); self.dd_bg.addItems([b[0] for b in BACKGROUNDS])
        self.dd_bg.currentTextChanged.connect(self.view.apply_background)
        self.sl_rot = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sl_rot.setRange(0, 360)
        self.sl_rot.setMinimumWidth(90)
        self.sl_rot.valueChanged.connect(self._rotate)
        self.lb_ang = QtWidgets.QLabel("0 deg"); self.lb_ang.setMinimumWidth(56)
        self.bt_reset = QtWidgets.QPushButton("Reset view"); self.bt_reset.clicked.connect(lambda: self.sl_rot.setValue(0))
        self.bt_mp4 = QtWidgets.QPushButton("Save MP4"); self.bt_mp4.clicked.connect(self.save_video)
        self.bt_mp4.setToolTip("Write a full turn of the volume as an MP4, captured from this renderer")
        bot.addWidget(label("Range", rtip)); bot.addWidget(self.sp_lo); bot.addWidget(self.sp_hi)
        bot.addWidget(label("Linearity")); bot.addWidget(self.sp_knee)
        bot.addWidget(label("Style")); bot.addWidget(self.dd_style)
        bot.addWidget(self.cb_largest)
        bot.addWidget(label("Background")); bot.addWidget(self.dd_bg)
        bot.addStretch(1)
        # turning and saving the volume sit under the volume
        self._rl.insertLayout(1, hbox(label("Rotation"), self.sl_rot, self.lb_ang, self.bt_reset, self.bt_mp4))
        bw = QtWidgets.QFrame(); bw.setFrameShape(QtWidgets.QFrame.StyledPanel); bw.setLayout(bot)
        root.addWidget(bw)
        self._paint_source_buttons()
        self._apply_alpha()

    def _build_sivv(self):
        gb = QtWidgets.QWidget()
        gb.setToolTip("Flow rate through a constant-range surface (SIVV)")
        lay = QtWidgets.QHBoxLayout(gb)
        lay.setContentsMargins(4, 2, 4, 2)
        c = QtWidgets.QGridLayout()
        ztip = ("Depth of the surface, mm, on the capture's depth axis. The surface is every lateral "
                "line of the colour box at that range from the mirror, across every angled position: "
                "an arc elevationally, straight laterally, and normal to the beam everywhere - so the "
                "Doppler velocity on it is the normal velocity SIVV needs. Pick it where Q vs depth is flat.")
        self.sp_sz = Spin(0, 500, 0, 0.25, 2, tip=ztip)
        self.sp_sz.valueEdited.connect(lambda v: self.set_sivv(v, None))
        ntip = ("Capture rows either side of the surface averaged into it - a slab 2n+1 rows thick - "
                "to steady Q against the speckle of the colour map. 0 is the one row.")
        self.sp_sn = Spin(0, 30, 2, 1, integer=True, tip=ntip)
        self.sp_sn.valueEdited.connect(lambda v: self.set_sivv(None, v))
        self.cb_sshow = QtWidgets.QCheckBox("Show in 3-D"); self.cb_sshow.setChecked(True)
        self.cb_sshow.toggled.connect(self._sivv_show)
        self.lb_q = QtWidgets.QLabel("Q  –")
        fq = self.lb_q.font(); fq.setPointSize(15); fq.setBold(True); self.lb_q.setFont(fq)
        self.lb_q.setStyleSheet("color:#bf1abf")
        self.lb_qinfo = QtWidgets.QLabel("")
        fi = self.lb_qinfo.font(); fi.setBold(False); fi.setPointSize(10); self.lb_qinfo.setFont(fi)
        self.lb_qinfo.setToolTip("Q in mL/s and its spread over the rows averaged; the parts towards "
                                 "and away from the probe; the area of the surface carrying flow and "
                                 "the surface's range from the mirror.")
        for w in (self.sp_sz, self.sp_sn, self.cb_sshow):
            f2 = w.font(); f2.setBold(False); w.setFont(f2)
        c.addWidget(label("Depth mm", ztip), 0, 0); c.addWidget(self.sp_sz, 0, 1)
        c.addWidget(label("± rows", ntip), 1, 0); c.addWidget(self.sp_sn, 1, 1)
        c.addWidget(self.cb_sshow, 2, 0, 1, 2)
        c.addWidget(self.lb_q, 3, 0, 1, 2)
        c.addWidget(self.lb_qinfo, 4, 0, 1, 2)
        c.setRowStretch(5, 1)
        wc = QtWidgets.QWidget(); wc.setLayout(c); wc.setFixedWidth(158)
        lay.addWidget(wc)
        # not aspect-locked: in a narrow panel the lock crops the surface, and
        # seeing all of it matters more than its exact proportions
        self.ax_surf = ImagePanel("Surface", "Lateral (mm)", "Arc (mm)",
                                  invert_y=False, lock_aspect=False)
        self.ax_q = ImagePanel("Q vs depth", "Depth (mm)", "Q (mL/min)", invert_y=False)
        self.ax_q.setToolTip("Q at every depth of the colour box. Read it where the curve is flat; "
                             "x marks depths where flow is cut off at an edge. Click to move the "
                             "surface to that depth.")
        self.ax_q.showGrid(x=True, y=True, alpha=0.3)
        self.ax_q.clicked.connect(lambda x, y, b: self.set_sivv(x, None) if self.sivv["prof"] is not None else None)
        lay.addWidget(self.ax_surf, 10)
        lay.addWidget(self.ax_q, 12)
        return gb

    # ================================================================ selection
    def _start_dir(self):
        return self.data_dir or self.data_root or os.path.expanduser("~")

    def _browse_cal(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick a calibration file", self._start_dir(),
                                                     "Calibration (*.mat)")
        if f:
            self.set_cal(f)

    def set_cal(self, path: str) -> bool:
        try:
            c = recon.load_calibration(path)
        except recon.CalibrationError as e:
            alert(self, "Not a usable calibration", str(e))
            return False
        self.cal = c
        self.lb_cal.setText(os.path.basename(path)); self.lb_cal.setStyleSheet("")
        self.lb_cal.setToolTip(path)
        msg = recon.describe_calibration(c)
        if self._have():
            if self._clean_active():
                self._apply_clean()
            self._invalidate("The calibration changed - press display3D")
            self._draw_map(); self._draw2d(); self._sivv_recompute(False)
            msg += " " + self._angles_note()
        self._set_status(msg)
        self._refresh()
        return True

    def _browse_set(self):
        d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick the folder of captures", self._start_dir())
        if d:
            self.set_data(d)

    def set_data(self, target: str) -> bool:
        target = str(target)
        if os.path.isfile(target):
            target = os.path.dirname(target)
        if not os.path.isdir(target):
            alert(self, "No such folder", f"{target} does not exist.")
            return False
        self.data_dir = target
        self.lb_set.setText(target); self.lb_set.setStyleSheet(""); self.lb_set.setToolTip(target)
        self._set_status(f"Set {target} selected. Press Load set.")
        self._refresh()
        return True

    # ================================================================ loading
    def load(self):
        if self.busy or not self.data_dir:
            if not self.data_dir:
                alert(self, "Nothing to load", "Pick the folder of captures first.")
            return
        self.busy = True
        self._refresh()
        dlg = Busy(self, "Loading", "Reading " + self.data_dir)
        try:
            ps = io.load_position_stack(self.data_dir, progress=lambda v, m: dlg(0.35 * v, m))
            src = self._decode(ps, dlg)
            P1 = src[next(iter(src))]
            if not (np.isfinite(P1.dz_mm) and P1.dz_mm > 0 and np.isfinite(P1.dx_mm) and P1.dx_mm > 0):
                raise io.ScanError("The DICOM does not state its pixel size in centimetres "
                                   "(PhysicalDeltaX/Y), so no depth could be put on it.")
            self._check_mirror(P1.t.shape[0], P1.dz_mm, self.sp_z0.value(), self.sp_mirror.value())
        except Exception as e:                      # keep the previous set whole
            dlg.close()
            self.busy = False
            extra = f"\n\nStill showing {self.loaded}." if self.loaded else ""
            alert(self, "Could not load this set", f"{e}{extra}")
            self._set_status("Load failed." + (f" Still showing {self.loaded}." if self.loaded else ""))
            self._refresh()
            return
        dlg.close()
        self.src = src
        self.N = ps.N
        self.files, self.times = ps.files, ps.time_s
        self.loaded = self.data_dir
        self.dz, self.dx = P1.dz_mm, P1.dx_mm
        self.z0, self.mirror = self.sp_z0.value(), self.sp_mirror.value()
        self.rec, self.vol, self.volB, self.shown = None, {}, None, ""
        self.sivv["prof"], self.sivv["at"] = None, None
        if self.src_name not in self.src:
            self.src_name = list(self.src)[-1]
        self.tv = {}
        self._apply_clean()
        self.sp_dz.set(self.dz); self.sp_dx.set(self.dx)
        nx = P1.t.shape[1]
        self.sp_lat.setRange(1, nx)
        b = self.src[self.src_name].box
        self.sp_lat.set(int(round((b[2] + b[3]) / 2)) if b else int(round(nx / 2)))
        self.sp_pos.setRange(1, max(2, self.N))
        self.pos = max(1, int(round(self.N / 2)))
        self.sp_pos.set(self.pos)
        m = max(1, self.N // 4)
        self.sp_shift.setRange(-m, m)
        name = os.path.basename(self.data_dir)
        self.gb_top.setTitle(f"Setup   –   {name}:  {self.N} positions, {P1.t.shape[0]} x {nx}, "
                             f"{self.dz:.4f} mm/pixel, ordered by {ps.order_by}")
        self.view.clear("Press display3D to render the volume")
        self.view.prepare()
        self.busy = False
        self.set_source(self.src_name)
        if self.mode == "cf":
            self._sivv_recompute(True)
            V = self.src["Velocity"]
            wf = np.flatnonzero(V.flow_frac > 0.002) + 1
            flow = (f"flow in {wf.size} positions ({wf[0]}:{wf[-1]})" if wf.size
                    else "no flow found in any position")
            msg = (f"{name}: {self.N} positions decoded. Box {self._box_text(V)}, bar misfit "
                   f"{V.resid:.1f}, {flow}. {self._mirror_text()}")
        else:
            parts = [f"{n}: box {self._box_text(s)}, opacity {s.alpha:.2f}, misfit {s.resid:.1f}"
                     for n, s in self.src.items()]
            msg = f"{name}: {self.N} positions decoded ({'; '.join(parts)}). {self._mirror_text()}"
        if self.cal is not None:
            msg += " " + self._angles_note()
        if self._clean_active():
            msg += " " + self._clean_text()
        self._set_status(msg)
        self._refresh()

    def _decode(self, ps, dlg) -> dict:
        excl = np.zeros(ps.screen.shape[:2], bool)
        for p in ps.panels:
            excl[np.ix_(p.rows, p.cols)] = True
        if self.mode == "cf":
            j = ps.flow_panel if ps.flow_panel is not None else 0
            P = ps.panels[j]
            box = None
            if ps.flow_panel is not None:
                fb = ps.flow_box
                box = [fb[0] - P.rows[0] + 1, fb[1] - P.rows[0] + 1, fb[2] - P.cols[0] + 1, fb[3] - P.cols[0] + 1]
            bar = decode.find_flow_bar(ps.screen, excl)
            if bar is None:
                raise io.ScanError("No colour-flow bar was found on the screen - a tall bar running "
                                   "from warm colours through a baseline to cool ones - so these are "
                                   "not colour-flow captures, or not ones this reads.")
            d = decode.cf_decode(P.rgb, bar, box=box, progress=lambda v, m: dlg(0.35 + 0.6 * v, m))
            common = dict(rgb=P.rgb, grey=d.grey, mask=d.mask, box=d.box, resid=d.resid,
                          dz_mm=P.dz_mm, dx_mm=P.dx_mm, flow_frac=d.flow_frac)
            V = Source(name="Velocity", t=(d.v + 1) / 2, cmap=bar_map(bar.colors), **common)
            S = Source(name="Speed", t=np.abs(d.v), cmap=bar_map(bar.colors[:bar.zero]), **common)
            self.bar = bar
            return {"Velocity": V, "Speed": S}

        bars = decode.find_colour_bars(ps.screen, excl)
        if not bars:
            raise io.ScanError("No colour bar was found on the screen, so these are not SWE captures "
                               "- or not ones this reads. B-Mode reconstructs B-mode.")
        colourful = []
        for p in ps.panels:
            I = p.rgb[:, :, :, 0].astype(float)
            mx, mn = I.max(2), I.min(2)
            colourful.append(np.mean(((mx - mn) / np.maximum(mx, 1) > 0.5) & (mx > 60)) > 0.002)
        pj = [j for j, c in enumerate(colourful) if c]
        if not pj:
            raise io.ScanError(f"None of the {len(ps.panels)} panel(s) on the screen carries a colour overlay.")
        if len(pj) >= 2 and len(bars) >= 2:
            pj, bars = [pj[0], pj[-1]], [bars[0], bars[-1]]
        else:
            pc = np.array([np.mean(ps.panels[j].cols) for j in pj])
            bc = np.array([np.mean(b.cols) for b in bars])
            D = np.abs(pc[:, None] - bc[None, :])
            jb = int(np.argmin(D.min(1)))
            pj, bars = [pj[jb]], [bars[int(np.argmin(D[jb]))]]
        out = {}
        for k, (j, b) in enumerate(zip(pj, bars)):
            nm = "Velocity" if b.has_blue else "Quality"
            if nm in out:
                raise io.ScanError(f"Two panels both look like {nm}: their colour bars are the same kind.")
            P = ps.panels[j]
            lo = 0.35 + 0.65 * k / len(pj)
            span = 0.65 / len(pj)
            d = decode.swi_decode(P.rgb, b.colors, progress=lambda v, m, lo=lo, span=span, nm=nm: dlg(lo + span * v, f"{nm}: {m}"))
            if d.box is None:
                raise io.ScanError(f"The {nm} panel decoded to nothing.")
            out[nm] = Source(name=nm, rgb=P.rgb, t=d.t, grey=d.grey, mask=d.mask, cmap=bar_map(b.colors),
                             box=d.box, resid=d.resid, dz_mm=P.dz_mm, dx_mm=P.dx_mm, alpha=d.alpha)
        return out

    # ================================================================ geometry
    def _have(self) -> bool:
        return self.N > 0 and self.src_name in self.src

    def _geom(self):
        nz = self.src[self.src_name].t.shape[0]
        zmm = self.z0 + np.arange(nz) * self.dz
        iz = int(np.flatnonzero(zmm >= self.mirror)[0]) + 1          # 1-based
        return zmm, iz, zmm[iz - 1:] - self.mirror

    def _check_mirror(self, nz, dz, z0, mirror):
        zmm = z0 + np.arange(nz) * dz
        idx = np.flatnonzero(zmm >= mirror)
        if idx.size == 0 or idx[0] + 1 >= zmm.size - 1:
            raise io.ScanError(f"The data covers {zmm[0]:.1f} to {zmm[-1]:.1f} mm but the mirror is set "
                               f"to {mirror:.1f} mm, so there is nothing below it to reconstruct. Check "
                               "Mirror mm and Starts at.")

    def _mirror_text(self):
        _, iz, _ = self._geom()
        return f"Mirror {self.mirror:.2f} mm = sample {iz}."

    def _geom_changed(self):
        if self.busy or not self._have():
            return
        try:
            self._check_mirror(self.src[self.src_name].t.shape[0], self.dz, self.sp_z0.value(),
                               self.sp_mirror.value())
        except io.ScanError as e:
            alert(self, "Mirror out of range", str(e))
            self.sp_mirror.set(self.mirror); self.sp_z0.set(self.z0)
            return
        self.mirror, self.z0 = self.sp_mirror.value(), self.sp_z0.value()
        if self._clean_active():
            self._apply_clean()
        self._invalidate("The geometry changed - press display3D")
        self._draw_frame(); self._draw_map(); self._draw2d(); self._sivv_recompute(False)
        self._set_status(self._mirror_text())

    # ================================================================ angles
    def angles(self):
        return recon.stepped_angles(self.cal, self.N, self.sp_shift.ival(), self.cb_flip.isChecked(),
                                    self.dd_angle.currentText())

    def _angles_note(self):
        ang, valid = self.angles()
        if valid.sum() < 2:
            return "The calibration gives fewer than two positions an angle - check Shift and Angle map."
        v = np.flatnonzero(valid)
        s = (f"{valid.sum()} of {self.N} positions angled ({v[0] + 1}:{v[-1] + 1}, "
             f"{ang[v[0]]:+.1f} to {ang[v[-1]]:+.1f} deg).")
        npos = self.cal.get("nPositions")
        if npos is not None and int(npos) != self.N:
            s += f" Calibration has {int(npos)} positions, this set {self.N} - {self.dd_angle.currentText()} mapping."
        return s

    def _settings_changed(self):
        if self.busy or not self._have():
            return
        if self._clean_active():
            self._apply_clean()
        self._invalidate("The angles changed - press display3D")
        self._draw_map(); self._draw2d(); self._sivv_recompute(False)
        if self.cal is not None:
            self._set_status(self._angles_note())

    # ================================================================ source + scale
    def set_source(self, name: str):
        if name not in self.names:
            return
        if self._have() and name not in self.src:
            alert(self, "Not in this set", f"This set has no {name}.")
            self._paint_source_buttons()
            return
        self.src_name = name
        self._paint_source_buttons()
        lo, hi = self.range[name]
        self.sp_lo.set(lo); self.sp_hi.set(hi)
        st = max(0.05, round(self.scale[name][1] / 32, 2)) if self.mode == "cf" else (0.25 if name == "Velocity" else 0.05)
        self.sp_lo.setSingleStep(st); self.sp_hi.setSingleStep(st)
        if not self._have():
            self._refresh()
            return
        self._draw_frame(); self._draw_map(); self._draw2d()
        if name in self.vol:
            self._show_volume(name)
        else:
            self.view.clear(f"Press display3D to render {name}")
            self.shown = ""
        self._refresh()

    def _paint_source_buttons(self):
        for nm, b in self.src_btn.items():
            on = nm == self.src_name
            b.setChecked(on)
            b.setStyleSheet("background:#296bcc; color:white; border-radius:4px;" if on else
                            "background:#f5f5f5; color:#262626; border:1px solid #bbb; border-radius:4px;")

    def set_scale(self, vmax: float):
        """Color flow: what the ends of the bar mean, +/- cm/s."""
        if self.mode != "cf" or not (vmax > 0):
            return
        old = self.scale["Speed"][1]
        self.vmax = vmax
        self.sp_vmax.set(vmax)
        self.scale = {"Velocity": [-vmax, vmax], "Speed": [0.0, vmax]}
        for nm in self.names:
            self.range[nm] = [v * vmax / old for v in self.range[nm]]
        self.sp_lo.set(self.range[self.src_name][0]); self.sp_hi.set(self.range[self.src_name][1])
        k = list(self.clean["keep"])
        if k[1] >= old:
            k[1] = vmax
        self.clean["keep"] = [min(k[0], vmax), min(k[1], vmax)]
        self.sp_klo.set(self.clean["keep"][0]); self.sp_khi.set(self.clean["keep"][1])
        if self._have():
            self._reclean()

    # ================================================================ clean-up
    def t_for(self, name: str) -> np.ndarray:
        return self.tv[name] if name in self.tv else self.src[name].t

    def _keep_cut(self):
        k = self.clean["keep"]
        if self.mode == "cf":
            vmax = self.scale["Speed"][1]
            tlo, thi = k[0] / vmax, k[1] / vmax
        else:
            sc = self.scale["Velocity"]
            tlo, thi = (k[0] - sc[0]) / (sc[1] - sc[0]), (k[1] - sc[0]) / (sc[1] - sc[0])
        return (tlo > 0 or thi < 1), tlo, thi

    def _clean_active(self) -> bool:
        need = "Speed" if self.mode == "cf" else "Velocity"
        return need in self.src and self._keep_cut()[0]

    def _apply_clean(self):
        self.rej_frac = 0.0
        self.tv = {}
        if not self._clean_active():
            return
        _, tlo, thi = self._keep_cut()
        if self.mode == "cf":
            sp = self.src["Speed"].t
            with np.errstate(invalid="ignore"):
                rej = (sp < tlo) | (sp > thi)
            for nm in self.names:
                t = self.src[nm].t.copy()
                t[rej] = np.nan
                self.tv[nm] = t
            self.rej_frac = rej.sum() / max(1, np.sum(~np.isnan(sp)))
            return
        src = self.src["Velocity"]
        t = src.t.copy()
        with np.errstate(invalid="ignore"):
            hot = t > thi
            cold = t < tlo
        b = src.box
        if self.clean["margin"] > 0 and hot.any() and b:
            rz, rx, rp = self._margin_px(self.clean["margin"])
            sub = hot[b[0] - 1:b[1], b[2] - 1:b[3], :]
            hot[b[0] - 1:b[1], b[2] - 1:b[3], :] = dilate_ellipsoid(sub, rz, rx, rp)
        rej = (hot | cold) & ~np.isnan(t)
        t[rej] = np.nan
        self.tv["Velocity"] = t
        self.rej_frac = rej.sum() / max(1, np.sum(~np.isnan(src.t)))

    def _margin_px(self, mm):
        rz, rx = mm / self.dz, mm / self.dx
        rp = rz
        ang, valid = self.angles()
        b = self.src["Velocity"].box
        if valid.sum() >= 2 and b:
            dth = np.median(np.abs(np.diff(ang[valid]))) * np.pi / 180
            dmid = self.z0 + ((b[0] + b[1]) / 2 - 1) * self.dz - self.mirror
            if dth > 0 and dmid > 0:
                rp = mm / (dmid * dth)
        return rz, rx, rp

    def _clean_changed(self):
        if self.busy or not self._have():
            return
        if self.mode == "swe":
            self.clean["margin"] = self.sp_margin.value()
        if self.sp_khi.value() <= self.sp_klo.value():
            self.sp_khi.set(self.sp_klo.value() + max(abs(self.sp_klo.value()) * 1e-3, 1e-3))
        self.clean["keep"] = [self.sp_klo.value(), self.sp_khi.value()]
        self._reclean()

    def _reclean(self):
        was = bool(self.shown) and self.view.active
        self._apply_clean()
        self.vol = {}
        self._draw_map(); self._draw2d(); self._sivv_recompute(False)
        self._set_status(self._clean_text())
        self._refresh()
        if was and self.shown == self.src_name and self.rec is not None:
            self.display3d(keep_view=True)
        elif was:
            self.view.clear("The values changed - press display3D")
            self.shown = ""

    def _smooth_changed(self):
        if self.busy or not self._have():
            return
        self.clean["smooth"] = self.sp_smooth.value()
        self._draw2d()
        self._rerender()
        self._set_status(self._clean_text())

    def _clean_text(self):
        c = self.clean
        what = []
        if self.mode == "cf":
            vmax = self.scale["Speed"][1]
            if c["keep"][0] > 0:
                what.append(f"|v| below {c['keep'][0]:.3g} cm/s")
            if c["keep"][1] < vmax:
                what.append(f"|v| above {c['keep'][1]:.3g} cm/s")
            label_ = "Flow"
        else:
            sc = self.scale["Velocity"]
            if c["keep"][1] < sc[1]:
                what.append(f"above {c['keep'][1]:.3g} m/s + {c['margin']:.3g} mm")
            if c["keep"][0] > sc[0]:
                what.append(f"below {c['keep'][0]:.3g} m/s")
            label_ = "Velocity"
        if not what:
            s = f"{label_}: nothing taken out."
            self.lb_clean.setText("")
        else:
            s = f"{label_}: {100 * self.rej_frac:.1f}% taken out ({'; '.join(what)})."
            self.lb_clean.setText(f"{100 * self.rej_frac:.0f}% out")
        if c["smooth"] > 0:
            s += f" Smoothing {c['smooth']:.3g} mm."
        return s

    # ================================================================ capture + map
    def set_position(self, v):
        if not self._have() or not np.isfinite(v):
            return
        p = int(min(max(round(v), 1), self.N))
        self.pos = p
        self.sp_pos.set(p)
        self._draw_frame()
        self._mark_position()

    def set_lat(self, v):
        if self.busy or not self._have() or not np.isfinite(v):
            return
        v = int(round(v))
        nx = self.src[self.src_name].t.shape[1]
        if v < 1 or v > nx:
            return
        self.sp_lat.set(v)
        self._draw_frame(); self._draw_map(); self._draw2d()

    def _paint(self, G, T, name):
        sc, rng, cm = self.scale[name], self.range[name], self.src[name].cmap
        V = sc[0] + T.astype(float) * (sc[1] - sc[0])
        u = (V - rng[0]) / (rng[1] - rng[0])
        with np.errstate(invalid="ignore"):
            ok = ~np.isnan(T) & (u >= 0) & (u <= 1)
        G = np.clip(G, 0, 1)
        RGB = np.repeat(G[:, :, None], 3, axis=2)
        idx = (np.floor(255 * u[ok] + 0.5)).astype(int)
        RGB[ok] = cm[idx]
        return RGB

    def _draw_frame(self):
        if not self._have():
            return
        src = self.src[self.src_name]
        zmm, _, _ = self._geom()
        nx = src.rgb.shape[1]
        p = min(max(self.pos, 1), self.N)
        ax = self.ax_frame
        ax.show_image(src.rgb[:, :, :, p - 1], 1, nx, zmm[0], zmm[-1])
        ax.setAspectLocked(True, ratio=self.dx)          # a lateral line is dx mm wide
        for t in ("marks", "sivv"):
            ax.clear_tag(t)
        L = self.sp_lat.ival()
        ax.line("marks", [L, L], [zmm[0], zmm[-1]], color=(255, 217, 51), width=1.4)
        ax.line("marks", [0.5, nx + 0.5], [self.mirror] * 2, color=(255, 255, 255), style=QtCore.Qt.DotLine)
        ax.set_title(f"{self.src_name}, position {p}")
        ax.setLabel("bottom", f"{self.files[p - 1]}  {self._clock(self.times[p - 1])}")
        self._mark_sivv()

    @staticmethod
    def _clock(t):
        if not np.isfinite(t):
            return ""
        s = t % 86400
        return f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"

    def _draw_map(self):
        if not self._have():
            return
        name = self.src_name
        src = self.src[name]
        zmm, _, _ = self._geom()
        L = min(max(self.sp_lat.ival(), 1), src.t.shape[1])
        G = src.grey[:, L - 1, :].astype(float) / 255
        T = self.t_for(name)[:, L - 1, :]
        ax = self.ax_map
        for t in ("marks", "sivv", "pos"):
            ax.clear_tag(t)
        ax.show_image(self._paint(G, T, name), 1, self.N, zmm[0], zmm[-1])
        ax.line("marks", [0.5, self.N + 0.5], [self.mirror] * 2, style=QtCore.Qt.DotLine)
        _, valid = self.angles()
        if valid.any():
            v = np.flatnonzero(valid) + 1
            y0, y1 = zmm[0] - self.dz / 2, zmm[-1] + self.dz / 2
            ax.line("marks", [v[0] - 0.5, v[-1] + 0.5, v[-1] + 0.5, v[0] - 0.5, v[0] - 0.5],
                    [y0, y0, y1, y1, y0], width=1.6, style=QtCore.Qt.DashLine)
        ax.set_title(f"{name} at lateral line {L}  –  " +
                     ("no calibration yet" if self.cal is None else "dashed: positions with an angle"))
        self._mark_position()
        self._mark_sivv()

    def _mark_position(self):
        if not self._have():
            return
        zmm, _, _ = self._geom()
        self.ax_map.clear_tag("pos")
        self.ax_map.line("pos", [self.pos, self.pos], [zmm[0], zmm[-1]], color=(255, 217, 51), width=1.2)

    # ================================================================ 2-D frame
    def recon2d(self, name, iLat):
        ang, valid = self.angles()
        pos = np.flatnonzero(valid)
        if pos.size < 2:
            raise ValueError(f"Only {pos.size} position(s) of this set fall inside the calibration's "
                             "angle range. Try a different Shift or Angle map.")
        a = ang[pos]
        if np.max(np.abs(a)) <= 1:
            raise ValueError(f"The angles span only {a.min():+.2f} to {a.max():+.2f} deg.")
        zmm, iz, d = self._geom()
        src = self.src[name]
        iLat = int(min(max(round(iLat), 1), src.t.shape[1]))
        g = SectorGrid(a, d, DEF_LAMBDA)
        T = self.t_for(name)[iz - 1:, iLat - 1, :][:, pos].astype(float)
        t = recon.sweep_values(T, g)
        grey = g.apply(src.grey[iz - 1:, iLat - 1, :][:, pos].astype(float))
        sg = self.clean["smooth"]
        if sg > 0:
            t = recon.nan_gauss(t, sg / np.array([np.mean(np.diff(g.zz)), np.mean(np.diff(g.yy))]))
        return dict(name=name, t=t, grey=grey, zz=g.zz + self.mirror, yy=g.yy, iLat=iLat,
                    pos=pos + 1, ang=a, smooth=sg)

    def _draw2d(self):
        if not self._have():
            return
        if self.cal is None:
            self.rec = None
            self.ax_rec.img.clear()
            self.ax_rec.set_title("Pick a calibration to reconstruct")
            return
        try:
            self.rec = self.recon2d(self.src_name, self.sp_lat.ival())
        except ValueError as e:
            self.rec = None
            self.ax_rec.img.clear()
            self.ax_rec.set_title(str(e))
            return
        self._paint2d()

    def _paint2d(self):
        R = self.rec
        if R is None:
            return
        name = R["name"]
        ax = self.ax_rec
        for t in ("probe", "sivv"):
            ax.clear_tag(t)
        ax.show_image(self._paint(R["grey"] / 255, R["t"], name), R["yy"][0], R["yy"][-1],
                      R["zz"][0], R["zz"][-1])
        # units only: the title above already names the source, and the full
        # label is clipped beside a narrow column
        ax.colorbar(float_lut(self.src[name].cmap), *self.range[name], self.units[name].split(" ")[0])
        title = (f"{name}   lateral line {R['iLat']}   positions {R['pos'][0]}:{R['pos'][-1]} "
                 f"({R['pos'].size} angled, {R['ang'].min():+.1f} to {R['ang'].max():+.1f} deg)")
        extra = []
        if self._clean_active():
            extra.append(f"cleaned: {100 * self.rej_frac:.0f}% taken out")
        if R["smooth"] > 0:
            extra.append(f"smoothed {R['smooth']:.3g} mm")
        ax.set_title(title + ("<br>" + ",  ".join(extra) if extra else ""))
        self._mark_sivv()

    def read_at(self, y, z):
        if self.rec is None:
            return float("nan")
        R = self.rec
        iy = int(np.argmin(np.abs(R["yy"] - y)))
        iz = int(np.argmin(np.abs(R["zz"] - z)))
        t = R["t"][iz, iy]
        name = R["name"]
        self.ax_rec.clear_tag("probe")
        self.ax_rec.points("probe", [R["yy"][iy]], [R["zz"][iz]], color=(255, 255, 255), size=14, symbol="+")
        if np.isnan(t):
            self._set_status(f"{name} at elevation {R['yy'][iy]:+.2f} mm, depth {R['zz'][iz]:.2f} mm: no value there.")
            return float("nan")
        sc, rng = self.scale[name], self.range[name]
        v = sc[0] + t * (sc[1] - sc[0])
        note = "  - outside Range, so not shown" if (v < rng[0] or v > rng[1]) else ""
        self._set_status(f"{name} at elevation {R['yy'][iy]:+.2f} mm, depth {R['zz'][iz]:.2f} mm, "
                         f"lateral line {R['iLat']}: {v:.3g} {self.units[name]}{note}")
        return float(v)

    # ================================================================ volume
    def display3d(self, keep_view: bool = False):
        if self.busy:
            return
        if not self._have() or self.cal is None:
            alert(self, "Nothing to render", "Load a set and pick a calibration first.")
            return
        self.busy = True
        self._refresh()
        name = self.src_name
        src = self.src[name]
        dlg = Busy(self, "display3D", "Reconstructing", cancelable=True)
        try:
            ang, valid = self.angles()
            pos = np.flatnonzero(valid)
            if pos.size < 2:
                raise ValueError("Fewer than two positions have an angle.")
            a = ang[pos]
            zmm, iz, _ = self._geom()
            if self.mode == "cf":
                b = src.box
                rr, cc = np.arange(b[0], b[1] + 1), np.arange(b[2], b[3] + 1)
            else:
                anyM = src.mask.any(axis=2)
                rr = np.flatnonzero(anyM.any(axis=1)) + 1
                cc = np.flatnonzero(anyM.any(axis=0)) + 1
            nx = src.t.shape[1]
            if self.full:
                vox = self.sp_vox.value()
                step = max(1, int(mround(vox / self.dx)))
                lat = np.arange(1, nx + 1, step)
                rows = np.arange(iz, src.t.shape[0] + 1)
                lam, dxe = 4 * vox, step * self.dx
                needB = self.volB is None or self.volB["key"] != vox
            else:
                r0, r1 = max(iz, rr[0] - 2), min(src.t.shape[0], rr[-1] + 2)
                if r1 - r0 < 2:
                    raise ValueError(f"The box lies above the mirror ({self.mirror:.1f} mm), so none of it "
                                     "is swept into the sector.")
                rows = np.arange(r0, r1 + 1)
                lat = np.arange(cc[0], cc[-1] + 1)
                # the data's own sample pitch, not lambda/4 then resampled
                lam, dxe, needB = 4 * max(DEF_LAMBDA / 4, min(self.dz, self.dx)), self.dx, False
            dsub = zmm[rows - 1] - self.mirror
            g = SectorGrid(a, dsub, lam)
            in_box = (lat >= cc[0]) & (lat <= cc[-1])
            Tn = self.t_for(name)
            nL = lat.size
            vol = np.full((g.zz.size, nL, g.yy.size), np.nan, np.float32)
            bvol = np.zeros((g.zz.size, nL, g.yy.size), np.float32) if needB else None
            chunk = 12
            for s in range(0, nL, chunk):
                if dlg.cancelled:
                    self._set_status("display3D cancelled.")
                    return
                sel = np.arange(s, min(nL, s + chunk))
                li = lat[sel] - 1
                if needB:
                    G = src.grey[np.ix_(rows - 1, li, pos)].astype(float).transpose(0, 2, 1)
                    bvol[:, sel, :] = g.apply(G).transpose(0, 2, 1)
                sb = sel[in_box[sel]]
                if sb.size:
                    T = Tn[np.ix_(rows - 1, lat[sb] - 1, pos)].astype(float).transpose(0, 2, 1)
                    vol[:, sb, :] = recon.sweep_values(T, g).transpose(0, 2, 1)
                dlg(0.8 * (sel[-1] + 1) / nL, f"{name}: lateral line {sel[-1] + 1} of {nL}")
            vox3 = [float(np.mean(np.diff(g.zz))), dxe, float(np.mean(np.diff(g.yy)))]
            if needB:
                self.volB = {"b": bvol, "vox": vox3, "iso": None, "key": self.sp_vox.value()}
            self.vol[name] = {"t": vol, "vox": vox3, "zz": g.zz + self.mirror, "yy": g.yy,
                              "x": (lat - 1) * self.dx, "lat": lat, "iso": None, "iso_s": None,
                              "full": self.full}
            dlg(0.85, "Rendering")
            if not keep_view:
                self.sl_rot.blockSignals(True); self.sl_rot.setValue(0); self.sl_rot.blockSignals(False)
                self.lb_ang.setText("0 deg")
                self.view.angle = 0
            self._show_volume(name, fresh=not keep_view)
        except Exception as e:
            self.vol.pop(name, None)
            self.view.clear("Rendering failed - press display3D to try again")
            alert(self, "display3D failed", str(e))
            return
        finally:
            dlg.close()
            self.busy = False
            self._refresh()
        Vn = self.vol[name]
        if Vn["full"]:
            pitch = Vn["lat"][1] - Vn["lat"][0] if Vn["lat"].size > 1 else 1
            self._set_status(f"Full volume: B-mode + {name}, voxels of {max(Vn['vox']):.3f} mm "
                             f"(lateral lines 1:{pitch}:{Vn['lat'][-1]}). Drag it, or use the slider.")
        else:
            self._set_status(f"{name} box (lateral lines {Vn['lat'][0]}:{Vn['lat'][-1]}), voxel "
                             f"{Vn['vox'][0]:.3f} x {Vn['vox'][1]:.3f} x {Vn['vox'][2]:.3f} mm. "
                             "Drag it, or use the slider.")

    def _iso_frame(self, Vn, sz):
        """Spacing and origin (depth, lateral, elevation) of the isotropic grid."""
        n0 = np.array(Vn["t"].shape, float)
        v = np.array(Vn["vox"])
        sp = n0 * v / np.array(sz)
        c0 = np.array([Vn["zz"][0], Vn["x"][0], Vn["yy"][0]])
        return sp, c0 - v / 2 + sp / 2

    def _vol_alpha(self, name):
        c = recon.alpha_curve(self.sp_knee.value(), self.sp_gamma.value())
        x = np.linspace(0, 1, 255)
        if self.mode == "cf" and name == "Velocity":
            rng = self.range["Velocity"]
            v = rng[0] + x * (rng[1] - rng[0])
            x = np.abs(v) / max(max(abs(rng[0]), abs(rng[1])), 1e-12)
        return np.r_[0.0, np.interp(x, np.linspace(0, 1, 256), c)]

    def _vol_cmap(self, name, iso_style=False):
        cm = self.src[name].cmap
        if iso_style:
            return np.repeat(cm[int(0.75 * 255)][None, :], 256, 0)
        xi = np.linspace(0, 1, 255)
        c = np.stack([np.interp(xi, np.linspace(0, 1, 256), cm[:, i]) for i in range(3)], 1)
        return np.r_[[[0, 0, 0]], c]

    def _show_volume(self, name, fresh=False):
        Vn = self.vol[name]
        sg = self.clean["smooth"]
        if Vn["iso"] is None or Vn["iso_s"] != sg:
            # nearly cubic voxels (the full view) are drawn on their own grid
            Vn["iso"], _ = recon.iso_volume(Vn["t"], Vn["vox"], sg, tolerance=1.25)
            Vn["iso_s"] = sg
        sc, rng = self.scale[name], self.range[name]
        vals = sc[0] + Vn["iso"].astype(float) * (sc[1] - sc[0])
        U = recon.window_to_u8(vals, rng)
        if self.cb_largest.isChecked():
            U = recon.keep_largest(U)
        spacing, origin = self._iso_frame(Vn, U.shape)

        full = Vn["full"] and self.volB is not None
        style = {"MIP": "mip", "Volume": "volume", "Isosurface": "iso", "Slices": "slices"}[self.dd_style.currentText()]
        bstyle = self.dd_bstyle.currentText()
        fused = full and style == "slices" and bstyle == "Slices"
        showB = full and not fused and bstyle != "Off"
        Ub = self._bmode_data(U.shape) if full else None

        v = self.view
        if fresh:
            v.clear()
        v._ensure()
        if not showB:
            v._remove("b")
        if showB:
            bs = {"MIP": "mip", "Volume": "volume", "Slices": "slices"}[bstyle]
            v.set_volume("b", Ub, spacing, np.repeat(np.linspace(0, 1, 256)[:, None], 3, 1),
                         self.sp_bopa.value() * np.linspace(0, 1, 256), style=bs, origin=origin)
        if fused:
            cm = self._vol_cmap(name)
            rgb = np.repeat(Ub[..., None], 3, axis=3)
            has = U > 0
            rgb[has] = np.clip(np.round(cm[U[has].astype(int)] * 255), 0, 255).astype(np.uint8)
            v.set_volume("flow", U, spacing, cm, self._vol_alpha(name), style="slices", origin=origin, rgb=rgb)
        else:
            v.set_volume("flow", U, spacing, self._vol_cmap(name, style == "iso"),
                         self._vol_alpha(name), style=style, origin=origin)
        self.shown = name
        n = np.array(U.shape)
        ext = origin + (n - 1) * spacing
        bounds = [origin[1], ext[1], origin[2], ext[2], origin[0], ext[0]]
        if fresh:
            v.frame(bounds)
        self._surface3d()

    def _bmode_data(self, sz):
        VB = self.volB
        if VB["iso"] is None or VB["iso"].shape != tuple(sz):
            b = VB["b"]
            VB["iso"] = imresize3(b, sz) if b.shape != tuple(sz) else b
        w = [self.sp_blo.value(), self.sp_bhi.value()]
        if w[1] <= w[0]:
            w[1] = w[0] + 1
        return (255 * np.clip((VB["iso"] - w[0]) / (w[1] - w[0]), 0, 1)).astype(np.uint8)

    def _rerender(self):
        if self.busy or not self.shown or self.shown not in self.vol or not self.view.active:
            return
        try:
            self._show_volume(self.shown)
        except Exception as e:
            alert(self, "Could not re-render the volume", str(e))

    def _apply_alpha(self):
        a = recon.alpha_curve(self.sp_knee.value(), self.sp_gamma.value())
        self.curve.set_curve(a)
        if self.shown and self.view.active and self.dd_style.currentText() in ("MIP", "Volume"):
            self.view.set_opacity("flow", self._vol_alpha(self.shown))

    def _display_changed(self):
        if self.busy:
            return
        if self.sp_hi.value() <= self.sp_lo.value():
            self.sp_hi.set(self.sp_lo.value() + max(abs(self.sp_lo.value()) * 1e-3, 1e-3))
        self.range[self.src_name] = [self.sp_lo.value(), self.sp_hi.value()]
        if not self._have():
            return
        self._draw_map(); self._paint2d(); self._rerender()
        if self.mode == "cf":
            self._draw_sivv()

    def _b_changed(self):
        if self.sp_bhi.value() <= self.sp_blo.value():
            self.sp_bhi.set(min(255, self.sp_blo.value() + 1))
        self._rerender()

    def _full_changed(self, on):
        self.full = bool(on)
        self.vol, self.volB = {}, None
        self.view.clear("Press display3D to render " + ("the full volume" if on else "the box"))
        self.shown = ""
        self._refresh()

    def _invalidate(self, msg):
        self.vol, self.volB = {}, None
        if self.view.active:
            self.view.clear(msg)
        self.shown = ""

    def _rotate(self, a):
        self.lb_ang.setText(f"{a} deg")
        if self.view.active:
            self.view.set_angle(float(a))

    def save_video(self):
        if not self.view.active or self.busy:
            return
        stem = f"{self.shown.lower()}_{'full_' if self.vol.get(self.shown, {}).get('full') else ''}volume"
        start = next_free(os.path.dirname(self.loaded) if self.loaded else os.path.expanduser("~"), stem, ".mp4")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save MP4", start, "MPEG-4 (*.mp4)")
        if not path:
            return
        self.busy = True
        dlg = Busy(self, "Save MP4", "Rendering the turn", cancelable=True)
        try:
            ok = self.view.save_turn(path, dlg, lambda: dlg.cancelled)
            self._set_status(f"Wrote {path}" if ok else "Save stopped - nothing written.")
        except Exception as e:
            alert(self, "Could not write the video", str(e))
        finally:
            dlg.close()
            self.busy = False

    # ================================================================ SIVV (colour flow)
    def _sivv_ready(self):
        return self.mode == "cf" and self._have() and self.cal is not None and self.angles()[1].sum() >= 2

    def _sivv_recompute(self, reset_depth):
        if self.mode != "cf":
            return
        self.sivv["prof"], self.sivv["at"] = None, None
        P = None
        if self._sivv_ready():
            ang, valid = self.angles()
            zmm, _, _ = self._geom()
            sc = self.scale["Velocity"]
            v = sc[0] + self.t_for("Velocity").astype(float) * (sc[1] - sc[0])
            P = recon.sivv_profile(v, self.src["Velocity"].box, zmm, self.mirror, self.dx, ang, valid)
        if P is None:
            self._sivv_clear()
            self._refresh()
            return
        self.sivv["prof"] = P
        lo, hi = float(P.z[0]), float(P.z[-1])
        if hi <= lo:
            hi = lo + self.dz
        self.sp_sz.setRange(lo, hi)
        self.sp_sz.setSingleStep(self.dz)
        d = self.sivv["depth"]
        if reset_depth or not np.isfinite(d) or d < P.z[0] or d > P.z[-1]:
            d = recon.sivv_default_depth(P)
        self.sivv["depth"] = d
        self.sp_sz.set(d)
        self._sivv_eval()
        self._refresh()

    def _sivv_eval(self):
        P = self.sivv["prof"]
        if P is None:
            return
        self.sivv["at"] = recon.sivv_at(P, self.sivv["depth"], self.sivv["n"])
        self._draw_sivv()
        self._mark_sivv()
        self._surface3d()

    def set_sivv(self, depth=None, n=None):
        if depth is not None and np.isfinite(depth):
            self.sivv["depth"] = float(depth)
            self.sp_sz.set(min(max(depth, self.sp_sz.minimum()), self.sp_sz.maximum()))
        if n is not None and np.isfinite(n):
            self.sivv["n"] = max(0, int(round(n)))
            self.sp_sn.set(self.sivv["n"])
        if self.sivv["prof"] is None:
            return
        self._sivv_eval()
        self._set_status(self.sivv_text())

    def sivv_text(self):
        at = self.sivv["at"]
        if at is None:
            return "SIVV: no surface."
        s = (f"SIVV at depth {at.z:.2f} mm (range {at.d:.2f} mm, {at.rows.size} rows): Q = {at.Q:.3f} mL/min "
             f"({at.Q / 60:.4f} mL/s), towards the probe {at.Qpos:+.3f}, away {at.Qneg:+.3f}, "
             f"flow area {at.area:.2f} mm^2.")
        if at.edge_lat or at.edge_pos:
            s += f" Flow reaches the edge of the {self._edge_what(at)} - Q is short there."
        return s

    @staticmethod
    def _edge_what(at):
        if at.edge_lat and at.edge_pos:
            return "box and the sweep"
        return "colour box" if at.edge_lat else "sweep"

    def _sivv_show(self, on):
        self.sivv["show"] = bool(on)
        self._surface3d()

    def _sivv_clear(self):
        if self.mode != "cf":
            return
        self.ax_surf.img.clear(); self.ax_surf.set_title("Surface")
        self.ax_q.clear_tag("q"); self.ax_q.set_title("Q vs depth" if self.cal is not None else "Pick a calibration")
        self.lb_q.setText("Q  –"); self.lb_qinfo.setText("")
        self._mark_sivv()
        self._surface3d()

    def _draw_sivv(self):
        P, at = self.sivv["prof"], self.sivv["at"]
        if P is None or at is None:
            return
        src = self.src["Velocity"]
        lat = (P.cols - 1) * self.dx
        arc = at.d * P.th
        if arc.size > 1 and arc[-1] > arc[0]:
            sg = np.linspace(arc[0], arc[-1], max(120, 2 * arc.size))
            idx = np.clip(np.round(np.interp(sg, arc, np.arange(arc.size))).astype(int), 0, arc.size - 1)
        else:
            sg, idx = arc, np.arange(arc.size)
        sc = self.scale["Velocity"]
        Vm = at.map[:, idx].T
        Tm = (Vm - sc[0]) / (sc[1] - sc[0])
        Tm[Vm == 0] = np.nan
        G = src.grey[np.ix_(at.rows - 1, P.cols - 1, P.pos - 1)].astype(float).mean(axis=0)
        G = G[:, idx].T / 255
        self.ax_surf.show_image(self._paint(G, Tm, "Velocity"), lat[0], lat[-1], sg[0], sg[-1])
        self.ax_surf.set_title(f"{at.Q:.1f} mL/min", color="#bf1abf")
        self.ax_surf.setToolTip(f"The surface unrolled flat - lateral across, elevational arc down - with "
                                f"the velocities on it. Q = {at.Q:.2f} mL/min at depth {at.z:.2f} mm.")

        ax = self.ax_q
        ax.clear_tag("q")
        ax.img.clear()
        zb = P.z[at.ii]
        import pyqtgraph as pg
        if zb.size > 1:
            ylo, yhi = min(P.Q.min(), 0), max(P.Q.max(), 0)
            ax.add("q", pg.LinearRegionItem((zb[0], zb[-1]), movable=False,
                                            brush=pg.mkBrush(191, 26, 191, 30), pen=pg.mkPen(None)))
        ax.add("q", pg.InfiniteLine(0, angle=0, pen=pg.mkPen((128, 128, 128), style=QtCore.Qt.DotLine)))
        ax.line("q", P.z, P.Q, color=(191, 26, 191), width=2)
        if P.edge.any():
            ax.points("q", P.z[P.edge], P.Q[P.edge], color=(115, 115, 115), size=6, symbol="x")
        ax.add("q", pg.InfiniteLine(at.z, angle=90, pen=pg.mkPen((191, 26, 191), style=QtCore.Qt.DashLine)))
        ax.points("q", [at.z], [at.Q], color=(191, 26, 191), size=9, edge=(0, 0, 0))
        ax.getViewBox().setRange(xRange=(P.z[0], P.z[-1]), yRange=(min(P.Q.min(), 0), max(P.Q.max(), 0) * 1.05 + 1e-9),
                                 padding=0.02)
        ax.set_title("Q vs depth")

        self.lb_q.setText(f"Q  {at.Q:.2f} mL/min")
        info = (f"{at.Q / 60:.3f} mL/s   sd {at.Qsd:.2f}\nto {at.Qpos:+.1f}  from {at.Qneg:+.1f}\n"
                f"{at.area:.1f} mm² at {at.d:.1f} mm")
        if at.edge_lat or at.edge_pos:
            info = f"CUT OFF: {self._edge_what(at)} edge\n" + info
        self.lb_qinfo.setText(info)

    def _mark_sivv(self):
        if self.mode != "cf":
            return
        for ax in (self.ax_frame, self.ax_map, self.ax_rec):
            ax.clear_tag("sivv")
        at, P = self.sivv["at"], self.sivv["prof"]
        if at is None or P is None or not self._have():
            return
        col = MAGENTA
        self.ax_frame.line("sivv", [P.cols[0] - 0.5, P.cols[-1] + 0.5], [at.z, at.z], color=col, width=1.6)
        self.ax_map.line("sivv", [P.pos.min() - 0.5, P.pos.max() + 0.5], [at.z, at.z], color=col, width=1.6)
        R = self.rec
        if R is not None and P.cols[0] <= R["iLat"] <= P.cols[-1]:
            th = np.linspace(P.th[0], P.th[-1], 120)
            self.ax_rec.line("sivv", at.d * np.sin(th), self.mirror + at.d * np.cos(th), color=col, width=1.8)

    def _surface3d(self):
        if self.mode != "cf":
            return
        at, P = self.sivv["at"], self.sivv["prof"]
        have = (self.sivv["show"] and at is not None and P is not None and self.view.active
                and self.shown in self.vol)
        if not have:
            self.view.set_surface("sivv", None, None)
            return
        th = np.linspace(P.th[0], P.th[-1], 72)
        x = np.array([(P.cols[0] - 1.5) * self.dx, (P.cols[-1] - 0.5) * self.dx])
        y = at.d * np.sin(th)
        z = self.mirror + at.d * np.cos(th)
        pts = np.r_[np.c_[np.full(72, x[0]), y, z], np.c_[np.full(72, x[1]), y, z]]
        k = np.arange(71)
        faces = np.r_[np.c_[k, k + 1, 72 + k], np.c_[k + 1, 73 + k, 72 + k]]
        self.view.set_surface("sivv", pts, faces)

    # ================================================================ housekeeping
    def _box_text(self, s: Source):
        b = s.box
        return (f"{(b[3] - b[2] + 1) * s.dx_mm:.1f} x {(b[1] - b[0] + 1) * s.dz_mm:.1f} mm at "
                f"{self.z0 + (b[0] - 1) * s.dz_mm:.1f}-{self.z0 + (b[1] - 1) * s.dz_mm:.1f} mm")

    def _set_status(self, msg):
        self.lb_status.setText(msg)
        self.status.emit(msg)

    def _refresh(self):
        have = self._have()
        busy = self.busy
        for w in (self.bt_cal, self.bt_set):
            w.setEnabled(not busy)
        self.bt_load.setEnabled(bool(self.data_dir) and not busy)
        for nm, b in self.src_btn.items():
            b.setEnabled(have and nm in self.src and not busy)
        for w in (self.sp_lat, self.sp_pos, self.sp_shift, self.cb_flip, self.dd_angle, self.sp_mirror,
                  self.sp_z0, self.sp_smooth, self.sp_klo, self.sp_khi):
            w.setEnabled(have and not busy)
        if self.mode == "cf":
            self.sp_vmax.setEnabled(have and not busy)
            q = self.sivv["prof"] is not None
            for w in (self.sp_sz, self.sp_sn, self.cb_sshow):
                w.setEnabled(q and not busy)
        else:
            self.sp_margin.setEnabled(have and not busy)
        for w in (self.dd_bstyle, self.sp_blo, self.sp_bhi, self.sp_bopa, self.sp_vox):
            w.setEnabled(self.full)
        self.bt_go.setEnabled(have and self.cal is not None and self.rec is not None and not busy)
        on3d = self.view.active
        for w in (self.sl_rot, self.bt_reset, self.bt_mp4):
            w.setEnabled(on3d and not busy)
