"""Index-tab calibration: stepped position sets (Calibration_CF) and cines
(Calibration_GUI_v1). One page class, two kinds:

  stepped   a folder of single-frame DICOMs, one per reflector position; the
            positions ARE the sweep, so there is one block: all of them. A
            capture viewer sits beside the position map.
  cine      one continuous cine; it is cut into sweep blocks at the turnarounds,
            one block is picked on the recording map, and the tabs are clicked
            on that block. Each block keeps its own tabs.

The clicking, snapping and fit are core.tabs, shared by both.
"""
from __future__ import annotations

import os

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from ..core import io, recon, sweep, tabs
from ..core.mlcompat import rgb2gray_u8
from .widgets import ElideLabel, Busy, ImagePanel, Spin, alert, hbox, label, lut_from

VIRIDIS = lut_from("viridis")


class CalibrationPage(QtWidgets.QWidget):
    status = QtCore.Signal(str)

    def __init__(self, kind: str, data_root: str = ""):
        super().__init__()
        assert kind in ("stepped", "cine")
        self.kind = kind
        self.data_root = data_root
        self.data_dir = ""
        self.files = []
        self.B = None                 # grey stack: depth x lateral x columns
        self.temp1 = None             # depth x columns map at the lateral line
        self.spans = []               # list of 1-based column arrays, one per block
        self.sw = None
        self.iblock = None
        self.marks: list[tabs.TabSet] = []
        self.last_win = None
        self.cal = None
        self.stage = "load"
        self.busy = False
        self.mm_per_sample = float("nan")
        self.pos = 1
        self.times = np.zeros(0)
        self.order_by = ""
        self.loaded = ""
        self.kind_loaded = ""
        self._build()
        self._set_stage("load")
        self._refresh()

    # ================================================================ layout
    def _build(self):
        st = self.kind == "stepped"
        unit = "pos" if st else "fr"
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)

        self.gb_top = QtWidgets.QGroupBox("Setup")
        g = QtWidgets.QGridLayout(self.gb_top)
        g.setContentsMargins(8, 4, 8, 4)
        self.lb_file = ElideLabel("(none)"); self.lb_file.setStyleSheet("color:#777")
        self.bt_browse = QtWidgets.QPushButton("Browse…"); self.bt_browse.clicked.connect(self._browse)
        self.bt_load = QtWidgets.QPushButton("Load positions" if st else "Load scan")
        self.bt_load.clicked.connect(self.load)
        g.addWidget(label("Position set" if st else "Calibration scan"), 0, 0)
        g.addWidget(self.lb_file, 0, 1)
        g.addWidget(self.bt_browse, 0, 2)
        g.addWidget(self.bt_load, 0, 3)
        self.dd_scan = None
        if not st:
            self.dd_scan = QtWidgets.QComboBox()
            self.dd_scan.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
            self.dd_scan.setMinimumContentsLength(10)
            self.dd_scan.currentTextChanged.connect(self._scan_changed)
            g.addWidget(label("Also here"), 1, 0)
            g.addWidget(self.dd_scan, 1, 1)
        self.lb_prompt = ElideLabel("", QtCore.Qt.ElideRight)
        f = self.lb_prompt.font(); f.setBold(True); f.setPointSize(12); self.lb_prompt.setFont(f)
        g.addWidget(self.lb_prompt, 2 if not st else 1, 0, 1, 4)
        g.setColumnStretch(1, 1)
        root.addWidget(self.gb_top)

        mid = QtWidgets.QHBoxLayout()
        root.addLayout(mid, 1)
        left = QtWidgets.QGroupBox("Positions  →  click the tabs" if st else
                                   "Recording  →  block  →  click the tabs")
        ll = QtWidgets.QVBoxLayout(left)
        self.ax_cross = ImagePanel("Load a position set" if st else "Block",
                                   "Position" if st else "Frame in block", "Axial sample",
                                   interactive=True)
        self.ax_cross.clicked.connect(self._block_click)
        if st:
            row = QtWidgets.QHBoxLayout()
            cap = QtWidgets.QVBoxLayout()
            self.ax_frame = ImagePanel("Capture", "Lateral line", "Axial sample", lock_aspect=True)
            self.ax_frame.setMaximumWidth(230)
            self.ax_frame.clicked.connect(lambda x, y, b: self.set_lat(x))
            self.sl_pos = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            self.sl_pos.valueChanged.connect(lambda v: self.set_position(v, "slider"))
            self.sp_pos = Spin(1, 2, 1, 1, integer=True, tip="Which position the capture view shows.")
            self.sp_pos.valueEdited.connect(lambda v: self.set_position(v, "spinner"))
            cap.addWidget(self.ax_frame, 1)
            cap.addLayout(hbox(self.sl_pos, self.sp_pos))
            wcap = QtWidgets.QWidget(); wcap.setLayout(cap); wcap.setMaximumWidth(240)
            row.addWidget(wcap)
            row.addWidget(self.ax_cross, 1)
            ll.addLayout(row, 1)
        else:
            self.ax_map = ImagePanel("Load a scan", "Frame", "Axial sample")
            self.ax_map.clicked.connect(self._map_click)
            ll.addWidget(self.ax_map, 85)

        stip = (f"The window a click searches for its tab, centred on the click: this many "
                f"{'positions' if st else 'frames'} either side, and the next number of axial samples "
                "above and below. The tab goes on the middle of the whole bright dash it finds. A dash "
                "that runs on past twice the window - the end of a row - is put by the spacing of the "
                "others instead and drawn hollow. The search runs with the stationary background "
                "removed. Search 0 puts the tab exactly where you clicked.")
        self.sp_lat = Spin(1, 4096, 10, 1, integer=True, tip="Which lateral line the maps are centred on.")
        self.sp_lat.valueEdited.connect(lambda v: self._view_changed())
        self.sp_avg = Spin(1, 4096, 1, 1, integer=True,
                           tip="How many lateral lines to average into the maps. 1 for a DICOM; every "
                               "line for a raw IQ capture.")
        self.sp_avg.valueEdited.connect(lambda v: self._view_changed())
        self.sp_dlo = Spin(0, 255, 60, 5, integer=True,
                           tip="Grey levels shown. Raise the floor until the tab layer separates from "
                               "the background behind it.")
        self.sp_dhi = Spin(1, 256, 256, 5, integer=True)
        self.sp_dlo.valueEdited.connect(lambda v: self._display_changed())
        self.sp_dhi.valueEdited.connect(lambda v: self._display_changed())
        self.sp_snap = Spin(0, 40, 10, 1, integer=True, prefix="±", tip=stip)
        self.sp_snapz = Spin(1, 40, 3, 1, integer=True, prefix="±", tip=stip)
        self.bt_undo = QtWidgets.QPushButton("Undo"); self.bt_undo.clicked.connect(self.undo)
        self.bt_clear = QtWidgets.QPushButton("Clear"); self.bt_clear.clicked.connect(self.clear)
        self.lb_count = QtWidgets.QLabel(""); f = self.lb_count.font(); f.setBold(True); self.lb_count.setFont(f)
        opts = hbox(label("Lat line"), self.sp_lat, label("Lat avg"), self.sp_avg,
                    label("Display"), self.sp_dlo, self.sp_dhi,
                    label(f"Search {unit}", stip), self.sp_snap, label("smp", stip), self.sp_snapz,
                    self.bt_undo, self.bt_clear, self.lb_count, None)
        ll.addLayout(opts)
        if not st:
            ll.addWidget(self.ax_cross, 125)
        mid.addWidget(left, 145)

        right = QtWidgets.QGroupBox("Position to angle" if st else "Frame to angle")
        rl = QtWidgets.QVBoxLayout(right)
        # an ImagePanel for its title, which shortens to fit rather than
        # pushing the plot past the panel
        self.ax_angle = ImagePanel("The angle map appears once two tabs are clicked",
                                   "Position" if st else "Frame in block", "Angle (degrees)",
                                   invert_y=False)
        self.ax_angle.showGrid(x=True, y=True, alpha=0.3)
        rl.addWidget(self.ax_angle, 115)
        self.table = QtWidgets.QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Tab", "Position" if st else "Frame", "Depth", "Angle",
                                              "Gap pos" if st else "Gap fr"])
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        rl.addWidget(self.table, 100)
        mid.addWidget(right, 100)

        bot = QtWidgets.QGridLayout()
        self.sp_step = Spin(0.1, 90, 5, 1, 3, tip="Angle between adjacent index tabs on the fixture.")
        self.sp_step.valueEdited.connect(lambda v: self.compute_angles())
        self.sp_zero = Spin(1, 64, 1, 1, integer=True, tip="The tab at normal incidence. Numbers are drawn on the map.")
        self.sp_neg = Spin(1, 64, 1, 1, integer=True,
                           tip="A tab one step on the negative side. Only which SIDE it is on matters.")
        for w in (self.sp_zero, self.sp_neg):
            f = w.font(); f.setBold(True); w.setFont(f)
            w.valueEdited.connect(lambda v: self.compute_angles())
        self.lb_qual = QtWidgets.QLabel("")
        self.lb_qual.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.ed_name = QtWidgets.QLineEdit("CF_Calibration" if st else "TabRow_Calibration")
        self.ed_name.setToolTip("Base name. The sweep direction measured is appended, -2+ or +2-.")
        self.lb_out = QtWidgets.QLabel("")
        self.bt_save = QtWidgets.QPushButton("Save calibration"); self.bt_save.clicked.connect(self.save)
        bot.addWidget(label("Step (deg)"), 0, 0); bot.addWidget(self.sp_step, 0, 1)
        bot.addWidget(label("0 deg tab", bold=True), 0, 2); bot.addWidget(self.sp_zero, 0, 3)
        bot.addWidget(label("-5 deg tab", bold=True), 0, 4); bot.addWidget(self.sp_neg, 0, 5)
        bot.addWidget(self.lb_qual, 0, 6, 1, 3)
        bot.addWidget(label("Save as"), 1, 0); bot.addWidget(self.ed_name, 1, 1, 1, 4)
        bot.addWidget(self.lb_out, 1, 5, 1, 3)
        bot.addWidget(self.bt_save, 1, 8)
        bot.setColumnStretch(7, 1)
        fb = QtWidgets.QFrame(); fb.setFrameShape(QtWidgets.QFrame.StyledPanel); fb.setLayout(bot)
        root.addWidget(fb)

    # ================================================================ stage
    def _set_stage(self, st):
        self.stage = st
        stepped = self.kind == "stepped"
        n = "2" if stepped else "3"
        txt = {
            "load": (f"Step 1 of {n}  –  pick the folder of calibration captures, then press Load positions."
                     if stepped else "Step 1 of 3  –  pick a calibration scan, then press Load scan."),
            "block": "Step 2 of 3  –  click the block you want, on the map above.",
            "marks": (f"Step {n} of {n}  –  click each index tab on the "
                      f"{'position map' if stepped else 'block view'}. Right click (or ctrl-click) removes one."),
            "ready": "Set the 0 deg and -5 deg tab numbers below, then Save calibration. Keep clicking to add more tabs.",
        }[st]
        self.lb_prompt.setText(txt)

    # ================================================================ selection
    def _start_dir(self):
        return self.data_dir or self.data_root or os.path.expanduser("~")

    def _browse(self):
        if self.kind == "stepped":
            d = QtWidgets.QFileDialog.getExistingDirectory(self, "Pick the folder of calibration captures",
                                                           self._start_dir())
            if d:
                self.set_data(d)
        else:
            f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick the calibration scan", self._start_dir())
            if f:
                self.set_data(f)

    def set_data(self, target: str) -> bool:
        target = str(target)
        if self.kind == "stepped":
            if os.path.isfile(target):
                target = os.path.dirname(target)
            if not os.path.isdir(target):
                alert(self, "No such folder", f"{target} does not exist.")
                return False
            self.data_dir = target
            self.lb_file.setText(target)
        else:
            if os.path.isdir(target):
                names = io.list_scan_files(target)
                if not names:
                    alert(self, "Nothing to load", f"Nothing readable in\n{target}\n\nExpected a Siemens "
                          "DICOM cine, or a GE IQ .dat with its IQSizeInfo.mat beside it.")
                    return False
                p, f = target, names[0]
            else:
                if not os.path.isfile(target):
                    alert(self, "No such file", f"{target} does not exist.")
                    return False
                p, f = os.path.dirname(target), os.path.basename(target)
                if not io.scan_file_kind(target)[0]:
                    alert(self, "Not a scan file", f"{f} is neither a DICOM cine nor a GE IQ capture.")
                    return False
                names = io.list_scan_files(p)
                if f not in names:
                    names = sorted(names + [f])
            self.data_dir, self.files = p, names
            self.dd_scan.blockSignals(True)
            self.dd_scan.clear(); self.dd_scan.addItems(names); self.dd_scan.setCurrentText(f)
            self.dd_scan.blockSignals(False)
            self.lb_file.setText(os.path.join(p, f))
        self.lb_file.setStyleSheet("")
        self._set_stage("load")
        self._refresh()
        return True

    def _scan_changed(self, name):
        if name and self.data_dir:
            self.lb_file.setText(os.path.join(self.data_dir, name))
            self._set_stage("load")

    # ================================================================ loading
    def load(self):
        if self.busy or not self.data_dir:
            return
        self.busy = True
        self._refresh()
        dlg = Busy(self, "Loading", "Reading " + self.data_dir)
        try:
            if self.kind == "stepped":
                ps = io.load_position_stack(self.data_dir, progress=dlg)
                P = ps.panels[0]
                N = ps.N
                dlg(1.0, "Converting to grey")
                B = np.stack([rgb2gray_u8(P.rgb[:, :, :, k]) for k in range(N)], axis=2)
                self.files, self.times, self.order_by = ps.files, ps.time_s, ps.order_by
                self.mm_per_sample, self.mm_per_line = P.dz_mm, P.dx_mm
                extra = f"   (panel 1 of {len(ps.panels)} on the screen)" if len(ps.panels) > 1 else ""
                title = (f"Setup   –   {os.path.basename(self.data_dir)}:  {N} positions, {B.shape[0]} x "
                         f"{B.shape[1]}, ordered by {ps.order_by}, {P.dz_mm:.4f} mm/pixel{extra}")
                kind = "dicomset"
            else:
                fname = os.path.join(self.data_dir, self.dd_scan.currentText())
                sc = io.load_scan_stack(fname, crop=(369, 656), progress=dlg)
                B = sc.B
                self.mm_per_sample = sc.dz_mm
                title = f"Setup   –   {os.path.basename(fname)}:  {B.shape[2]} frames, {B.shape[0]} x {B.shape[1]}"
                kind = sc.kind
        except Exception as e:
            dlg.close()
            self.busy = False
            alert(self, "Could not load this set",
                  f"{e}" + (f"\n\nStill showing {self.loaded}." if self.loaded else ""))
            self._refresh()
            return

        prev = self.kind_loaded
        self.B = B
        self.kind_loaded = kind
        self.loaded = self.lb_file.text()
        self.cal = None
        self.gb_top.setTitle(title)
        nx = B.shape[1]
        self.sp_lat.setRange(1, nx); self.sp_avg.setRange(1, nx)
        if kind == "geiq" and prev != kind:
            self.sp_lat.set(round(nx / 2)); self.sp_avg.set(nx)
            v = np.sort(B.ravel()[::max(1, B.size // 2_000_000)])
            self.sp_dlo.set(min(255, max(0, round(float(v[max(0, round(0.55 * v.size) - 1)])))))
            self.sp_dhi.set(256)
        elif prev != kind:
            self.sp_lat.set(min(max(self.sp_lat.ival(), 1), nx)); self.sp_avg.set(1)
        self.temp1 = self._elev_map()

        if self.kind == "stepped":
            N = B.shape[2]
            self.spans = [np.arange(1, N + 1)]
            self.iblock = 0
            self.marks = [tabs.TabSet()]
            self.sl_pos.blockSignals(True)
            self.sl_pos.setRange(1, max(N, 2))
            self.pos = min(max(self.pos, 1), N)
            self.sl_pos.setValue(self.pos)
            self.sl_pos.blockSignals(False)
            self.sp_pos.setRange(1, max(N, 2)); self.sp_pos.set(self.pos)
            self.last_win = None
            self.busy = False
            dlg.close()
            self._clear_right("The angle map appears once two tabs are clicked")
            self._draw_frame()
            self._draw_cross()
            self._set_stage("marks")
        else:
            dlg(1.0, "Finding the sweep blocks")
            try:
                self.sw = sweep.auto_spans(self.temp1)
            except sweep.SweepError as e:
                self.sw, self.spans, self.marks = None, [], []
                self.busy = False
                dlg.close()
                self._draw_map()
                alert(self, "No blocks found", f"{e}\n\nThe recording could not be cut into sweep "
                                               "blocks, so there is nothing to click on.")
                self._refresh()
                return
            self.spans = self.sw.spans
            self.marks = [tabs.TabSet() for _ in self.spans]
            self.iblock = None
            self.last_win = None
            self.busy = False
            dlg.close()
            self.ax_cross.img.clear(); self.ax_cross.set_title("Click a block on the map above")
            for t in ("tabs", "win", "pos"):
                self.ax_cross.clear_tag(t)
            self._clear_right("The angle map appears once two tabs are clicked")
            self._draw_map()
            self._set_stage("block")
            self.status.emit(f"Period {self.sw.Tmech:.2f} frames, {len(self.spans)} blocks.")
        self._refresh()

    def _elev_map(self, iLat=None, nAvg=None):
        iLat = self.sp_lat.ival() if iLat is None else iLat
        nAvg = self.sp_avg.ival() if nAvg is None else nAvg
        nx = self.B.shape[1]
        iLat = min(max(iLat, 1), nx)
        nAvg = min(max(nAvg, 1), nx)
        if nAvg <= 1:
            return self.B[:, iLat - 1, :].astype(float)
        half = nAvg // 2
        lo = max(1, iLat - half)
        hi = min(nx, lo + nAvg - 1)
        lo = max(1, hi - nAvg + 1)
        return self.B[:, lo - 1:hi, :].astype(float).mean(axis=1)

    def _view_changed(self):
        if self.busy or self.B is None:
            return
        self.temp1 = self._elev_map()
        if self.kind == "stepped":
            self._draw_frame()
        else:
            self._draw_map()
        self._draw_cross()

    def set_lat(self, v):
        if self.B is None or not np.isfinite(v):
            return
        self.sp_lat.set(min(max(round(v), 1), self.B.shape[1]))
        self._view_changed()

    # ================================================================ display
    def _disp(self):
        lo, hi = sorted((self.sp_dlo.value(), self.sp_dhi.value()))
        return (lo, hi if hi > lo else lo + 1)

    def _display_changed(self):
        if self.temp1 is None:
            return
        lo, hi = self._disp()
        for ax in (self.ax_cross, getattr(self, "ax_frame", None), getattr(self, "ax_map", None)):
            if ax is not None and ax.img.image is not None:
                ax.set_levels(lo, hi)

    # ================================================================ stepped: capture
    def set_position(self, v, frm=""):
        if self.B is None or not np.isfinite(v):
            return
        N = self.B.shape[2]
        p = int(min(max(round(v), 1), N))
        if frm != "spinner":
            self.sp_pos.set(p)
        if frm != "slider":
            self.sl_pos.blockSignals(True); self.sl_pos.setValue(p); self.sl_pos.blockSignals(False)
        self.pos = p
        self._draw_frame()
        self._mark_position()

    def _draw_frame(self):
        if self.kind != "stepped" or self.B is None:
            return
        nz, nx, N = self.B.shape
        p = min(max(self.pos, 1), N)
        ax = self.ax_frame
        ax.show_image(self.B[:, :, p - 1].astype(float), 1, nx, 1, nz, levels=self._disp())
        ax.clear_tag("lat")
        iLat, nAvg = self.sp_lat.ival(), self.sp_avg.ival()
        if nAvg > 1:
            half = nAvg // 2
            lo = max(1, iLat - half); hi = min(nx, lo + nAvg - 1); lo = max(1, hi - nAvg + 1)
            rg = pg.LinearRegionItem((lo - 0.5, hi + 0.5), movable=False,
                                     brush=pg.mkBrush(255, 217, 51, 46), pen=pg.mkPen(None))
            ax.add("lat", rg)
        ax.line("lat", [iLat, iLat], [0.5, nz + 0.5], color=(255, 217, 51), width=1.4)
        ax.set_title(f"Position {p} of {N}")
        t = self.times[p - 1] if self.times.size >= p else np.nan
        clock = ""
        if np.isfinite(t):
            s = t % 86400
            clock = f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}:{s % 60:05.2f}"
        ax.setLabel("bottom", f"{self.files[p - 1]}   {clock}")

    def _mark_position(self):
        if self.kind != "stepped" or self.temp1 is None:
            return
        self.ax_cross.clear_tag("pos")
        self.ax_cross.line("pos", [self.pos, self.pos], [0.5, self.temp1.shape[0] + 0.5],
                           color=(255, 255, 255), width=0.8, style=QtCore.Qt.DashLine)

    # ================================================================ cine: map and blocks
    def _draw_map(self):
        if self.kind != "cine" or self.temp1 is None:
            return
        ax = self.ax_map
        for t in ("blocks",):
            ax.clear_tag(t)
        nz, nf = self.temp1.shape
        ax.show_image(self.temp1, 1, nf, 1, nz, levels=self._disp(), lut=VIRIDIS)
        for i, fr in enumerate(self.spans):
            ax.line("blocks", [fr[0] - 0.5] * 2, [0.5, nz + 0.5], color=(255, 0, 0), width=1)
            ax.text("blocks", (fr[0] + fr[-1]) / 2, 0.02 * nz, str(i + 1), anchor=(0.5, 0.0))
            m = self.marks[i].m
            if len(m):
                ax.points("blocks", m[:, 0] + fr[0] - 1, m[:, 1], color=(255, 0, 0), size=8, symbol="x")
        if self.spans:
            fr = self.spans[-1]
            ax.line("blocks", [fr[-1] + 0.5] * 2, [0.5, nz + 0.5], color=(255, 0, 0), width=1)
        if self.iblock is not None:
            fr = self.spans[self.iblock]
            ax.line("blocks", [fr[0], fr[-1], fr[-1], fr[0], fr[0]], [0.5, 0.5, nz + 0.5, nz + 0.5, 0.5],
                    color=(255, 255, 255), width=1.4, style=QtCore.Qt.DashLine)
        ax.set_title("Click the block you want")

    def _map_click(self, x, y, btn):
        if self.busy or not self.spans:
            return
        for i, fr in enumerate(self.spans):
            if fr[0] - 0.5 <= x <= fr[-1] + 0.5:
                self.pick_block(i + 1)
                return

    def pick_block(self, b: int):
        if not self.spans or not 1 <= b <= len(self.spans):
            alert(self, "No such block", f"There is no block {b}; this recording has {len(self.spans)}.")
            return
        self.iblock = b - 1
        self.cal = None
        self.last_win = None
        self._draw_map()
        self._draw_cross(fresh=True)
        self.compute_angles()
        if len(self.marks[self.iblock]) < 2:
            self._set_stage("marks")
        self._refresh()

    # ================================================================ the tabs
    def _tabset(self):
        return self.marks[self.iblock] if self.iblock is not None and self.marks else None

    def _block_map(self):
        fr = self.spans[self.iblock]
        return self.temp1[:, fr - 1]

    def _block_click(self, x, y, btn):
        if self.busy or self.iblock is None:
            return
        if self.kind == "stepped":
            self.set_position(x)
        if btn == "right":
            self.remove_nearest(x, y)
        else:
            self.add_tab(x, y)

    def add_tab(self, col, row):
        ts = self._tabset()
        if ts is None:
            return
        n = len(self.spans[self.iblock])
        nz = self.temp1.shape[0]
        if not (0.5 <= col <= n + 0.5 and 0.5 <= row <= nz + 0.5):
            return
        M = tabs.snap_map(self._block_map())
        found, k = ts.add(M, col, row, self.sp_snap.ival(), self.sp_snapz.ival())
        self.last_win = found.win
        if found.status == "placed":
            self.lb_prompt.setText(f"Tab {k} is already on that one - right click removes it. Click the tab "
                                   "you want, or narrow Search.")
            self._draw_cross()
            return
        self._after_marks()

    def remove_nearest(self, col, row):
        ts = self._tabset()
        if ts is None:
            return
        vr = self.ax_cross.getViewBox().viewRange()
        if ts.remove_nearest(col, row, vr[0][1] - vr[0][0], vr[1][1] - vr[1][0]):
            ts.resnap_open(tabs.snap_map(self._block_map()), self.sp_snap.ival(), self.sp_snapz.ival())
            self._after_marks()

    def undo(self):
        ts = self._tabset()
        if ts is not None and len(ts):
            ts.undo()
            ts.resnap_open(tabs.snap_map(self._block_map()), self.sp_snap.ival(), self.sp_snapz.ival())
            self._after_marks()

    def clear(self):
        ts = self._tabset()
        if ts is not None:
            ts.clear()
            self._after_marks()

    def _after_marks(self):
        n = len(self._tabset())
        if n < 2:
            self.cal = None
            self._clear_right(f"{n} tab(s) clicked - two are needed for an angle map")
            self._set_stage("marks")
        else:
            for w in (self.sp_zero, self.sp_neg):
                w.setRange(1, n)
            if self.sp_zero.ival() > n:
                self.sp_zero.set(min(max(round((n + 1) / 2), 1), n))
            if self.sp_neg.ival() > n:
                self.sp_neg.set(max(1, self.sp_zero.ival() - 1))
            if self.sp_neg.ival() == self.sp_zero.ival():
                self.sp_neg.set(self.sp_zero.ival() - 1 if self.sp_zero.ival() > 1 else min(2, n))
            self.compute_angles()
        if self.kind == "cine":
            self._draw_map()
        self._draw_cross()
        self._refresh()

    def _signal_rows(self, M):
        nz = M.shape[0]
        prof = M.mean(axis=1)
        yl = (0.5, nz + 0.5)
        med, hi = np.median(prof), prof.max()
        if not (np.isfinite(med) and np.isfinite(hi)) or hi <= med:
            return yl
        on = np.flatnonzero(prof >= med + 0.25 * (hi - med)) + 1
        if on.size < 2:
            return yl
        pad = max(20, 0.6 * (on[-1] - on[0]))
        yl2 = (max(0.5, on[0] - pad), min(nz + 0.5, on[-1] + pad))
        if yl2[1] - yl2[0] < 20 or yl2[1] - yl2[0] > 0.75 * nz:
            return yl
        return yl2

    def _draw_cross(self, fresh=False):
        if self.iblock is None or self.temp1 is None:
            return
        ax = self.ax_cross
        M = self._block_map()
        vb = ax.getViewBox()
        keep = not fresh and ax.img.image is not None and getattr(self, "_cross_block", None) == self.iblock
        old = vb.viewRange()
        ax.show_image(M, 1, M.shape[1], 1, M.shape[0], levels=self._disp(), lut=VIRIDIS, keep_view=True)
        if keep:
            vb.setRange(xRange=old[0], yRange=old[1], padding=0)
        else:
            vb.setRange(xRange=(0.5, M.shape[1] + 0.5), yRange=self._signal_rows(M), padding=0)
        self._cross_block = self.iblock
        for t in ("tabs", "win"):
            ax.clear_tag(t)
        if self.last_win is not None:
            w = self.last_win
            ax.line("win", [w[0] - 0.5, w[1] + 0.5, w[1] + 0.5, w[0] - 0.5, w[0] - 0.5],
                    [w[2] - 0.5, w[2] - 0.5, w[3] + 0.5, w[3] + 0.5, w[2] - 0.5],
                    color=(217, 217, 217), width=0.8, style=QtCore.Qt.DotLine)
        m = self._tabset().sorted()
        if len(m):
            solid = m[:, 4] == 0
            if solid.any():
                ax.points("tabs", m[solid, 0], m[solid, 1], color=(255, 0, 0), size=9)
            if (~solid).any():
                ax.points("tabs", m[~solid, 0], m[~solid, 1], color=(255, 0, 0), size=11, hollow=True)
            yr = vb.viewRange()[1]
            for i, (c, r) in enumerate(m[:, :2]):
                ax.text("tabs", c, r - 0.035 * (yr[1] - yr[0]), str(i + 1), size=12)
        fr = self.spans[self.iblock]
        if self.kind == "stepped":
            ax.set_title(f"Positions 1:{fr.size}, lateral line {self.sp_lat.ival()}   –   {len(m)} tab(s).  "
                         "Left click adds, right click removes.")
            self.lb_count.setText(f"{len(m)} tabs")
            self._mark_position()
        else:
            ax.set_title(f"Block {self.iblock + 1}   frames {fr[0]}:{fr[-1]}   –   {len(m)} tab(s).  "
                         "Left click adds, right click removes.")
            self.lb_count.setText(f"{len(m)} tabs, block {self.iblock + 1}")

    # ================================================================ angles
    def compute_angles(self):
        ts = self._tabset()
        if ts is None:
            return
        m = ts.sorted()
        if len(m) < 2:
            self.cal = None
            self._clear_right(f"{len(m)} tab(s) clicked - two are needed for an angle map")
            self._refresh()
            return
        try:
            a = tabs.compute_angles(m, self.sp_step.value(), self.sp_zero.ival(), self.sp_neg.ival())
        except tabs.AngleError as e:
            self.cal = None
            self._clear_right(str(e))
            self.lb_qual.setText(str(e)); self.lb_out.setText("")
            self._refresh()
            return
        fr = self.spans[self.iblock]
        a.update(FrameLine=fr.astype(float), lat_idx=self.sp_lat.ival(), latAvg=self.sp_avg.ival(),
                 dispRange=list(self._disp()), blockChoice=self.iblock + 1)
        self.cal = a
        self._draw_angles()
        self._fill_table()
        self._set_stage("ready")
        n = len(m)
        sp = np.flatnonzero(a["bySpacing"]) + 1
        note = "" if sp.size == 0 else (f" | tab {sp[0]} by spacing" if sp.size == 1 else
                                        f" | tabs {','.join(map(str, sp))} by spacing")
        if self.kind == "stepped":
            self.lb_qual.setText(f"{n} tabs, {a['theta'].min():+g} to {a['theta'].max():+g} deg over positions "
                                 f"{int(a['colFit'][0])}:{int(a['colFit'][-1])} ({a['colFit'].size} of {fr.size}) | "
                                 f"{tabs.short_dir(a['direction'])}{note}")
        else:
            self.lb_qual.setText(f"{n} tabs, {a['theta'].min():+g} to {a['theta'].max():+g} deg over "
                                 f"{a['colFit'].size} fr | {tabs.short_dir(a['direction'])}{note}")
        self.lb_out.setText("→ " + self._out_name())
        self._refresh()

    def _draw_angles(self):
        c = self.cal
        p = self.ax_angle
        p.clear()
        p.plot(c["colFit"], c["thetaFit"], pen=pg.mkPen((51, 102, 230), width=2))
        p.plot(c["cols"], c["theta"], pen=None, symbol="o", symbolBrush=(230, 77, 51), symbolPen="k", symbolSize=8)
        p.addItem(pg.InfiniteLine(0, angle=0, pen=pg.mkPen("k", style=QtCore.Qt.DotLine)))
        if np.isfinite(c["col0"]):
            p.addItem(pg.InfiniteLine(c["col0"], angle=90, pen=pg.mkPen("k", style=QtCore.Qt.DotLine)))
        if self.kind == "stepped":
            p.setXRange(0.5, c["FrameLine"].size + 0.5, padding=0)
            p.set_title(f"{c['direction']}   (positions outside {int(c['colFit'][0])}:{int(c['colFit'][-1])} "
                        "get no angle)")
        else:
            p.set_title(f"Block {self.iblock + 1}: {c['direction']}")

    def _fill_table(self):
        c = self.cal
        n = c["cols"].size
        self.table.setRowCount(n)
        for i in range(n):
            depth = (f"{c['rows'][i]:.0f}  ({c['rows'][i] * self.mm_per_sample:.2f} mm)"
                     if np.isfinite(self.mm_per_sample) else f"{c['rows'][i]:.0f}")
            vals = [str(i + 1), f"{c['cols'][i]:.2f}", depth, f"{c['theta'][i]:+.1f}",
                    "-" if i == 0 else f"{c['cols'][i] - c['cols'][i - 1]:.2f}"]
            for j, v in enumerate(vals):
                self.table.setItem(i, j, QtWidgets.QTableWidgetItem(v))

    def _clear_right(self, msg):
        self.ax_angle.clear()
        self.ax_angle.set_title(msg)
        self.table.setRowCount(0)
        self.lb_qual.setText(""); self.lb_out.setText("")

    def set_tabs(self, zero, neg):
        self.sp_zero.set(zero); self.sp_neg.set(neg)
        self.compute_angles()

    # ================================================================ saving
    def _out_name(self):
        base = self.ed_name.text().strip() or ("CF_Calibration" if self.kind == "stepped" else "TabRow_Calibration")
        d = tabs.short_dir(self.cal["direction"]) if self.cal else "+2-"
        return f"{base}_{d}.mat"

    def _session_dir(self):
        """Where a calibration belongs in the Data layout: the session folder,
        one up from the calibration captures."""
        d = self.data_dir
        return os.path.dirname(d) if d else os.path.expanduser("~")

    def save(self):
        if self.cal is None:
            alert(self, "Nothing to save", "There is no calibration to save yet.")
            return
        start = os.path.join(self._session_dir(), self._out_name())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save calibration", start, "Calibration (*.mat)")
        if not path:
            return
        self.save_to(path)

    def save_to(self, path):
        c = self.cal
        ts = self._tabset()
        fields = dict(
            cols=c["cols"], theta=c["theta"], colFit=c["colFit"], thetaFit=c["thetaFit"],
            rows=c["rows"], col0=c["col0"], direction=c["direction"], step=c["step"],
            FrameLine=c["FrameLine"], lat_idx=c["lat_idx"], latAvg=c["latAvg"], ladderIdx=c["rung"],
            blockChoice=c["blockChoice"], dispRangeUsed=np.array(c["dispRange"], float),
            zeroRung=c["zeroRung"], negRung=c["negRung"], mmPerSample=self.mm_per_sample,
            fixture="index tabs clicked by hand", tabBySpacing=c["bySpacing"].astype(float),
            searchWin=np.array([self.sp_snap.ival(), self.sp_snapz.ival()], float),
            BeamScopeImg=self.temp1[:, self.spans[self.iblock] - 1],
            climUsed=np.array(c["dispRange"], float),
            tabClicks=ts.sorted()[:, 2:4],
        )
        if self.kind == "stepped":
            n = c["FrameLine"].size
            pa = np.full(n, np.nan)
            pa[c["colFit"].astype(int) - 1] = c["thetaFit"]
            fields.update(pickedBy="FASTER Fusion app - stepped calibration (manual, quasi-static positions)",
                          posAngle=pa, positionFiles=list(self.files), positionTimes=self.times,
                          positionOrder=self.order_by, nPositions=float(n), dataFolder=self.data_dir,
                          acqMode="quasi-static")
        else:
            fields.update(pickedBy="FASTER Fusion app - cine calibration (manual)",
                          dataFolder=self.data_dir, scanFile=self.dd_scan.currentText())
        try:
            recon.save_calibration(path, fields)
        except Exception as e:
            alert(self, "Could not save", str(e))
            return False
        self.lb_out.setText("saved " + os.path.basename(path))
        self.status.emit(f"Wrote {path}")
        return True

    # ================================================================ housekeeping
    def _refresh(self):
        busy = self.busy
        have = self.B is not None
        block = self.iblock is not None
        n = len(self._tabset()) if block and self.marks else 0
        self.bt_browse.setEnabled(not busy)
        self.bt_load.setEnabled(bool(self.data_dir) and not busy)
        if self.dd_scan is not None:
            self.dd_scan.setEnabled(bool(self.files) and not busy)
        for w in (self.sp_lat, self.sp_avg, self.sp_dlo, self.sp_dhi, self.sp_snap, self.sp_snapz):
            w.setEnabled(have and not busy)
        if self.kind == "stepped":
            self.sl_pos.setEnabled(have and not busy); self.sp_pos.setEnabled(have and not busy)
        self.bt_undo.setEnabled(block and n > 0 and not busy)
        self.bt_clear.setEnabled(block and n > 0 and not busy)
        for w in (self.sp_step, self.sp_zero, self.sp_neg):
            w.setEnabled(n >= 2 and not busy)
        self.ed_name.setEnabled(self.cal is not None)
        self.bt_save.setEnabled(self.cal is not None and not busy)
