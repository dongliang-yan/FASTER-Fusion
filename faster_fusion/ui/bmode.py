"""B-Mode: 3-D reconstruction of a cine sweep (Recon_v1).

Pick a calibration, pick a scan - a Siemens DICOM cine or a Verasonics IQ
capture - click one sweep block on the map, and get the 2-D elevational-axial
frame and the 3-D volume. A DICOM states its pixel size and it is locked; an IQ
capture does not, so mm/sample and mm/line are editable and flagged as assumed
until set, or read from an IQGeom.mat beside the capture.
"""
from __future__ import annotations

import os

import numpy as np
from PySide6 import QtCore, QtWidgets

from ..core import io, recon, sweep
from ..core.mlcompat import imresize3, mround
from ..core.sector import SectorGrid
from .widgets import (ElideLabel, BACKGROUNDS, Busy, CurvePlot, ImagePanel, Spin, Volume3D, alert, hbox, label,
                      lut_from, next_free)

DEF_MIRROR = 23.0
DEF_LAMBDA = 1540 / 5.2083e6 * 1000
CROP = (369, 656)
VIRIDIS = lut_from("viridis")
GRAY = lut_from("gray")


class BModePage(QtWidgets.QWidget):
    status = QtCore.Signal(str)

    def __init__(self, data_root: str = ""):
        super().__init__()
        self.data_root = data_root
        self.cal = None
        self.data_dir, self.files = "", []
        self.sc, self.sc_file = None, ""
        self.B = None
        self.Mdet = None
        self.blocks = None
        self.sweep = None
        self.iz = None
        self.zmm = None
        self.d = None
        self.iblock = None
        self.vol = None
        self.vox = None
        self.loaded = ""
        self.kind = ""
        self.z0, self.dz, self.dx = 0.0, float("nan"), float("nan")
        self.lam = DEF_LAMBDA
        self.mirror = DEF_MIRROR
        self.geom_assumed, self.geom_from, self.geom_set = False, "", False
        self.busy = False
        self._build()
        self._refresh()

    def _build(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        self.gb_top = QtWidgets.QGroupBox("Setup")
        g = QtWidgets.QGridLayout(self.gb_top)
        self.lb_cal = ElideLabel("(none)"); self.lb_cal.setStyleSheet("color:#777")
        self.lb_file = ElideLabel("(none)"); self.lb_file.setStyleSheet("color:#777")
        self.bt_cal = QtWidgets.QPushButton("Browse…"); self.bt_cal.clicked.connect(self._browse_cal)
        self.bt_scan = QtWidgets.QPushButton("Browse…"); self.bt_scan.clicked.connect(self._browse_scan)
        self.bt_load = QtWidgets.QPushButton("Load scan"); self.bt_load.clicked.connect(self.load)
        self.dd_scan = QtWidgets.QComboBox(); self.dd_scan.currentTextChanged.connect(self._scan_changed)
        self.dd_scan.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon); self.dd_scan.setMinimumContentsLength(10)
        # one line, ended with an ellipsis when long: the whole text is in the
        # tooltip and the status bar. (A wrapping label here made the scroll area
        # think the page needed three times its height.)
        self.lb_status = ElideLabel("Choose a calibration .mat and a scan file.", QtCore.Qt.ElideRight)
        g.addWidget(label("Calibration"), 0, 0); g.addWidget(self.lb_cal, 0, 1); g.addWidget(self.bt_cal, 0, 2)
        g.addWidget(label("Patient scan"), 0, 3); g.addWidget(self.lb_file, 0, 4); g.addWidget(self.bt_scan, 0, 5)
        g.addWidget(self.bt_load, 0, 6)
        g.addWidget(label("Also here"), 1, 0); g.addWidget(self.dd_scan, 1, 1)
        g.addWidget(self.lb_status, 1, 2, 1, 5)
        g.setColumnStretch(1, 1); g.setColumnStretch(4, 1)
        root.addWidget(self.gb_top)

        mid = QtWidgets.QHBoxLayout()
        root.addLayout(mid, 1)
        left = QtWidgets.QGroupBox("Sweep map  →  2-D reconstruction")
        ll = QtWidgets.QVBoxLayout(left)
        self.ax_map = ImagePanel("Load a scan, then click the block you want", "Frame", "Axial sample")
        self.ax_map.clicked.connect(self._map_click)
        ll.addWidget(self.ax_map, 100)
        grid = QtWidgets.QGridLayout()
        self.sp_lat = Spin(1, 4096, 120, 5, integer=True,
                           tip="Which lateral line the 2-D preview is cut from. The volume uses all of them.")
        self.sp_lat.valueEdited.connect(self._lat_changed)
        self.sp_shift = Spin(-200, 200, 0, 1, integer=True,
                             tip="Slide the block in frames. 0 sits on the detected turnarounds.")
        self.sp_shift.valueEdited.connect(lambda v: self._settings_changed())
        self.cb_flip = QtWidgets.QCheckBox("Flip sweep"); self.cb_flip.toggled.connect(lambda v: self._settings_changed())
        self.dd_angle = QtWidgets.QComboBox(); self.dd_angle.addItems(["fraction", "index"])
        self.dd_angle.setToolTip("fraction rescales the calibration by block length; index pairs frame k with k")
        self.dd_angle.currentTextChanged.connect(lambda v: self._settings_changed())
        self.lb_block = QtWidgets.QLabel("")
        grid.addWidget(label("Lat line"), 0, 0); grid.addWidget(self.sp_lat, 0, 1)
        grid.addWidget(label("Shift"), 0, 2); grid.addWidget(self.sp_shift, 0, 3)
        grid.addWidget(label("Angle map"), 0, 4); grid.addWidget(self.dd_angle, 0, 5)
        grid.addWidget(self.cb_flip, 1, 0, 1, 2)
        grid.addWidget(self.lb_block, 1, 2, 1, 2)
        self.sp_mirror = Spin(0, 500, DEF_MIRROR, 0.5, 2,
                              tip="Depth of the mirror centre, in mm. Everything above it is thrown away.")
        self.sp_z0 = Spin(0, 500, 0, 1, 2, tip="Depth of the FIRST axial sample, in mm.")
        self.sp_dz = Spin(0, 10, 0, 0.005, 4, tip="Axial millimetres per sample.")
        self.sp_dx = Spin(0, 10, 0, 0.005, 4, tip="Lateral millimetres per line.")
        for w in (self.sp_mirror, self.sp_z0, self.sp_dz, self.sp_dx):
            w.valueEdited.connect(lambda v: self._geom_changed())
        self.lb_geom = QtWidgets.QLabel(""); self.lb_geom.setStyleSheet("color:#8c4d00")
        grid.addWidget(label("Mirror mm"), 1, 4); grid.addWidget(self.sp_mirror, 1, 5)
        grid.addWidget(label("Starts at"), 2, 0); grid.addWidget(self.sp_z0, 2, 1)
        grid.addWidget(label("mm/sample"), 2, 2); grid.addWidget(self.sp_dz, 2, 3)
        grid.addWidget(label("mm/line"), 2, 4); grid.addWidget(self.sp_dx, 2, 5)
        grid.addWidget(self.lb_geom, 2, 6)
        grid.setColumnStretch(7, 1)
        ll.addLayout(grid)
        self.ax_rec = ImagePanel("2-D elevational-axial frame", "Elevational position (mm)",
                                 "Axial depth (mm)", lock_aspect=True)
        ll.addWidget(self.ax_rec, 115)
        mid.addWidget(left, 9)       # the images need less width than the 3-D view

        arrow = QtWidgets.QVBoxLayout(); arrow.addStretch(1)
        self.bt_go = QtWidgets.QPushButton("▶ ▶\ndisplay3D")
        f = self.bt_go.font(); f.setPointSize(15); f.setBold(True); self.bt_go.setFont(f)
        self.bt_go.setMinimumSize(84, 90)
        self.bt_go.clicked.connect(self.display3d)
        arrow.addWidget(self.bt_go); arrow.addStretch(1)
        mid.addLayout(arrow)

        right = QtWidgets.QGroupBox("3-D volume")
        rl = QtWidgets.QVBoxLayout(right)
        self.view = Volume3D("Pick a block, then press display3D")
        rl.addWidget(self.view, 1)
        opa = QtWidgets.QHBoxLayout()
        lb = label("Opacity", bold=True)
        lb.setToolTip("How opaque the dark end of the range is made. 1 is no extra shaping.")
        self.sp_gamma = Spin(0.1, 20, 1, 0.5, 3)
        self.sp_gamma.valueEdited.connect(lambda v: self._apply_alpha())
        opa.addWidget(lb); opa.addWidget(self.sp_gamma)
        self.curve = CurvePlot("intensity")
        self.curve.setMaximumHeight(84)
        self.curve.setMinimumHeight(64)
        opa.addWidget(self.curve, 1)
        rl.addLayout(opa)
        self._rl = rl
        mid.addWidget(right, 13)

        bot = QtWidgets.QHBoxLayout()
        self.sp_lo = Spin(0, 255, 0, 5, integer=True, tip="Everything at or below this renders black")
        self.sp_hi = Spin(1, 256, 256, 5, integer=True, tip="Everything at or above this renders white")
        self.sp_lo.valueEdited.connect(lambda v: self._display_changed())
        self.sp_hi.valueEdited.connect(lambda v: self._display_changed())
        self.sp_knee = Spin(0, 1, 0.66, 0.05, 2, tip="How hard the dark end is rolled off before the projection.")
        self.sp_knee.valueEdited.connect(lambda v: self._apply_alpha())
        self.dd_bg = QtWidgets.QComboBox(); self.dd_bg.addItems([b[0] for b in BACKGROUNDS])
        self.dd_bg.currentTextChanged.connect(self.view.apply_background)
        self.sl_rot = QtWidgets.QSlider(QtCore.Qt.Horizontal); self.sl_rot.setRange(0, 360)
        self.sl_rot.setMinimumWidth(90)
        self.sl_rot.valueChanged.connect(self._rotate)
        self.lb_ang = QtWidgets.QLabel("0 deg"); self.lb_ang.setMinimumWidth(56)
        self.bt_reset = QtWidgets.QPushButton("Reset view"); self.bt_reset.clicked.connect(lambda: self.sl_rot.setValue(0))
        self.bt_mp4 = QtWidgets.QPushButton("Save MP4"); self.bt_mp4.clicked.connect(self.save_video)
        # turning and saving the volume sit under the volume
        self._rl.insertLayout(1, hbox(label("Rotation"), self.sl_rot, self.lb_ang, self.bt_reset, self.bt_mp4))
        for w in (label("Range", "Display window in grey levels."), self.sp_lo, self.sp_hi,
                  label("Linearity"), self.sp_knee, label("Background"), self.dd_bg, None):
            if w is None:
                bot.addStretch(1)
            else:
                bot.addWidget(w, 1 if w is self.sl_rot else 0)
        fb = QtWidgets.QFrame(); fb.setFrameShape(QtWidgets.QFrame.StyledPanel); fb.setLayout(bot)
        root.addWidget(fb)
        self._apply_alpha()

    # ---------------------------------------------------------------- selection
    def _start(self):
        return self.data_dir or self.data_root or os.path.expanduser("~")

    def _browse_cal(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick a calibration file", self._start(), "Calibration (*.mat)")
        if f:
            self.set_cal(f)

    def set_cal(self, path):
        try:
            self.cal = recon.load_calibration(path)
        except recon.CalibrationError as e:
            alert(self, "Not a usable calibration", str(e))
            return False
        self.lb_cal.setText(os.path.basename(path)); self.lb_cal.setStyleSheet("")
        self._set_status(recon.describe_calibration(self.cal))
        self._refresh()
        return True

    def _browse_scan(self):
        f, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Pick the scan to reconstruct", self._start())
        if f:
            self.set_scan(f)

    def set_scan(self, target):
        target = str(target)
        if os.path.isdir(target):
            names = io.list_scan_files(target)
            if not names:
                alert(self, "Nothing to load", f"Nothing readable in\n{target}")
                return False
            p, f = target, names[0]
        else:
            if not os.path.isfile(target):
                alert(self, "No such file", f"{target} does not exist.")
                return False
            p, f = os.path.dirname(target), os.path.basename(target)
            if not io.scan_file_kind(target)[0]:
                alert(self, "Not a scan file", f"{f} is neither a DICOM cine nor a GE IQ capture.\n\nPick "
                      "the scan itself - a Siemens DICOM, or the IQ .dat with its IQSizeInfo.mat beside it.")
                return False
            names = io.list_scan_files(p)
            if f not in names:
                names = sorted(names + [f])
        self.data_dir, self.files = p, names
        self.geom_set = False
        self.dd_scan.blockSignals(True)
        self.dd_scan.clear(); self.dd_scan.addItems(names); self.dd_scan.setCurrentText(f)
        self.dd_scan.blockSignals(False)
        self.lb_file.setText(os.path.join(p, f)); self.lb_file.setStyleSheet("")
        self._set_status(f"Scan {f} selected ({len(names)} in this folder). Press Load scan.")
        self._refresh()
        return True

    def _scan_changed(self, name):
        if name:
            self.geom_set = False
            self.lb_file.setText(os.path.join(self.data_dir, name))
            self._set_status(f"Scan {name} selected. Press Load scan.")

    # ---------------------------------------------------------------- loading
    def load(self):
        if self.busy:
            return
        if self.cal is None or not self.files:
            alert(self, "Nothing to load", "Pick a calibration and a scan first.")
            return
        fname = os.path.join(self.data_dir, self.dd_scan.currentText())
        self.busy = True
        self._refresh()
        dlg = Busy(self, "Loading", "Reading " + self.dd_scan.currentText())
        try:
            sc = self.sc if (self.sc is not None and self.sc_file == fname) else \
                io.load_scan_stack(fname, crop=CROP, progress=dlg)
            dz, dx = sc.dz_mm, sc.dx_mm
            lam = sc.geom.get("lambda_mm", DEF_LAMBDA)
            mirror = sc.geom.get("mirrorDepth_mm", DEF_MIRROR)
            if sc.kind == "geiq" and "mirrorDepth_mm" not in sc.geom:
                mirror = 76 * dz
            z0 = sc.geom.get("z0_mm", 0.0)
            if self.geom_set:
                dz, dx, mirror, z0 = self.sp_dz.value(), self.sp_dx.value(), self.sp_mirror.value(), self.sp_z0.value()
            if not np.isfinite(z0) or z0 < 0:
                z0 = 0.0
            if not (np.isfinite(dz) and dz > 0):
                raise io.ScanError(f"The axial sample size is {dz} mm, which cannot be used.")
            zmm = z0 + np.arange(sc.B.shape[0]) * dz
            idx = np.flatnonzero(zmm >= mirror)
            if idx.size == 0 or idx[0] + 1 >= zmm.size - 1:
                raise io.ScanError(f"The data covers {zmm[0]:.1f} to {zmm[-1]:.1f} mm but the mirror is set to "
                                   f"{mirror:.1f} mm, so there is nothing below it to reconstruct.")
            iz = int(idx[0]) + 1
            dlg(0.95, "Finding the sweep blocks")
            Mdet = sc.B.astype(float).mean(axis=1)
            blk = sweep.find_sweep_blocks(Mdet, iz)
        except Exception as e:
            dlg.close()
            self.busy = False
            alert(self, "Could not load this scan", f"{e}" + (f"\n\nStill showing {self.loaded}." if self.loaded else ""))
            self._refresh()
            return
        dlg.close()
        self.sc, self.sc_file = sc, fname
        self.B, self.zmm, self.iz = sc.B, zmm, iz
        self.d = zmm[iz - 1:] - mirror
        self.Mdet, self.blocks, self.sweep = Mdet, blk.blocks, blk
        self.iblock, self.vol = None, None
        self.loaded = self.dd_scan.currentText()
        self.kind, self.dz, self.dx, self.lam, self.z0, self.mirror = sc.kind, dz, dx, lam, z0, mirror
        self.geom_assumed = sc.assumed and not self.geom_set
        self.geom_from = "the values set in the GUI" if self.geom_set else sc.geom_from
        self.sp_mirror.set(mirror); self.sp_z0.set(z0); self.sp_dz.set(dz); self.sp_dx.set(max(0, dx))
        locked = sc.kind == "dicom"
        self.sp_dz.setReadOnly(locked); self.sp_dx.setReadOnly(locked)
        self.lb_geom.setText("assumed" if self.geom_assumed else "")
        self.lb_geom.setToolTip(f"mm/sample, mm/line and the mirror depth are {self.geom_from}." if self.geom_assumed else "")
        nz, nx, nf = sc.B.shape
        self.gb_top.setTitle(f"Setup   –   {'Verasonics IQ' if sc.kind == 'geiq' else 'Siemens DICOM'}   {nz} x {nx} x {nf}")
        self.sp_lat.setRange(1, nx); self.sp_lat.set(min(max(self.sp_lat.ival(), 1), nx))
        m = max(1, nf // 4); self.sp_shift.setRange(-m, m); self.sp_shift.set(0)
        self.lb_block.setText("")
        self.view.clear("Pick a block, then press display3D")
        self.view.prepare()
        self.ax_rec.img.clear(); self.ax_rec.set_title("2-D elevational-axial frame")
        self._draw_map()
        msg = (f"{self.loaded}: {nf} frames, period {blk.T}, {self.blocks.shape[0]} complete sweep block(s). "
               f"Data {zmm[0]:.1f}-{zmm[-1]:.1f} mm, mirror {mirror:.2f} mm = sample {iz}, so {zmm.size - iz + 1} "
               f"of {zmm.size} samples reconstruct from {self.d[0]:.1f} mm below the mirror. Click a block on the map.")
        if self.geom_assumed:
            msg += f"  GEOMETRY ASSUMED ({self.geom_from}) - check mm/sample and Mirror mm."
        self.busy = False
        self._set_status(msg)
        self._refresh()

    def _geom_changed(self):
        if self.busy or self.B is None:
            return
        self.geom_set = True
        self.load()

    # ---------------------------------------------------------------- map and blocks
    def _draw_map(self, used=None):
        ax = self.ax_map
        ax.clear_tag("blocks")
        nz, nf = self.Mdet.shape
        ax.show_image(self.Mdet, 1, nf, 1, nz, lut=VIRIDIS)
        for b in range(self.blocks.shape[0]):
            for x in self.blocks[b]:
                ax.line("blocks", [x, x], [0.5, nz + 0.5], color=(255, 0, 0), width=1.5)
            ax.text("blocks", self.blocks[b].mean(), 0.02 * nz, f"{b + 1} ({self.sweep.cls[b]})", anchor=(0.5, 0.0))
        if used is not None:
            import pyqtgraph as pg
            ax.add("blocks", pg.LinearRegionItem((used[0], used[1]), movable=False,
                                                 brush=pg.mkBrush(255, 255, 255, 50),
                                                 pen=pg.mkPen((255, 255, 255), style=QtCore.Qt.DashLine)))
        ax.set_title("Click the block you want")

    def _map_click(self, x, y, btn):
        if self.busy or self.blocks is None:
            return
        self.pick_block(int(np.argmin(np.abs(self.blocks.mean(axis=1) - x))) + 1)

    def pick_block(self, b):
        if self.blocks is None or not 1 <= b <= self.blocks.shape[0]:
            alert(self, "No such block", f"There is no block {b}.")
            return
        self.iblock = b
        ok, used = self._draw2d(b)
        if not ok:
            return
        self._draw_map(used)
        self.vol = None
        self.view.clear("Press display3D to render this block")
        self._refresh()

    def _recon_src(self, b):
        s = self.blocks[b - 1, 0] + self.sp_shift.ival()
        e = self.blocks[b - 1, 1] + self.sp_shift.ival()
        nf = self.B.shape[2]
        if s < 1 or e > nf:
            raise ValueError(f"Shift {self.sp_shift.ival():+d} pushes block {b} to {s}:{e}, outside 1:{nf}.")
        FL = np.arange(s, e + 1)
        if FL.size < 2:
            raise ValueError(f"Block {b} is {FL.size} frame(s) long; at least 2 are needed.")
        ang, valid = recon.cine_angles(self.cal, FL.size, self.dd_angle.currentText(), self.cb_flip.isChecked())
        if valid.sum() < 2:
            raise ValueError(f"Only {valid.sum()} frame(s) of this block fall inside the calibration angle "
                             "range. Try a different shift, or the other angle map.")
        frames = FL[valid]
        used = (frames[0], frames[-1])
        return FL, ang, frames, used

    def _draw2d(self, b, iLat=None):
        iLat = self.sp_lat.ival() if iLat is None else iLat
        iLat = min(max(iLat, 1), self.B.shape[1])
        try:
            FL, ang, frames, used = self._recon_src(b)
        except ValueError as e:
            alert(self, "Cannot reconstruct this block", str(e))
            return False, None
        g = SectorGrid(ang, self.d, self.lam)
        img = g.apply(self.B[self.iz - 1:, iLat - 1, :][:, frames - 1].astype(float))
        rng = self._range()
        self.ax_rec.show_image(img, g.yy[0], g.yy[-1], g.zz[0] + self.mirror, g.zz[-1] + self.mirror,
                               levels=rng, lut=GRAY)
        self.ax_rec.set_title(f"Block {b}   frames {FL[0]}:{FL[-1]}   lateral line {iLat}")
        self.lb_block.setText(f"block {b} ({self.sweep.cls[b - 1]})")
        self._set_status(f"Block {b} -> frames {FL[0]}:{FL[-1]}, {ang.size} of them angled ({ang[0]:+.1f} to "
                         f"{ang[-1]:+.1f} deg, absolute {used[0]}:{used[1]}), lateral line {iLat}.")
        return True, used

    def _settings_changed(self):
        if self.busy or self.iblock is None:
            return
        self.pick_block(self.iblock)

    def _lat_changed(self, v):
        if self.busy or self.iblock is None or self.B is None:
            return
        v = int(round(v))
        if 1 <= v <= self.B.shape[1]:
            self._draw2d(self.iblock, v)

    # ---------------------------------------------------------------- volume
    def display3d(self):
        if self.busy:
            return
        if self.iblock is None:
            alert(self, "Nothing to render", "Click a block on the map first.")
            return
        self.busy = True
        self._refresh()
        dlg = Busy(self, "display3D", "Reconstructing", cancelable=True)
        try:
            FL, ang, frames, used = self._recon_src(self.iblock)
            # The 2-D frame is swept on the lambda/4 grid, as in MATLAB. The
            # volume is swept on the data's own sample pitch instead: MATLAB went
            # to lambda/4 and then shrank the result to cubic voxels, which is
            # finer than the scanner sampled and a hundred million voxels to
            # resample. VTK draws unequal voxel sizes as they are, so nothing
            # needs resampling after.
            pitch = max(self.lam / 4, min(self.dz, self.dx))
            g = SectorGrid(ang, self.d, 4 * pitch)
            nx = self.B.shape[1]
            src_all = self.B[self.iz - 1:, :, :][:, :, frames - 1]
            vol = np.zeros((g.zz.size, nx, g.yy.size), np.float32)
            chunk = 16
            for s in range(0, nx, chunk):
                if dlg.cancelled:
                    self._set_status("display3D cancelled.")
                    return
                sel = np.arange(s, min(nx, s + chunk))
                src = src_all[:, sel, :].astype(float).transpose(0, 2, 1)
                vol[:, sel, :] = g.apply(src).transpose(0, 2, 1)
                dlg(0.75 * (sel[-1] + 1) / nx, f"Reconstructing lateral line {sel[-1] + 1} of {nx}")
            vox = [float(np.mean(np.diff(g.zz))), self.dx, float(np.mean(np.diff(g.yy)))]
            if not all(np.isfinite(vox)) or min(vox) <= 0:
                raise ValueError(f"Voxel size came out as {vox} mm. mm/line under the map sets the middle one.")
            self.vol, self.vox = vol, vox
            self.zz0 = g.zz[0] + self.mirror
            self.yy0 = g.yy[0]
            dlg(0.8, "Rendering")
            self.sl_rot.blockSignals(True); self.sl_rot.setValue(0); self.sl_rot.blockSignals(False)
            self.lb_ang.setText("0 deg")
            self.view.angle = 0
            self._show_volume(fresh=True)
        except Exception as e:
            self.vol = None
            self.view.clear("Rendering failed - press display3D to try again")
            alert(self, "display3D failed", str(e))
            return
        finally:
            dlg.close()
            self.busy = False
            self._refresh()
        self._set_status(f"Volume {list(self.vol.shape)}, voxel {self.vox[0]:.3f} x {self.vox[1]:.3f} x "
                         f"{self.vox[2]:.3f} mm. Drag it, or use the slider.")

    def _show_volume(self, fresh=False):
        rng = self._range()
        V = np.nan_to_num(self.vol.astype(np.float32))
        V = np.clip((V - rng[0]) / (rng[1] - rng[0]), 0, 1)
        V = (255 * V / max(V.max(), np.finfo(float).eps)).astype(np.uint8)
        sz = np.array(V.shape)                             # drawn on its own grid
        U = V
        n0 = np.array(V.shape, float)
        sp = n0 * np.array(self.vox) / sz
        origin = np.array([self.zz0, 0.0, self.yy0]) - np.array(self.vox) / 2 + sp / 2
        a = recon.alpha_curve(self.sp_knee.value(), self.sp_gamma.value())
        self.view.set_volume("b", U, sp, np.repeat(np.linspace(0, 1, 256)[:, None], 3, 1), a,
                             style="mip", origin=origin)
        if fresh:
            ext = origin + (np.array(U.shape) - 1) * sp
            self.view.frame([origin[1], ext[1], origin[2], ext[2], origin[0], ext[0]])

    def _range(self):
        if self.sp_hi.value() <= self.sp_lo.value():
            self.sp_hi.set(min(self.sp_lo.value() + 1, 256))
            self.sp_lo.set(self.sp_hi.value() - 1)
        return (self.sp_lo.value(), self.sp_hi.value())

    def _display_changed(self):
        rng = self._range()
        if self.ax_rec.img.image is not None:
            self.ax_rec.set_levels(*rng)
        if self.vol is not None and self.view.active and not self.busy:
            self._show_volume()

    def _apply_alpha(self):
        a = recon.alpha_curve(self.sp_knee.value(), self.sp_gamma.value())
        self.curve.set_curve(a)
        if self.view.active:
            self.view.set_opacity("b", a)

    def _rotate(self, a):
        self.lb_ang.setText(f"{a} deg")
        if self.view.active:
            self.view.set_angle(float(a))

    def save_video(self):
        if not self.view.active or self.busy:
            return
        start = next_free(os.path.dirname(self.data_dir) if self.data_dir else os.path.expanduser("~"), "volume", ".mp4")
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

    def _set_status(self, msg):
        self.lb_status.setText(msg)
        self.status.emit(msg)

    def _refresh(self):
        busy = self.busy
        have = self.B is not None
        self.bt_cal.setEnabled(not busy); self.bt_scan.setEnabled(not busy)
        self.bt_load.setEnabled(bool(self.files) and not busy)
        self.dd_scan.setEnabled(bool(self.files) and not busy)
        for w in (self.sp_lat, self.sp_shift, self.cb_flip, self.dd_angle, self.sp_mirror, self.sp_z0):
            w.setEnabled(have and not busy)
        self.sp_dz.setEnabled(have and not busy); self.sp_dx.setEnabled(have and not busy)
        self.bt_go.setEnabled(self.iblock is not None and not busy)
        on3d = self.view.active
        for w in (self.sl_rot, self.bt_reset, self.bt_mp4):
            w.setEnabled(on3d and not busy)
