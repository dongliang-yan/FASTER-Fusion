"""Records docs/FASTER_Fusion_demo.mp4: the app driven through its main workflow
on the real data, frame by frame, with an animated cursor and captions.

    .venv/bin/python docs/make_demo.py [path/to/Data] [--part colorflow|swe|bmode]

With --part, a clip of that one mode alone goes to docs/FASTER_Fusion_<part>.mp4.

Every frame is the real window: Qt draws the widgets, VTK the 3-D view, and the
two are composited, then a cursor, click ripples and captions are painted on top.
Nothing in the Data folder is written - the calibration saved in the demo goes to
a temporary folder.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

from PySide6 import QtCore, QtGui, QtWidgets  # noqa: E402

FPS = 30
PART = "all"                                   # or one mode: colorflow, swe, bmode
OUT_W, OUT_H = 1920, 1080
WIN_W, WIN_H = 1440, 810                      # 16:9, fits a laptop screen
NAVY = QtGui.QColor(26, 36, 56)
BLUE = QtGui.QColor(41, 107, 204)
MAGENTA = QtGui.QColor(255, 64, 255)


def ease(t):
    return t * t * (3 - 2 * t)


class Recorder:
    def __init__(self, app, win, path):
        import imageio.v2 as iio
        self.app, self.win = app, win
        self.writer = iio.get_writer(path, fps=FPS, codec="libx264", quality=8,
                                     macro_block_size=8, pixelformat="yuv420p")   # 1080 is a multiple of 8
        self.base = None
        self.cursor = QtCore.QPointF(WIN_W * 0.6, WIN_H * 0.5)
        self.ripples = []                     # [point, age in frames]
        self.caption = ("", "")
        self.frames = 0

    # ---------------------------------------------------------------- capture
    def snap(self):
        """The window as it is now, the 3-D view pasted in from VTK."""
        for _ in range(3):
            self.app.processEvents()
        w = self.win
        img = w.grab().toImage().convertToFormat(QtGui.QImage.Format_RGB32)
        page = w.pages.get(w.key)
        view = getattr(page, "view", None)
        if view is not None and view.active:
            view.plotter.ren_win.Render()
            shot = np.ascontiguousarray(view.plotter.screenshot(return_img=True))
            h, wd = shot.shape[:2]
            q = QtGui.QImage(shot.data, wd, h, 3 * wd, QtGui.QImage.Format_RGB888).copy()
            inter = view.plotter.interactor
            p = QtGui.QPainter(img)
            dpr = img.devicePixelRatio()
            tl = inter.mapTo(w, QtCore.QPoint(0, 0))
            p.drawImage(QtCore.QRectF(tl.x(), tl.y(), inter.width(), inter.height()), q)
            p.end()
        self.base = img

    # ---------------------------------------------------------------- drawing
    def _cursor_path(self, pt, s=1.25):
        pts = [(0, 0), (0, 17), (4.2, 13), (7.2, 19.6), (9.6, 18.6), (6.6, 12), (12, 12)]
        return QtGui.QPolygonF([QtCore.QPointF(pt.x() + x * s, pt.y() + y * s) for x, y in pts])

    def _compose(self):
        img = self.base.copy()
        p = QtGui.QPainter(img)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        for pt, age in self.ripples:
            t = age / 14.0
            r = 8 + 26 * t
            c = QtGui.QColor(BLUE); c.setAlphaF(max(0.0, 0.55 * (1 - t)))
            p.setPen(QtGui.QPen(c, 3))
            p.setBrush(QtCore.Qt.NoBrush)
            p.drawEllipse(pt, r, r)
        p.setPen(QtGui.QPen(QtGui.QColor(0, 0, 0), 1.2))
        p.setBrush(QtGui.QColor(255, 255, 255))
        p.drawPolygon(self._cursor_path(self.cursor))
        title, sub = self.caption
        if title:
            f = QtGui.QFont(self.app.font()); f.setPointSizeF(17); f.setBold(True)
            f2 = QtGui.QFont(self.app.font()); f2.setPointSizeF(12.5)
            fm, fm2 = QtGui.QFontMetrics(f), QtGui.QFontMetrics(f2)
            wbox = max(fm.horizontalAdvance(title), fm2.horizontalAdvance(sub)) + 44
            hbox = fm.height() + (fm2.height() + 6 if sub else 0) + 26
            x0, y0 = 150, WIN_H - hbox - 70
            bg = QtGui.QColor(NAVY); bg.setAlphaF(0.92)
            p.setPen(QtCore.Qt.NoPen); p.setBrush(bg)
            p.drawRoundedRect(QtCore.QRectF(x0, y0, wbox, hbox), 12, 12)
            p.setBrush(MAGENTA)
            p.drawRoundedRect(QtCore.QRectF(x0, y0, 7, hbox), 3, 3)
            p.setPen(QtGui.QColor(255, 255, 255)); p.setFont(f)
            p.drawText(QtCore.QPointF(x0 + 24, y0 + 12 + fm.ascent()), title)
            if sub:
                p.setPen(QtGui.QColor(200, 212, 235)); p.setFont(f2)
                p.drawText(QtCore.QPointF(x0 + 24, y0 + 18 + fm.height() + fm2.ascent()), sub)
        p.end()
        return img

    def _write(self, img):
        img = img.scaled(OUT_W, OUT_H, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
        img = img.convertToFormat(QtGui.QImage.Format_RGB888)
        a = np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())
        self.writer.append_data(a[:, :img.width() * 3].reshape(img.height(), img.width(), 3).copy())
        self.frames += 1

    def emit(self, n=1, regrab=False):
        for _ in range(n):
            if regrab:
                self.snap()
            self._write(self._compose())
            self.ripples = [[pt, a + 1] for pt, a in self.ripples if a < 14]

    # ---------------------------------------------------------------- actions
    def hold(self, secs, regrab=False):
        self.emit(max(1, int(round(secs * FPS))), regrab)

    def say(self, title, sub=""):
        self.caption = (title, sub)

    def move(self, target, secs=0.6):
        a = QtCore.QPointF(self.cursor)
        b = QtCore.QPointF(target)
        n = max(2, int(secs * FPS))
        for i in range(1, n + 1):
            t = ease(i / n)
            self.cursor = a + (b - a) * t
            self.emit(1)

    def click(self, target, action=None, secs=0.55, after=0.5):
        self.move(target, secs)
        self.ripples.append([QtCore.QPointF(target), 0])
        self.emit(4)
        if action is not None:
            action()
        self.snap()
        self.hold(after)

    def close(self):
        self.writer.close()


# -------------------------------------------------------------------- helpers
def centre(win, widget):
    return QtCore.QPointF(widget.mapTo(win, widget.rect().center()))


def data_pt(win, ax, x, y):
    sp = ax.getViewBox().mapViewToScene(QtCore.QPointF(x, y))
    vp = ax.mapFromScene(sp)
    return QtCore.QPointF(ax.mapTo(win, vp))


def slider_pt(win, sl, v):
    lo, hi = sl.minimum(), sl.maximum()
    x = 10 + (sl.width() - 20) * (v - lo) / max(1, hi - lo)
    return QtCore.QPointF(sl.mapTo(win, QtCore.QPoint(int(x), sl.height() // 2)))


def card(rec, lines, secs, icon):
    """A full-frame title card."""
    img = QtGui.QImage(WIN_W * 2, WIN_H * 2, QtGui.QImage.Format_RGB32)
    img.setDevicePixelRatio(2)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    g = QtGui.QLinearGradient(0, 0, 0, WIN_H)
    g.setColorAt(0, QtGui.QColor(30, 42, 66)); g.setColorAt(1, QtGui.QColor(14, 20, 34))
    p.fillRect(QtCore.QRectF(0, 0, WIN_W, WIN_H), g)
    if icon is not None and not icon.isNull():
        p.drawImage(QtCore.QRectF(WIN_W / 2 - 80, 120, 160, 160), icon)
    y = 330
    for text, size, bold, col in lines:
        f = QtGui.QFont(QtWidgets.QApplication.font()); f.setPointSizeF(size); f.setBold(bold)
        p.setFont(f); p.setPen(col)
        fm = QtGui.QFontMetrics(f)
        p.drawText(QtCore.QPointF((WIN_W - fm.horizontalAdvance(text)) / 2, y + fm.ascent()), text)
        y += fm.height() + 14
    p.end()
    saved = rec.base, rec.caption, rec.cursor
    rec.base, rec.caption, rec.cursor = img, ("", ""), QtCore.QPointF(-100, -100)
    rec.hold(secs)
    rec.base, rec.caption, rec.cursor = saved


def tidy_paths(page, data_root):
    """Show paths from the Data folder on, not the home folder they live in."""
    for name in ("lb_cal", "lb_set", "lb_file"):
        lb = getattr(page, name, None)
        if lb is not None and hasattr(lb, "text"):
            t = lb.text()
            if data_root and t.startswith(os.path.dirname(data_root)):
                lb.setText(os.path.relpath(t, os.path.dirname(data_root)))


# -------------------------------------------------------------------- the demo
def main():
    args = sys.argv[1:]
    global PART
    PART = "all"
    if "--part" in args:
        i = args.index("--part"); PART = args[i + 1]; del args[i:i + 2]
    assert PART in ("all", "colorflow", "swe", "bmode"), PART
    data = args[0] if args else os.path.abspath(os.path.join(HERE, "..", "..", "Data"))
    out = os.path.join(HERE, "FASTER_Fusion_demo.mp4" if PART == "all" else f"FASTER_Fusion_{PART}.mp4")
    app = QtWidgets.QApplication(sys.argv[:1])
    f = app.font(); f.setPointSizeF(f.pointSizeF() - 1); app.setFont(f)
    from faster_fusion.ui.main_window import MainWindow
    from faster_fusion import __version__
    from scipy.io import loadmat
    win = MainWindow("cal_stepped" if PART == "all" else PART)
    win.resize(WIN_W, WIN_H)
    win.move(20, 30)
    win.show()
    win.statusBar().showMessage("Data folder: Data/")
    icon = QtGui.QImage(os.path.join(HERE, "..", "faster_fusion", "resources", "icon.png"))
    rec = Recorder(app, win, out)
    W = lambda: QtGui.QColor(255, 255, 255)
    S = lambda: QtGui.QColor(190, 205, 235)

    CF = os.path.join(data, "3_ColorFlow_Stepped", "2026-09-28_ColorFlow")
    SW = os.path.join(data, "2_SWE_Stepped", "2026-09-23_SWE")
    IQ = os.path.join(data, "1_BMode_Cine", "FASTER-AIR_GE9LD_Verasonics_IQ")
    REF = os.path.join(HERE, "..", "tests", "ref", "tabs.mat")

    def run():
        try:
            if PART != "all":
                # a clip of one mode: start on it, the cursor at rest
                rec.snap()
                rec.hold(0.4)

            # ---------------- title
            if PART == "all":
                card(rec, [("FASTER Fusion", 46, True, W()),
                           ("3-D ultrasound from a mirror sweep:", 20, False, S()),
                           ("calibration · B-mode · colour flow with SIVV flow rate · shear-wave elastography", 17, False, S()),
                           (f"v{__version__}", 13, False, QtGui.QColor(140, 160, 200))], 2.2, icon)
                rec.snap()
                rec.say("FASTER Fusion", "one window, a mode for each job - chosen on the rail at the left")
                rec.hold(1.0)
                rec.move(centre(win, win.bt_cal), 0.8)
                rec.hold(1.0)

            # ---------------- 1 calibration
            if PART == "all":
                c = win.page("cal_stepped")
                rec.say("1  Calibrate the sweep", "a stepped set: one DICOM per reflector position")
                rec.click(centre(win, c.bt_browse), lambda: (c.set_data(os.path.join(CF, "Calibration_-2+")),
                                                             tidy_paths(c, data)))
                rec.click(centre(win, c.bt_load), lambda: (c.load(), tidy_paths(c, data)), after=0.8)
                r = loadmat(REF, squeeze_me=True)
                c.set_lat(int(r["lat"]))
                c.sp_snap.set(int(r["searchWin"][0])); c.sp_snapz.set(int(r["searchWin"][1]))
                rec.snap()
                rec.say("Click each index tab on the position map",
                        "a click says which tab - where it is gets measured: the middle of its bright dash")
                clicks = sorted(r["clicks"].tolist(), key=lambda p: p[0])
                for x, y in clicks:
                    rec.click(data_pt(win, c.ax_cross, x, y), lambda x=x, y=y: c.add_tab(x, y), secs=0.32, after=0.18)
                rec.hold(0.8)
                rec.say("Number the 0° and −5° tabs", "the angle of every position comes out, interpolated between tabs")
                rec.click(centre(win, c.sp_zero), lambda: c.set_tabs(int(r["zeroRung"]), int(r["negRung"])), after=1.6)
                tmp = tempfile.mkdtemp()
                def save():
                    c.save_to(os.path.join(tmp, "CAL_2026-09-28_CF_-2+.mat"))
                    win.statusBar().showMessage("Wrote CAL_2026-09-28_CF_-2+.mat into the session folder")
                rec.say("Save the calibration", "next to the scans it belongs to; MATLAB reads the same file")
                rec.click(centre(win, c.bt_save), save, after=1.4)

            # ---------------- 2 colour flow
            if PART in ("all", "colorflow"):
                rec.say(("2  " if PART == "all" else "") + "Color Flow", "3-D colour flow and the volumetric flow rate")
                rec.click(centre(win, win.bt_cf), lambda: win.set_mode("colorflow"), after=0.6)
                p = win.page("colorflow")
                rec.say("Load the calibration and the colour-flow set",
                        "the velocities are read back out of the colour of every pixel")
                rec.click(centre(win, p.bt_cal), lambda: (p.set_cal(os.path.join(CF, "CAL_2026-09-28_CF_-2+.mat")),
                                                         tidy_paths(p, data)), after=0.3)
                rec.click(centre(win, p.bt_set), lambda: (p.set_data(os.path.join(CF, "Phantom_-2+_v2")),
                                                         tidy_paths(p, data)), after=0.3)
                rec.click(centre(win, p.bt_load), lambda: (p.load(), tidy_paths(p, data)), after=1.0)
                rec.say("Click the capture to pick a lateral line", "the 2-D elevational frame through the vessel")
                rec.click(data_pt(win, p.ax_frame, 202, 66), lambda: p.set_lat(202), after=1.2)
                rec.say("display3D", "the whole field: B-mode, colour flow, and the SIVV surface in magenta")
                rec.click(centre(win, p.bt_go), lambda: p.display3d(), after=1.0)
                rec.say("Turn the volume", "the vessel, and the constant-range surface the flow rate is measured on")
                sl = p.sl_rot
                rec.move(slider_pt(win, sl, 0), 0.5)
                for a in range(0, 361, 3):
                    sl.setValue(a)
                    rec.cursor = slider_pt(win, sl, a)
                    rec.snap(); rec.emit(1)
                sl.setValue(40)
                rec.cursor = slider_pt(win, sl, 40)
                rec.snap(); rec.hold(0.6)
                rec.say("SIVV: flow rate through a constant-range surface",
                        "normal to the beam everywhere, so the Doppler velocity is v·n - Q in mL/min")
                rec.move(centre(win, p.lb_q), 0.7)
                rec.hold(1.6)
                rec.say("Q at every depth through the colour box",
                        "a vessel cut whole gives the same Q at every depth - click the curve to move the surface")
                for z in (66.5, 71.5, 69.2):
                    rec.click(data_pt(win, p.ax_q, z, 20), lambda z=z: p.set_sivv(z, None), secs=0.6, after=1.1)

            # ---------------- 3 SWE
            if PART in ("all", "swe"):
                rec.say(("3  " if PART == "all" else "") + "SWE", "shear-wave elastography: velocity and quality")
                rec.click(centre(win, win.bt_swe), lambda: win.set_mode("swe"), after=0.5)
                s = win.page("swe")
                rec.click(centre(win, s.bt_cal), lambda: (s.set_cal(os.path.join(SW, "SWE_Calibration_-2+.mat")),
                                                         tidy_paths(s, data)), after=0.3)
                rec.click(centre(win, s.bt_set), lambda: (s.set_data(os.path.join(SW, "Phantom")),
                                                         tidy_paths(s, data)), after=0.3)
                rec.click(centre(win, s.bt_load), lambda: (s.load(), tidy_paths(s, data)), after=1.0)
                rec.say("Mirror depth and Style: Volume", "the mirror at 20.5 mm; composite volume rendering")
                rec.click(centre(win, s.sp_mirror), lambda: s.sp_mirror.setValue(20.5), after=0.4)
                rec.click(centre(win, s.dd_style), lambda: s.dd_style.setCurrentText("Volume"), after=0.4)
                rec.say("display3D", "the whole velocity map over the B-mode")
                rec.click(centre(win, s.bt_go), lambda: s.display3d(), after=0.6)
                sl = s.sl_rot
                rec.move(slider_pt(win, sl, 0), 0.4)
                for a in range(0, 112, 3):
                    sl.setValue(a); rec.cursor = slider_pt(win, sl, a); rec.snap(); rec.emit(1)
                sl.setValue(111); rec.snap(); rec.hold(0.8)
                rec.say("Range: paint only the inclusion's values", "2.75 to 5 m/s")
                rec.click(centre(win, s.sp_lo), lambda: s.sp_lo.setValue(2.75), after=0.4)
                rec.click(centre(win, s.sp_hi), lambda: s.sp_hi.setValue(5.0), after=1.0)
                rec.say("Largest blob", "keep the biggest connected region: the stiff ball")
                rec.click(centre(win, s.cb_largest), lambda: s.cb_largest.setChecked(True), after=1.0)
                rec.say("Opacity", "lift the low end of the window so the ball reads solid; B-mode at full opacity")
                rec.click(centre(win, s.sp_gamma), lambda: s.sp_gamma.setValue(4.0), after=0.4)
                if hasattr(s, "tabs"):
                    pass
                rec.click(centre(win, s.sp_bopa), lambda: s.sp_bopa.setValue(1.0), after=1.0)
                rec.say("Turn the volume", "the stiff ball inside the phantom")
                rec.move(slider_pt(win, sl, 111), 0.4)
                for a in range(111, 111 + 361, 3):
                    sl.setValue(a % 360); rec.cursor = slider_pt(win, sl, a % 360); rec.snap(); rec.emit(1)
                rec.hold(0.8)

            # ---------------- 4 B-mode
            if PART in ("all", "bmode"):
                rec.say(("4  " if PART == "all" else "") + "B-Mode", "a continuous cine - here a Verasonics IQ capture from FASTER-AIR")
                rec.click(centre(win, win.bt_b), lambda: win.set_mode("bmode"), after=0.5)
                b = win.page("bmode")
                rec.click(centre(win, b.bt_cal), lambda: (b.set_cal(os.path.join(IQ, "CAL_FASTER-AIR_-2+.mat")),
                                                         tidy_paths(b, data)), after=0.3)
                rec.click(centre(win, b.bt_scan), lambda: (b.set_scan(os.path.join(IQ, "Phantom_Dark_v3", "IQ_2.dat")),
                                                          tidy_paths(b, data)), after=0.3)
                rec.click(centre(win, b.bt_load), lambda: (b.load(), tidy_paths(b, data)), after=0.8)
                rec.say("Set the geometry", "an IQ capture does not record it: mirror at 20.5 mm, data starting at 40 mm")
                rec.click(centre(win, b.sp_mirror), lambda: (b.sp_mirror.setValue(20.5), tidy_paths(b, data)), after=0.5)
                rec.click(centre(win, b.sp_z0), lambda: (b.sp_z0.setValue(40.0), tidy_paths(b, data)), after=0.9)
                rec.say("Click a sweep block", "the cine is cut into single sweeps at the mirror's turnarounds")
                bx = float(b.blocks[0].mean())
                rec.click(data_pt(win, b.ax_map, bx, b.Mdet.shape[0] * 0.5), lambda: b.pick_block(1), after=1.0)
                rec.click(centre(win, b.bt_go), lambda: b.display3d(), after=0.6)
                rec.say("Turn the volume", "maximum-intensity projection of the swept B-mode")
                sl = b.sl_rot
                rec.move(slider_pt(win, sl, 0), 0.4)
                for a in range(0, 361, 3):
                    sl.setValue(a); rec.cursor = slider_pt(win, sl, a); rec.snap(); rec.emit(1)
                rec.hold(0.8)

        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            rec.close()
            print(f"wrote {out}: {rec.frames} frames, {rec.frames / FPS:.1f} s")
            app.quit()

    QtCore.QTimer.singleShot(600, run)
    app.exec()


if __name__ == "__main__":
    main()
