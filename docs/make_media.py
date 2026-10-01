"""Regenerates the screenshots and GIFs in docs/img from the real app and the Data folder.

    .venv/Scripts/python docs/make_media.py ../Data

Needs a desktop session (it grabs the real window, OpenGL included).
"""
import glob
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image
from PySide6 import QtCore, QtGui, QtWidgets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "img")
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "Data")


def pump(app, s=0.4):
    t = time.time()
    while time.time() - t < s:
        app.processEvents()
        time.sleep(0.02)


def grab(app, win):
    pump(app, 0.3)
    scr = win.screen()
    return scr.grabWindow(win.winId()).toImage()


def to_pil(qimg):
    qimg = qimg.convertToFormat(QtGui.QImage.Format_RGB888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.constBits()
    arr = np.frombuffer(ptr, np.uint8, h * qimg.bytesPerLine()).reshape(h, qimg.bytesPerLine())[:, :w * 3]
    return Image.fromarray(arr.reshape(h, w, 3).copy())


def save_png(img, name):
    img.save(os.path.join(OUT, name), optimize=True)
    print("wrote", name)


def save_gif(frames, name, width, fps=12):
    fr = []
    for f in frames:
        h = round(f.height * width / f.width)
        fr.append(f.resize((width, h), Image.LANCZOS).quantize(128, dither=Image.Dither.NONE))
    fr[0].save(os.path.join(OUT, name), save_all=True, append_images=fr[1:],
               duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    print("wrote", name, round(os.path.getsize(os.path.join(OUT, name)) / 1e6, 2), "MB")


def turn_frames(app, view, n=48):
    a0 = view.angle
    out = []
    for i in range(n):
        view.set_angle(a0 + 360.0 * i / n)
        view.plotter.ren_win.Render()
        out.append(Image.fromarray(view.plotter.screenshot(return_img=True)))
        app.processEvents()
    view.set_angle(a0)
    return out


def step(name, fn):
    try:
        fn()
    except Exception:
        print("FAILED", name)
        traceback.print_exc()


def run(win, app):
    win.resize(1500, 930)
    win.move(20, 20)
    pump(app, 1.0)

    swe = os.path.join(DATA, "2_SWE_Stepped", "2026-09-23_SWE")
    cf = os.path.join(DATA, "3_ColorFlow_Stepped", "2026-09-28_ColorFlow")

    def cal_stepped():
        win.set_mode("cal_stepped")
        p = win.page("cal_stepped")
        p.set_data(os.path.join(swe, "Calibration"))
        p.load()
        save_png(to_pil(grab(app, win)), "calibration.png")
    step("calibration", cal_stepped)

    def swe_page():
        win.set_mode("swe")
        p = win.page("swe")
        p.set_cal(os.path.join(swe, "SWE_Calibration_-2+.mat"))
        p.set_data(os.path.join(swe, "Phantom"))
        p.load()
        n = p.N
        # scanning through the captured positions
        frames = []
        for k in list(range(max(1, n // 5), n - n // 5, max(1, n // 24))):
            p.set_position(k)
            frames.append(to_pil(grab(app, win)))
        save_gif(frames, "swe_positions.gif", 900, fps=8)
        p.set_position(int(n * 0.76))
        p.display3d()
        pump(app, 1.0)
        save_png(to_pil(grab(app, win)), "swe.png")
        save_gif(turn_frames(app, p.view), "swe_3d_turn.gif", 520, fps=15)
    step("swe", swe_page)

    def cf_page():
        win.set_mode("colorflow")
        p = win.page("colorflow")
        p.set_cal(os.path.join(cf, "CAL_2026-09-28_CF_-2+.mat"))
        ph = sorted(glob.glob(os.path.join(cf, "Phantom_-2+*")))[-1]
        p.set_data(ph)
        p.load()
        p.display3d()
        pump(app, 1.0)
        save_png(to_pil(grab(app, win)), "colorflow.png")
        save_gif(turn_frames(app, p.view), "colorflow_3d_turn.gif", 520, fps=15)
    step("colorflow", cf_page)

    def bmode_page():
        bm = os.path.join(DATA, "1_BMode_Cine", "2026-09-22_Siemens_10Vpp")
        win.set_mode("bmode")
        p = win.page("bmode")
        p.set_cal(sorted(glob.glob(os.path.join(bm, "CAL_*-2+.mat")))[0])
        p.set_scan(sorted(glob.glob(os.path.join(bm, "Phantom", "*")))[0])
        p.load()
        p.pick_block(2)
        p.display3d()
        pump(app, 1.0)
        save_png(to_pil(grab(app, win)), "bmode.png")
        save_gif(turn_frames(app, p.view), "bmode_3d_turn.gif", 520, fps=15)
    step("bmode", bmode_page)
    app.quit()


def main():
    os.environ.setdefault("QT_API", "pyside6")
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    app = QtWidgets.QApplication(sys.argv[:1])
    f = app.font(); f.setPointSizeF(max(8.0, f.pointSizeF() - 0.5)); app.setFont(f)
    sys.path.insert(0, os.path.join(HERE, ".."))
    from faster_fusion.ui.main_window import MainWindow
    win = MainWindow("swe")
    win.show()
    QtCore.QTimer.singleShot(500, lambda: run(win, app))
    app.exec()


if __name__ == "__main__":
    main()
