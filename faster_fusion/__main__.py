"""FASTER Fusion.

    python -m faster_fusion [mode]
    FASTER Fusion.app / FASTER Fusion.exe [mode]

mode: cal_stepped (default), cal_cine, bmode, colorflow, swe.

    ... --selftest DATA_DIR REPORT

checks an installation end to end without a person at it: opens the window,
loads the colour-flow set in DATA_DIR (a session folder holding Phantom_* and a
CAL_*.mat, or the Data folder itself), reconstructs, renders the volume, computes
the SIVV flow rate, writes a short MP4, writes what happened to REPORT and quits.
"""
import os
import sys
import traceback


def _selftest(win, app, data_dir, report):
    from PySide6 import QtCore
    lines = []

    def run():
        try:
            import glob
            sess = data_dir
            if not glob.glob(os.path.join(sess, "CAL_*CF*.mat")):
                hits = sorted(glob.glob(os.path.join(data_dir, "**", "CAL_*CF*.mat"), recursive=True))
                sess = os.path.dirname(hits[0])
            cal = sorted(glob.glob(os.path.join(sess, "CAL_*CF_-2+*.mat")) or
                         glob.glob(os.path.join(sess, "CAL_*CF*.mat")))[0]
            ph = sorted(d for d in glob.glob(os.path.join(sess, "Phantom_*"))
                        if cal.endswith("-2+.mat") == ("-2+" in os.path.basename(d)))[-1]
            win.set_mode("colorflow")
            p = win.page("colorflow")
            p.set_cal(cal)
            p.set_data(ph)
            p.load()
            p.display3d()
            at = p.sivv["at"]
            # the MP4 writer: a short turn into a temporary file
            import tempfile
            mp4 = os.path.join(tempfile.mkdtemp(), "selftest.mp4")
            ok = p.view.save_turn(mp4, n_frames=8)
            mp4_size = os.path.getsize(mp4) if ok and os.path.isfile(mp4) else 0
            # and that it really turns: frames a half turn apart must differ
            turns = False
            if mp4_size:
                import imageio.v2 as iio
                import numpy as np
                fr = [f.astype(int) for f in iio.get_reader(mp4)]
                turns = len(fr) >= 5 and np.abs(fr[4] - fr[0]).mean() > 1.0
            lines.append(f"calibration  {cal}")
            lines.append(f"set          {ph}")
            lines.append(f"positions    {p.N}")
            lines.append(f"volume       {'rendered' if p.view.active else 'NOT rendered'}")
            lines.append(f"SIVV         Q = {at.Q:.2f} mL/min at {at.z:.2f} mm")
            lines.append(f"MP4          {'written, %d bytes: %s' % (mp4_size, mp4) if mp4_size else 'NOT written'}")
            lines.append(f"MP4 turns    {'yes' if turns else 'NO - every frame is the same view'}")
            lines.append("RESULT       OK" if mp4_size and turns and p.view.active else "RESULT       FAILED")
        except Exception:
            lines.append(traceback.format_exc())
            lines.append("RESULT       FAILED")
        with open(report, "w") as f:
            f.write("\n".join(lines) + "\n")
        app.quit()

    QtCore.QTimer.singleShot(300, run)


def main():
    os.environ.setdefault("QT_API", "pyside6")
    from PySide6 import QtCore, QtGui, QtWidgets
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, True)
    args = sys.argv[1:]
    app = QtWidgets.QApplication(sys.argv[:1])
    # A size smaller than the platform default, so the whole interface fits
    # a laptop screen: 12 pt on macOS (default 13), 8.5 pt on Windows (9).
    f = app.font()
    f.setPointSizeF(max(8.0, f.pointSizeF() - (1.0 if sys.platform == "darwin" else 0.5)))
    app.setFont(f)
    app.setApplicationName("FASTER Fusion")
    app.setOrganizationName("FASTER")
    here = os.path.dirname(os.path.abspath(__file__))
    icon = os.path.join(here, "resources", "icon.png")
    if os.path.isfile(icon):
        app.setWindowIcon(QtGui.QIcon(icon))
    from .ui.main_window import MainWindow
    test = None
    if "--selftest" in args:
        i = args.index("--selftest")
        test = args[i + 1:i + 3]
        args = args[:i]
    mode = args[0] if args and not args[0].startswith("-") else "cal_stepped"
    w = MainWindow(mode)
    w.show()
    if test and len(test) == 2:
        _selftest(w, app, *test)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
