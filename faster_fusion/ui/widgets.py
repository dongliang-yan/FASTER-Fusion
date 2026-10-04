"""Widgets every page uses: image panels, labelled spinners, progress, the 3-D view."""
from __future__ import annotations

import os
from typing import Callable, Optional

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets

pg.setConfigOptions(imageAxisOrder="row-major", antialias=True, background="w", foreground="k")

NAVY = "#1a2438"
BLUE = "#296bcc"
MAGENTA = (255, 77, 255)


# --------------------------------------------------------------------------
# small controls
# --------------------------------------------------------------------------

def label(text: str, tip: str = "", align=QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
          bold: bool = False) -> QtWidgets.QLabel:
    w = QtWidgets.QLabel(text)
    w.setAlignment(align)
    if tip:
        w.setToolTip(tip)
    if bold:
        f = w.font(); f.setBold(True); w.setFont(f)
    return w


class ElideLabel(QtWidgets.QLabel):
    """A label for file paths: it takes whatever width it is given and shortens
    its text with an ellipsis in the MIDDLE, where a path is least telling,
    instead of pushing the window wider. The whole text is in the tooltip."""

    def __init__(self, text="", mode=QtCore.Qt.ElideMiddle):
        super().__init__()
        self._full = ""
        self._mode = mode
        self.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Preferred)
        self.setMinimumWidth(40)
        self.setText(text)

    def setText(self, text):
        self._full = str(text)
        self.setToolTip(self._full)
        self._elide()

    def text(self):
        return self._full

    def _elide(self):
        fm = self.fontMetrics()
        super().setText(fm.elidedText(self._full, self._mode, max(20, self.width() - 4)))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._elide()


class Spin(QtWidgets.QDoubleSpinBox):
    """A spinner that says when the user changes it (valueEdited) and can be set
    silently from code (set)."""
    valueEdited = QtCore.Signal(float)

    def __init__(self, lo, hi, value, step=1.0, decimals=2, suffix="", tip="", integer=False,
                 prefix=""):
        super().__init__()
        self.setRange(lo, hi)
        self.setDecimals(0 if integer else decimals)
        self.setSingleStep(step)
        self.setValue(value)
        self.setKeyboardTracking(False)
        if suffix:
            self.setSuffix(suffix)
        if prefix:
            self.setPrefix(prefix)
        if tip:
            self.setToolTip(tip)
        self.setAccelerated(True)
        self.valueChanged.connect(self._changed)
        self._silent = False
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)

    def _text_width(self) -> int:
        """Room for the numbers this box really shows, not for its range: a
        spinner allowed +/-1e6 with three decimals would otherwise reserve
        eleven digits for a value like 8.000."""
        fm = self.fontMetrics()
        vals = [self.value(), max(min(self.maximum(), 999), -99), max(self.minimum(), -99)]
        w = max(fm.horizontalAdvance(self.prefix() + self.textFromValue(v) + self.suffix()) for v in vals)
        return w + 2 * fm.horizontalAdvance("0")

    def _chrome_width(self) -> int:
        """Frame, text margins and the up/down buttons as this platform's style
        draws them (side by side on Windows 11, so wider than on macOS)."""
        opt = QtWidgets.QStyleOptionSpinBox()
        self.initStyleOption(opt)
        opt.rect = QtCore.QRect(0, 0, 400, 30)
        st = self.style()
        edit = st.subControlRect(QtWidgets.QStyle.CC_SpinBox, opt,
                                 QtWidgets.QStyle.SC_SpinBoxEditField, self)
        return max(30, 400 - edit.width()) + 10

    def sizeHint(self):
        h = super().sizeHint()
        return QtCore.QSize(self._text_width() + self._chrome_width(), h.height())

    def minimumSizeHint(self):
        h = super().minimumSizeHint()
        return QtCore.QSize(self._text_width() + self._chrome_width(), h.height())

    def _changed(self, v):
        if not self._silent:
            self.valueEdited.emit(v)

    def set(self, v):
        self._silent = True
        try:
            self.setValue(v)
        finally:
            self._silent = False

    def ival(self) -> int:
        return int(round(self.value()))


def hbox(*items, spacing=6, margins=(0, 0, 0, 0)) -> QtWidgets.QHBoxLayout:
    lay = QtWidgets.QHBoxLayout()
    lay.setSpacing(spacing)
    lay.setContentsMargins(*margins)
    for it in items:
        if it is None:
            lay.addStretch(1)
        elif isinstance(it, int):
            lay.addSpacing(it)
        elif isinstance(it, QtWidgets.QLayout):
            lay.addLayout(it)
        else:
            lay.addWidget(it)
    return lay


def group(title: str, layout: QtWidgets.QLayout) -> QtWidgets.QGroupBox:
    g = QtWidgets.QGroupBox(title)
    g.setLayout(layout)
    return g


def alert(parent, title: str, msg: str):
    QtWidgets.QMessageBox.warning(parent, title, msg)


class Busy:
    """A modal progress dialog for a long operation run on the GUI thread, as
    the MATLAB GUIs do. It appears only if the operation is still running after
    a moment, and repaints at most five times a second: every repaint redraws
    the whole window - the 3-D view included - and doing that on each progress
    step made the operation itself slower."""

    def __init__(self, parent, title, msg="", cancelable=False):
        import time
        self._time = time.monotonic
        self.dlg = QtWidgets.QProgressDialog(msg, "Cancel" if cancelable else None, 0, 1000, parent)
        self.dlg.setWindowTitle(title)
        self.dlg.setWindowModality(QtCore.Qt.WindowModal)
        self.dlg.setMinimumDuration(400)
        self.dlg.setAutoClose(False)
        self.dlg.setAutoReset(False)
        self.dlg.setValue(0)
        self._last = 0.0
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        self._cursor = True

    def __call__(self, v: float, msg: str = ""):
        now = self._time()
        if now - self._last < 0.2 and v < 1:
            return
        self._last = now
        if msg:
            self.dlg.setLabelText(msg)
        self.dlg.setValue(int(1000 * min(max(v, 0), 1)))
        QtWidgets.QApplication.processEvents(QtCore.QEventLoop.ExcludeUserInputEvents)

    @property
    def cancelled(self) -> bool:
        return self.dlg.wasCanceled()

    def close(self):
        self.dlg.close()
        if self._cursor:
            QtWidgets.QApplication.restoreOverrideCursor()
            self._cursor = False

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()


# --------------------------------------------------------------------------
# 2-D image panel
# --------------------------------------------------------------------------

class ImagePanel(pg.PlotWidget):
    """An image in data coordinates with overlays by tag and click reporting.

    clicked(x, y, button) is emitted in data units; button is 'left' or 'right'.
    """
    clicked = QtCore.Signal(float, float, str)

    def __init__(self, title="", xlabel="", ylabel="", invert_y=True, lock_aspect=False,
                 interactive=False):
        super().__init__()
        self._title, self._title_color = "", None
        self.set_title(title)
        self.setLabel("bottom", xlabel)
        self.setLabel("left", ylabel)
        self.invertY(invert_y)
        self.setAspectLocked(lock_aspect)
        self.setMenuEnabled(False)
        vb = self.getViewBox()
        vb.setMouseEnabled(interactive, interactive)
        self.img = pg.ImageItem()
        self.addItem(self.img)
        self._tags: dict[str, list] = {}
        self._cbar = None
        self.scene().sigMouseClicked.connect(self._on_click)

    def set_title(self, t: str, color: Optional[str] = None):
        """The title, shortened with an ellipsis to the width the plot has.
        pyqtgraph makes a plot at least as wide as its title, so a long title in
        a narrow panel pushed the plot past the panel's edge and cut it off. The
        whole title is in the tooltip."""
        self._title, self._title_color = t, color
        self.setToolTip(t.replace("<br>", "\n"))
        self._show_title()

    def _show_title(self):
        f = QtGui.QFont(self.font())
        f.setPointSizeF(10)
        fm = QtGui.QFontMetrics(f)
        # the title sits over the plot area only, right of the left axis
        axis = self.plotItem.getAxis("left").width() if self.plotItem.getAxis("left").isVisible() else 0
        avail = max(40, self.width() - axis - 24)
        lines = [fm.elidedText(l, QtCore.Qt.ElideRight, avail) for l in self._title.split("<br>")]
        t = "<br>".join(lines)
        if self._title_color:
            t = f"<span style='color:{self._title_color}'>{t}</span>"
        self.plotItem.setTitle(t, size="10pt")
        self.plotItem.titleLabel.setMinimumWidth(0)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.__dict__.get("_title"):        # resized during construction, before a title
            self._show_title()

    def show_image(self, data: np.ndarray, x0: float, x1: float, y0: float, y1: float,
                   levels=None, lut=None, keep_view: bool = False):
        """data is rows x cols (x cols, y rows) or rows x cols x 3 in 0..1/uint8.
        x0/x1 are the CENTRES of the first/last column, likewise y."""
        ny, nx = data.shape[:2]
        dx = (x1 - x0) / max(nx - 1, 1) if nx > 1 else 1.0
        dy = (y1 - y0) / max(ny - 1, 1) if ny > 1 else 1.0
        rect = QtCore.QRectF(x0 - dx / 2, y0 - dy / 2, dx * nx, dy * ny)
        if data.ndim == 3:
            d = data
            if d.dtype != np.uint8:
                d = np.clip(np.nan_to_num(d) * 255, 0, 255).astype(np.uint8)
            self.img.setLookupTable(None)
            self.img.setImage(d, levels=(0, 255), autoLevels=False)
        else:
            self.img.setLookupTable(lut)
            self.img.setImage(np.nan_to_num(data.astype(np.float32)),
                              levels=levels if levels is not None else None,
                              autoLevels=levels is None)
        self.img.setRect(rect)
        if not keep_view:
            self.getViewBox().setRange(rect, padding=0)
        return rect

    def set_levels(self, lo, hi):
        self.img.setLevels((lo, hi))

    def clear_tag(self, tag: str):
        for it in self._tags.pop(tag, []):
            self.removeItem(it)

    def add(self, tag: str, item):
        self.addItem(item)
        self._tags.setdefault(tag, []).append(item)
        return item

    def line(self, tag, xs, ys, color=(255, 255, 255), width=1.2, style=QtCore.Qt.SolidLine):
        pen = pg.mkPen(color=color, width=width, style=style)
        return self.add(tag, pg.PlotDataItem(np.asarray(xs, float), np.asarray(ys, float), pen=pen))

    def points(self, tag, xs, ys, color=(255, 0, 0), size=8, symbol="o", hollow=False,
               edge=(255, 255, 255)):
        brush = None if hollow else pg.mkBrush(color)
        pen = pg.mkPen(color if hollow else edge, width=1.5 if hollow else 0.8)
        return self.add(tag, pg.ScatterPlotItem(np.asarray(xs, float), np.asarray(ys, float),
                                                size=size, symbol=symbol, brush=brush, pen=pen))

    def text(self, tag, x, y, s, color=(255, 255, 255), bold=True, size=11, anchor=(0.5, 1.0)):
        t = pg.TextItem(s, color=color, anchor=anchor)
        f = QtGui.QFont(); f.setBold(bold); f.setPointSize(size)
        t.setFont(f)
        t.setPos(x, y)
        return self.add(tag, t)

    def colorbar(self, lut, lo, hi, title=""):
        """One colour bar beside the plot, created once and updated after."""
        cmap = pg.ColorMap(np.linspace(0, 1, lut.shape[0]), lut)
        if self._cbar is None:
            self._cbar = pg.ColorBarItem(values=(lo, hi), colorMap=cmap, interactive=False, width=14)
            self._cbar.axis.setLabel(title)
            self.plotItem.layout.addItem(self._cbar, 2, 5)
        else:
            self._cbar.setColorMap(cmap)
            self._cbar.setLevels((lo, hi))
            self._cbar.axis.setLabel(title)

    def _on_click(self, ev):
        vb = self.getViewBox()
        if not vb.sceneBoundingRect().contains(ev.scenePos()):
            return
        p = vb.mapSceneToView(ev.scenePos())
        btn = "right" if ev.button() == QtCore.Qt.RightButton else "left"
        if ev.button() == QtCore.Qt.LeftButton and ev.modifiers() & QtCore.Qt.ControlModifier:
            btn = "right"                       # ctrl-click on a one-button trackpad
        ev.accept()
        self.clicked.emit(float(p.x()), float(p.y()), btn)


def lut_from(name: str) -> np.ndarray:
    """A 256 x 3 uint8 lookup table from a pyqtgraph/matplotlib colormap name."""
    if name == "gray":
        g = np.arange(256, dtype=np.uint8)
        return np.stack([g, g, g], 1)
    cm = pg.colormap.get(name)
    return cm.getLookupTable(0, 1, 256)[:, :3].astype(np.uint8)


def float_lut(cm01: np.ndarray) -> np.ndarray:
    return np.clip(np.round(cm01 * 255), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# opacity curve
# --------------------------------------------------------------------------

class CurvePlot(pg.PlotWidget):
    def __init__(self, xlabel="value", ylabel="opacity"):
        super().__init__()
        self.setMenuEnabled(False)
        self.getViewBox().setMouseEnabled(False, False)
        self.setXRange(0, 1, padding=0.02)
        self.setYRange(0, 1, padding=0.05)
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setLabel("bottom", xlabel)
        self.setLabel("left", ylabel)
        self.plot([0, 1], [0, 1], pen=pg.mkPen((150, 150, 150), style=QtCore.Qt.DotLine))
        self.curve = self.plot(np.linspace(0, 1, 256), np.linspace(0, 1, 256),
                               pen=pg.mkPen((0, 114, 189), width=2.5))
        self.setMinimumHeight(90)

    def set_curve(self, a):
        self.curve.setData(np.linspace(0, 1, a.size), a)


# --------------------------------------------------------------------------
# 3-D view
# --------------------------------------------------------------------------

BACKGROUNDS = [
    ("Black", (0, 0, 0), None),
    ("Deep blue", (0.04, 0.09, 0.26), None),
    ("White", (1, 1, 1), None),
    ("Grey to black", (0.72, 0.72, 0.75), (0, 0, 0)),
    ("White to black", (1, 1, 1), (0, 0, 0)),
    ("Blue to black", (0.10, 0.35, 0.75), (0, 0, 0)),
    ("Scanner blue", (0, 0.561, 1), (0, 0.329, 0.529)),
    ("Black to grey", (0, 0, 0), (0.72, 0.72, 0.75)),
    ("Black to white", (0, 0, 0), (1, 1, 1)),
    ("Black to blue", (0, 0, 0), (0.10, 0.35, 0.75)),
]


def _hex(c) -> str:
    """A colour as '#rrggbb'. pyvista reads a tuple holding any integer - (1, 0.25, 1) -
    as 0..255, so every colour goes to it as hex to leave no room for that."""
    c = np.clip(np.asarray(c, float), 0, 1)
    return "#%02x%02x%02x" % tuple(int(round(v * 255)) for v in c[:3])


class Volume3D(QtWidgets.QWidget):
    """The 3-D viewer: a flow/value volume and an optional B-mode volume in the
    same millimetre frame, rendered by VTK, plus an optional surface.

    Volumes are given as uint8 arrays (depth, lateral, elevation) with 0 = no
    value, and the spacing of those three axes in mm. World axes are
    x = lateral, y = elevation, z = depth; depth is drawn downwards.
    """

    def __init__(self, hint="Press display3D"):
        super().__init__()
        self._lay = QtWidgets.QStackedLayout(self)
        self._hint = QtWidgets.QLabel(hint)
        self._hint.setAlignment(QtCore.Qt.AlignCenter)
        self._hint.setStyleSheet("color:#777")
        self._lay.addWidget(self._hint)
        self.plotter = None
        self.actors = {}
        self._vols = {}              # name -> the VTK pipeline of a ray-cast volume
        self.target = np.zeros(3)
        self.radius = 1.0            # bounding-sphere radius of what is framed, mm
        self.dist = 1.0              # camera distance that fits it in the view
        self.angle = 0.0
        self.bg = BACKGROUNDS[0]

    # ---- life cycle
    def prepare(self):
        """Create the renderer now, behind the hint - it costs over half a
        second the first time, better spent while a set is loading than after
        display3D is pressed."""
        if self.plotter is None:
            cur = self._lay.currentIndex()
            self._ensure()
            self._lay.setCurrentIndex(cur)

    def _remove(self, name):
        act = self.actors.pop(name, None)
        if act is None:
            return
        if name in self._vols:
            self.plotter.renderer.RemoveVolume(self._vols.pop(name)["vol"])
        else:
            self.plotter.remove_actor(act, render=False)

    def _ensure(self):
        if self.plotter is None:
            from pyvistaqt import QtInteractor
            self.plotter = QtInteractor(self)
            self.plotter.enable_anti_aliasing("msaa") if hasattr(self.plotter, "enable_anti_aliasing") else None
            self._lay.addWidget(self.plotter.interactor)
            self.apply_background()
        self._lay.setCurrentIndex(1)

    def clear(self, hint: str = ""):
        if self.plotter is not None:
            for name in list(self.actors):
                self._remove(name)
            self.plotter.render()
        self.actors = {}
        self._vols = {}
        if hint:
            self._hint.setText(hint)
        self._lay.setCurrentIndex(0)

    @property
    def active(self) -> bool:
        return self.plotter is not None and self._lay.currentIndex() == 1 and bool(self.actors)

    # ---- content
    @staticmethod
    def _grid(U: np.ndarray, spacing, origin=(0, 0, 0)):
        import pyvista as pv
        nz, nx, ny = U.shape
        g = pv.ImageData(dimensions=(nx, ny, nz), spacing=(spacing[1], spacing[2], spacing[0]),
                         origin=(origin[1], origin[2], origin[0]))
        g.point_data["v"] = np.transpose(U, (1, 2, 0)).ravel(order="F")
        return g

    def set_volume(self, name: str, U: np.ndarray, spacing, cmap: np.ndarray, alpha: np.ndarray,
                   style: str = "mip", origin=(0, 0, 0), rgb: Optional[np.ndarray] = None):
        """style: mip | volume | iso | slices. cmap: 256 x 3 in 0..1, alpha 256."""
        import pyvista as pv
        self._ensure()
        p = self.plotter
        if style in ("mip", "volume"):
            self._raycast(name, U, spacing, origin, cmap, alpha, style)
            p.render()
            return
        self._remove(name)
        if style == "slices":
            if rgb is not None:
                g = self._grid(rgb[..., 0], spacing, origin)
                g.point_data["rgb"] = np.transpose(rgb, (1, 2, 0, 3)).reshape(-1, 3, order="F")
                sl = g.slice_orthogonal()
                self.actors[name] = p.add_mesh(sl, scalars="rgb", rgb=True, show_scalar_bar=False,
                                               render=False)
            else:
                g = self._grid(U, spacing, origin)
                sl = g.slice_orthogonal()
                lut = pv.LookupTable(values=np.c_[np.clip(cmap * 255, 0, 255),
                                                  np.full(256, 255)].astype(np.uint8))
                lut.scalar_range = (0, 255)
                self.actors[name] = p.add_mesh(sl, scalars="v", cmap=lut, clim=(0, 255),
                                               show_scalar_bar=False, render=False)
        elif style == "iso":
            g = self._grid(U, spacing, origin)
            surf = g.contour([0.5], scalars="v")
            col = _hex(cmap[int(0.75 * 255)])
            if surf.n_points:
                self.actors[name] = p.add_mesh(surf, color=col, smooth_shading=True, render=False,
                                               specular=0.3)
        p.render()

    def _raycast(self, name, U, spacing, origin, cmap, alpha, style):
        """A GPU ray-cast volume, built on VTK directly. When the new volume has
        the shape of the one on screen - a new window, a new smoothing, a new
        Range - only the voxels are swapped in place: no new pipeline and no new
        upload of the geometry, just the texture."""
        from vtkmodules.util import numpy_support
        from vtkmodules.vtkCommonCore import VTK_UNSIGNED_CHAR
        from vtkmodules.vtkCommonDataModel import vtkImageData
        from vtkmodules.vtkRenderingCore import vtkVolume, vtkVolumeProperty
        from vtkmodules.vtkRenderingVolume import vtkGPUVolumeRayCastMapper
        import vtkmodules.vtkRenderingVolumeOpenGL2  # noqa: F401  registers the GPU mapper

        nz, nx, ny = U.shape
        flat = np.ascontiguousarray(np.transpose(U, (0, 2, 1))).ravel()     # x fastest, then y, z
        sp = (float(spacing[1]), float(spacing[2]), float(spacing[0]))
        org = (float(origin[1]), float(origin[2]), float(origin[0]))
        v = self._vols.get(name)
        if v is not None and v["dims"] == (nx, ny, nz):
            v["flat"][:] = flat
            v["arr"].Modified()
            v["img"].SetSpacing(sp); v["img"].SetOrigin(org)
            v["img"].Modified()
        else:
            self._remove(name)
            img = vtkImageData()
            img.SetDimensions(nx, ny, nz)
            img.SetSpacing(sp)
            img.SetOrigin(org)
            arr = numpy_support.numpy_to_vtk(flat, deep=False, array_type=VTK_UNSIGNED_CHAR)
            img.GetPointData().SetScalars(arr)
            mapper = vtkGPUVolumeRayCastMapper()
            mapper.SetInputData(img)
            prop = vtkVolumeProperty()
            prop.SetInterpolationTypeToLinear()
            prop.ShadeOff()
            vol = vtkVolume()
            vol.SetMapper(mapper)
            vol.SetProperty(prop)
            self.plotter.renderer.AddVolume(vol)
            v = dict(img=img, arr=arr, flat=flat, mapper=mapper, prop=prop, vol=vol, dims=(nx, ny, nz))
            self._vols[name] = v
            self.actors[name] = vol
        if style == "mip":
            v["mapper"].SetBlendModeToMaximumIntensity()
        else:
            v["mapper"].SetBlendModeToComposite()
        self._set_tf(v["prop"], alpha, cmap)

    @staticmethod
    def _set_tf(prop, alpha, cmap=None):
        from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
        from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
        otf = vtkPiecewiseFunction()
        for i, a in enumerate(np.asarray(alpha, float)):
            otf.AddPoint(float(i), float(a))
        prop.SetScalarOpacity(otf)
        if cmap is not None:
            ctf = vtkColorTransferFunction()
            for i, c in enumerate(np.asarray(cmap, float)):
                ctf.AddRGBPoint(float(i), float(c[0]), float(c[1]), float(c[2]))
            prop.SetColor(ctf)

    def set_opacity(self, name: str, alpha: np.ndarray, cmap: Optional[np.ndarray] = None):
        """Change the opacity (and colours) of a rendered volume without rebuilding."""
        v = self._vols.get(name)
        if v is None:
            return
        self._set_tf(v["prop"], alpha, cmap)
        self.plotter.render()

    def set_surface(self, name: str, points: Optional[np.ndarray], faces: Optional[np.ndarray],
                    color=(1, 0.25, 1), opacity=0.45):
        """points in world mm (x lateral, y elevation, z depth); faces n x 3."""
        import pyvista as pv
        if self.plotter is None:
            return
        self._remove(name)
        if points is not None and faces is not None and len(faces):
            f = np.c_[np.full(len(faces), 3), faces].ravel()
            mesh = pv.PolyData(points, f)
            self.actors[name] = self.plotter.add_mesh(mesh, color=_hex(color), opacity=opacity,
                                                      render=False, lighting=False)
        self.plotter.render()

    # ---- camera
    def frame(self, bounds):
        """Aim at the middle of bounds (xmin xmax ymin ymax zmin zmax) and stand
        back just far enough that the volume stays wholly in view at EVERY angle
        of the rotation slider, whatever shape the panel is."""
        b = np.asarray(bounds, float)
        self.target = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2])
        self.ext = np.array([b[1] - b[0], b[3] - b[2], b[5] - b[4]])
        self.radius = max(np.linalg.norm(self.ext / 2), 1e-6)
        self._fit()
        self.set_angle(self.angle)

    def _fit(self):
        """The volume turns about its depth axis, so across the turn it sweeps a
        cylinder: radius r = half the lateral-elevational diagonal, height h =
        its depth. The camera looks at it from the side, in perspective:
          across   the cylinder's silhouette is its circle seen from distance d,
                   inside the horizontal field when r/d <= sin(hfov/2)
          up/down  its near edge, d - r away, reaches h/2 above and below,
                   inside the vertical field when (h/2)/(d - r) <= tan(vfov/2)
        A bounding sphere would do too, but it wastes the corners: this fills
        the view about a quarter more and still never clips at any angle."""
        if self.plotter is None or not hasattr(self, "ext"):
            return
        vfov = np.deg2rad(self.plotter.camera.view_angle)
        w = max(1, self.plotter.interactor.width())
        h = max(1, self.plotter.interactor.height())
        hfov = 2 * np.arctan(np.tan(vfov / 2) * w / h)
        r = max(np.hypot(self.ext[0], self.ext[1]) / 2, 1e-6)
        hz = self.ext[2] / 2
        self.dist = 1.04 * max(r / np.sin(hfov / 2), r + hz / np.tan(vfov / 2))

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        if self.active:
            self._fit()
            self.set_angle(self.angle)

    def reset(self):
        """Back to the fitted view, face on - whatever the mouse has done to the
        camera since: turned, tilted, panned or zoomed."""
        if self.plotter is None:
            return
        # a scroll zoom narrows the field of view rather than moving the camera
        self.plotter.camera.view_angle = 30.0
        self._fit()
        self.set_angle(0.0)

    def set_angle(self, a: float):
        """Turn about the depth axis; 0 deg looks along the lateral axis, so the
        sector is face on."""
        self.angle = a
        if self.plotter is None:
            return
        cam = self.plotter.camera
        cam.focal_point = tuple(self.target)
        cam.position = tuple(self.target + self.dist * np.array([np.cos(np.deg2rad(a)),
                                                                 np.sin(np.deg2rad(a)), 0.0]))
        cam.up = (0.0, 0.0, -1.0)
        self.plotter.reset_camera_clipping_range()
        self.plotter.render()

    def apply_background(self, name: Optional[str] = None):
        if name is not None:
            self.bg = next((b for b in BACKGROUNDS if b[0] == name), BACKGROUNDS[0])
        if self.plotter is None:
            return
        _, top, bottom = self.bg
        if bottom is None:
            self.plotter.set_background(_hex(top))
        else:
            self.plotter.set_background(_hex(bottom), top=_hex(top))
        self.plotter.render()

    def save_turn(self, path: str, progress: Callable[[float, str], None] = lambda v, m: None,
                  cancelled: Callable[[], bool] = lambda: False, n_frames: int = 120, fps: int = 20):
        """A full turn about the depth axis, captured from this renderer."""
        import imageio.v2 as imageio
        a0 = self.angle
        w = imageio.get_writer(path, fps=fps, codec="libx264", quality=9, macro_block_size=16)
        try:
            for i in range(n_frames):
                if cancelled():
                    w.close()
                    os.remove(path)
                    return False
                self.set_angle(a0 + 360.0 * i / n_frames)
                # pyvistaqt defers a render to the next idle moment, so draw
                # now: otherwise every frame is the angle drawn last, not this one
                self.plotter.ren_win.Render()
                img = self.plotter.screenshot(return_img=True)
                h, wd = (img.shape[0] // 16) * 16, (img.shape[1] // 16) * 16
                w.append_data(img[:h, :wd])
                if (i + 1) % 4 == 0:
                    progress((i + 1) / n_frames, f"Frame {i + 1} of {n_frames}")
        finally:
            try:
                w.close()
            except Exception:
                pass
            self.set_angle(a0)
        return True


def next_free(folder: str, stem: str, ext: str) -> str:
    n = 1
    while os.path.exists(os.path.join(folder, f"{stem}_{n}{ext}")):
        n += 1
    return os.path.join(folder, f"{stem}_{n}{ext}")
