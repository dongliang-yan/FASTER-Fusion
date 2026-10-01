"""Reading scans: stepped position sets, B-mode cines and Verasonics IQ.

Ports of loadPositionStack / loadCFStack, loadScanStack and scanFileKind.

Axis order follows the MATLAB code throughout, so the algorithms port line for
line: a panel's pixels are (depth, lateral, 3, position), a grey stack is
(depth, lateral, frame). Grey levels stay uint8 wherever the scanner wrote
uint8 - the MATLAB code held them as double, but they are whole numbers 0..255
and a cine held as double is eight times the memory.
"""
from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError
from scipy.io import loadmat

from .mlcompat import mround, rgb2gray_u8

Progress = Callable[[float, str], None]


def _noprog(v: float, msg: str) -> None:  # pragma: no cover - trivial
    pass


class ScanError(Exception):
    """A scan that cannot be read, with a message meant for the user."""


# --------------------------------------------------------------------------
# helpers shared by both loaders
# --------------------------------------------------------------------------

def _read_header(path: str):
    try:
        return pydicom.dcmread(path, stop_before_pixels=True)
    except (InvalidDicomError, OSError, ValueError, KeyError, AttributeError):
        return None


def _regions(ds, name: str):
    """The 2-D ultrasound regions, left to right, as plain dicts."""
    seq = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not seq:
        raise ScanError(
            f"{name} has no SequenceOfUltrasoundRegions, so where the ultrasound "
            "picture is on the screen and how big a pixel is are both unknown. It "
            "is probably not a capture from the scanner.")
    want = ["RegionSpatialFormat", "RegionDataType", "RegionLocationMinX0",
            "RegionLocationMinY0", "RegionLocationMaxX1", "RegionLocationMaxY1",
            "ReferencePixelX0", "ReferencePixelY0", "PhysicalUnitsXDirection",
            "PhysicalUnitsYDirection", "PhysicalDeltaX", "PhysicalDeltaY"]
    regs = []
    for item in seq:
        if not hasattr(item, "RegionLocationMinX0"):
            continue
        fmt = getattr(item, "RegionSpatialFormat", None)
        if fmt is not None and int(fmt) != 1:
            continue
        r = {}
        for w in want:
            v = getattr(item, w, None)
            r[w] = float(v) if v is not None else float("nan")
        regs.append(r)
    if not regs:
        raise ScanError(f"{name} has no 2-D ultrasound region.")
    regs.sort(key=lambda r: r["RegionLocationMinX0"])    # stable, as MATLAB sort
    return regs


def _phys_delta(r: dict, ax: str) -> float:
    """PhysicalUnits 3 is centimetres; anything else leaves the scale unknown."""
    if r[f"PhysicalUnits{ax}Direction"] == 3 and np.isfinite(r[f"PhysicalDelta{ax}"]):
        return abs(r[f"PhysicalDelta{ax}"]) * 10.0
    return float("nan")


def _hhmmss(tm: str) -> float:
    tm = str(tm).strip().replace(":", "")
    if len(tm) < 6:
        return float("nan")
    try:
        return 3600 * float(tm[0:2]) + 60 * float(tm[2:4]) + float(tm[4:])
    except ValueError:
        return float("nan")


def _acq_seconds(ds) -> float:
    """Acquisition time in seconds with the date folded in; content time if not."""
    for dkey, tkey in (("AcquisitionDate", "AcquisitionTime"), ("ContentDate", "ContentTime")):
        t = getattr(ds, tkey, None)
        if t:
            sec = _hhmmss(t)
            if not np.isfinite(sec):
                continue
            day = 0
            d = str(getattr(ds, dkey, "") or "")
            if len(d) >= 8:
                try:
                    day = _dt.date(int(d[0:4]), int(d[4:6]), int(d[6:8])).toordinal()
                except ValueError:
                    day = 0
            return day * 86400 + sec
    return float("nan")


def _num_field(ds, f: str) -> float:
    v = getattr(ds, f, None)
    try:
        return float(v) if v is not None and str(v) != "" else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _read_rgb(path: str, ds_head=None) -> np.ndarray:
    """One frame as H x W x 3 uint8 RGB."""
    ds = pydicom.dcmread(path)
    try:
        I = ds.pixel_array                      # YBR is converted to RGB here
    except Exception as e:                      # pragma: no cover - codec issue
        raise ScanError(f"{os.path.basename(path)}: its pixels could not be decoded ({e}).")
    if I.ndim == 2:
        I = np.repeat(I[:, :, None], 3, axis=2)
    if I.dtype != np.uint8:
        I = (255.0 * I.astype(float) / max(1.0, float(I.max()))).astype(np.uint8)
    if I.shape[0] != int(ds.Rows) or I.shape[1] != int(ds.Columns):
        raise ScanError(f"{os.path.basename(path)} is not {ds.Rows} x {ds.Columns} "
                        "as its header says.")
    return I


def _lit_run(is_lit: np.ndarray, c0: int, max_gap: int):
    """The run of lit columns through c0 (0-based), bridging gaps of max_gap."""
    n = is_lit.size
    if not is_lit[c0]:
        lit = np.flatnonzero(is_lit)
        if lit.size == 0:
            return None
        c0 = int(lit[np.argmin(np.abs(lit - c0))])
    lo = c0
    while True:
        a = max(0, lo - 1 - max_gap)
        k = np.flatnonzero(is_lit[a:lo])
        if k.size == 0:
            break
        lo = a + int(k[-1])
    hi = c0
    while True:
        k = np.flatnonzero(is_lit[hi + 1:min(n, hi + 2 + max_gap)])
        if k.size == 0:
            break
        hi = hi + 1 + int(k[0])
    return np.arange(lo, hi + 1)


# --------------------------------------------------------------------------
# stepped position sets (SWE, colour flow, stepped calibration)
# --------------------------------------------------------------------------

@dataclass
class Panel:
    rgb: np.ndarray            # nz x nx x 3 x N uint8
    rows: np.ndarray           # screen rows (0-based)
    cols: np.ndarray           # screen columns (0-based)
    dz_mm: float
    dx_mm: float
    region: dict


@dataclass
class PositionStack:
    folder: str
    files: list
    time_s: np.ndarray
    instance: np.ndarray
    order_by: str
    screen: np.ndarray         # first screen, H x W x 3 uint8
    panels: list
    flow_box: Optional[list] = None   # [r0 r1 c0 c1] SCREEN pixels, 0-based inclusive
    flow_panel: Optional[int] = None
    kind: str = "dicomset"

    @property
    def N(self) -> int:
        return len(self.files)


def _split_regions(regs, H, W):
    """loadCFStack: panels are the tissue regions (type 1); the colour box is the
    smallest type-2 region that states a pixel size."""
    typ = np.array([r["RegionDataType"] for r in regs])
    isT = typ == 1
    if not isT.any():
        isT = ~(typ == 2)
    if not isT.any():
        isT = np.ones(len(regs), bool)
    tissue, seen = [], set()
    for r, t in zip(regs, isT):
        if not t:
            continue
        key = (r["RegionLocationMinX0"], r["RegionLocationMaxX1"],
               r["RegionLocationMinY0"], r["RegionLocationMaxY1"])
        if key in seen:
            continue
        seen.add(key)
        tissue.append(r)
    flow = None
    cand = [r for r, t in zip(regs, typ)
            if t == 2 and np.isfinite(r["PhysicalDeltaX"]) and r["PhysicalDeltaX"] != 0]
    if cand:
        area = [(r["RegionLocationMaxX1"] - r["RegionLocationMinX0"] + 1) *
                (r["RegionLocationMaxY1"] - r["RegionLocationMinY0"] + 1) for r in cand]
        r = cand[int(np.argmin(area))]
        flow = [int(max(0, r["RegionLocationMinY0"])), int(min(H - 1, r["RegionLocationMaxY1"])),
                int(max(0, r["RegionLocationMinX0"])), int(min(W - 1, r["RegionLocationMaxX1"]))]
    return tissue, flow


def load_position_stack(target: str, progress: Progress = _noprog,
                        min_positions: int = 3) -> PositionStack:
    """A folder of single-frame DICOMs, one per reflector position, in
    acquisition order. Colour-flow screens are handled as loadCFStack does: only
    tissue regions become panels, the colour box is reported beside them."""
    target = str(target)
    if os.path.isdir(target):
        folder = target
    elif os.path.isfile(target):
        folder = os.path.dirname(target) or os.getcwd()
    else:
        raise ScanError(f"{target} does not exist.")

    names = sorted(n for n in os.listdir(folder)
                   if not n.startswith(".") and os.path.isfile(os.path.join(folder, n)))
    infos, keep = [], []
    for i, n in enumerate(names):
        ds = _read_header(os.path.join(folder, n))
        if ds is None or not hasattr(ds, "Rows") or not hasattr(ds, "Columns"):
            continue
        nf = _num_field(ds, "NumberOfFrames")
        if np.isfinite(nf) and nf > 1:
            raise ScanError(
                f"{n} is a cine of {int(nf)} frames. A stepped set is one single-frame "
                "DICOM per reflector position, all in one folder. For a cine use the "
                "Cine calibration or B-Mode.")
        infos.append(ds)
        keep.append(n)
        if (i + 1) % 10 == 0:
            progress(0.25 * (i + 1) / len(names), f"Reading headers  {i + 1} of {len(names)}")
    names = keep
    N = len(names)
    if N < min_positions:
        raise ScanError(f"{folder} holds {N} single-frame DICOM(s). A position set "
                        f"needs at least {min_positions}, one per reflector position.")

    # acquisition order
    t = np.array([_acq_seconds(d) for d in infos])
    inst = np.array([_num_field(d, "InstanceNumber") for d in infos])
    if np.all(np.isfinite(t)) and np.unique(t).size == N:
        ordr = np.lexsort((inst, t))
        order_by = "acquisition time"
    elif np.all(np.isfinite(inst)) and np.unique(inst).size == N:
        ordr = np.argsort(inst, kind="stable")
        order_by = "instance number"
    else:
        ordr = np.arange(N)                  # names are sorted already
        order_by = "file name"
    names = [names[i] for i in ordr]
    infos = [infos[i] for i in ordr]
    t, inst = t[ordr], inst[ordr]

    # one layout for every file
    ref = infos[0]
    regs = _regions(ref, names[0])
    loc = lambda rr: [(r["RegionLocationMinX0"], r["RegionLocationMaxX1"],
                       r["RegionLocationMinY0"], r["RegionLocationMaxY1"]) for r in rr]
    for i in range(1, N):
        d = infos[i]
        if d.Rows != ref.Rows or d.Columns != ref.Columns:
            raise ScanError(f"Position {i + 1} ({names[i]}) is {d.Rows} x {d.Columns}; the "
                            f"rest of the set is {ref.Rows} x {ref.Columns}. Every position "
                            "must be captured with the same screen layout.")
        if loc(_regions(d, names[i])) != loc(regs):
            raise ScanError(f"Position {i + 1} ({names[i]}) has a different screen layout "
                            f"from position 1 ({names[0]}): its ultrasound regions are not in "
                            "the same place. Every position must be captured with the same "
                            "preset and display mode.")

    H, W = int(ref.Rows), int(ref.Columns)
    tissue, flow = _split_regions(regs, H, W)

    # where each panel's picture is, measured on a few positions
    probe = np.unique(mround(np.linspace(1, N, min(N, 9))).astype(int)) - 1
    lit = np.zeros((H, W))
    first_screens = {}
    for k in probe:
        I = _read_rgb(os.path.join(folder, names[k]))
        first_screens[k] = I
        lit += I.max(axis=2) > 8
    lit /= probe.size

    panels = []
    for j, r in enumerate(tissue):
        rows = np.arange(int(r["RegionLocationMinY0"]), int(r["RegionLocationMaxY1"]) + 1)
        xr = np.arange(int(r["RegionLocationMinX0"]), int(r["RegionLocationMaxX1"]) + 1)
        rows = rows[(rows >= 0) & (rows < H)]
        xr = xr[(xr >= 0) & (xr < W)]
        if rows.size == 0 or xr.size == 0:
            raise ScanError(f"Ultrasound region {j + 1} lies outside the {H} x {W} screen.")
        col_lit = lit[np.ix_(rows, xr)].mean(axis=0) > 0.2
        c0 = None
        if np.isfinite(r["ReferencePixelX0"]):
            c0 = int(r["ReferencePixelX0"] + r["RegionLocationMinX0"] - xr[0])
            if c0 < 0 or c0 >= xr.size:
                c0 = None
        if c0 is None:
            c0 = int(mround(xr.size / 2)) - 1
        run = _lit_run(col_lit, c0, 3)
        if run is None:
            raise ScanError(f"Ultrasound region {j + 1} (columns {xr[0] + 1} to {xr[-1] + 1}) "
                            "is dark in every position looked at, so there is no picture to crop.")
        cols = xr[run]
        panels.append(Panel(rgb=np.zeros((rows.size, cols.size, 3, N), np.uint8),
                            rows=rows, cols=cols, dz_mm=_phys_delta(r, "Y"),
                            dx_mm=_phys_delta(r, "X"), region=r))

    screen = None
    for k in range(N):
        I = first_screens.get(k)
        if I is None:
            I = _read_rgb(os.path.join(folder, names[k]))
        if k == 0:
            screen = I
        for p in panels:
            p.rgb[:, :, :, k] = I[np.ix_(p.rows, p.cols)]
        if (k + 1) % 5 == 0 or k == N - 1:
            progress(0.25 + 0.75 * (k + 1) / N, f"Reading position {k + 1} of {N}")

    ps = PositionStack(folder=folder, files=names, time_s=t, instance=inst,
                       order_by=order_by, screen=screen, panels=panels)
    if flow is not None:
        for j, p in enumerate(panels):
            rr = [max(flow[0], p.rows[0]), min(flow[1], p.rows[-1])]
            cc = [max(flow[2], p.cols[0]), min(flow[3], p.cols[-1])]
            if rr[1] > rr[0] and cc[1] > cc[0]:
                ps.flow_box = [int(rr[0]), int(rr[1]), int(cc[0]), int(cc[1])]
                ps.flow_panel = j
                break
    return ps


# --------------------------------------------------------------------------
# cines and IQ captures (B-mode, cine calibration)
# --------------------------------------------------------------------------

def scan_file_kind(f: str):
    """('dicom' | 'geiq' | '', aux). By content, not extension."""
    f = str(f)
    if not os.path.isfile(f):
        return "", {}
    base, ext = os.path.splitext(f)
    if ext.lower() == ".dat":
        szf = os.path.join(os.path.dirname(f), "IQSizeInfo.mat")
        if os.path.isfile(szf):
            try:
                m = loadmat(szf)
            except Exception:
                return "", {}
            if "IQSizeInfo" in m:
                sz = np.asarray(m["IQSizeInfo"], float).ravel()
                if sz.size == 3 and np.all(np.isfinite(sz)) and np.all(sz >= 2):
                    if os.path.getsize(f) == 16 * int(np.prod(sz)):
                        return "geiq", {"sz": sz.astype(int), "sizeFile": szf}
        return "", {}
    try:
        with open(f, "rb") as fh:
            head = fh.read(132)
        if len(head) == 132 and head[128:132] == b"DICM":
            return "dicom", {}
    except OSError:
        return "", {}
    return ("dicom", {}) if _read_header(f) is not None else ("", {})


def list_scan_files(folder: str) -> list:
    names = [n for n in sorted(os.listdir(folder))
             if not n.startswith(".") and os.path.isfile(os.path.join(folder, n))]
    return [n for n in names if scan_file_kind(os.path.join(folder, n))[0]]


@dataclass
class ScanStack:
    B: np.ndarray              # depth x lateral x frames; uint8 (DICOM) or float32 (IQ)
    kind: str
    dz_mm: float
    dx_mm: float
    assumed: bool = False
    geom_from: str = ""
    region: Optional[dict] = None
    geom: dict = field(default_factory=dict)
    dyn_range: float = float("nan")


# GE 9L-D defaults, used only when nothing else says otherwise
DEF_DZ_MM = 0.1185
DEF_DX_MM = 0.2292


def load_scan_stack(fname: str, crop=(369, 656), dyn_range: float = 60,
                    dz_mm: float = float("nan"), dx_mm: float = float("nan"),
                    progress: Progress = _noprog, min_frames: int = 80) -> ScanStack:
    """A Siemens DICOM cine or a GE/Verasonics IQ capture as one grey stack.
    crop is the DICOM column crop, 1-based inclusive as in the MATLAB GUIs."""
    kind, aux = scan_file_kind(fname)
    if kind == "dicom":
        s = _load_dicom_cine(fname, crop, progress)
    elif kind == "geiq":
        s = _load_ge_iq(fname, aux, dyn_range, dz_mm, dx_mm, progress)
    else:
        raise ScanError(f"{fname} is neither a DICOM cine nor a GE IQ capture.\n\n"
                        "A DICOM is the scan file itself, for example 01462453. A GE "
                        "capture is the IQ .dat file, with its IQSizeInfo.mat in the "
                        "same folder.")
    if s.B.shape[2] < min_frames:
        raise ScanError(f"Only {s.B.shape[2]} frames. The sweep period cannot be measured "
                        "in less than about one mechanical period.")
    return s


def _load_dicom_cine(fname, crop, progress) -> ScanStack:
    from pydicom.pixels import iter_pixels
    progress(0.1, "Reading the DICOM")
    ds = pydicom.dcmread(fname, stop_before_pixels=True)
    seq = getattr(ds, "SequenceOfUltrasoundRegions", None)
    if not seq:
        raise ScanError("This file has no SequenceOfUltrasoundRegions, so the ultrasound "
                        "window and the pixel size are unknown. It is probably not a "
                        "B-mode cine from the scanner.")
    r0 = seq[0]
    r = {k: float(getattr(r0, k)) if getattr(r0, k, None) is not None else float("nan")
         for k in ("RegionLocationMinX0", "RegionLocationMinY0", "RegionLocationMaxX1",
                   "RegionLocationMaxY1", "PhysicalDeltaX", "PhysicalDeltaY",
                   "PhysicalUnitsXDirection", "PhysicalUnitsYDirection", "RegionDataType")}
    ymin, ymax = int(r["RegionLocationMinY0"]), int(r["RegionLocationMaxY1"])   # 0-based
    H, W = int(ds.Rows), int(ds.Columns)
    if crop is None:
        xmin, xmax = 0, W - 1
    else:
        xmin, xmax = int(crop[0]) - 1, int(crop[1]) - 1
    if xmin < 0 or ymin < 0 or xmax >= W or ymax >= H or xmin >= xmax or ymin >= ymax:
        raise ScanError(f"The crop columns {xmin + 1}:{xmax + 1}, rows {ymin + 1}:{ymax + 1} "
                        f"does not fit this {W} x {H} image.")
    nf = int(_num_field(ds, "NumberOfFrames")) if np.isfinite(_num_field(ds, "NumberOfFrames")) else 1
    B = np.zeros((ymax - ymin + 1, xmax - xmin + 1, nf), np.uint8)
    # A JPEG cine arrives as YCbCr. Converting whole frames to RGB and cropping
    # after is three quarters wasted work, so the window is cut out first and
    # only it is converted - with pydicom's own conversion, so the result is
    # exactly what converting the whole frame would give.
    pi = str(getattr(ds, "PhotometricInterpretation", ""))
    ybr = pi.startswith("YBR") and int(getattr(ds, "SamplesPerPixel", 1)) == 3
    if ybr:
        from pydicom.pixels.processing import convert_color_space
    for k, fr in enumerate(iter_pixels(fname, as_rgb=not ybr)):
        fr = fr[ymin:ymax + 1, xmin:xmax + 1]
        if ybr:
            fr = convert_color_space(fr, "YBR_FULL", "RGB")
        B[:, :, k] = rgb2gray_u8(fr) if fr.ndim == 3 and fr.shape[2] == 3 else fr
        if (k + 1) % 20 == 0 or k == nf - 1:
            progress(0.1 + 0.85 * (k + 1) / nf, f"Decoding frame {k + 1} of {nf}")
    return ScanStack(B=B, kind="dicom", dz_mm=_phys_delta(r, "Y"), dx_mm=_phys_delta(r, "X"),
                     assumed=False, geom_from="DICOM", region=r)


def _load_ge_iq(fname, aux, dyn_range, dz_o, dx_o, progress) -> ScanStack:
    nz, nx, nf = (int(v) for v in aux["sz"])
    nzx = nz * nx
    mm = np.memmap(fname, dtype="<f8", mode="r", shape=(2 * nf * nzx,))
    env = np.empty((nz, nx, nf), np.float64)
    for k in range(nf):
        re = mm[k * nzx:(k + 1) * nzx]
        im = mm[(nf + k) * nzx:(nf + k + 1) * nzx]
        env[:, :, k] = np.hypot(re, im).reshape((nz, nx), order="F")
        if (k + 1) % 20 == 0 or k == nf - 1:
            progress(0.85 * (k + 1) / nf, f"Reading IQ frame {k + 1} of {nf}")
    del mm
    progress(0.9, "Log compressing")
    # 99.9th centile of a column-major subsample, as robustPeak
    flat = env.ravel(order="F")
    step = max(1, int(np.floor(flat.size / 2e6)))
    x = np.sort(flat[::step])
    ref = x[min(x.size, max(1, int(np.ceil(0.999 * x.size)))) - 1] if x.size else np.nan
    if not np.isfinite(ref) or ref <= 0:
        raise ScanError("This capture is all zeros; there is nothing to show.")
    dB = 20 * np.log10(np.maximum(env, np.finfo(float).tiny) / ref)
    B = np.clip(255 * (dB + dyn_range) / dyn_range, 0, 255).astype(np.float32)
    del env, dB

    g = {}
    gf = os.path.join(os.path.dirname(fname), "IQGeom.mat")
    if os.path.isfile(gf):
        try:
            m = loadmat(gf, squeeze_me=True)
            for k in ("dz_mm", "dx_mm", "z0_mm", "mirrorDepth_mm", "lambda_mm"):
                if k in m and np.size(m[k]) == 1 and np.isfinite(float(m[k])):
                    g[k] = float(m[k])
        except Exception:
            g = {}

    def first(*vals):
        for v in vals:
            if v is not None and np.isfinite(v) and v > 0:
                return float(v)
        return float("nan")

    dz = first(dz_o, g.get("dz_mm"), DEF_DZ_MM)
    dx = first(dx_o, g.get("dx_mm"), DEF_DX_MM)
    if np.isfinite(dz_o) or np.isfinite(dx_o):
        assumed, frm = False, "the values set in the GUI"
    elif "dz_mm" in g or "dx_mm" in g:
        assumed, frm = False, "IQGeom.mat"
    else:
        assumed, frm = True, "GE 9L-D defaults, NOT measured from this file"
    progress(1.0, "Loaded")
    return ScanStack(B=B, kind="geiq", dz_mm=dz, dx_mm=dx, assumed=assumed,
                     geom_from=frm, geom=g, dyn_range=dyn_range)
