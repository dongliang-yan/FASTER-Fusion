"""Calibrations, angles, sweeping into the sector, volumes and SIVV.

What Recon_v1, Recon_SWE and Recon_CF_v2 share, without any GUI:

  load_calibration / save_calibration   the .mat files, interchangeable with MATLAB
  stepped_angles                         anglesFor (Shift, Flip, Angle map, unique)
  cine_angles                            the angle part of Recon_v1's recon2D
  sweep_values / sweep_grey              NaN-aware sweep of one or all lateral lines
  iso_volume                             resample to isotropic voxels, NaN-aware
  nan_gauss, keep_largest                smoothing and the Largest-blob filter
  sivv_profile / sivv_at                 flow rate through constant-range surfaces
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage
from scipy.io import loadmat, savemat

from .mlcompat import imgaussfilt, imresize3, interp1, mround
from .sector import SectorGrid

EDGE_COVER = 0.5          # see recon2D: where the edge of a masked region falls


# --------------------------------------------------------------------------
# calibration files
# --------------------------------------------------------------------------

class CalibrationError(Exception):
    pass


def _mat_value(v):
    if isinstance(v, np.ndarray) and v.dtype.kind in "US" and v.size == 1:
        return str(v.item())
    if isinstance(v, np.ndarray) and v.size == 1 and v.dtype.kind in "fiub":
        return v.item()
    return v


def load_calibration(path: str) -> dict:
    """Read and check a calibration from any of the calibration GUIs, MATLAB's or
    this app's. Raises CalibrationError with a message for the user."""
    short = os.path.basename(path)
    try:
        m = loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception as e:
        raise CalibrationError(f"Could not read {short}: {e}")
    c = {k: _mat_value(v) for k, v in m.items() if not k.startswith("__")}
    need = ["colFit", "thetaFit", "FrameLine", "direction"]
    miss = [k for k in need if k not in c]
    if miss:
        raise CalibrationError(f"{short} is missing {', '.join(miss)}.\n\nPick a file saved "
                               "by one of the calibration GUIs.")
    cf = np.atleast_1d(np.asarray(c["colFit"], float)).ravel()
    tf = np.atleast_1d(np.asarray(c["thetaFit"], float)).ravel()
    fl = np.atleast_1d(np.asarray(c["FrameLine"], float)).ravel()
    why = ""
    if cf.size != tf.size:
        why = f"colFit has {cf.size} points and thetaFit has {tf.size}."
    elif cf.size < 2:
        why = f"colFit has {cf.size} point(s); at least 2 are needed to interpolate."
    elif np.any(np.diff(cf) <= 0):
        why = "colFit is not strictly increasing, so it cannot be interpolated against."
    elif fl.size < 2:
        why = f"FrameLine covers {fl.size} position(s)."
    elif np.all(np.isnan(tf)):
        why = "thetaFit is all NaN, so no position would get an angle."
    if why:
        raise CalibrationError(f"{short} is not usable:\n\n{why}")
    c["colFit"], c["thetaFit"], c["FrameLine"] = cf, tf, fl
    c["_file"] = path
    return c


def describe_calibration(c: dict) -> str:
    short = os.path.basename(c.get("_file", ""))
    how = f" [{c['pickedBy']}]" if isinstance(c.get("pickedBy"), str) else ""
    tf = c["thetaFit"]
    if c.get("acqMode") == "quasi-static":
        what = f"{c['FrameLine'].size} positions"
    else:
        what = f"a {c['FrameLine'].size}-frame block"
    return (f"Calibration {short}{how}: {c['direction']}, {what}, "
            f"{np.nanmin(tf):+.0f} to {np.nanmax(tf):+.0f} deg.")


def save_calibration(path: str, fields: dict):
    """Write a calibration as MATLAB does (-v5/-v7 readable by both)."""
    out = {}
    for k, v in fields.items():
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and v and all(isinstance(s, str) for s in v):
            v = np.array(v, dtype=object)
        out[k] = v
    savemat(path, out, do_compression=True, oned_as="column")


# --------------------------------------------------------------------------
# angles
# --------------------------------------------------------------------------

def stepped_angles(cal: dict, N: int, shift: int = 0, flip: bool = False,
                   angle_map: str = "fraction"):
    """anglesFor: the angle of each of N positions, NaN where the calibration
    says nothing; valid marks the ones used (unique angles only)."""
    ang = np.full(N, np.nan)
    valid = np.zeros(N, bool)
    if cal is None or N < 2:
        return ang, valid
    n_cal = cal["FrameLine"].size
    cf, tf = cal["colFit"], cal["thetaFit"]
    if angle_map == "fraction" and n_cal > 1:
        col_axis = 1 + (cf - 1) * (N - 1) / (n_cal - 1)
    else:
        col_axis = cf
    k = np.arange(1, N + 1) - shift
    ang = interp1(col_axis, tf, k, "linear")
    if flip:
        ang = -ang
    valid = np.isfinite(ang)
    v = np.flatnonzero(valid)
    _, first = np.unique(mround(ang[v] * 1e6), return_index=True)
    keep = np.zeros(v.size, bool)
    keep[first] = True
    valid[v[~keep]] = False
    ang[~valid] = np.nan
    return ang, valid


def cine_angles(cal: dict, n_frames: int, angle_map: str = "fraction", flip: bool = False):
    """Recon_v1: angle of each frame of a block of n_frames, and which are valid."""
    n_cal = cal["FrameLine"].size
    if angle_map == "fraction":
        col_axis = 1 + (cal["colFit"] - 1) * (n_frames - 1) / (n_cal - 1)
    else:
        col_axis = cal["colFit"]
    ang = interp1(col_axis, cal["thetaFit"], np.arange(1, n_frames + 1), "linear")
    valid = ~np.isnan(ang)
    a = ang[valid]
    if flip:
        a = -a
    return a, valid


# --------------------------------------------------------------------------
# sweeping values into the sector
# --------------------------------------------------------------------------

def sweep_values(T: np.ndarray, grid: SectorGrid, edge_cover: float = EDGE_COVER):
    """NaN-aware sweep. T is (range, angle[, lines]) with NaN = no value: the
    values and their mask are swept apart and divided through, and a swept pixel
    that is less than edge_cover from real values is NaN."""
    M = ~np.isnan(T)
    T0 = np.where(M, T, 0.0)
    num = grid.apply(T0)
    den = grid.apply(M.astype(float))
    with np.errstate(invalid="ignore", divide="ignore"):
        v = num / np.maximum(den, np.finfo(float).eps)
    v[den < edge_cover] = np.nan
    return v


def nan_gauss(V: np.ndarray, sig) -> np.ndarray:
    """Gaussian smoothing that ignores NaN and keeps the NaN where it was."""
    has = ~np.isnan(V)
    V0 = np.where(has, V, 0).astype(V.dtype if V.dtype.kind == "f" else np.float32)
    num = imgaussfilt(V0, sig)
    den = imgaussfilt(has.astype(V0.dtype), sig)
    out = num / np.maximum(den, np.finfo(V0.dtype).eps)
    out[~has] = np.nan
    return out


def iso_volume(t: np.ndarray, vox, smooth_mm: float = 0.0, tolerance: float = 1.0):
    """Resample a NaN-bearing volume to isotropic voxels (the largest of vox),
    values and mask apart, then smooth; as Recon_SWE's showVolume.

    MATLAB's volshow could only draw cubic voxels, so it always resampled. VTK
    takes the spacing as it is, so when the voxels are already within
    `tolerance` of cubic (largest/smallest) the resample is skipped and the
    volume is used on its own grid - the smoothing then works in millimetres
    along each axis. tolerance 1 always resamples, as MATLAB.
    Returns the volume and its voxel size per axis."""
    vox = np.asarray(vox, float)
    if vox.max() / vox.min() <= tolerance:
        tI = t.astype(np.float32, copy=True)
        if smooth_mm > 0:
            tI = nan_gauss(tI, smooth_mm / vox)
        return tI, vox
    iso = vox.max()
    sz = np.maximum(mround(np.array(t.shape) * vox / iso), 1).astype(int)
    m = (~np.isnan(t)).astype(np.float32)
    t0 = np.where(np.isnan(t), 0, t).astype(np.float32)
    if tuple(sz) != t.shape:
        t0 = imresize3(t0, sz)
        m = imresize3(m, sz)
    with np.errstate(invalid="ignore", divide="ignore"):
        tI = t0 / np.maximum(m, np.finfo(np.float32).eps)
    tI[m < 0.5] = np.nan
    if smooth_mm > 0:
        tI = nan_gauss(tI, smooth_mm / iso)
    return tI.astype(np.float32), np.array(t.shape) * vox / sz


def keep_largest(U: np.ndarray) -> np.ndarray:
    """Zero every 26-connected region of U >= 1 except the largest."""
    shown = U >= 1
    lab, n = ndimage.label(shown, structure=np.ones((3, 3, 3), int))
    if n < 2:
        return U
    sizes = np.bincount(lab.ravel())[1:]
    k = int(np.argmax(sizes)) + 1
    U = U.copy()
    U[shown & (lab != k)] = 0
    return U


def window_to_u8(values: np.ndarray, rng) -> np.ndarray:
    """1..255 across the window, 0 outside it or where NaN (the renderer's
    'no value')."""
    u = (values - rng[0]) / (rng[1] - rng[0])
    out = np.isnan(u) | (u < 0) | (u > 1)
    U = (1 + mround(254 * np.clip(np.nan_to_num(u), 0, 1))).astype(np.uint8)
    U[out] = 0
    return U


def alpha_curve(knee: float, boost: float) -> np.ndarray:
    """Opacity per value, 256 entries over the window (Linearity + Opacity)."""
    t = np.linspace(0, 1, 256)
    a = t * np.minimum(t / knee, 1) if knee > 0 else t.copy()
    g = 1 / max(boost, np.finfo(float).eps)
    if g != 1:
        a = a ** g
    if a.max() > 0:
        a = a / a.max()
    return np.clip(a, 0, 1)


# --------------------------------------------------------------------------
# SIVV
# --------------------------------------------------------------------------

@dataclass
class SivvProfile:
    rows: np.ndarray        # 1-based capture rows
    z: np.ndarray           # depth, mm
    d: np.ndarray           # range from the mirror, mm
    Q: np.ndarray           # mL/min, net towards the probe
    Qpos: np.ndarray
    Qneg: np.ndarray
    edge: np.ndarray
    edge_lat: np.ndarray
    edge_pos: np.ndarray
    area: np.ndarray        # mm^2 of flow on the surface
    pos: np.ndarray         # 1-based positions, sorted by angle
    th: np.ndarray          # their angles, rad, increasing
    w: np.ndarray           # trapezoidal angle weights, rad
    cols: np.ndarray        # 1-based lateral lines of the box
    V: np.ndarray           # rows x cols x positions, cm/s (0 = no flow)


def sivv_profile(v_cms: np.ndarray, box, zmm: np.ndarray, mirror_mm: float, dx_mm: float,
                 ang_deg: np.ndarray, valid: np.ndarray) -> Optional[SivvProfile]:
    """Q at every depth of the colour box. v_cms is (depth, lateral, position)
    velocity in cm/s with NaN for no flow; box is 1-based [r0 r1 c0 c1]."""
    pos = np.flatnonzero(valid) + 1
    if pos.size < 2:
        return None
    th = np.deg2rad(ang_deg[pos - 1])
    o = np.argsort(th, kind="stable")
    th, pos = th[o], pos[o]
    dth = np.diff(th)
    w = (np.r_[dth, 0] + np.r_[0, dth]) / 2
    iz = int(np.flatnonzero(zmm >= mirror_mm)[0]) + 1 if np.any(zmm >= mirror_mm) else None
    if iz is None:
        return None
    rows = np.arange(max(box[0], iz), box[1] + 1)
    rows = rows[zmm[rows - 1] > mirror_mm]
    if rows.size == 0:
        return None
    cols = np.arange(box[2], box[3] + 1)
    V = v_cms[np.ix_(rows - 1, cols - 1, pos - 1)].astype(float)
    V[np.isnan(V)] = 0
    d = zmm[rows - 1] - mirror_mm
    k = 0.6 * dx_mm                                   # cm/s x mm^2 -> mL/min
    sj = lambda X: X.sum(axis=1)                      # rows x positions
    Q = k * d * (sj(V) @ w)
    Qpos = k * d * (sj(np.maximum(V, 0)) @ w)
    Qneg = k * d * (sj(np.minimum(V, 0)) @ w)
    F = V != 0
    lat_idx = np.r_[0:min(2, cols.size), max(0, cols.size - 2):cols.size]
    eL = F[:, lat_idx, :].any(axis=(1, 2))
    eP = F[:, :, [0, -1]].any(axis=(1, 2))
    area = (k / 0.6) * d * (sj(F.astype(float)) @ w)
    return SivvProfile(rows=rows, z=zmm[rows - 1], d=d, Q=Q, Qpos=Qpos, Qneg=Qneg,
                       edge=eL | eP, edge_lat=eL, edge_pos=eP, area=area, pos=pos,
                       th=th, w=w, cols=cols, V=V.astype(np.float32))


def sivv_default_depth(P: SivvProfile) -> float:
    ok = ~P.edge & (P.Q != 0)
    if not ok.any():
        ok = P.Q != 0
    if not ok.any():
        return float(P.z[int(mround(P.z.size / 2)) - 1])
    q = np.abs(P.Q)
    q[~ok] = 0
    near = np.flatnonzero(q >= 0.8 * q.max())
    return float(P.z[near[int(np.ceil(near.size / 2)) - 1]])


@dataclass
class SivvAt:
    i0: int
    ii: np.ndarray
    z: float
    d: float
    rows: np.ndarray
    Q: float
    Qpos: float
    Qneg: float
    Qsd: float
    area: float
    edge_lat: bool
    edge_pos: bool
    map: np.ndarray          # cols x positions, cm/s


def sivv_at(P: SivvProfile, depth: float, n_avg: int) -> SivvAt:
    i0 = int(np.argmin(np.abs(P.z - depth)))
    ii = np.arange(max(0, i0 - n_avg), min(P.z.size, i0 + n_avg + 1))
    return SivvAt(i0=i0, ii=ii, z=float(P.z[i0]), d=float(P.d[i0]), rows=P.rows[ii],
                  Q=float(P.Q[ii].mean()), Qpos=float(P.Qpos[ii].mean()),
                  Qneg=float(P.Qneg[ii].mean()),
                  Qsd=float(P.Q[ii].std(ddof=1)) if ii.size > 1 else 0.0,
                  area=float(P.area[ii].mean()), edge_lat=bool(P.edge_lat[ii].any()),
                  edge_pos=bool(P.edge_pos[ii].any()), map=P.V[ii].mean(axis=0))
