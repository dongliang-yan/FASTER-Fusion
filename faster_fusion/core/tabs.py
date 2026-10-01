"""Index-tab clicking and the frame/position-to-angle fit.

The logic both calibration GUIs share (Calibration_GUI_v1 for cines,
Calibration_SWE / Calibration_CF for stepped sets): a click says WHICH tab, and
where it is gets measured - the middle of the whole bright dash the tab draws on
the map, found with the stationary background taken out. Ends of a row that run
into a flat streak are put by the spacing of the others (open tabs), and moved
again as tabs come and go. See snapToPeak in the MATLAB files for the reasoning
and the measurements behind each rule; this is a line-for-line port.

Columns ("col") are frames of a block for a cine and positions for a stepped
set; rows are axial samples. All coordinates here are 1-based, as on the MATLAB
axes, so saved calibrations are interchangeable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy import ndimage

from .mlcompat import argmax_f, interp1, iround


def snap_map(block_map: np.ndarray) -> np.ndarray:
    """The map with what does not move removed: minus the median along columns."""
    M = block_map - np.median(block_map, axis=1, keepdims=True)
    M[M < 0] = 0
    return M


@dataclass
class Found:
    status: str = "empty"       # ok | placed | empty | off
    open: bool = False
    win: Optional[list] = None  # [c0 c1 r0 r1], 1-based
    match: float = float("nan")


def _on_blob(F, kr, kc) -> bool:
    rr = slice(max(0, kr - 1), min(F.shape[0], kr + 2))
    cc = slice(max(0, kc - 1), min(F.shape[1], kc + 2))
    return bool(F[rr, cc].any())


def _climb(W, taken, kr, kc):
    while True:
        rr = np.arange(max(0, kr - 1), min(W.shape[0], kr + 2))
        cc = np.arange(max(0, kc - 1), min(W.shape[1], kc + 2))
        sub = W[np.ix_(rr, cc)].astype(float).copy()
        sub[taken[np.ix_(rr, cc)]] = -np.inf
        a, b = argmax_f(sub)
        if sub[a, b] <= W[kr, kc]:
            return kr, kc
        kr, kc = int(rr[a]), int(cc[b])


def _step_on(t, g, p, s, dirn, lim):
    step = s[-1]
    if s.size >= 2:
        step = step + max(-0.25 * s[-1], min(0.25 * s[-1], s[-1] - s[-2]))
    while (dirn > 0 and p < lim) or (dirn < 0 and p > lim):
        p = p + dirn * step
        t.append(p)
        g.append(step)
    return t, g


def spacing_teeth(placed, lo, hi):
    """Where the spacing of the placed tabs says tabs fall over [lo hi]."""
    placed = np.unique(np.asarray(placed, float)[np.isfinite(placed)])
    if placed.size < 2:
        return np.zeros(0), np.zeros(0)
    d = np.diff(placed)
    nd = d.size
    n = np.ones(nd)
    for i in range(nd):
        j = [k for k in (i - 1, i + 1) if 0 <= k < nd]
        if j:
            n[i] = min(12, max(1, np.floor(d[i] / min(d[j]) + 0.5)))
    s = d / n
    t, g = [], []
    for i in np.flatnonzero(n >= 2):
        t.extend(placed[i] + np.arange(1, int(n[i])) * s[i])
        g.extend([s[i]] * (int(n[i]) - 1))
    t, g = _step_on(t, g, placed[-1], s, +1, hi)
    t, g = _step_on(t, g, placed[0], s[::-1], -1, lo)
    return np.array(t), np.array(g)


def _spaced_along(Wf, CC, span, placed, col, wF):
    c = col
    if len(placed) < 2:
        return c
    t, g = spacing_teeth(placed, span[0] - wF, span[1] + wF)
    if t.size == 0:
        return c
    inn = np.flatnonzero((t >= span[0]) & (t <= span[1]))
    if inn.size == 0:
        k = int(np.argmin(np.abs(t - col)))
    else:
        dist = [np.min(np.abs(np.asarray(placed) - v)) for v in t[inn]]
        k = int(inn[int(np.argmin(dist))])
    w = Wf * np.exp(-0.5 * ((CC - t[k]) / (0.3 * g[k])) ** 2)
    if w.sum() > 0:
        c = float((w * CC).sum() / w.sum())
    return c


def snap_to_peak(M, col, row, others, wF: int, wZ: int):
    """snapToPeak on the snap map M (rows x cols). Returns (c, r, Found)."""
    c, r = col, row
    found = Found()
    if wF <= 0:
        found.status = "off"
        return c, r, found
    nz, nf = M.shape
    ci = min(max(iround(col), 1), nf)
    ri = min(max(iround(row), 1), nz)
    c0, c1 = max(1, ci - wF), min(nf, ci + wF)
    e0, e1 = max(1, ci - 2 * wF), min(nf, ci + 2 * wF)
    r0, r1 = max(1, ri - wZ), min(nz, ri + wZ)
    found.win = [c0, c1, r0, r1]
    W = M[r0 - 1:r1, e0 - 1:e1].astype(float)
    fr = np.arange(e0, e1 + 1)
    in_win = (fr >= c0) & (fr <= c1)
    top = W[:, in_win].max() if in_win.any() else 0
    if not top > 0:
        return c, r, found

    placed = np.sort(np.asarray(others, float).ravel())
    rows = np.arange(r0, r1 + 1)[:, None]
    score = W * (np.exp(-0.5 * ((rows - row) / (2 * wZ / 3)) ** 2) *
                 (np.exp(-0.5 * ((fr - col) / (wF / 4)) ** 2) * in_win)[None, :])
    same = 2.0
    if placed.size >= 2:
        same = min(2.0, 0.4 * np.min(np.diff(placed)))
    CC = np.broadcast_to(fr[None, :], W.shape).astype(float)
    RR = np.broadcast_to(rows, W.shape).astype(float)
    taken = np.zeros(W.shape, bool)
    for _ in range(12):
        score[taken] = 0
        kr, kc = argmax_f(score)
        if not score[kr, kc] > 0:
            return c, r, found
        kr, kc = _climb(W, taken, int(kr), int(kc))
        Pv = W[kr, kc]
        if Pv < 0.2 * top:
            return c, r, found
        lab, _ = ndimage.label((W >= 0.75 * Pv) & ~taken, structure=np.ones((3, 3), int))
        F = lab == lab[kr, kc]
        lit = np.flatnonzero(F.any(axis=0))
        is_open = lit[0] == 0 or lit[-1] == fr.size - 1
        Wf = W * F
        c_new = float((Wf * CC).sum() / Wf.sum())
        r_new = float((Wf * RR).sum() / Wf.sum())
        if is_open:
            c_new = _spaced_along(Wf, CC, fr[lit[[0, -1]]], placed, col, wF)
        if placed.size:
            j = int(np.argmin(np.abs(placed - c_new)))
            if abs(placed[j] - c_new) < same:
                found.status = "placed"
                found.match = float(placed[j])
                if _on_blob(F, ri - r0, ci - e0):
                    return c, r, found
                taken |= F
                continue
        found.status, found.open = "ok", bool(is_open)
        return c_new, r_new, found
    return c, r, found


# --------------------------------------------------------------------------
# a set of tabs on one block / position map
# --------------------------------------------------------------------------

@dataclass
class TabSet:
    """Tabs in CLICK order (Undo undoes the last click). Each row is
    [col, row, clicked col, clicked row, 1 if put by spacing]."""
    m: np.ndarray = field(default_factory=lambda: np.zeros((0, 5)))

    def sorted(self) -> np.ndarray:
        if self.m.size == 0:
            return np.zeros((0, 5))
        return self.m[np.argsort(self.m[:, 0], kind="stable")]

    def __len__(self):
        return self.m.shape[0]

    def add(self, M, col, row, wF, wZ):
        """Add the tab clicked at (col,row). Returns (Found, index of an existing
        tab it refused on, 1-based in sorted order, or None)."""
        c, r, found = snap_to_peak(M, col, row, self.m[:, 0], wF, wZ)
        if found.status == "placed":
            ms = self.sorted()
            k = int(np.argmin(np.abs(ms[:, 0] - found.match))) + 1
            return found, k
        self.m = np.vstack([self.m, [c, r, col, row, float(found.open)]])
        self.resnap_open(M, wF, wZ)
        return found, None

    def resnap_open(self, M, wF, wZ):
        m = self.m
        for k in np.flatnonzero(m[:, 4] != 0):
            others = np.delete(m[:, 0], k)
            c, r, found = snap_to_peak(M, m[k, 2], m[k, 3], others, wF, wZ)
            if found.status == "ok" and found.open:
                m[k, 0:2] = [c, r]
        self.m = m

    def remove_nearest(self, col, row, xspan, yspan) -> bool:
        if self.m.size == 0:
            return False
        d = np.hypot((self.m[:, 0] - col) / max(xspan, 1e-12),
                     (self.m[:, 1] - row) / max(yspan, 1e-12))
        i = int(np.argmin(d))
        if d[i] > 0.05:
            return False
        self.m = np.delete(self.m, i, axis=0)
        return True

    def undo(self):
        if self.m.size:
            self.m = self.m[:-1]

    def clear(self):
        self.m = np.zeros((0, 5))


class AngleError(Exception):
    pass


def compute_angles(ms: np.ndarray, step_deg: float, zero_tab: int, neg_tab: int):
    """computeAngles: tabs sorted by column, tab numbers 1-based.
    Returns a dict with cols, rows, theta, colFit, thetaFit, direction, col0..."""
    n = ms.shape[0]
    if n < 2:
        raise AngleError(f"{n} tab(s) clicked - two are needed for an angle map")
    cols, rows = ms[:, 0], ms[:, 1]
    z = min(int(zero_tab), n)
    g = min(int(neg_tab), n)
    if z == g:
        raise AngleError("0 deg and -5 deg are the same tab. Set -5 deg to a tab on one "
                         "side of it.")
    sgn = -np.sign(g - z)
    theta = sgn * step_deg * (np.arange(1, n + 1) - z)
    if np.any(np.diff(cols) <= 0):
        raise AngleError("Two tabs are on the same column, so the fit cannot be "
                         "interpolated. Remove one of them.")
    col_fit = np.arange(int(np.ceil(cols.min())), int(np.floor(cols.max())) + 1).astype(float)
    if col_fit.size < 2:
        raise AngleError("The tabs span less than two whole columns.")
    theta_fit = interp1(cols, theta, col_fit, "pchip" if n >= 3 else "linear")
    if np.any(~np.isfinite(theta_fit)):
        raise AngleError("The interpolated angles came out non-finite.")
    direction = ("negative -> positive" if (cols[-1] - cols[0]) * sgn > 0
                 else "positive -> negative")
    col0 = float(interp1(theta[::-1] if theta[0] > theta[-1] else theta,
                         cols[::-1] if theta[0] > theta[-1] else cols, 0.0))
    return dict(cols=cols, rows=rows, theta=theta, rung=np.arange(1, n + 1),
                bySpacing=ms[:, 4] != 0, colFit=col_fit, thetaFit=theta_fit,
                direction=direction, step=step_deg, zeroRung=z, negRung=g, col0=col0)


def short_dir(direction: str) -> str:
    return "-2+" if direction == "negative -> positive" else "+2-"
