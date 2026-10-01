"""Scan conversion of the mirror sweep into the elevational-axial sector.

SectorRecon.m, with one change of shape: the MATLAB function is called once per
lateral line and rebuilds the Cartesian grid and the interpolation every time.
Here the grid and the bilinear weights are built ONCE for a set of angles and
ranges (SectorGrid) and applied to every lateral line at once - the same
numbers, a fraction of the work.

The samples sit on a regular (range x angle) grid, so each Cartesian point is
looked up at (hypot(z,y), atan2(y,z)) by bilinear interpolation in (angle,
range), exactly as interp2(..., 'linear', NaN); outside the swept sector the
result is 0.
"""
from __future__ import annotations

import numpy as np

from .mlcompat import matlab_colon


class SectorError(Exception):
    pass


class SectorGrid:
    """Cartesian grid and interpolation weights for one sweep.

    angles  degrees (or radians if all |a| <= 1, as SectorRecon), one per column
    d       ranges below the mirror, mm, one per row
    lam     wavelength, mm; the grid pitch is lam/4
    """

    def __init__(self, angles, d, lam: float):
        a = np.asarray(angles, float).ravel()
        if np.max(np.abs(a)) > 1:
            a = np.deg2rad(a)
        d = np.asarray(d, float).ravel()
        if a.size < 2:
            raise SectorError(f"Need at least 2 angles, got {a.size}.")
        self.n_ang = a.size
        self.n_d = d.size
        amin, amax = a.min(), a.max()
        A, D = np.meshgrid(a, d)
        z = D * np.cos(A)
        y = D * np.sin(A)
        step = lam / 4
        self.zz = matlab_colon(z.min(), step, z.max())
        self.yy = matlab_colon(y.min(), step, y.max())
        Y, Z = np.meshgrid(self.yy, self.zz)
        ang = np.arctan2(Y, Z)
        rad = np.hypot(Y, Z)
        mask = (ang < amin) | (ang > amax) | (rad > d[-1]) | (rad < d[0])

        order = np.argsort(a, kind="stable")
        self.order = order
        asrt = a[order]
        # bilinear weights on the (range, sorted angle) grid; interp2 includes
        # both end points, and anything outside it is NaN -> 0
        ok = ~mask & (ang >= asrt[0]) & (ang <= asrt[-1]) & (rad >= d[0]) & (rad <= d[-1])
        ia = np.clip(np.searchsorted(asrt, ang, side="right") - 1, 0, asrt.size - 2)
        idd = np.clip(np.searchsorted(d, rad, side="right") - 1, 0, d.size - 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            wa = (ang - asrt[ia]) / (asrt[ia + 1] - asrt[ia])
            wd = (rad - d[idd]) / (d[idd + 1] - d[idd])
        self.shape = Z.shape
        self.ok = ok.ravel()
        sel = np.flatnonzero(self.ok)
        self.sel = sel
        ia, idd, wa, wd = ia.ravel()[sel], idd.ravel()[sel], wa.ravel()[sel], wd.ravel()[sel]
        self.i00 = idd * asrt.size + ia          # flat indices into (d, sorted angle)
        self.w00 = (1 - wd) * (1 - wa)
        self.w01 = (1 - wd) * wa
        self.w10 = wd * (1 - wa)
        self.w11 = wd * wa
        self.na = asrt.size

    def apply(self, src: np.ndarray) -> np.ndarray:
        """src is (range, angle) or (range, angle, k) in the ORIGINAL angle order.
        Returns (len(zz), len(yy)) or (len(zz), len(yy), k); 0 outside."""
        src = np.asarray(src)
        two_d = src.ndim == 2
        s = src[:, self.order] if two_d else src[:, self.order, :]
        s = s.reshape(self.n_d * self.na, -1).astype(np.float64, copy=False)
        v = (self.w00[:, None] * s[self.i00] + self.w01[:, None] * s[self.i00 + 1] +
             self.w10[:, None] * s[self.i00 + self.na] + self.w11[:, None] * s[self.i00 + self.na + 1])
        out = np.zeros((self.ok.size, v.shape[1]))
        out[self.sel] = v
        out[~np.isfinite(out)] = 0          # NaN in the source -> no data, as SectorRecon
        out = out.reshape(self.shape + (v.shape[1],))
        return out[:, :, 0] if two_d else out


def sector_recon(source, frame_selection, angles, d, lam, i_lat):
    """Drop-in for SectorRecon(source, frameSelection, angles, d, lambda, i_lat):
    source is (range, lateral, frame); i_lat and frame_selection are 1-based.
    Returns (recon, zz, yy, mask)."""
    g = SectorGrid(angles, d, lam)
    fs = np.asarray(frame_selection, int).ravel() - 1
    src = np.asarray(source)[:, int(i_lat) - 1, :][:, fs]
    return g.apply(src), g.zz, g.yy, ~g.ok.reshape(g.shape)
