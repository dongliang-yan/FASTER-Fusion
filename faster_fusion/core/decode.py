"""Reading values back out of the scanner's colour overlays.

Colour flow   find_flow_bar + cf_decode      (findFlowBar.m, cfDecode.m)
              opaque paint: each pixel is matched to the nearest bar colour,
              interpolated between rows but never across the baseline
SWE           find_colour_bars + swi_decode  (findColourBars.m, swiDecode.m)
              a blend a*bar(t) + (1-a)*g*tint over the B-mode, inverted in the
              plane perpendicular to the B-mode's tint

Arrays are (depth, lateral, 3, position) in, (depth, lateral, position) out, as
in MATLAB. See the MATLAB files for the measurements behind each constant.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy import ndimage

from .mlcompat import conv_same_any, mround

Progress = Callable[[float, str], None]


def _noprog(v, msg):
    pass


def _null_perp(u: np.ndarray) -> np.ndarray:
    """3 x 2 orthonormal basis of the plane perpendicular to unit vector u."""
    _, _, vt = np.linalg.svd(u.reshape(1, 3))
    return vt[1:].T


def grey_tint(rgb: np.ndarray) -> np.ndarray:
    """The B-mode grey's colour ([r g b], largest 1); Siemens is slightly blue."""
    N = rgb.shape[3]
    fr = np.unique(mround(np.linspace(1, N, min(N, 5))).astype(int)) - 1
    X = []
    for f in fr:
        P = rgb[:, :, :, f].reshape(-1, 3, order="F").astype(float)
        mx, mn = P.max(1), P.min(1)
        X.append(P[(mx >= 20) & ((mx - mn) / np.maximum(mx, 1) < 0.3)])
    X = np.concatenate(X) if X else np.zeros((0, 3))
    if X.shape[0] < 100:
        return np.array([1.0, 1.0, 1.0])
    t = np.median(X / X.max(1, keepdims=True), axis=0)
    return t / t.max()


def _fill_holes(G: np.ndarray, passes: int) -> np.ndarray:
    k = np.ones((5, 5))
    for _ in range(passes):
        hole = np.isnan(G)
        if not hole.any():
            return G
        V = np.where(hole, 0.0, G)
        s = ndimage.convolve(V, k, mode="constant", cval=0.0)
        w = ndimage.convolve((~hole).astype(float), k, mode="constant", cval=0.0)
        fill = hole & (w > 0)
        if not fill.any():
            break
        G[fill] = s[fill] / w[fill]
    G[np.isnan(G)] = 0
    return G


def _regions(S: np.ndarray):
    """regionprops BoundingBox + Area over 8-connected components of S, as
    (r0, r1, c0, c1, area) 0-based inclusive."""
    lab, n = ndimage.label(S, structure=np.ones((3, 3), int))
    out = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None:
            continue
        area = int((lab[sl] == i).sum())
        out.append((sl[0].start, sl[0].stop - 1, sl[1].start, sl[1].stop - 1, area))
    return out


# ==========================================================================
# colour flow
# ==========================================================================

@dataclass
class FlowBar:
    rows: np.ndarray
    cols: np.ndarray
    colors: np.ndarray     # K x 3, TOP to BOTTOM
    zero: int              # baseline sits under this 1-based bar row


def find_flow_bar(screen: np.ndarray, exclude: Optional[np.ndarray] = None) -> Optional[FlowBar]:
    I = screen.astype(float)
    if exclude is None:
        exclude = np.zeros(I.shape[:2], bool)
    mx, mn = I.max(2), I.min(2)
    sat = (mx - mn) / np.maximum(mx, 1)
    S = (sat > 0.35) & (mx > 50) & ~exclude          # the cyan end is only 0.41
    cands = []
    for r0, r1, c0, c1, area in _regions(S):
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 40 or h < 4 * w or w < 2 or area < 0.8 * w * h:
            continue
        cols = np.arange(c0, c1 + 1)
        if cols.size >= 5:
            cols = cols[1:-1]
        rows = np.arange(r0, r1 + 1)
        rows = rows[S[np.ix_(rows, cols)].all(axis=1)]
        if rows.size < 40:
            continue
        rows = np.arange(rows[0], rows[-1] + 1)
        C = I[np.ix_(rows, cols)].mean(axis=1)
        warm = C[:, 0] > C[:, 2] + 40
        cool = C[:, 2] > C[:, 0] + 40
        if warm.sum() < 10 or cool.sum() < 10:
            continue
        step = np.sqrt((np.diff(C, axis=0) ** 2).sum(1))
        z = int(np.argmax(step)) + 1
        cands.append((rows.size, FlowBar(rows=rows, cols=cols, colors=C, zero=z)))
    if not cands:
        return None
    return max(cands, key=lambda c: c[0])[1]


def _match_bar(P, C, z):
    """Nearest bar colour, fractional row (1-based) and misfit; never across the
    baseline."""
    n, K = P.shape[0], C.shape[0]
    cc = (C ** 2).sum(1)
    kf = np.zeros(n)
    res = np.zeros(n)
    chunk = max(1, int(3e6 // K))
    for s in range(0, n, chunk):
        Q = P[s:s + chunk]
        D = np.maximum((Q ** 2).sum(1)[:, None] - 2 * Q @ C.T + cc[None, :], 0)
        k = np.argmin(D, axis=1)                 # 0-based; first minimum
        lin = np.arange(Q.shape[0])
        d0 = D[lin, k]
        km, kp = np.maximum(k - 1, 0), np.minimum(k + 1, K - 1)
        Dm, Dp = D[lin, km], D[lin, kp]
        den = Dm - 2 * d0 + Dp
        k1 = k + 1                               # 1-based
        inn = (k1 > 1) & (k1 < K) & (k1 != z) & (k1 != z + 1) & (den > 0)
        dl = np.zeros(k.size)
        dl[inn] = 0.5 * (Dm[inn] - Dp[inn]) / den[inn]
        kf[s:s + chunk] = k1 + np.clip(dl, -0.5, 0.5)
        res[s:s + chunk] = np.sqrt(d0)
    return kf, res


def _bar_value(kf, K, z):
    up = kf <= z + 0.5
    v = np.where(up, (z + 0.5 - kf) / z, -(kf - z - 0.5) / (K - z))
    return np.clip(v, -1, 1)


@dataclass
class CFDecoded:
    v: np.ndarray          # nz x nx x N float32, -1..1 of the bar, NaN = no flow
    grey: np.ndarray       # nz x nx x N uint8
    mask: np.ndarray       # nz x nx x N bool
    box: list              # [r0 r1 c0 c1] 1-based inclusive, in the panel
    static: np.ndarray
    tint: np.ndarray
    resid: float
    flow_frac: np.ndarray


def cf_decode(rgb: np.ndarray, bar: FlowBar, box=None, edge_trim: int = 2,
              max_resid: float = 24, min_chroma: float = 20, static_frac: float = 0.9,
              progress: Progress = _noprog) -> CFDecoded:
    """box is [r0 r1 c0 c1], 1-based inclusive in the panel, as the MATLAB code."""
    C = bar.colors.astype(float)
    K, z = C.shape[0], int(bar.zero)
    if K < 4 or not (1 <= z < K):
        raise ValueError("The flow bar must be K x 3 with a baseline inside it.")
    nz, nx, _, N = rgb.shape
    tint = grey_tint(rgb)
    E = _null_perp(tint / np.linalg.norm(tint))
    tt = float(tint @ tint)

    chroma = np.zeros((nz, nx, N), bool)
    for f in range(N):
        P = rgb[:, :, :, f].reshape(-1, 3).astype(float)
        chroma[:, :, f] = (((P @ E) ** 2).sum(1) > min_chroma ** 2).reshape(nz, nx)
    static = chroma.mean(axis=2) >= static_frac
    static = conv_same_any(static, 3)

    if box is not None:
        b = [int(round(v)) for v in box]
        tr = edge_trim
        b = [max(1, b[0] + tr), min(nz, b[1] - tr), max(1, b[2] + tr), min(nx, b[3] - tr)]
        if b[1] < b[0] or b[3] < b[2]:
            raise ValueError("The colour box is empty once its edges are trimmed.")
    else:
        b = [1, nz, 1, nx]
    in_box = np.zeros((nz, nx), bool)
    in_box[b[0] - 1:b[1], b[2] - 1:b[3]] = True
    look = in_box & ~static

    v_out = np.full((nz, nx, N), np.nan, np.float32)
    grey = np.zeros((nz, nx, N), np.uint8)
    mask = np.zeros((nz, nx, N), bool)
    res50 = np.full(N, np.nan)
    for f in range(N):
        P = rgb[:, :, :, f].reshape(-1, 3).astype(float)
        g = (P @ tint) / tt
        cand = np.flatnonzero((chroma[:, :, f] & look).ravel())
        v = np.full(nz * nx, np.nan)
        ok = np.zeros(nz * nx, bool)
        if cand.size:
            kf, res = _match_bar(P[cand], C, z)
            good = res <= max_resid
            v[cand[good]] = _bar_value(kf[good], K, z)
            ok[cand[good]] = True
            if good.any():
                res50[f] = np.median(res[good])
        hole = chroma[:, :, f] | ok.reshape(nz, nx)
        hole = conv_same_any(hole, 3)
        G = g.reshape(nz, nx).copy()
        G[hole] = np.nan
        G = _fill_holes(G, 40)
        v_out[:, :, f] = v.reshape(nz, nx)
        grey[:, :, f] = np.clip(np.floor(G + 0.5), 0, 255).astype(np.uint8)
        mask[:, :, f] = ok.reshape(nz, nx)
        if (f + 1) % 5 == 0 or f == N - 1:
            progress((f + 1) / N, f"Decoding position {f + 1} of {N}")

    if box is None and mask.any():
        rr, cc = np.nonzero(mask.any(axis=2))
        b = [int(rr.min()) + 1, int(rr.max()) + 1, int(cc.min()) + 1, int(cc.max()) + 1]
    bx = mask[b[0] - 1:b[1], b[2] - 1:b[3], :]
    flow_frac = bx.sum(axis=(0, 1)) / max(1, (b[1] - b[0] + 1) * (b[3] - b[2] + 1))
    return CFDecoded(v=v_out, grey=grey, mask=mask, box=b, static=static, tint=tint,
                     resid=float(np.nanmedian(res50)) if np.isfinite(res50).any() else float("nan"),
                     flow_frac=flow_frac)


# ==========================================================================
# shear-wave elastography
# ==========================================================================

@dataclass
class ColourBar:
    rows: np.ndarray
    cols: np.ndarray
    colors: np.ndarray     # K x 3, TOP to BOTTOM
    has_blue: bool


def find_colour_bars(screen: np.ndarray, exclude: Optional[np.ndarray] = None) -> list:
    I = screen.astype(float)
    if exclude is None:
        exclude = np.zeros(I.shape[:2], bool)
    mx, mn = I.max(2), I.min(2)
    sat = (mx - mn) / np.maximum(mx, 1)
    S = (sat > 0.6) & (mx > 90) & ~exclude
    bars = []
    for r0, r1, c0, c1, area in _regions(S):
        h, w = r1 - r0 + 1, c1 - c0 + 1
        if h < 40 or h < 4 * w or w < 2 or area < 0.8 * w * h:
            continue
        cols = np.arange(c0, c1 + 1)
        if cols.size >= 5:
            cols = cols[1:-1]
        rows = np.arange(r0, r1 + 1)
        rows = rows[S[np.ix_(rows, cols)].all(axis=1)]
        if rows.size < 40:
            continue
        rows = np.arange(rows[0], rows[-1] + 1)
        C = I[np.ix_(rows, cols)].mean(axis=1)
        blue = np.any((C[:, 2] > 150) & (C[:, 2] > C[:, 0] + 60) & (C[:, 2] > C[:, 1] + 60))
        bars.append(ColourBar(rows=rows, cols=cols, colors=C, has_blue=bool(blue)))
    bars.sort(key=lambda b: b.cols[0])
    return bars


def _fit_pixels(P, P2, C, C2, cc, tint, tt, a):
    n, K = P2.shape[0], C2.shape[0]
    kf = np.zeros(n)
    chunk = max(1, int(3e6 // K))
    for s in range(0, n, chunk):
        Q = P2[s:s + chunk]
        D = (Q ** 2).sum(1)[:, None] - 2 * a * (Q @ C2.T) + (a ** 2) * cc[None, :]
        k = np.argmin(D, axis=1)
        lin = np.arange(Q.shape[0])
        d0 = D[lin, k]
        km, kp = np.maximum(k - 1, 0), np.minimum(k + 1, K - 1)
        Dm, Dp = D[lin, km], D[lin, kp]
        den = Dm - 2 * d0 + Dp
        k1 = k + 1
        inn = (k1 > 1) & (k1 < K) & (den > 0)
        dl = np.zeros(k.size)
        dl[inn] = 0.5 * (Dm[inn] - Dp[inn]) / den[inn]
        kf[s:s + chunk] = k1 + np.clip(dl, -0.5, 0.5)
    k0 = np.floor(kf).astype(int)
    k1 = np.minimum(k0 + 1, K)
    w = kf - k0
    k0 = np.maximum(k0, 1)
    Ck = C[k0 - 1] * (1 - w)[:, None] + C[k1 - 1] * w[:, None]
    base = P - a * Ck
    g = np.clip((base @ tint) / ((1 - a) * tt), 0, 255)
    res = np.sqrt(((base - (1 - a) * g[:, None] * tint[None, :]) ** 2).sum(1))
    return kf, g, res


def _fit_alpha(rgb, C, E, C2, cc, tint, tt, min_chroma):
    N = rgb.shape[3]
    fr = np.unique(mround(np.linspace(1, N, min(N, 6))).astype(int)) - 1
    P = []
    for f in fr:
        Pf = rgb[:, :, :, f].reshape(-1, 3, order="F").astype(float)
        P.append(Pf[((Pf @ E) ** 2).sum(1) > min_chroma ** 2])
    P = np.concatenate(P)
    if P.shape[0] < 50:
        return 0.6
    if P.shape[0] > 20000:
        P = P[mround(np.linspace(1, P.shape[0], 20000)).astype(int) - 1]
    P2 = P @ E
    score = lambda a: np.median(_fit_pixels(P, P2, C, C2, cc, tint, tt, a)[2])
    aa = np.arange(0.20, 0.95 + 1e-9, 0.05)
    s = np.array([score(a) for a in aa])
    i = int(np.argmin(s))
    aa = np.arange(max(0.05, aa[i] - 0.05), min(0.97, aa[i] + 0.05) + 1e-9, 0.01)
    s = np.array([score(a) for a in aa])
    i = int(np.argmin(s))
    a = aa[i]
    if 0 < i < aa.size - 1:
        den = s[i - 1] - 2 * s[i] + s[i + 1]
        if den > 0:
            a = a + 0.01 * 0.5 * (s[i - 1] - s[i + 1]) / den
    return float(a)


@dataclass
class SWEDecoded:
    t: np.ndarray          # nz x nx x N float32, 0 bottom .. 1 top of bar, NaN no value
    grey: np.ndarray
    mask: np.ndarray
    alpha: float
    tint: np.ndarray
    resid: float
    box: Optional[list]    # 1-based inclusive
    recovered: float


def swi_decode(rgb, colors, alpha=None, max_resid=12, min_chroma=12, edge_trim=1,
               in_box_all=True, progress: Progress = _noprog) -> SWEDecoded:
    C = np.asarray(colors, float)
    K = C.shape[0]
    nz, nx, _, N = rgb.shape
    tint = grey_tint(rgb)
    E = _null_perp(tint / np.linalg.norm(tint))
    C2 = C @ E
    cc = (C2 ** 2).sum(1)
    tt = float(tint @ tint)
    if alpha is None:
        progress(0, "Measuring the overlay opacity")
        a = _fit_alpha(rgb, C, E, C2, cc, tint, tt, min_chroma)
    else:
        a = float(alpha)
    if not 0 < a < 1:
        raise ValueError(f"The overlay opacity came out as {a}; it must be between 0 and 1.")

    t_out = np.full((nz, nx, N), np.nan, np.float32)
    grey = np.zeros((nz, nx, N), np.uint8)
    mask = np.zeros((nz, nx, N), bool)
    chroma = np.zeros((nz, nx, N), bool)
    res50 = np.full(N, np.nan)
    for f in range(N):
        P = rgb[:, :, :, f].reshape(-1, 3).astype(float)
        g = (P @ tint) / tt
        tv = np.full(P.shape[0], np.nan)
        ok = np.zeros(P.shape[0], bool)
        P2 = P @ E
        ch = (P2 ** 2).sum(1) > min_chroma ** 2
        cand = np.flatnonzero(ch)
        if cand.size:
            kf, gc, res = _fit_pixels(P[cand], P2[cand], C, C2, cc, tint, tt, a)
            good = res <= max_resid
            tv[cand] = (K - kf) / (K - 1)
            g[cand[good]] = gc[good]
            ok[cand[good]] = True
            bad = np.zeros(P.shape[0], bool)
            bad[cand[~good]] = True
            bad = conv_same_any(bad.reshape(nz, nx), 5).ravel()
            bad[cand[good]] = False
            g[bad] = np.nan
            if good.any():
                res50[f] = np.median(res[good])
        G = g.reshape(nz, nx)
        if np.isnan(G).any():
            G = _fill_holes(G, 4)
        t_out[:, :, f] = tv.reshape(nz, nx)
        grey[:, :, f] = np.clip(np.floor(np.clip(G, 0, 255) + 0.5), 0, 255).astype(np.uint8)
        mask[:, :, f] = ok.reshape(nz, nx)
        chroma[:, :, f] = ch.reshape(nz, nx)
        if (f + 1) % 5 == 0 or f == N - 1:
            progress((f + 1) / N, f"Decoding position {f + 1} of {N}")

    box, recovered = None, 0.0
    cnt = mask.sum(axis=2)
    if cnt.any():
        core = cnt >= 0.5 * cnt.max()
        rr, cc2 = np.nonzero(core)
        tr = edge_trim
        bx = [int(rr.min()) + 1 + tr, int(rr.max()) + 1 - tr, int(cc2.min()) + 1 + tr, int(cc2.max()) + 1 - tr]
        if bx[1] >= bx[0] and bx[3] >= bx[2]:
            inb = np.zeros((nz, nx), bool)
            inb[bx[0] - 1:bx[1], bx[2] - 1:bx[3]] = True
            if in_box_all:
                keep = (mask | chroma) & inb[:, :, None]
                recovered = float((keep & ~mask).sum() / max(1, inb.sum() * N))
                mask = keep
            else:
                mask = mask & inb[:, :, None]
            box = bx
    t_out[~mask] = np.nan
    return SWEDecoded(t=t_out, grey=grey, mask=mask, alpha=a, tint=tint,
                      resid=float(np.nanmedian(res50)) if np.isfinite(res50).any() else float("nan"),
                      box=box, recovered=recovered)
