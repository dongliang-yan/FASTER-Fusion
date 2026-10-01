"""Cutting a continuous back-and-forth cine into single sweeps.

find_sweep_blocks  - findSweepBlocks.m, used by B-Mode
beamscope_sweep    - beamscopeSweep.m, used by the cine calibration

Both find the turnarounds as axes of TIME SYMMETRY: where the mirror reverses,
frames t-k and t+k see the same angle, and nowhere else in a sweep is symmetric
like that. See the MATLAB files for the measurements behind the choices.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mlcompat import findpeaks, iround, movmean, movmedian, mround, xcorr_coeff


class SweepError(Exception):
    pass


@dataclass
class Blocks:
    blocks: np.ndarray       # nB x 2, 1-based [start end] frames, as in MATLAB
    turns: np.ndarray        # 1-based turnaround frames
    T: int
    L: float
    P: np.ndarray
    lag_profile: np.ndarray
    mirror_band: np.ndarray
    slope: np.ndarray
    cls: str


def find_sweep_blocks(M: np.ndarray, iz_mirror: int) -> Blocks:
    """M is axial x frames; iz_mirror is 1-based, as in MATLAB."""
    M = np.asarray(M, float)
    Nf = M.shape[1]
    X = M - M.mean(axis=1, keepdims=True)
    X = X / np.maximum(X.std(axis=1, ddof=1, keepdims=True), np.finfo(float).eps)
    X = X / np.maximum(np.linalg.norm(X, axis=0, keepdims=True), np.finfo(float).eps)
    S = X.T @ X

    max_lag = Nf - 60
    if max_lag < 20:
        raise SweepError(f"Need at least 80 frames, got {Nf}.")
    lag = np.array([np.mean(np.diagonal(S, lg)) for lg in range(1, max_lag + 1)])
    pk_val, pk_loc = findpeaks(lag, min_prominence=0.03, min_distance=20)
    if pk_loc.size == 0:
        raise SweepError("No repeating pattern in this recording, so the sweep length is "
                         "unknown.")
    T = int(np.min(pk_loc[pk_val >= 0.9 * pk_val.max()]) + 1)    # lag = index + 1
    L = T / 2

    K = max(8, iround(0.2 * L))
    P = np.full(Nf, np.nan)
    need = iround(0.6 * K)
    for t in range(1, Nf + 1):
        kmax = min(K, t - 1, Nf - t)
        if kmax >= need:
            k = np.arange(1, kmax + 1)
            P[t - 1] = np.mean(S[t - 1 - k, t - 1 + k])
    P = movmean(P, 5, omitnan=True)

    _, turns = findpeaks(P, min_prominence=0.02, min_distance=iround(0.6 * L))
    turns = turns + 1
    if turns.size < 2:
        raise SweepError(f"Found {turns.size} turnaround(s), so no complete sweep could be "
                         "cut out. The recording may hold less than one full period.")
    blocks = np.column_stack([turns[:-1], turns[1:]])

    nB = blocks.shape[0]
    A = M[:max(1, iz_mirror - 1), :]
    W = A - np.median(A)
    W[W < 0] = 0
    z = np.arange(1, A.shape[0] + 1)
    band = movmean((z @ W) / np.maximum(W.sum(axis=0), np.finfo(float).eps), 7)
    slope = np.zeros(nB)
    for b in range(nB):
        idx = np.arange(blocks[b, 0], blocks[b, 1] + 1) - 1
        slope[b] = np.polyfit(np.arange(1, idx.size + 1), band[idx], 1)[0]
    cls = "".join("B" if s < 0 else "A" for s in slope)
    return Blocks(blocks=blocks, turns=turns, T=T, L=L, P=P, lag_profile=lag,
                  mirror_band=band, slope=slope, cls=cls)


@dataclass
class Sweep:
    Tmech: float
    L: float
    turns: np.ndarray            # 1-based, sub-frame
    spans: list                  # list of 1-based frame index arrays
    sym_curve: np.ndarray
    sym_frame: np.ndarray
    quality: float
    turns_seen: np.ndarray
    Tprior: int
    mode: str = "cine"


def beamscope_sweep(temp1: np.ndarray, depth_range=None, clutter_win: int = 31,
                    half_width=None, min_quality: float = 0.4) -> Sweep:
    temp1 = np.asarray(temp1, float)
    Nz, Nf = temp1.shape

    zc = np.arange(1, Nz + 1)
    Wc = temp1 - temp1.min()
    cen = movmean((zc @ Wc) / np.maximum(Wc.sum(axis=0), np.finfo(float).eps), 5)
    ac = xcorr_coeff(cen - cen.mean())[Nf - 1:]
    ac_pk, ac_loc = findpeaks(ac, min_prominence=0.15, min_distance=30)
    if ac_loc.size == 0:
        raise SweepError(f"Could not measure the sweep period from {Nf} frames.")
    # MATLAB takes the 1-based index into ac - which starts at lag 0 - as the
    # period, so it is one more than the lag. Kept: the rest was tuned on it.
    Tprior = int(ac_loc[int(np.argmax(ac_pk))]) + 1
    Lprior = iround(Tprior / 2)

    S = temp1 - movmedian(temp1, clutter_win, axis=1)
    S[S < 0] = 0
    if depth_range is not None:
        zlo = max(1, int(np.floor(depth_range[0])))
        zhi = min(Nz, int(np.ceil(depth_range[1])))
        S = S[zlo - 1:zhi, :]

    w = Lprior if half_width is None else half_width
    w = max(10, min(w, (Nf - 1) // 2))
    cand = np.arange(w + 1, Nf - w + 1)             # 1-based frames
    if cand.size == 0:
        raise SweepError(f"Recording is {Nf} frames, too short for a +/-{w} frame "
                         "symmetry test.")
    sym = np.full(cand.size, np.nan)
    for i, f0 in enumerate(cand):
        A = S[:, f0 - w - 1:f0 - 1]
        Bm = S[:, f0:f0 + w][:, ::-1]
        a = A.ravel(order="F") - A.mean()
        b = Bm.ravel(order="F") - Bm.mean()
        den = np.sqrt((a @ a) * (b @ b))
        if den > 0:
            sym[i] = (a @ b) / den

    pk, loc = findpeaks(sym, min_prominence=0.05, min_distance=max(10, iround(0.6 * Lprior)))
    good = pk >= min_quality
    if not good.any():
        raise SweepError(f"No frame looks like a turnaround (best symmetry "
                         f"{max(np.r_[pk, 0]):.2f}, need {min_quality:.2f}).")
    loc, pk = loc[good], pk[good]

    t_ref = np.zeros(loc.size)
    for i, j in enumerate(loc):
        if 0 < j < sym.size - 1 and np.all(np.isfinite(sym[j - 1:j + 2])):
            y1, y2, y3 = sym[j - 1], sym[j], sym[j + 1]
            den = y1 - 2 * y2 + y3
            t_ref[i] = cand[j] + (0.5 * (y1 - y3) / den if den != 0 else 0.0)
        else:
            t_ref[i] = cand[j]

    L = float(np.mean(np.diff(t_ref))) if t_ref.size >= 2 else float(Lprior)
    k_idx = mround((t_ref - t_ref[0]) / L)
    phase = float(np.mean(t_ref - k_idx * L))
    k_lo = int(np.floor((1 - phase) / L))
    k_hi = int(np.ceil((Nf - phase) / L))
    turns = phase + np.arange(k_lo, k_hi + 1) * L
    turns = turns[(turns > -L / 2) & (turns < Nf + L / 2)]

    spans = []
    for i in range(turns.size - 1):
        a = int(np.ceil(turns[i]))
        b = int(np.floor(turns[i + 1]))
        if a >= 1 and b <= Nf and (b - a + 1) >= 0.8 * L:
            spans.append(np.arange(a, b + 1))
    return Sweep(Tmech=2 * L, L=L, turns=turns, spans=spans, sym_curve=sym,
                 sym_frame=cand, quality=float(pk.max()), turns_seen=t_ref, Tprior=Tprior)


def auto_spans(temp1: np.ndarray) -> Sweep:
    """Calibration_GUI_v1's autoSpans: a second pass with the clutter window set
    from the period the first pass measured, kept if it is better."""
    sw = beamscope_sweep(temp1)
    try:
        cw = max(5, iround(iround(sw.Tmech / 2)))
        if cw % 2 == 0:
            cw += 1
        sw2 = beamscope_sweep(temp1, clutter_win=cw)
        if sw2.quality > sw.quality:
            sw = sw2
    except SweepError:
        pass
    return sw
