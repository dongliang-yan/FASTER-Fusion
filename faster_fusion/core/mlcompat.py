"""MATLAB built-ins that numpy/scipy do not reproduce exactly.

The reconstruction was developed and validated in MATLAB, and several of the
built-ins it leans on differ from their obvious numpy/scipy counterparts in ways
that change results, not just the last digit:

  findpeaks   filters by prominence BEFORE distance (scipy does distance first),
              and puts a flat peak on its first sample (scipy: the middle)
  round       rounds halves away from zero (numpy: to even)
  max(A(:))   returns the first maximum in COLUMN-major order
  movmean /   shrink the window at the ends rather than padding
  movmedian
  imresize3   anti-aliases when shrinking, with its own kernel and mirror padding
  imgaussfilt filter half-width ceil(2*sigma), replicate padding
  interp1     linear or pchip, NaN outside

Each is reproduced here as MATLAB documents or implements it, so the modules
that use them read like the originals and give the same numbers.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage
from scipy.interpolate import PchipInterpolator
from scipy.signal import peak_prominences


# --------------------------------------------------------------------------
# rounding and indexing
# --------------------------------------------------------------------------

def mround(x):
    """MATLAB round: halves go away from zero."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.floor(np.abs(x) + 0.5)


def iround(x) -> int:
    """MATLAB round of a scalar, as a Python int."""
    return int(mround(x))


def argmax_f(a):
    """Index (as a tuple) of the first maximum in column-major order, like
    [~,k] = max(A(:)); [i,j] = ind2sub(size(A),k)."""
    a = np.asarray(a)
    k = int(np.argmax(a.ravel(order="F")))
    return np.unravel_index(k, a.shape, order="F")


def matlab_colon(lo: float, step: float, hi: float) -> np.ndarray:
    """lo:step:hi, with MATLAB's tolerance at the far end."""
    if step == 0 or (hi - lo) / step < 0:
        return np.zeros(0)
    n = int(np.floor((hi - lo) / step + 1e-10))
    return lo + step * np.arange(n + 1)


# --------------------------------------------------------------------------
# colour
# --------------------------------------------------------------------------

_GRAY = np.array([0.298936021293776, 0.587043074451121, 0.114020904255103])


def rgb2gray_u8(rgb: np.ndarray) -> np.ndarray:
    """rgb2gray on uint8 RGB (last axis = channels) -> uint8, as MATLAB rounds."""
    g = np.asarray(rgb, dtype=np.float64) @ _GRAY
    return np.clip(np.floor(g + 0.5), 0, 255).astype(np.uint8)


def ycbcr2rgb_u8(ycc: np.ndarray) -> np.ndarray:
    """ycbcr2rgb for uint8 video-range YCbCr (MATLAB's), -> uint8 RGB."""
    y = np.asarray(ycc, dtype=np.float64)
    T = np.array([[65.481, 128.553, 24.966],
                  [-37.797, -74.203, 112.0],
                  [112.0, -93.786, -18.214]])
    off = np.array([16.0, 128.0, 128.0])
    Ti = np.linalg.inv(T) * 255.0
    rgb = (y - off) @ Ti.T
    return np.clip(np.floor(rgb + 0.5), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# moving windows
# --------------------------------------------------------------------------

def _window_bounds(n: int, k: int):
    """Start/stop of MATLAB's centred window of length k at every index,
    shrunk at the ends ('shrink', the default)."""
    if k % 2 == 1:
        b = f = (k - 1) // 2
    else:
        b, f = k // 2, k // 2 - 1
    i = np.arange(n)
    return np.maximum(i - b, 0), np.minimum(i + f, n - 1) + 1


def movmean(x, k: int, axis: int = -1, omitnan: bool = False):
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    lo, hi = _window_bounds(n, k)
    if omitnan:
        v = np.where(np.isnan(x), 0.0, x)
        c = (~np.isnan(x)).astype(float)
        cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(v, -1)], -1)
        cc = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(c, -1)], -1)
        s = cs[..., hi] - cs[..., lo]
        m = cc[..., hi] - cc[..., lo]
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(m > 0, s / np.maximum(m, 1), np.nan)
    else:
        cs = np.concatenate([np.zeros(x.shape[:-1] + (1,)), np.cumsum(x, -1)], -1)
        out = (cs[..., hi] - cs[..., lo]) / (hi - lo)
    return np.moveaxis(out, -1, axis)


def movmedian(x, k: int, axis: int = -1):
    """movmedian with shrinking ends. NaN-free input assumed."""
    x = np.asarray(x, dtype=float)
    x = np.moveaxis(x, axis, -1)
    n = x.shape[-1]
    lo, hi = _window_bounds(n, k)
    out = np.empty_like(x)
    full = (hi - lo) == k
    if full.any():
        from numpy.lib.stride_tricks import sliding_window_view
        win = sliding_window_view(x, k, axis=-1)          # (..., n-k+1, k)
        idx = np.nonzero(full)[0]
        out[..., idx] = np.median(win[..., lo[idx], :], axis=-1)
    for i in np.nonzero(~full)[0]:
        out[..., i] = np.median(x[..., lo[i]:hi[i]], axis=-1)
    return np.moveaxis(out, -1, axis)


# --------------------------------------------------------------------------
# findpeaks
# --------------------------------------------------------------------------

def _local_maxima(y: np.ndarray) -> np.ndarray:
    """findpeaks' findLocalMaxima: NaN-bookended, a flat peak at its first index,
    a peak next to a NaN is not a peak."""
    yt = np.concatenate([[np.nan], y, [np.nan]])
    it = np.arange(yt.size)
    fin = ~np.isnan(yt)
    neq = (yt[:-1] != yt[1:]) & (fin[:-1] | fin[1:])
    ineq = np.concatenate([[0], 1 + np.nonzero(neq)[0]])
    it = it[ineq]
    with np.errstate(invalid="ignore"):
        s = np.sign(np.diff(yt[it]))
        d = np.diff(s)
    imax = 1 + np.nonzero(d < 0)[0]
    return it[imax] - 1


def findpeaks(y, min_prominence: float = 0.0, min_distance: float = 0.0):
    """[pks, locs] = findpeaks(y, 'MinPeakProminence', p, 'MinPeakDistance', d)
    with x = 1:N. Returns (pks, locs) with locs 0-based.

    Prominence is measured on the finite run of samples holding each peak - a
    NaN bounds it as findpeaks' inflection points do - and peaks are filtered
    by it before the distance rule, which keeps the tallest first."""
    y = np.asarray(y, dtype=float).ravel()
    locs = _local_maxima(y)
    if locs.size and min_prominence > 0:
        prom = np.full(locs.size, np.nan)
        fin = np.isfinite(y)
        # contiguous finite runs
        edges = np.flatnonzero(np.diff(np.concatenate([[0], fin.astype(int), [0]])))
        for a, b in zip(edges[::2], edges[1::2]):
            sel = (locs >= a) & (locs < b)
            if sel.any():
                prom[sel] = peak_prominences(y[a:b], locs[sel] - a)[0]
        locs = locs[prom >= min_prominence]
    if locs.size and min_distance > 0:
        order = np.argsort(-y[locs], kind="stable")
        lt = locs[order].astype(float)
        dele = np.zeros(lt.size, bool)
        for i in range(lt.size):
            if not dele[i]:
                dele |= (lt >= lt[i] - min_distance) & (lt <= lt[i] + min_distance)
                dele[i] = False
        locs = np.sort(locs[order[~dele]])
    return y[locs], locs


# --------------------------------------------------------------------------
# interpolation and resampling
# --------------------------------------------------------------------------

def interp1(x, v, xq, method: str = "linear"):
    """interp1 with NaN outside [x(1) x(end)]; x strictly increasing."""
    x = np.asarray(x, float).ravel()
    v = np.asarray(v, float).ravel()
    xq = np.asarray(xq, float)
    if method == "pchip":
        out = PchipInterpolator(x, v, extrapolate=False)(xq)
    else:
        out = np.interp(xq, x, v, left=np.nan, right=np.nan)
    return out


def _resize_weights(in_len: int, out_len: int, antialias: bool = True):
    """imresize's contributions() for the triangle ('linear') kernel."""
    scale = out_len / in_len
    kw = 2.0
    if scale < 1 and antialias:
        kw = kw / scale
    x = np.arange(1, out_len + 1, dtype=float)
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kw / 2)
    P = int(np.ceil(kw) + 2)
    ind = left[:, None] + np.arange(P)[None, :]
    d = u[:, None] - ind
    if scale < 1 and antialias:
        w = scale * np.maximum(0.0, 1 - np.abs(scale * d))
    else:
        w = np.maximum(0.0, 1 - np.abs(d))
    w = w / w.sum(axis=1, keepdims=True)
    # mirror-symmetric padding, as imresize: aux = [1:n, n:-1:1]
    aux = np.concatenate([np.arange(1, in_len + 1), np.arange(in_len, 0, -1)])
    ind = aux[np.mod(ind - 1, aux.size).astype(int)] - 1
    keep = np.any(w != 0, axis=0)
    return ind[:, keep].astype(int), w[:, keep]


def imresize_axis(a: np.ndarray, axis: int, out_len: int, antialias: bool = True):
    """Resize one axis with imresize's weights, as one sparse matrix product:
    the same weights as applying them tap by tap, without copying the whole
    volume once per tap."""
    from scipy import sparse
    in_len = a.shape[axis]
    if out_len == in_len:
        return a
    ind, w = _resize_weights(in_len, out_len, antialias)
    rows = np.repeat(np.arange(out_len), ind.shape[1])
    W = sparse.csr_matrix((w.ravel().astype(np.float32), (rows, ind.ravel())), shape=(out_len, in_len))
    am = np.moveaxis(a, axis, 0)
    shp = am.shape
    out = W @ np.ascontiguousarray(am, dtype=np.float32).reshape(in_len, -1)
    return np.moveaxis(np.asarray(out).reshape((out_len,) + shp[1:]), 0, axis)


def imresize3(v: np.ndarray, size) -> np.ndarray:
    """imresize3(V, SZ, 'linear'), antialiased. Like imresize, the dimensions are
    resized in order of scale factor, the most shrunk first."""
    size = [int(s) for s in size]
    scales = [size[i] / v.shape[i] for i in range(3)]
    order = np.argsort(scales, kind="stable")            # smallest scale first
    out = v.astype(np.float32, copy=False)
    for ax in order:
        out = imresize_axis(out, int(ax), size[ax])
    return out


def imgaussfilt(v: np.ndarray, sigma) -> np.ndarray:
    """imgaussfilt / imgaussfilt3: filter size 2*ceil(2*sigma)+1, replicate edges."""
    sig = np.broadcast_to(np.asarray(sigma, float), (v.ndim,))
    rad = [int(np.ceil(2 * s)) for s in sig]
    return ndimage.gaussian_filter(v, sigma=list(sig), mode="nearest", radius=rad)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def conv_same_any(mask: np.ndarray, k: int) -> np.ndarray:
    """conv2(double(mask), ones(k), 'same') > 0, for odd k."""
    return ndimage.binary_dilation(mask, structure=np.ones((k, k), bool),
                                   border_value=0)


def xcorr_coeff(x: np.ndarray) -> np.ndarray:
    """xcorr(x, 'coeff') for real x: the full autocorrelation over r(0)."""
    x = np.asarray(x, float).ravel()
    r = np.correlate(x, x, mode="full")
    return r / r[x.size - 1] if r[x.size - 1] != 0 else r
