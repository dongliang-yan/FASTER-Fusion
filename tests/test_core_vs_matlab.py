"""The Python core against the MATLAB originals, on the real data.

tests/ref/*.mat were written by the MATLAB code (see tests/make_reference.m) on
the sets in ../Data. Each test runs the Python port on the same input and
compares. Skipped if the data folder is not there.

MATLAB indices are 1-based; where an index is compared it is converted here.
"""
import os

import numpy as np
import pytest
from scipy.io import loadmat

from faster_fusion.core import decode, io, recon, sector, sweep, tabs

HERE = os.path.dirname(__file__)
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
DATA = os.path.join(PKG, "Data")
REF = os.path.join(HERE, "ref")
CF = os.path.join(DATA, "3_ColorFlow_Stepped", "2026-09-28_ColorFlow")
SW = os.path.join(DATA, "2_SWE_Stepped", "2026-09-23_SWE")
CINE = os.path.join(DATA, "1_BMode_Cine", "2026-09-14_Siemens_10Vpp")
IQ = os.path.join(DATA, "1_BMode_Cine", "FASTER-AIR_GE9LD_Verasonics_IQ", "Phantom_Dark_v3", "IQ_2.dat")

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA), reason="no Data folder")


def ref(name):
    return loadmat(os.path.join(REF, name), squeeze_me=True, struct_as_record=False)


def files_in(d):
    return sorted(os.path.join(d, n) for n in os.listdir(d) if not n.startswith("."))


# ---------------------------------------------------------------- colour flow

@pytest.fixture(scope="module")
def cf_stack():
    return io.load_position_stack(os.path.join(CF, "Phantom_-2+_v2"))


@pytest.fixture(scope="module")
def cf_decoded(cf_stack):
    ps = cf_stack
    P = ps.panels[ps.flow_panel]
    excl = np.zeros(ps.screen.shape[:2], bool)
    excl[np.ix_(P.rows, P.cols)] = True
    bar = decode.find_flow_bar(ps.screen, excl)
    fb = ps.flow_box
    box = [fb[0] - P.rows[0] + 1, fb[1] - P.rows[0] + 1, fb[2] - P.cols[0] + 1, fb[3] - P.cols[0] + 1]
    return bar, decode.cf_decode(P.rgb, bar, box=box)


def test_cf_loader(cf_stack):
    r = ref("cf.mat")
    ps = cf_stack
    assert list(ps.files) == list(r["files"])
    assert ps.order_by == r["orderBy"]
    P = ps.panels[0]
    np.testing.assert_array_equal(P.rows + 1, r["rows"])
    np.testing.assert_array_equal(P.cols + 1, r["cols"])
    assert ps.flow_panel + 1 == r["flowPanel"]
    np.testing.assert_array_equal(np.array(ps.flow_box) + 1, r["flowBox"])
    assert P.dz_mm == pytest.approx(r["dz"]) and P.dx_mm == pytest.approx(r["dx"])
    np.testing.assert_array_equal(P.rgb[:, :, :, 47], r["rgb48"])
    assert P.rgb.astype(float).sum() == r["rgbsum"]


def test_cf_bar_and_decode(cf_decoded):
    r = ref("cf.mat")
    bar, d = cf_decoded
    np.testing.assert_array_equal(bar.rows + 1, r["barRows"])
    np.testing.assert_array_equal(bar.cols + 1, r["barCols"])
    np.testing.assert_allclose(bar.colors, r["barColors"])
    assert bar.zero == r["barZero"]
    assert d.box == list(r["box"])
    np.testing.assert_array_equal(d.static, r["static"].astype(bool))
    np.testing.assert_allclose(d.tint, r["tint"], rtol=1e-12)
    assert d.resid == pytest.approx(r["resid"], rel=1e-9)
    np.testing.assert_allclose(d.flow_frac, r["flowFrac"], rtol=1e-12)
    assert int(d.mask.sum()) == r["nmask"]
    k = np.atleast_1d(r["k"]) - 1
    np.testing.assert_array_equal(d.mask[:, :, k], r["mask"].astype(bool))
    np.testing.assert_allclose(d.v[:, :, k], r["v"], atol=1e-6, equal_nan=True)
    np.testing.assert_array_equal(d.grey[:, :, k], r["grey"])
    # MATLAB summed a single array in single precision: good to ~1e-5 only
    assert np.nansum(d.v.astype(float)) == pytest.approx(r["vsum"], rel=2e-5)


# ---------------------------------------------------------------- SWE

def test_swe_loader_and_decode():
    r = ref("swe.mat")
    ps = io.load_position_stack(os.path.join(SW, "Phantom"))
    assert len(ps.panels) == r["nPanels"]
    np.testing.assert_array_equal(ps.panels[0].rows + 1, r["rows1"])
    np.testing.assert_array_equal(ps.panels[0].cols + 1, r["cols1"])
    P = ps.panels[-1]
    np.testing.assert_array_equal(P.cols + 1, r["cols2"])
    excl = np.zeros(ps.screen.shape[:2], bool)
    for p in ps.panels:
        excl[np.ix_(p.rows, p.cols)] = True
    bars = decode.find_colour_bars(ps.screen, excl)
    assert len(bars) == r["nBars"]
    b = bars[-1]
    np.testing.assert_allclose(b.colors, r["barColors"])
    assert b.has_blue == bool(r["barBlue"])
    d = decode.swi_decode(P.rgb, b.colors)
    assert d.alpha == pytest.approx(r["alpha"], abs=1e-9)
    assert d.resid == pytest.approx(r["resid"], rel=1e-6)
    assert d.box == list(r["box"])
    assert d.recovered == pytest.approx(r["recovered"], rel=1e-9)
    assert int(d.mask.sum()) == r["nmask"]
    k = np.atleast_1d(r["k"]) - 1
    np.testing.assert_allclose(d.t[:, :, k], r["t"], atol=1e-5, equal_nan=True)
    np.testing.assert_array_equal(d.grey[:, :, k], r["grey"])


# ---------------------------------------------------------------- cine

@pytest.fixture(scope="module")
def cine_cal_stack():
    return io.load_scan_stack(files_in(os.path.join(CINE, "Calibration"))[0], crop=(369, 656))


def test_cine_loader(cine_cal_stack):
    r = ref("cine.mat")
    s = cine_cal_stack
    assert list(s.B.shape) == list(r["size"])
    assert s.dz_mm == pytest.approx(r["dz"])
    # JPEG: a different decoder from MATLAB's, so allow a grey level
    d1 = np.abs(s.B[:, :, 0].astype(int) - r["f1"].astype(int))
    print("cine frame 1: max |diff|", d1.max(), " mean", d1.mean())
    assert d1.mean() < 0.5 and d1.max() <= 3
    assert np.abs(s.B.mean(axis=2) - r["Bmean"]).max() < 0.5


def test_beamscope_sweep():
    r = ref("cine.mat")
    temp1 = r["temp1"].astype(float)            # MATLAB's own map: tests the algorithm alone
    sw = sweep.beamscope_sweep(temp1)
    assert sw.Tprior == r["Tprior"]
    np.testing.assert_allclose(sw.turns_seen, np.atleast_1d(r["turnsSeen"]), atol=1e-9)
    np.testing.assert_allclose(sw.turns, np.atleast_1d(r["turns"]), atol=1e-9)
    assert sw.quality == pytest.approx(r["quality"], abs=1e-12)
    assert len(sw.spans) == r["nspans"]
    np.testing.assert_array_equal(sw.spans[0], np.atleast_1d(r["span1"]))
    np.testing.assert_allclose(sw.sym_curve, r["sym"], atol=1e-12, equal_nan=True)
    sw2 = sweep.auto_spans(temp1)
    best = r["turns2"] if r["quality2"] > r["quality"] else r["turns"]
    np.testing.assert_allclose(sw2.turns, np.atleast_1d(best), atol=1e-9)


def test_find_sweep_blocks():
    r = ref("cine.mat")
    b = sweep.find_sweep_blocks(r["Mdet"].astype(float), int(r["iz"]))
    np.testing.assert_array_equal(b.blocks, np.atleast_2d(r["blocks"]))
    np.testing.assert_array_equal(b.turns, np.atleast_1d(r["bturns"]))
    assert b.T == r["bT"]
    np.testing.assert_allclose(b.slope, np.atleast_1d(r["bslope"]), rtol=1e-9)
    assert b.cls == "".join(np.atleast_1d(r["bclass"]))   # a char array in the .mat


def test_sector_recon_bmode():
    """Recon_v1's recon2D on the MATLAB-loaded cine: the scan conversion alone."""
    r = ref("cine.mat")
    # rebuild src exactly as the reference did, from the reference's own inputs
    cal = recon.load_calibration(os.path.join(CINE, "CAL_2026-09-14_10Vpp_-2+.mat"))
    blocks = np.atleast_2d(r["blocks"])
    FL = np.arange(blocks[0, 0], blocks[0, 1] + 1)
    ang, valid = recon.cine_angles(cal, FL.size, "fraction")
    np.testing.assert_allclose(ang, r["ang"], rtol=1e-12)
    s = io.load_scan_stack(files_in(os.path.join(CINE, "Phantom_v1"))[0], crop=(369, 656))
    dz = s.dz_mm
    zmm = np.arange(s.B.shape[0]) * dz
    iz = int(np.flatnonzero(zmm >= 23)[0]) + 1
    assert iz == r["iz"]
    src = s.B[iz - 1:, 119, FL[valid] - 1].astype(float)
    g = sector.SectorGrid(ang, zmm[iz - 1:] - 23, 1540 / 5.2083e6 * 1000)
    img = g.apply(src)
    np.testing.assert_allclose(g.zz, r["zz"], atol=1e-9)
    np.testing.assert_allclose(g.yy, r["yy"], atol=1e-9)
    diff = np.abs(img - r["img"])
    print("B-mode 2-D frame vs MATLAB: max |diff|", diff.max(), " mean", diff.mean())
    assert diff.mean() < 0.3          # JPEG decode differences only


def test_iq_loader():
    r = ref("iq.mat")
    s = io.load_scan_stack(IQ)
    assert list(s.B.shape) == list(r["size"])
    np.testing.assert_allclose(s.B[:, :, 0], r["f1"], atol=1e-4)
    np.testing.assert_allclose(s.B[:, :, 49], r["f50"], atol=1e-4)
    assert s.dz_mm == pytest.approx(r["dz"]) and s.dx_mm == pytest.approx(r["dx"])


# ---------------------------------------------------------------- tabs and angles

def test_tab_snapping_and_fit():
    r = ref("tabs.mat")
    M = tabs.snap_map(r["temp1"].astype(float))
    ts = tabs.TabSet()
    wF, wZ = (int(v) for v in r["searchWin"])
    for c, rw in r["clicks"]:
        ts.add(M, c, rw, wF, wZ)
    got = ts.sorted()
    np.testing.assert_allclose(got[:, :2], r["tabs"], atol=1e-9)
    np.testing.assert_array_equal(got[:, 4] != 0, r["bySpacing"].astype(bool))
    a = tabs.compute_angles(got, 5.0, int(r["zeroRung"]), int(r["negRung"]))
    np.testing.assert_allclose(a["colFit"], r["colFit"])
    np.testing.assert_allclose(a["thetaFit"], r["thetaFit"], atol=1e-9)
    assert a["direction"] == r["direction"]
    assert a["col0"] == pytest.approx(r["col0"], abs=1e-9)


# ---------------------------------------------------------------- colour-flow reconstruction

def test_cf_recon_2d_sivv_and_volume(cf_stack, cf_decoded):
    r = ref("cfrec.mat")
    bar, d = cf_decoded
    P = cf_stack.panels[cf_stack.flow_panel]
    cal = recon.load_calibration(os.path.join(CF, "CAL_2026-09-28_CF_-2+.mat"))
    N = cf_stack.N
    ang, valid = recon.stepped_angles(cal, N)
    pos = np.flatnonzero(valid)
    np.testing.assert_array_equal(pos + 1, r["pos"])
    np.testing.assert_allclose(ang[pos], r["ang"], rtol=1e-12)

    vmax = 8.0
    # the clean-up: keep 0.5..8 cm/s of speed
    t_vel = (d.v + 1) / 2
    spd = np.abs(d.v)
    rej = (spd < 0.5 / vmax) | (spd > 8 / vmax)
    t_vel = np.where(rej, np.nan, t_vel)
    assert np.sum(rej & ~np.isnan(spd)) / np.sum(~np.isnan(spd)) == pytest.approx(r["rejFrac"], rel=1e-9)

    zmm = np.arange(P.rgb.shape[0]) * P.dz_mm
    iz = int(np.flatnonzero(zmm >= 23)[0])
    lam = 1540 / 5.2083e6 * 1000
    g = sector.SectorGrid(ang[pos], zmm[iz:] - 23, lam)
    T = t_vel[iz:, 201, :][:, pos].astype(float)
    t2 = recon.sweep_values(T, g)
    grey2 = g.apply(d.grey[iz:, 201, :][:, pos].astype(float))
    np.testing.assert_allclose(t2, r["t"], atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(grey2, r["grey"], atol=1e-9)
    np.testing.assert_allclose(g.zz + 23, r["zz"], atol=1e-9)

    # SIVV on the cleaned velocity
    v_cms = -vmax + t_vel.astype(float) * (2 * vmax)
    prof = recon.sivv_profile(v_cms, d.box, zmm, 23.0, P.dx_mm, ang, valid)
    np.testing.assert_allclose(prof.z, r["z"], atol=1e-9)
    np.testing.assert_allclose(prof.Q, r["Q"], rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(prof.edge, r["edge"].astype(bool))
    assert recon.sivv_default_depth(prof) == pytest.approx(r["depth"])
    at = recon.sivv_at(prof, r["depth"], 2)
    assert at.Q == pytest.approx(r["atQ"], rel=1e-9)

    # box volume and its isotropic resample
    b = d.box
    r0, r1 = max(iz + 1, b[0] - 2), min(t_vel.shape[0], b[1] + 2)
    rows = np.arange(r0, r1 + 1)
    lat = np.arange(b[2], b[3] + 1)
    np.testing.assert_array_equal(lat, np.atleast_1d(r["volLat"]))
    g3 = sector.SectorGrid(ang[pos], zmm[rows - 1] - 23, lam)
    T3 = t_vel[np.ix_(rows - 1, lat - 1, pos)].astype(float)
    T3 = np.transpose(T3, (0, 2, 1))                     # range x angle x lateral
    vol = recon.sweep_values(T3, g3)                     # zz x yy x lateral
    vol = np.transpose(vol, (0, 2, 1))                   # zz x lateral x yy, as MATLAB
    np.testing.assert_allclose(vol, r["volT"], atol=1e-6, equal_nan=True)
    vox = [float(np.mean(np.diff(g3.zz))), P.dx_mm, float(np.mean(np.diff(g3.yy)))]
    np.testing.assert_allclose(vox, r["volVox"], rtol=1e-12)
    iso, _ = recon.iso_volume(vol.astype(np.float32), vox, 0.0)
    assert iso.shape == r["volIso"].shape
    diff = np.abs(iso - r["volIso"])
    print("iso volume vs MATLAB: max |diff|", np.nanmax(diff),
          " NaN pattern equal:", np.array_equal(np.isnan(iso), np.isnan(r["volIso"])))
    assert np.nanmax(diff) < 1e-4
    # A voxel is empty when the resampled mask is under 0.5. A handful land on
    # 0.5 exactly in MATLAB and on 0.49999994 in float32 here - rounding at the
    # threshold (7 of 11 million on this set), not a difference in behaviour.
    assert np.sum(np.isnan(iso) != np.isnan(r["volIso"])) <= 20
