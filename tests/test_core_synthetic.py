"""Checks that need no data: MATLAB semantics, and SIVV against a flow rate known
in closed form."""
import numpy as np
import pytest

from faster_fusion.core import mlcompat as ml
from faster_fusion.core import recon, sector, tabs


def test_round_half_away_from_zero():
    np.testing.assert_array_equal(ml.mround([0.5, 1.5, 2.5, -0.5, -2.5]), [1, 2, 3, -1, -3])


def test_findpeaks_matlab_rules():
    # a flat peak goes on its FIRST sample; a peak next to NaN is no peak
    y = np.array([0, 1, 3, 3, 3, 1, 0, 2, np.nan, 5, 0])
    _, loc = ml.findpeaks(y)
    assert list(loc) == [2]
    # prominence is applied before distance: the tall narrow peak survives,
    # and suppresses its lower neighbour inside the distance
    y = np.array([0, 5, 0, 4, 0, 0, 0, 0, 3, 0])
    _, loc = ml.findpeaks(y, min_prominence=1, min_distance=3)
    assert list(loc) == [1, 8]


def test_movmean_shrinks_at_ends():
    np.testing.assert_allclose(ml.movmean(np.arange(5.0), 3), [0.5, 1, 2, 3, 3.5])


def test_sector_grid_reproduces_a_constant():
    g = sector.SectorGrid(np.linspace(-20, 20, 41), np.linspace(1, 30, 60), 0.3)
    out = g.apply(np.full((60, 41), 7.0))
    inside = g.ok.reshape(g.shape)
    np.testing.assert_allclose(out[inside], 7.0)
    assert np.all(out[~inside] == 0)


def test_sivv_uniform_flow_gives_v_times_area():
    """Flow of v cm/s towards the probe over a patch W lateral lines wide and
    spanning angles th0..th1 at range d: Q = 0.6 * v * (W*dx) * (d*(th1-th0))."""
    dz, dx, mirror = 0.1, 0.15, 10.0
    nz, nx, N = 400, 120, 81
    zmm = np.arange(nz) * dz
    ang = np.linspace(-20, 20, N)                  # evenly spaced, degrees
    valid = np.ones(N, bool)
    v = 5.0
    V = np.full((nz, nx, N), np.nan)
    c0, c1 = 40, 69                                # 30 lateral lines with flow
    p0, p1 = 30, 50                                # positions with flow
    V[:, c0:c1 + 1, p0:p1 + 1] = v
    box = [1, nz, 1, nx]
    P = recon.sivv_profile(V, box, zmm, mirror, dx, ang, valid)
    th = np.deg2rad(ang)
    for i in (50, 150, 250):
        d = P.d[i]
        # the trapezoid gives each position the arc half way to its neighbours
        expect = 0.6 * v * (30 * dx) * d * (th[1] - th[0]) * (p1 - p0 + 1)
        assert P.Q[i] == pytest.approx(expect, rel=1e-12)
        assert P.Qneg[i] == 0 and P.Qpos[i] == pytest.approx(expect)
    assert not P.edge.any()                        # the patch touches no edge
    # every row carries the same flow through a surface it cuts whole, so the
    # mean over a slab is that flow; Q grows with range only through the arc
    at = recon.sivv_at(P, P.z[150], 0)
    assert at.Q == pytest.approx(P.Q[150])


def test_sivv_flags_flow_cut_off_by_the_box():
    zmm = np.arange(100) * 0.1
    ang = np.linspace(-10, 10, 21)
    V = np.full((100, 30, 21), np.nan)
    V[:, 0:5, 5:10] = 3.0                          # touches the lateral edge
    P = recon.sivv_profile(V, [1, 100, 1, 30], zmm, 1.0, 0.2, ang, np.ones(21, bool))
    assert P.edge_lat.all() and not P.edge_pos.any()


def test_tabs_fit_with_evenly_spaced_dashes():
    """Five bright dashes on a flat map: each click snaps to its dash centre and
    the fit is a straight line through the tab angles."""
    M = np.zeros((60, 100))
    centres = [15, 32, 49, 66, 83]
    for c in centres:
        M[30:32, c - 3:c + 4] = 100.0
    M += 1.0                                       # a stationary background
    ts = tabs.TabSet()
    SM = tabs.snap_map(M)
    for c in centres:
        ts.add(SM, c + 2 + 1, 30.5 + 1, 10, 3)     # clicks 1-based, off centre
    m = ts.sorted()
    np.testing.assert_allclose(m[:, 0], np.array(centres) + 1, atol=1e-9)
    a = tabs.compute_angles(m, 5.0, 3, 2)              # tab 3 is 0 deg, tab 2 is -5 deg
    np.testing.assert_allclose(a["theta"], [-10, -5, 0, 5, 10])
    assert a["direction"] == "negative -> positive"
    a = tabs.compute_angles(m, 5.0, 3, 4)              # -5 deg on the other side
    np.testing.assert_allclose(a["theta"], [10, 5, 0, -5, -10])
    assert a["direction"] == "positive -> negative"
