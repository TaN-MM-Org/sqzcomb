"""`optimal_band_angle` (new in 0.19): the best fixed angle for a band
given as any function of the angle (the photonic molecule's ports
here), and the faster worst-case search of `_minimax_angle`.

Independent checks: a scan of the angle refined by a bounded minimizer
on the port variance itself, the single-ring function (exact equality
for a single-ring variance, and agreement through a molecule with
J = 0, a separate code path), and the 0.18 search, kept below.
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

import sqzcomb as sc
from sqzcomb.spectra import _minimax_angle


def _minimax_angle_018(c0, c1, c2, ref):
    """The search of sqzcomb 0.18.0, unchanged: every candidate angle is
    evaluated against every curve."""
    a0, a1, a2 = (np.asarray(c, dtype=float) / ref for c in (c0, c1, c2))
    k = a0.size
    cands = [np.zeros(1)]
    r = np.hypot(a1, a2)
    cands.append(np.arctan2(-a2, -a1)[r > 0.0])
    i, j = np.triu_indices(k, 1)
    d0, d1, d2 = a0[i] - a0[j], a1[i] - a1[j], a2[i] - a2[j]
    rr = np.hypot(d1, d2)
    ok = (rr > 0.0) & (np.abs(d0) <= rr)
    base = np.arctan2(d2[ok], d1[ok])
    off = np.arccos(np.clip(-d0[ok] / rr[ok], -1.0, 1.0))
    cands += [base + off, base - off]
    th = np.concatenate(cands)
    best_val, best_t = np.inf, 0.0
    chunk = max(1, 2_000_000 // k)
    for s0 in range(0, th.size, chunk):
        t = th[s0:s0 + chunk]
        env = np.max(a0[:, None] + a1[:, None] * np.cos(t)[None, :]
                     + a2[:, None] * np.sin(t)[None, :], axis=0)
        m = int(np.argmin(env))
        if env[m] < best_val:
            best_val, best_t = float(env[m]), float(t[m])
    return float(np.mod(0.5 * best_t, np.pi))


# an illustrative detuned molecule read at the auxiliary ring
_M, _G = sc.photonic_molecule(0.6, 1.2, delta_a=0.3, gamma=2.0)
_OM = np.linspace(0.0, 2.0, 15)


def _ports(p):
    return np.array([sc.output_variance_ports(_M, _G, 0.9, 1, w, phi=p)
                     for w in _OM])


def _refined_scan(fun, n=721):
    """Smallest value of a function of the angle on [0, pi): a scan,
    then a bounded minimizer around the best scan point."""
    ph = np.linspace(0.0, np.pi, n, endpoint=False)
    vals = np.array([fun(p) for p in ph])
    k = int(np.argmin(vals))
    step = ph[1] - ph[0]
    res = minimize_scalar(fun, bounds=(ph[k] - step, ph[k] + step),
                          method="bounded", options={"xatol": 1e-12})
    return min(float(res.fun), float(vals[k])), float(res.x)


@pytest.mark.parametrize("goal", ["mean", "worst", "penalty"])
def test_ports_band_angle_matches_a_refined_scan(goal):
    r = sc.optimal_band_angle(_ports, goal=goal)
    # each frequency's own optimum, found by minimizing the port
    # variance at that frequency alone
    v_opt = np.array([_refined_scan(
        lambda p, w=w: sc.output_variance_ports(_M, _G, 0.9, 1, w, phi=p),
        n=181)[0] for w in _OM])
    np.testing.assert_allclose(r["v_opt"], v_opt, rtol=1e-9)   # measured 4e-16
    if goal == "mean":
        obj = lambda p: float(np.mean(_ports(p)))
        got = r["v_band"]
    elif goal == "worst":
        obj = lambda p: float(np.max(_ports(p)))
        got = float(np.max(r["v"]))
    else:
        obj = lambda p: float(np.max(_ports(p) / v_opt))
        got = float(np.max(r["v"] / r["v_opt"]))
    best, phi = _refined_scan(obj)
    # never worse than the scan, and equal to its refined minimum
    assert got <= best * (1.0 + 1e-12)
    assert got == pytest.approx(best, rel=1e-9)      # measured 9e-16 (penalty: 4e-11 below)
    d = abs(r["phi"] - phi)
    assert min(d, np.pi - d) < 1e-5                   # measured below 1e-9
    # the reported values are the port variances at that angle
    np.testing.assert_allclose(r["v"], _ports(r["phi"]), rtol=0, atol=0)


@pytest.mark.parametrize("goal", ["mean", "worst", "penalty"])
def test_single_ring_variance_gives_optimal_band_quadrature(goal):
    M = sc.single_mode_parametric(0.6, delta=0.3)
    band = np.linspace(0.0, 2.0, 21)
    wt = np.linspace(1.0, 2.0, 21)
    a = sc.optimal_band_quadrature(M, 0.8, band, 0, 1, weights=wt, goal=goal)
    b = sc.optimal_band_angle(
        lambda p: np.array([sc.output_quadrature_variance(M, 0.8, w, 0, 1,
                                                          phi=p)
                            for w in band]), weights=wt, goal=goal)
    for key in a:
        assert np.array_equal(a[key], b[key]), key


@pytest.mark.parametrize("goal", ["mean", "worst", "penalty"])
def test_uncoupled_molecule_port_equals_the_single_ring(goal):
    # with J = 0 the main ring's port is the single ring with gamma = 1
    M2, g2 = sc.photonic_molecule(0.6, 0.0, delta_a=0.3, gamma=3.0)
    M1 = sc.single_mode_parametric(0.6, delta=0.3)
    # the same mode: both builders put -1 + i delta on the diagonal
    assert np.array_equal(M2[np.ix_([0, 2], [0, 2])], M1)
    band = np.linspace(0.0, 2.0, 21)
    a = sc.optimal_band_quadrature(M1, 0.8, band, 0, 1, goal=goal)
    b = sc.optimal_band_angle(
        lambda p: np.array([sc.output_variance_ports(M2, g2, 0.8, 0, w,
                                                     phi=p)
                            for w in band]), goal=goal)
    assert b["phi"] == pytest.approx(a["phi"], abs=1e-9)    # measured 3e-15
    np.testing.assert_allclose(b["v"], a["v"], rtol=1e-12)   # measured 5e-15
    np.testing.assert_allclose(b["v_opt"], a["v_opt"],
                               rtol=1e-12)              # measured 7e-16


def test_band_angle_refusals():
    with pytest.raises(ValueError, match="goal"):
        sc.optimal_band_angle(_ports, goal="median")
    with pytest.raises(ValueError, match="form"):
        sc.optimal_band_angle(lambda p: sc.squeezing_db(_ports(p)))
    with pytest.raises(ValueError, match="non-finite"):
        sc.optimal_band_angle(lambda p: np.full(3, np.nan))
    with pytest.raises(ValueError, match="1-D"):
        sc.optimal_band_angle(lambda p: 0.7)
    with pytest.raises(ValueError, match="different lengths"):
        sc.optimal_band_angle(lambda p: np.full(3 if p == 0.0 else 4, 0.7))
    with pytest.raises(ValueError, match="weights"):
        sc.optimal_band_angle(_ports, weights=np.ones(3))
    with pytest.raises(ValueError, match="weights"):
        sc.optimal_band_angle(_ports, weights=-np.ones(_OM.size))


def test_faster_search_returns_the_018_angle():
    """Bit-identical angles on random curve families (including tied,
    flat and nearly flat curves) and on the band of README example 27
    at 400 frequencies."""
    rng = np.random.default_rng(7)
    for trial in range(400):
        n = int(rng.integers(1, 50))
        c0 = rng.uniform(0.3, 3.0, n)
        c1 = rng.normal(0.0, 1.0, n)
        c2 = rng.normal(0.0, 1.0, n)
        if trial % 5 == 0:
            c1, c2 = 1e-4 * c1, 1e-4 * c2          # nearly flat
        if trial % 7 == 0:
            c0[:] = c0[0]                          # equal means
            c1[n // 2:], c2[n // 2:] = c1[0], c2[0]   # repeated curves
        if trial % 11 == 0:
            c1[:] = c2[:] = 0.0                    # all flat
        ref = rng.uniform(0.2, 1.0, n) if trial % 2 else np.full(n, 0.5)
        assert _minimax_angle(c0, c1, c2, ref) \
            == _minimax_angle_018(c0, c1, c2, ref)
    M = sc.single_mode_parametric(0.6, delta=0.3)
    band = np.linspace(0.0, 2.0, 400)
    r = sc.optimal_quadrature(M, 0.8, band, 0, 1)
    v0, v45, v90 = (np.array([sc.output_quadrature_variance(
        M, 0.8, w, 0, 1, phi=p) for w in band])
        for p in (0.0, np.pi / 4, np.pi / 2))
    c0 = 0.5 * (v0 + v90)
    for ref in (np.full(band.size, 0.5), r["v_min"]):
        args = (c0, 0.5 * (v0 - v90), v45 - c0, ref)
        assert _minimax_angle(*args) == _minimax_angle_018(*args)
