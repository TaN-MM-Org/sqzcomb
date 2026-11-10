"""0.18: the fixed local-oscillator angle with the best worst case over
a band (`optimal_band_quadrature(..., goal="worst" or "penalty")`).

References: hand-solvable envelopes of two sinusoids for the angle
search itself; a dense scan of the angle, refined by a bounded
one-dimensional minimizer, on variances built from the 2x2 blocks of
`output_covariance_xxpp` (a separate code path from the one under
test); the optimality conditions of a minimax over smooth curves; and
the limits in which every frequency shares one best angle.
"""
import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from sqzcomb import (optimal_band_quadrature, optimal_quadrature,
                     output_covariance_xxpp, single_mode_parametric,
                     squeezing_db)
from sqzcomb.spectra import _minimax_angle

M_DET = single_mode_parametric(0.6, delta=0.3)
BAND = np.linspace(0.0, 2.0, 21)


def test_angle_search_on_hand_solvable_envelopes():
    one = np.ones(1)
    # 1 + cos t and 1 - cos t: the envelope 1 + |cos t| is lowest at
    # t = pi/2 or 3 pi/2 (phi = pi/4 or 3 pi/4), where the two cross
    phi = _minimax_angle(np.r_[1.0, 1.0], np.r_[1.0, -1.0], np.r_[0.0, 0.0],
                         np.r_[1.0, 1.0])
    assert min(abs(phi - np.pi / 4), abs(phi - 3 * np.pi / 4)) < 1e-12
    # 1 + 0.5 cos t lies below 2 + 0.1 sin t everywhere, so the envelope
    # is the second curve, lowest at t = 3 pi / 2 (phi = 3 pi / 4)
    phi = _minimax_angle(np.r_[1.0, 2.0], np.r_[0.5, 0.0], np.r_[0.0, 0.1],
                         np.r_[1.0, 1.0])
    assert phi == pytest.approx(3 * np.pi / 4, abs=1e-12)
    # dividing by a reference changes which curve is on top: with the
    # second one halved the first one leads, lowest at t = pi
    phi = _minimax_angle(np.r_[1.0, 2.0], np.r_[0.5, 0.0], np.r_[0.0, 0.1],
                         np.r_[1.0, 4.0])
    assert phi == pytest.approx(np.pi / 2, abs=1e-12)
    # flat curves: every angle is as good, and 0 comes back
    assert _minimax_angle(2 * one, 0 * one, 0 * one, one) == 0.0


def _scan_reference(goal):
    """Envelope minimum from a dense angle scan of u^T S_k u (S_k the
    2x2 output covariance block at omega_k, u = (cos phi, sin phi)),
    refined by a bounded minimizer around the best grid point."""
    S = [output_covariance_xxpp(M_DET, 0.8, w, 1) for w in BAND]
    v_opt = np.array([np.linalg.eigvalsh(s)[0] for s in S])
    ref = np.full(BAND.size, 0.5) if goal == "worst" else v_opt

    def env(p):
        c, s = np.cos(p), np.sin(p)
        v = np.array([m[0, 0] * c * c + 2 * m[0, 1] * c * s + m[1, 1] * s * s
                      for m in S])
        return np.max(v / (ref[:, None] if np.ndim(p) else ref), axis=0)

    grid = np.linspace(0.0, np.pi, 200001)
    e = env(grid)
    k = int(np.argmin(e))
    d = grid[1] - grid[0]
    res = minimize_scalar(lambda p: float(env(np.array([p]))[0]),
                          bounds=(grid[k] - 2 * d, grid[k] + 2 * d),
                          method="bounded", options={"xatol": 1e-13})
    return e[k], float(res.fun), float(res.x)


@pytest.mark.parametrize("goal,key", [("worst", "worst_db"),
                                      ("penalty", "worst_penalty_db")])
def test_worst_case_angle_beats_a_dense_scan(goal, key):
    r = optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1, goal=goal)
    grid_min, refined, phi_ref = _scan_reference(goal)
    # back from dB to the ratio the scan minimizes: max_k V_k / ref_k
    exact = 10.0 ** (r[key] / 10.0)
    # never worse than any scanned angle; the refined scan comes within
    # 1e-8 (at a corner of the envelope, where two curves cross, the
    # bounded minimizer stops a few 1e-9 above it: 3e-9 for "penalty")
    assert exact <= grid_min * (1 + 1e-12)
    assert exact <= refined * (1 + 1e-12)
    assert refined - exact <= 1e-8 * exact
    assert r["phi"] == pytest.approx(phi_ref, abs=1e-6)


@pytest.mark.parametrize("goal", ["worst", "penalty"])
def test_optimality_conditions(goal):
    """At the best worst-case angle either the leading frequency sits at
    its own best angle (a smooth minimum of one curve), or two
    frequencies tie for the lead (a corner of the envelope)."""
    r = optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1, goal=goal)
    vals = r["squeezing_db"] if goal == "worst" else r["penalty_db"]
    lead = np.argsort(vals)[::-1]
    at_own_best = abs(r["penalty_db"][lead[0]]) < 1e-9
    tie = abs(vals[lead[0]] - vals[lead[1]]) < 1e-9
    assert at_own_best or tie
    # for this detuned squeezer "worst" lands on the top frequency's own
    # best angle (a smooth minimum), "penalty" on a corner (a tie)
    if goal == "worst":
        assert lead[0] == BAND.size - 1 and at_own_best
        assert r["phi"] == pytest.approx(r["phi_opt"][-1], abs=1e-9)
    else:
        assert tie and not at_own_best


def test_each_goal_is_best_at_its_own_number():
    rs = {g: optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1, goal=g)
          for g in ("mean", "worst", "penalty")}
    for g in ("mean", "penalty"):
        assert rs["worst"]["worst_db"] <= rs[g]["worst_db"] + 1e-12
    for g in ("mean", "worst"):
        assert rs["penalty"]["worst_penalty_db"] <= \
            rs[g]["worst_penalty_db"] + 1e-12
        assert rs["mean"]["v_band"] <= rs[g]["v_band"] * (1 + 1e-12)
    # the reported summaries are those of the arrays, and v_band is the
    # average of v (same arithmetic as the default goal)
    for r in rs.values():
        assert r["worst_db"] == np.max(r["squeezing_db"])
        assert r["worst_penalty_db"] == np.max(r["penalty_db"])
        assert r["v_band"] == pytest.approx(np.mean(r["v"]), rel=1e-14)
    # goal="mean" is the default, the same computation
    d = optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1)
    assert d["phi"] == rs["mean"]["phi"] and d["goal"] == "mean"
    assert np.array_equal(d["v"], rs["mean"]["v"])


@pytest.mark.parametrize("goal", ["mean", "worst", "penalty"])
def test_limits_with_one_best_angle(goal):
    """With a real gain and no detuning every frequency is best at
    phi = pi/2, with the closed-form variance; with one frequency every
    goal is optimal_quadrature."""
    M = single_mode_parametric(0.5)
    r = optimal_band_quadrature(M, 0.7, BAND, 0, 1, goal=goal)
    assert r["phi"] == pytest.approx(np.pi / 2, abs=1e-9)
    closed = 0.5 * (1 - 4 * 0.7 * 0.5 / (1.5 ** 2 + BAND ** 2))
    assert np.allclose(r["v"], closed, rtol=1e-12, atol=0)
    assert np.max(np.abs(r["penalty_db"])) < 1e-9
    one = optimal_band_quadrature(M_DET, 0.8, [0.7], 0, 1, goal=goal)
    opt = optimal_quadrature(M_DET, 0.8, 0.7, 0, 1)
    assert one["phi"] == pytest.approx(opt["phi_min"], abs=1e-9)
    assert one["worst_db"] == pytest.approx(opt["squeezing_db"], abs=1e-9)


def test_refusals_and_weights():
    with pytest.raises(ValueError, match="goal"):
        optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1, goal="max")
    # weights enter the band average only, not the worst case
    w = np.linspace(1.0, 3.0, BAND.size)
    a = optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1, goal="penalty")
    b = optimal_band_quadrature(M_DET, 0.8, BAND, 0, 1, goal="penalty",
                                weights=w)
    assert a["phi"] == b["phi"]
    assert b["v_band"] == pytest.approx(np.average(b["v"], weights=w),
                                        rel=1e-14)
    assert squeezing_db(b["v_band"]) == pytest.approx(
        b["squeezing_db_band"], abs=1e-12)
