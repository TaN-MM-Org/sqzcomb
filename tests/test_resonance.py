"""0.16 anchors: linewidth and escape efficiency from a transmission scan.

References: the through-port transmission derived again here from the
package's drift matrix and resolvent (T_ex = 2 eta G - 1, a separate
code path from `through_transmission`), the on-resonance closed form
((kappa_0 - kappa_ex) / (kappa_0 + kappa_ex))^2, noise-free recovery of
the generating values, and seeded Monte Carlo for the error bars.
"""
import numpy as np
import pytest

from sqzcomb import (fit_resonance_transmission, single_mode_parametric,
                     through_transmission)

F0, KAP, ETA, BASE = 12e6, 100e6, 0.8, 2.5
SCAN = np.linspace(-400e6, 400e6, 161)


def _t_from_drift(eta, delta):
    """Mean-field through-port amplitude of a passive mode detuned by
    delta (units of kappa/2): (T_ex)_00 = 2 eta G_00 - 1 at omega = 0,
    G = (-M)^-1, from the package's own drift matrix."""
    M = single_mode_parametric(0.0, delta)
    G = np.linalg.inv(-M)
    return 2.0 * eta * G[0, 0] - 1.0


def test_model_is_the_input_output_relation():
    for eta in (0.0, 0.1, 0.5, 0.8, 1.0):
        for delta in (-7.0, -1.0, -0.3, 0.0, 0.4, 2.0, 15.0):
            f = F0 + delta * KAP / 2.0
            ref = abs(_t_from_drift(eta, delta)) ** 2
            assert through_transmission(f, F0, KAP, eta) == \
                pytest.approx(ref, rel=1e-14, abs=1e-15)
        k_ex, k_0 = eta, 1.0 - eta
        assert through_transmission(F0, F0, KAP, eta) == pytest.approx(
            ((k_0 - k_ex) / (k_0 + k_ex)) ** 2, abs=1e-15)
    # FWHM of the dip is the loaded linewidth: half depth at f0 +/- kappa/2
    D = 4 * ETA * (1 - ETA)
    half = through_transmission(F0 + KAP / 2, F0, KAP, ETA)
    assert half == pytest.approx(1.0 - D / 2.0, rel=1e-14)


def test_phase_winds_only_when_over_coupled():
    """The docstring's way to pick the branch: across an over-coupled
    resonance the phase of t turns by 2 pi, across an under-coupled one
    it returns to where it started."""
    deltas = np.linspace(-2000.0, 2000.0, 400001)
    for eta, turns in ((0.2, 0.0), (0.8, 1.0)):
        t = np.array([_t_from_drift(eta, d) for d in deltas[::100]])
        t_fine = 2.0 * eta / (1.0 - 1j * deltas) - 1.0
        assert np.allclose(t, t_fine[::100], rtol=1e-12, atol=1e-14)
        ph = np.unwrap(np.angle(t_fine))
        assert abs(ph[-1] - ph[0]) / (2 * np.pi) == pytest.approx(
            turns, abs=1e-3)


@pytest.mark.parametrize("eta,branch", [(0.8, "over"), (0.3, "under"),
                                        (0.95, "over"), (0.1, "under")])
def test_noise_free_scan_returns_the_true_values(eta, branch):
    T = through_transmission(SCAN, F0, KAP, eta, BASE)
    r = fit_resonance_transmission(SCAN, T, branch=branch)
    assert r["f0_hz"] == pytest.approx(F0, abs=1e-6 * KAP)
    assert r["kappa_hz"] == pytest.approx(KAP, rel=1e-9)
    assert r["eta_esc"] == pytest.approx(eta, rel=1e-9)
    assert r["eta_under"] + r["eta_over"] == pytest.approx(1.0, abs=1e-15)
    assert r["baseline"] == pytest.approx(BASE, rel=1e-9)
    assert r["kappa_ex_hz"] == pytest.approx(eta * KAP, rel=1e-8)
    assert r["kappa_0_hz"] == pytest.approx((1 - eta) * KAP, rel=1e-8)
    assert np.allclose(r["model"], T, rtol=1e-9)
    # a laser power drifting linearly across the sweep
    tilt = 1.0 + 2e-10 * (SCAN - SCAN.mean())
    r = fit_resonance_transmission(SCAN, T * tilt, branch=branch,
                                   fit_slope=True)
    assert r["kappa_hz"] == pytest.approx(KAP, rel=1e-8)
    assert r["eta_esc"] == pytest.approx(eta, rel=1e-8)
    assert r["slope_per_hz"] == pytest.approx(BASE * 2e-10, rel=1e-7)


def test_error_bars_match_monte_carlo():
    rng = np.random.default_rng(7)
    T = through_transmission(SCAN, F0, KAP, ETA, BASE)
    sig = 0.01
    keys = ("f0_hz", "kappa_hz", "depth")
    draws = {k: [] for k in keys + ("eta", "k_ex")}
    reported = None
    for _ in range(120):
        r = fit_resonance_transmission(SCAN, T + rng.normal(0, sig, T.size),
                                       sigma=sig, branch="over")
        for k in keys:
            draws[k].append(r[k])
        draws["eta"].append(r["eta_esc"])
        draws["k_ex"].append(r["kappa_ex_hz"])
        if reported is None:
            reported = r
    for k in keys:
        ratio = np.std(draws[k], ddof=1) / reported["sigma"][k]
        assert 0.75 < ratio < 1.3, (k, ratio)
    assert 0.75 < np.std(draws["eta"], ddof=1) / reported["eta_sigma"] < 1.3
    assert 0.75 < np.std(draws["k_ex"], ddof=1) \
        / reported["kappa_ex_hz_sigma"] < 1.3
    # unbiased within the scatter of the mean
    assert abs(np.mean(draws["kappa_hz"]) - KAP) < \
        4 * reported["sigma"]["kappa_hz"] / np.sqrt(120)
    assert reported["p_value"] > 1e-6


def test_critical_coupling_merges_the_branches():
    rng = np.random.default_rng(2)
    T = through_transmission(SCAN, F0, KAP, 0.5, BASE)
    r = fit_resonance_transmission(SCAN, T + rng.normal(0, 0.01, T.size),
                                   sigma=0.01, branch="over")
    assert r["critical_consistent"]
    assert r["eta_sigma"] is None and r["eta_esc_sigma"] is None
    assert r["eta_under"] <= 0.5 <= r["eta_over"]


def test_refusals():
    T = through_transmission(SCAN, F0, KAP, ETA, BASE)
    with pytest.raises(ValueError, match="one linewidth"):
        narrow = np.abs(SCAN - F0) < 0.6 * KAP
        fit_resonance_transmission(SCAN[narrow], T[narrow])
    with pytest.raises(ValueError, match="not resolved"):
        coarse = np.linspace(-400e6, 400e6, 17)
        fit_resonance_transmission(
            coarse, through_transmission(coarse, F0, KAP, ETA, BASE))
    rng = np.random.default_rng(1)
    with pytest.raises(ValueError, match="no resolved dip"):
        fit_resonance_transmission(SCAN, BASE + rng.normal(0, 0.01,
                                                           SCAN.size),
                                   sigma=0.01)
    # a doublet split by backscattering: two dips one linewidth apart
    split = BASE * (through_transmission(SCAN, F0 - 0.5 * KAP, KAP, 0.3)
                    + through_transmission(SCAN, F0 + 0.5 * KAP, KAP, 0.3)
                    - 1.0)
    with pytest.raises(ValueError, match="single Lorentzian"):
        fit_resonance_transmission(SCAN, split + rng.normal(0, 0.005,
                                                            SCAN.size),
                                   sigma=0.005)
    for bad in (dict(branch="critical"), dict(sigma=0.0),
                dict(sigma=np.nan), dict(min_p_value=1.0)):
        with pytest.raises(ValueError):
            fit_resonance_transmission(SCAN, T, **bad)
    with pytest.raises(ValueError):
        fit_resonance_transmission(SCAN[:7], T[:7])
    with pytest.raises(ValueError):
        fit_resonance_transmission(SCAN, np.where(SCAN > 0, np.nan, T))
    with pytest.raises(ValueError):
        through_transmission(SCAN, F0, KAP, 1.2)
    with pytest.raises(ValueError):
        through_transmission(SCAN, F0, -KAP, 0.5)
