"""0.16 anchors: the gain a target squeezing needs (`gain_for_squeezing`).

References: the package's own spectra (`output_quadrature_variance` on
`single_mode_parametric(mu)`, a separate code path from the closed
forms the planner uses), the limit 10 log10(1 - 4 eta / (4 + omega^2))
at threshold, and a dense scan of the gain for the jitter case.
"""
import numpy as np
import pytest

from sqzcomb import (detected_variance, gain_for_squeezing,
                     output_quadrature_variance, phase_noise_variance,
                     single_mode_parametric)


def _detected_db(mu, eta, omega, theta=0.0, dark=0.0, ref=False):
    """Detected squeezing through the package's spectra path."""
    M = single_mode_parametric(mu)
    vs = output_quadrature_variance(M, eta, omega, 0, 1, phi=np.pi / 2)
    va = output_quadrature_variance(M, eta, omega, 0, 1, phi=0.0)
    v = detected_variance(phase_noise_variance(vs, va, theta), 1.0, dark,
                          ref)
    return 10 * np.log10(v / 0.5)


@pytest.mark.parametrize("target,eta,omega,dark", [
    (-3.0, 0.9, 0.0, 0.0), (-6.0, 0.9, 0.5, 0.0), (-8.0, 0.95, 0.1, 0.005),
    (-0.5, 0.4, 2.0, 0.0)])
def test_round_trip_without_jitter(target, eta, omega, dark):
    r = gain_for_squeezing(target, eta, omega, dark_noise=dark)
    assert 0.0 < r["mu"] < 1.0
    assert _detected_db(r["mu"], eta, omega, dark=dark) == pytest.approx(
        target, abs=1e-10)
    assert r["detected_db"] == pytest.approx(target, abs=1e-12)
    # the smallest such gain: a little less does not reach the target
    assert _detected_db(r["mu"] * (1 - 1e-6), eta, omega, dark=dark) \
        > target
    assert r["best_at_limit"]
    # deepest squeezing at threshold: 1/2 [1 - 4 eta / (4 + omega^2)] + dark
    v_lim = 0.5 * (1 - 4 * eta / (4 + omega ** 2)) + dark
    assert r["best_db"] == pytest.approx(10 * np.log10(v_lim / 0.5),
                                         abs=1e-6)
    M = single_mode_parametric(r["mu"])
    va = output_quadrature_variance(M, eta, omega, 0, 1, phi=0.0) + dark
    assert r["anti_db"] == pytest.approx(10 * np.log10(va / 0.5),
                                         abs=1e-10)


@pytest.mark.parametrize("eta,omega,theta,dark,ref", [
    (0.9, 0.0, 0.03, 0.0, False), (0.9, 0.3, 0.05, 0.01, True),
    (0.8, 1.0, 0.1, 0.0, False)])
def test_jitter_has_a_best_gain_found_by_a_scan(eta, omega, theta, dark,
                                                ref):
    mus = np.linspace(1e-4, 1 - 1e-6, 1501)
    scan = np.array([_detected_db(m, eta, omega, theta, dark, ref)
                     for m in mus])
    target = float(scan.min()) + 0.5          # reachable, with margin
    r = gain_for_squeezing(target, eta, omega, theta, dark, ref)
    k = int(np.argmin(scan))
    step = mus[1] - mus[0]
    assert r["best_db"] <= scan.min() + 1e-9
    if k < mus.size - 1:
        assert abs(r["mu_best"] - mus[k]) <= 2 * step
    # the smallest gain reaching the target: first crossing of the scan
    first = int(np.argmax(scan <= target))
    assert mus[first - 1] <= r["mu"] <= mus[first]
    assert _detected_db(r["mu"], eta, omega, theta, dark, ref) == \
        pytest.approx(target, abs=1e-9)
    with pytest.raises(ValueError, match="cannot be reached"):
        gain_for_squeezing(scan.min() - 0.05, eta, omega, theta, dark, ref)


def test_refusals():
    for bad in (dict(target_db=0.5), dict(target_db=np.nan),
                dict(eta=0.0), dict(eta=1.2), dict(omega=np.inf),
                dict(theta_rms=-0.1), dict(dark_noise=-1e-3),
                dict(mu_max=1.0)):
        kw = dict(target_db=-3.0, eta=0.9)
        kw.update(bad)
        with pytest.raises(ValueError):
            gain_for_squeezing(**kw)
    with pytest.raises(ValueError, match="cannot be reached"):
        gain_for_squeezing(-10.5, 0.9)          # beyond 10 log10(0.1)
