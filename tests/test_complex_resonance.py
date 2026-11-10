"""0.18: the escape efficiency and the coupling regime from the complex
(amplitude and phase) through-port transmission.

References: the through-port amplitude derived again here from the
package's drift matrix and resolvent (T_ex = 2 eta G - 1 of a passive
mode, a separate code path from `through_amplitude`), the circle the
amplitude traces (center 1 - eta, radius eta: an identity of the
closed form), the power fit of the same scan (an independent fit of
|t|^2 with its own model), noise-free recovery of the generating values,
and seeded Monte Carlo for the error bars.
"""
import numpy as np
import pytest

from sqzcomb import (fit_complex_transmission, fit_resonance_transmission,
                     single_mode_parametric, through_amplitude,
                     through_transmission)

F0, KAP = 12e6, 100e6
SCAN = np.linspace(-400e6, 400e6, 161)


def _t_from_drift(eta, delta):
    """(T_ex)_00 = 2 eta G_00 - 1 at omega = 0 for a passive mode
    detuned by delta (units of kappa/2), G = (-M)^-1."""
    G = np.linalg.inv(-single_mode_parametric(0.0, delta))
    return 2.0 * eta * G[0, 0] - 1.0


def _noise(rng, sig, n):
    return sig * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def test_amplitude_is_the_input_output_relation():
    """through_amplitude is -T_ex (an overall sign, a phase of pi), its
    conjugate convention is the complex conjugate, and its squared
    modulus is through_transmission."""
    for eta in (0.0, 0.1, 0.5, 0.8, 1.0):
        for delta in (-7.0, -1.0, -0.3, 0.0, 0.4, 2.0, 15.0):
            f = F0 + delta * KAP / 2.0
            ref = -_t_from_drift(eta, delta)
            t = through_amplitude(f, F0, KAP, eta)
            assert abs(t - ref) <= 1e-14 * max(1.0, abs(ref))
            tc = through_amplitude(f, F0, KAP, eta, conjugate=True)
            assert abs(tc - np.conj(ref)) <= 1e-14 * max(1.0, abs(ref))
            assert abs(t) ** 2 == pytest.approx(
                through_transmission(f, F0, KAP, eta), rel=1e-13,
                abs=1e-15)
    # level, phase and delay multiply the resonance factor
    t = through_amplitude(SCAN, F0, KAP, 0.7, 2.0, 0.3, 4e-9)
    t1 = through_amplitude(SCAN, F0, KAP, 0.7)
    lead = 2.0 * np.exp(1j * (0.3 + 2 * np.pi * 4e-9 * (SCAN - F0)))
    assert np.allclose(t, lead * t1, rtol=1e-13, atol=0)


def test_the_amplitude_traces_a_circle_around_or_beside_the_origin():
    """|t - (1 - eta)| = eta at every detuning, with t(f0) = 1 - 2 eta:
    the circle holds the origin exactly when eta > 1/2, which is why the
    complex scan decides the coupling regime."""
    f = np.linspace(-5e9, 5e9, 2001)
    for eta in (0.2, 0.5, 0.8):
        t = through_amplitude(f, 0.0, KAP, eta)
        assert np.allclose(np.abs(t - (1.0 - eta)), eta, rtol=0,
                           atol=1e-14)
        assert through_amplitude(0.0, 0.0, KAP, eta) == pytest.approx(
            1.0 - 2.0 * eta, abs=1e-15)


@pytest.mark.parametrize("eta", [0.1, 0.3, 0.5, 0.7, 0.95])
@pytest.mark.parametrize("conjugate", [False, True])
def test_noise_free_scan_returns_the_true_values(eta, conjugate):
    T = through_amplitude(SCAN, F0, KAP, eta, 1.7, 2.5, 3e-9, conjugate)
    r = fit_complex_transmission(SCAN, T)
    assert r["eta_esc"] == pytest.approx(eta, rel=1e-9)
    assert r["kappa_hz"] == pytest.approx(KAP, rel=1e-9)
    assert r["f0_hz"] == pytest.approx(F0, abs=1e-6 * KAP)
    assert r["amplitude"] == pytest.approx(1.7, rel=1e-9)
    assert r["phase_rad"] == pytest.approx(2.5, abs=1e-9)
    assert r["delay_s"] == pytest.approx(3e-9, rel=1e-8)
    assert r["conjugate"] is conjugate
    assert r["kappa_ex_hz"] == pytest.approx(eta * KAP, rel=1e-8)
    assert r["kappa_0_hz"] == pytest.approx((1 - eta) * KAP, rel=1e-8)
    assert r["regime"] == {0.5: "critical"}.get(
        eta, "over" if eta > 0.5 else "under")
    assert np.allclose(r["model"], T, rtol=1e-9, atol=1e-12)


def test_without_a_delay_and_on_a_narrow_scan():
    """fit_delay=False holds the phase constant; a scan only a little
    wider than the linewidth still converges to the true values."""
    T = through_amplitude(SCAN, F0, KAP, 0.8, 0.5, -1.0)
    r = fit_complex_transmission(SCAN, T, fit_delay=False)
    assert r["eta_esc"] == pytest.approx(0.8, rel=1e-9)
    assert r["delay_s"] == 0.0 and "delay_s" not in r["sigma"]
    assert r["chi2_dof"] is None
    f = np.linspace(-150e6, 150e6, 161)
    T = through_amplitude(f, -7e6, KAP, 0.7, 0.3, -1.0, 1e-9)
    r = fit_complex_transmission(f, T)
    assert r["eta_esc"] == pytest.approx(0.7, rel=1e-9)
    assert r["kappa_hz"] == pytest.approx(KAP, rel=1e-9)


@pytest.mark.parametrize("eta", [0.3, 0.8])
def test_agrees_with_the_power_fit_and_picks_its_branch(eta):
    """The power fit of |t|^2 (its own model and code) offers eta and
    1 - eta; the complex fit of the same noisy scan lands on the right
    one, within the combined error bars, and its linewidth agrees."""
    rng = np.random.default_rng(11)
    sig = 0.004
    T = through_amplitude(SCAN, F0, KAP, eta, 1.0, 0.4, 2e-9)
    Tn = T + _noise(rng, sig, SCAN.size)
    r = fit_complex_transmission(SCAN, Tn, sigma=sig)
    # |T|^2 of the noisy field: error of the power is about 2 |T| sig
    pw = fit_resonance_transmission(SCAN, np.abs(Tn) ** 2,
                                    sigma=2 * sig * np.abs(T))
    right = pw["eta_over"] if eta > 0.5 else pw["eta_under"]
    wrong = pw["eta_under"] if eta > 0.5 else pw["eta_over"]
    tol = 3.0 * np.hypot(r["sigma"]["eta_esc"], pw["eta_sigma"])
    assert abs(r["eta_esc"] - right) < tol
    assert abs(r["eta_esc"] - wrong) > 20 * tol
    assert abs(r["kappa_hz"] - pw["kappa_hz"]) < 3.0 * np.hypot(
        r["sigma"]["kappa_hz"], pw["sigma"]["kappa_hz"])


@pytest.mark.parametrize("eta", [0.3, 0.8])
def test_error_bars_match_monte_carlo(eta):
    rng = np.random.default_rng(2)
    sig = 0.01
    T = through_amplitude(SCAN, F0, KAP, eta, 1.0, 0.4, 2e-9)
    ref = fit_complex_transmission(SCAN, T, sigma=sig)
    keys = ("eta_esc", "kappa_hz", "f0_hz", "kappa_ex_hz", "kappa_0_hz",
            "delay_s")
    draws = {k: [] for k in keys}
    chi2 = []
    for _ in range(80):
        r = fit_complex_transmission(SCAN, T + _noise(rng, sig, SCAN.size),
                                     sigma=sig)
        for k in keys:
            draws[k].append(r[k])
        chi2.append(r["chi2"])
        assert r["regime"] == ("over" if eta > 0.5 else "under")
    for k in keys:
        ratio = np.std(draws[k], ddof=1) / ref["sigma"][k]
        assert 0.75 < ratio < 1.3, (k, ratio)
    # chi-square averages to its degrees of freedom (2 N - 6)
    assert np.mean(chi2) == pytest.approx(2 * SCAN.size - 6, rel=0.05)


def test_regime_is_called_only_when_resolved():
    rng = np.random.default_rng(4)
    sig = 0.01
    T = through_amplitude(SCAN, F0, KAP, 0.5, 1.0, 0.0, 1e-9)
    r = fit_complex_transmission(SCAN, T + _noise(rng, sig, SCAN.size),
                                 sigma=sig)
    assert r["regime"] == "critical" and abs(r["regime_z"]) <= 3.0
    T = through_amplitude(SCAN, F0, KAP, 0.55, 1.0, 0.0, 1e-9)
    r = fit_complex_transmission(SCAN, T + _noise(rng, sig, SCAN.size),
                                 sigma=sig)
    assert r["regime"] == "over" and r["regime_z"] > 3.0


def test_refusals():
    T = through_amplitude(SCAN, F0, KAP, 0.8, 1.0, 0.0, 1e-9)
    with pytest.raises(ValueError, match="complex"):
        fit_complex_transmission(SCAN, np.abs(T) ** 2)
    with pytest.raises(ValueError):
        fit_complex_transmission(SCAN[:7], T[:7])
    bad = T.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        fit_complex_transmission(SCAN, bad)
    with pytest.raises(ValueError):
        fit_complex_transmission(np.r_[SCAN[:1], SCAN[:-1]], T)
    with pytest.raises(ValueError):
        fit_complex_transmission(SCAN, T, sigma=0.0)
    with pytest.raises(ValueError):
        fit_complex_transmission(SCAN, T, sigma=0.01, min_p_value=1.0)
    # the scan must reach a linewidth beyond the resonance on both sides
    with pytest.raises(ValueError, match="Widen"):
        fit_complex_transmission(SCAN, through_amplitude(
            SCAN, 330e6, KAP, 0.8, 1.0, 0.0, 1e-9))
    # no coupling: no resonance to fit
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="no resolved resonance"):
        fit_complex_transmission(SCAN, 1.0 + _noise(rng, 0.01, SCAN.size),
                                 sigma=0.01)
    # a doublet split by backscattering (Kippenberg et al., the model of
    # split_transmission) is not one resonance: refused by chi-square
    x = 2.0 * (SCAN - F0) / KAP
    b = 1.2
    dbl = 1.0 - 0.8 / (1 - 1j * (x - b)) - 0.8 / (1 - 1j * (x + b))
    with pytest.raises(ValueError, match="does not describe"):
        fit_complex_transmission(SCAN, dbl + _noise(rng, 0.005, SCAN.size),
                                 sigma=0.005)
