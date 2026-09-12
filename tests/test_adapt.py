"""v0.9 adaptability anchors: the lab-units bridge round-trips and is
cross-validated through the package's own flat-state cubic; the
noise-spectrum fit recovers generating parameters, matches the
independent closed form, reports honest uncertainties, and refuses
the under-determined and unphysical cases."""
import dataclasses

import numpy as np
import pytest

from sqzcomb import (RingSpec, fit_noise_spectra, homogeneous_steady_states,
                     intracavity_photons, noise_spectrum_db,
                     normalized_detuning, normalized_dispersion,
                     normalized_frequency, normalized_pump,
                     physical_frequency, threshold_power)

SPEC = RingSpec(kappa_hz=100e6, eta_esc=0.8, g0_hz=10.0,
                lambda_pump_m=1.55e-6,
                reference="synthetic test values for the test suite")


# ------------------------- physical bridge -------------------------


def test_conversions_round_trip_and_scale():
    f = np.array([1e6, 25e6, 400e6])
    assert np.allclose(physical_frequency(SPEC, normalized_frequency(SPEC, f)),
                       f, rtol=1e-15)
    # F proportional to sqrt(P); kappa^{-3/2} at fixed eta_esc, g0
    P = 2e-3
    F1 = normalized_pump(SPEC, P)
    assert np.isclose(normalized_pump(SPEC, 4 * P), 2.0 * F1, rtol=1e-12)
    # F^2 = 8 g0 (eta kappa) P / (kappa^3 hbar w) = 8 g0 eta P/(kappa^2 hbar w):
    # at FIXED eta_esc one power of kappa cancels, so 2x kappa halves F
    spec2 = dataclasses.replace(SPEC, kappa_hz=2 * SPEC.kappa_hz)
    assert np.isclose(normalized_pump(spec2, P), F1 / 2.0, rtol=1e-12)
    # detuning and dispersion scale as 1/kappa, signs as documented
    assert np.isclose(normalized_detuning(SPEC, 50e6),
                      2 * (2 * np.pi * 50e6) / SPEC.kappa, rtol=1e-15)
    assert normalized_dispersion(SPEC, D2_hz=1e5) < 0    # anomalous -> d2 < 0


def test_threshold_power_hits_the_cubic_root_exactly():
    """At threshold_power(alpha), the normalized pump satisfies
    F^2 = 1 + (alpha-1)^2, and the package's own flat-state cubic has
    the rho = 1 root -- the identity is checked through
    `homogeneous_steady_states`, not restated."""
    for alpha in (0.7, 1.0, 1.8):
        P_th = threshold_power(SPEC, alpha)
        F = normalized_pump(SPEC, P_th)
        assert np.isclose(F ** 2, 1.0 + (alpha - 1.0) ** 2, rtol=1e-12)
        roots = homogeneous_steady_states(F, alpha)
        assert np.min(np.abs(np.asarray(roots) - 1.0)) < 1e-10


def test_photon_number_and_refusals():
    N = intracavity_photons(SPEC, rho=1.0)
    assert np.isclose(N, SPEC.kappa / (2 * SPEC.g0), rtol=1e-15)
    with pytest.raises(ValueError):
        RingSpec(-1.0, 0.8, 10.0, 1.55e-6, "x")
    with pytest.raises(ValueError):
        RingSpec(1e8, 1.5, 10.0, 1.55e-6, "x")     # eta_esc > 1
    with pytest.raises(ValueError):
        RingSpec(1e8, 0.8, 10.0, 1.55e-6, "  ")    # missing reference
    with pytest.raises(ValueError):
        normalized_pump(SPEC, -1.0)


# ------------------------- noise-spectrum fit ----------------------

MU, ETA, KAP = 0.6, 0.72, 80e6
F_AX = np.linspace(2e6, 4e8, 60)


def _clean():
    sq = noise_spectrum_db(MU, ETA, KAP, F_AX, "squeezed")
    an = noise_spectrum_db(MU, ETA, KAP, F_AX, "anti")
    return sq, an


def test_model_equals_independent_closed_form():
    """The fit's model comes from the package solver; the closed form
    S = 0.5 [1 -/+ 4 eta mu / ((1 +/- mu)^2 + Omega^2)] is the
    independent path."""
    om = normalized_frequency(RingSpec(KAP, 0.9, 10.0, 1.55e-6, "x"),
                              F_AX)
    sq, an = _clean()
    v_sq = 0.5 * (1 - 4 * ETA * MU / ((1 + MU) ** 2 + om ** 2))
    v_an = 0.5 * (1 + 4 * ETA * MU / ((1 - MU) ** 2 + om ** 2))
    assert np.allclose(sq, 10 * np.log10(v_sq / 0.5), atol=1e-10)
    assert np.allclose(an, 10 * np.log10(v_an / 0.5), atol=1e-10)


def test_noise_free_recovery_both_traces():
    sq, an = _clean()
    fit = fit_noise_spectra(F_AX, sq, an)
    assert abs(fit.mu - MU) < 1e-6
    assert abs(fit.eta - ETA) < 1e-6
    assert abs(fit.kappa_hz - KAP) / KAP < 1e-6
    # single trace + known kappa also identified
    fit1 = fit_noise_spectra(F_AX, sq_db=sq, kappa_hz=KAP)
    assert abs(fit1.mu - MU) < 1e-5 and abs(fit1.eta - ETA) < 1e-5


def test_monte_carlo_scatter_matches_reported_sigma():
    sq, an = _clean()
    rng = np.random.default_rng(0)
    sd = 0.1                                   # dB
    fits = []
    for _ in range(40):
        f = fit_noise_spectra(F_AX, sq + rng.normal(0, sd, sq.size),
                              an + rng.normal(0, sd, an.size),
                              sigma_db=sd)
        fits.append([f.mu, f.eta, f.kappa_hz])
    rep = fit_noise_spectra(F_AX, sq + rng.normal(0, sd, sq.size),
                            an + rng.normal(0, sd, an.size),
                            sigma_db=sd)
    reported = np.array([rep.sigma["mu"], rep.sigma["eta"],
                         rep.sigma["kappa_hz"]])
    emp = np.asarray(fits).std(axis=0)
    assert np.all(emp < 2.0 * reported) and np.all(emp > 0.4 * reported)
    assert rep.chi2 / rep.chi2_dof < 2.0


def test_dark_floor_fitted_when_asked():
    sq, an = _clean()
    dark = 0.05
    sq_d = noise_spectrum_db(MU, ETA, KAP, F_AX, "squeezed", dark=dark)
    an_d = noise_spectrum_db(MU, ETA, KAP, F_AX, "anti", dark=dark)
    fit = fit_noise_spectra(F_AX, sq_d, an_d, fit_dark=True)
    assert abs(fit.dark - dark) < 1e-4
    assert abs(fit.eta - ETA) < 1e-4
    # ignoring the floor would bias eta low; fitting it recovers truth
    fit0 = fit_noise_spectra(F_AX, sq_d, an_d)
    assert abs(fit0.eta - ETA) > abs(fit.eta - ETA)


def test_refusals():
    sq, an = _clean()
    with pytest.raises(ValueError, match="single Lorentzian"):
        fit_noise_spectra(F_AX, sq_db=sq)          # under-determined
    with pytest.raises(ValueError, match="below shot"):
        fit_noise_spectra(F_AX, sq, -np.abs(an))   # impossible trace
    with pytest.raises(ValueError):
        fit_noise_spectra(F_AX[:3], sq[:3], an[:3])  # too few points
    with pytest.raises(ValueError):
        fit_noise_spectra(F_AX, sq, an, sigma_db=0.0)
