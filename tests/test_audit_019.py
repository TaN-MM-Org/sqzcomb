"""Inputs that gave a wrong or NaN answer without an error up to
0.18.0, now refused (0.19). Each test fails on 0.18.0."""
import numpy as np
import pytest

import sqzcomb as sc


def test_noise_spectrum_db_refuses_a_negative_linewidth_gain_or_floor():
    f = np.array([1e6, 2e6])
    good = sc.noise_spectrum_db(0.5, 0.8, 1e7, f)
    # 0.18.0 returned exactly the curve of +1e7 for -1e7 (the spectrum
    # is even in the frequency), the antisqueezed curve as "squeezed"
    # for mu = -0.5 (+8.14 dB), and a floor that lowered the noise
    with pytest.raises(ValueError, match="kappa_hz"):
        sc.noise_spectrum_db(0.5, 0.8, -1e7, f)
    with pytest.raises(ValueError, match="kappa_hz"):
        sc.noise_spectrum_db(0.5, 0.8, 0.0, f)
    with pytest.raises(ValueError, match="mu"):
        sc.noise_spectrum_db(-0.5, 0.8, 1e7, f)
    with pytest.raises(ValueError, match="dark"):
        sc.noise_spectrum_db(0.5, 0.8, 1e7, f, dark=-0.01)
    with pytest.raises(ValueError, match="finite"):
        sc.noise_spectrum_db(0.5, 0.8, 1e7, [np.nan, 2e6])
    # the valid call is unchanged: the closed form of the fitnoise module
    om = 4.0 * np.pi * f / (2.0 * np.pi * 1e7)
    v = 0.5 * (1.0 - 4.0 * 0.8 * 0.5 / (1.5 ** 2 + om ** 2))
    np.testing.assert_allclose(good, 10 * np.log10(v / 0.5),
                               atol=1e-12)             # measured 4e-15


def test_spectra_models_refuse_a_negative_linewidth():
    names, model = sc.parametric_spectra_model()
    with pytest.raises(ValueError, match="kappa_hz"):
        model(dict(mu=0.5, eta=0.8, kappa_hz=-1e7), [1e6])
    names, model = sc.molecule_spectra_model()
    with pytest.raises(ValueError, match="kappa_hz"):
        model(dict(mu=0.5, J=1.0, gamma=1.0, eta=0.8, kappa_hz=-1e7),
              [1e6])


def test_spectra_refuse_nan_frequency_angle_and_occupation():
    M = sc.single_mode_parametric(0.5)
    Mm, g = sc.photonic_molecule(0.5, 1.0)
    calls = [
        lambda: sc.output_quadrature_variance(M, 0.8, np.nan, 0, 1),
        lambda: sc.output_quadrature_variance(M, 0.8, np.inf, 0, 1),
        lambda: sc.output_quadrature_variance(M, 0.8, 0.1, 0, 1,
                                              phi=np.nan),
        lambda: sc.output_quadrature_variance(M, 0.8, 0.1, 0, 1,
                                              n_th_loss=np.nan),
        lambda: sc.output_covariance_xxpp(M, 0.8, np.nan, 1),
        lambda: sc.output_covariance_xxpp(M, 0.8, 0.1, 1,
                                          n_th_port=np.nan),
        lambda: sc.output_variance_ports(Mm, g, 0.8, 1, np.nan),
        lambda: sc.output_variance_ports(Mm, g, 0.8, 1, 0.1, phi=np.nan),
        lambda: sc.output_variance_ports(Mm, g, 0.8, 1, 0.1,
                                         n_th_port=np.nan),
    ]
    for call in calls:
        with pytest.raises(ValueError):
            call()


def test_phase_noise_variance_refuses_negative_variances_and_nan():
    # 0.18.0: -0.079, a negative "detected variance", and NaN
    with pytest.raises(ValueError, match="v_squeezed"):
        sc.phase_noise_variance(-0.1, 2.0, 0.1)
    with pytest.raises(ValueError, match="v_antisqueezed"):
        sc.phase_noise_variance(0.1, np.nan, 0.1)
    with pytest.raises(ValueError, match="static_offset"):
        sc.phase_noise_variance(0.1, 2.0, 0.1, static_offset=np.nan)
    with pytest.raises(ValueError, match="finite"):
        sc.detected_variance(np.nan, 0.8)
    # valid calls unchanged: the exact Gaussian average
    v = sc.phase_noise_variance(0.1, 2.0, 0.1, static_offset=0.05)
    c = np.cos(0.1) * np.exp(-0.02)
    assert v == pytest.approx(0.1 * (1 + c) / 2 + 2.0 * (1 - c) / 2,
                              rel=1e-14)               # measured 0 (exact)


def test_unit_helpers_refuse_nan():
    ring = sc.RingSpec(kappa_hz=100e6, eta_esc=0.8, g0_hz=10.0,
                       lambda_pump_m=1.55e-6, reference="test")
    with pytest.raises(ValueError, match="frequency"):
        sc.thermal_occupation(np.nan, 300.0)
    with pytest.raises(ValueError, match="temperature"):
        sc.thermal_occupation(1e12, np.nan)
    with pytest.raises(ValueError, match="alpha"):
        sc.threshold_power(ring, np.nan)
    with pytest.raises(ValueError, match="rho"):
        sc.intracavity_photons(ring, np.nan)
    assert sc.thermal_occupation(1e12, 0.0) == 0.0


def test_planner_with_a_dark_floor_still_works():
    # the planner steps the dark floor of 0 to -1e-12 and back; it uses
    # the unchecked model path and gives the numbers it gave before
    f = np.linspace(0.5e6, 30e6, 25)
    p = sc.plan_noise_measurement(0.55, 0.62, 12e6, f, sigma_db=0.05,
                                  fit_dark=True)
    assert p["names"][-1] == "dark"
    assert p["identifiable"]
