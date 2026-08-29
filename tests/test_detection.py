"""Detection-model tests: vacuum fixed point, exact loss composition,
physicality of the channel, scalar/matrix agreement, closed-form
numbers."""
import numpy as np
import pytest

from sqzcomb import (covariance_xxpp, intracavity_covariance,
                     single_mode_parametric, symplectic_eigenvalues)
from sqzcomb.detection import (dark_from_clearance_db, detected_squeezing_db,
                               detected_variance, lossy_channel_xxpp,
                               required_efficiency, required_efficiency_db)


def _squeezed_sigma(hbar=2.0):
    # a genuinely squeezed intracavity state from the package's own
    # machinery: below-threshold single-mode parametric interaction
    M = single_mode_parametric(mu=0.6)
    V = intracavity_covariance(M, gammas=np.ones(1))
    return covariance_xxpp(V, hbar=hbar)


def test_vacuum_is_a_fixed_point_of_pure_loss():
    for eta in (0.1, 0.37, 0.9, 1.0):
        assert detected_variance(0.5, eta) == pytest.approx(0.5, abs=1e-15)
    sigma_vac = np.eye(2)
    for eta in (0.1, 0.37, 0.9, 1.0):
        out = lossy_channel_xxpp(sigma_vac, eta, hbar=2.0)
        assert np.allclose(out, sigma_vac, atol=1e-15)


def test_loss_composition_is_exact():
    e1, e2 = 0.83, 0.61
    V = 0.19
    once = detected_variance(V, e1 * e2)
    twice = detected_variance(detected_variance(V, e1), e2)
    assert once == pytest.approx(twice, abs=1e-15)
    sigma = _squeezed_sigma()
    a = lossy_channel_xxpp(sigma, e1 * e2)
    b = lossy_channel_xxpp(lossy_channel_xxpp(sigma, e1), e2)
    assert np.allclose(a, b, atol=1e-13)


def test_channel_preserves_physicality():
    sigma = _squeezed_sigma()
    # the source really is squeezed below vacuum
    assert sigma.min() < 1.0 or np.linalg.eigvalsh(sigma).min() < 1.0
    for eta in (0.05, 0.5, 0.95):
        out = lossy_channel_xxpp(sigma, eta, hbar=2.0)
        nu = symplectic_eigenvalues(out, hbar=2.0)
        assert np.all(nu >= 1.0 - 1e-10)   # hbar/2 = 1 in this scaling


def test_scalar_and_channel_forms_agree_on_one_mode():
    # hbar = 1 makes the xxpp vacuum variance 1/2, the scalar convention
    sigma = _squeezed_sigma(hbar=1.0)
    eta = 0.44
    out = lossy_channel_xxpp(sigma, eta, hbar=1.0)
    for q in (0, 1):                        # x and p quadratures
        assert out[q, q] == pytest.approx(
            detected_variance(sigma[q, q], eta), abs=1e-13)


def test_three_db_at_half_efficiency_closed_form():
    V = 0.25                                # exactly 3.010 dB squeezed
    got = detected_squeezing_db(V, efficiency=0.5)
    assert got == pytest.approx(10.0 * np.log10(0.75), abs=1e-12)


def test_dark_noise_conversion_and_effect():
    dark = dark_from_clearance_db(10.0)
    assert dark == pytest.approx(0.05, abs=1e-15)
    # infinitely clean receiver adds nothing
    assert dark_from_clearance_db(300.0) == pytest.approx(0.0, abs=1e-15)
    V = detected_variance(0.05, 1.0, dark)  # 10 dB source, 10 dB clearance
    assert 10.0 * np.log10(V / 0.5) == pytest.approx(
        10.0 * np.log10(0.2), abs=1e-12)


def test_required_efficiency_round_trip():
    Vs = 0.05
    for eta in (0.3, 0.62, 0.99):
        Vt = detected_variance(Vs, eta)
        assert required_efficiency(Vt, Vs) == pytest.approx(eta, abs=1e-13)
    assert required_efficiency_db(-3.0, -10.0) == pytest.approx(
        (0.5 - 0.5 * 10 ** -0.3) / (0.5 - 0.05), abs=1e-13)


def test_per_mode_efficiencies():
    sigma = np.diag([0.4, 0.6, 2.0, 1.2])   # 2 modes, hbar = 1 units
    out = lossy_channel_xxpp(sigma, [1.0, 0.5], hbar=1.0)
    # mode 1 untouched, mode 2 pulled halfway toward vacuum 0.5
    assert out[0, 0] == pytest.approx(0.4, abs=1e-15)
    assert out[2, 2] == pytest.approx(2.0, abs=1e-15)
    assert out[1, 1] == pytest.approx(0.5 * 0.6 + 0.25, abs=1e-15)
    assert out[3, 3] == pytest.approx(0.5 * 1.2 + 0.25, abs=1e-15)


def test_input_validation():
    with pytest.raises(ValueError):
        detected_variance(0.3, efficiency=0.0)
    with pytest.raises(ValueError):
        detected_variance(0.3, efficiency=1.2)
    with pytest.raises(ValueError):
        detected_variance(-0.1)
    with pytest.raises(ValueError):
        detected_variance(0.3, dark_noise=-0.01)
    with pytest.raises(ValueError):
        dark_from_clearance_db(-3.0)
    with pytest.raises(ValueError):
        lossy_channel_xxpp(np.eye(3), 0.9)
    with pytest.raises(ValueError):
        lossy_channel_xxpp(np.eye(4), [0.9, 0.5, 0.2])
    with pytest.raises(ValueError):
        required_efficiency(0.6, 0.05)
    with pytest.raises(ValueError):
        required_efficiency(0.02, 0.05)
