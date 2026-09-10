"""v0.8 anchors: detected entanglement -- the output-field covariance
and the logarithmic negativity a two-line homodyne pair would certify
at the detector, held to exact identities: every quadrature read off
the matrix equals the independent scalar spectra path; passive
thermal outputs are exactly ((2 nbar + 1)/2) I and separable; the
symmetric twin-beam E_N(omega) equals -ln(2 V_EPR_min(omega)) from
the independent joint-quadrature path; E_N vanishes at large omega
and is monotone under detection loss."""
import numpy as np
import pytest

from sqzcomb.spectra import (output_covariance_xxpp, output_entanglement,
                             output_entanglement_spectrum,
                             output_quadrature_variance)


def _twin(mu):
    """Nondegenerate parametric (twin-beam) drift in (a1, a2, a1+, a2+):
    da1 = -a1 + mu a2+, da2 = -a2 + mu a1+; stable for |mu| < 1."""
    return np.array([[-1, 0, 0, mu], [0, -1, mu, 0],
                     [0, mu, -1, 0], [mu, 0, 0, -1]], dtype=complex)


def test_matrix_and_scalar_spectra_paths_agree_exactly():
    M = _twin(0.5)
    sig = output_covariance_xxpp(M, 0.8, 0.3, 2)
    for k in range(2):
        for phi in (0.0, np.pi / 2, 0.7):
            v = output_quadrature_variance(M, 0.8, 0.3, k, 2, phi=phi)
            u = np.zeros(4)
            u[k] = np.cos(phi)
            u[2 + k] = np.sin(phi)
            assert abs(u @ sig @ u - v) < 1e-13
    for phi in (0.0, 1.1):
        v = output_quadrature_variance(M, 0.8, 0.3, 0, 2, phi=phi,
                                       mode_index_b=1)
        u = np.zeros(4)
        u[0] = u[1] = np.cos(phi) / np.sqrt(2)
        u[2] = u[3] = np.sin(phi) / np.sqrt(2)
        assert abs(u @ sig @ u - v) < 1e-13


def test_passive_thermal_output_is_exactly_thermal_and_separable():
    Mp = -np.eye(4, dtype=complex)
    for nb in (0.0, 1.7):
        for eta, w in ((0.3, 0.0), (0.9, 2.3)):
            sig = output_covariance_xxpp(Mp, eta, w, 2, n_th_port=nb,
                                         n_th_loss=nb)
            assert np.abs(sig - (2 * nb + 1) / 2 *
                          np.eye(4)).max() < 1e-12
            assert output_entanglement(Mp, eta, w, 0, 1, 2,
                                       n_th_port=nb,
                                       n_th_loss=nb) < 1e-12


def test_twin_beam_entanglement_equals_minus_log_min_epr_variance():
    """For the symmetric twin beam the PPT symplectic eigenvalue IS
    the minimal joint quadrature variance, so E_N(omega) must equal
    -ln(2 V_EPR_min(omega)) computed by the independent, previously
    anchored scalar spectra path."""
    M = _twin(0.5)
    for eta, w in ((0.8, 0.3), (0.6, 1.0)):
        EN = output_entanglement(M, eta, w, 0, 1, 2)
        assert EN > 0.0
        vmin = min(output_quadrature_variance(M, eta, w, 0, 2, phi=phi,
                                              mode_index_b=1)
                   for phi in np.linspace(0.0, np.pi, 4001))
        assert abs(EN + np.log(2.0 * vmin)) < 1e-6


def test_entanglement_dies_at_large_omega_and_zero_pump():
    M = _twin(0.5)
    assert output_entanglement(M, 0.8, 1e6, 0, 1, 2) < 1e-12
    assert output_entanglement(_twin(0.0), 0.8, 0.0, 0, 1, 2) < 1e-12
    spec = output_entanglement_spectrum(M, 0.8, [0.0, 1.0, 5.0, 50.0],
                                        0, 1, 2)
    assert np.all(np.diff(spec) <= 1e-12)      # decays away from carrier


def test_detection_loss_never_creates_entanglement():
    M = _twin(0.6)
    EN = output_entanglement(M, 0.9, 0.2, 0, 1, 2)
    for eta_d in (0.9, 0.5, 0.2):
        ENl = output_entanglement(M, 0.9, 0.2, 0, 1, 2,
                                  detection_efficiency=eta_d)
        assert ENl <= EN + 1e-12
        EN = ENl


def test_unstable_and_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        output_covariance_xxpp(_twin(1.5), 0.8, 0.3, 2)   # above threshold
    with pytest.raises(ValueError):
        output_covariance_xxpp(_twin(0.5), 0.8, 0.3, 3)   # wrong n_modes
    with pytest.raises(ValueError):
        output_entanglement(_twin(0.5), 0.8, 0.3, 0, 1, 2,
                            n_th_port=-1.0)
