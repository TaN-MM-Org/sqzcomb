"""0.14 anchors: inputs that used to give silent wrong answers are
refused, and the molecule's port spectra get the thermal baths and the
marginal-stability handling of the single-port spectra, checked
against that independent single-port path and closed forms."""
import warnings

import numpy as np
import pytest

from sqzcomb import (classical_noise_variance,
                     classical_noise_variance_ports, output_covariance_xxpp,
                     output_entanglement, output_entanglement_spectrum,
                     output_quadrature_variance, output_variance_ports,
                     photonic_molecule, single_mode_parametric)

TWIN = np.array([[-1, 0, 0, 0.5], [0, -1, 0.5, 0],
                 [0, 0.5, -1, 0], [0.5, 0, 0, -1]], dtype=complex)


def test_efficiency_outside_zero_one_is_refused_not_nan():
    M = single_mode_parametric(0.5)
    for eta in (1.2, -0.1, np.nan):
        with pytest.raises(ValueError, match="eta"):
            output_quadrature_variance(M, eta, 0.3, 0, 1)
        with pytest.raises(ValueError, match="eta"):
            output_covariance_xxpp(M, eta, 0.3, 1)
        # 0.13.0 returned E_N = 0 here ("not entangled") with a warning
        with pytest.raises(ValueError, match="eta"):
            output_entanglement_spectrum(TWIN, eta, [0.0], 0, 1, 2)
    # the edges stay allowed: eta = 0 is a cavity with no port (vacuum)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert abs(output_quadrature_variance(M, 0.0, 0.3, 0, 1)
                   - 0.5) < 1e-15
        assert output_entanglement(TWIN, 1.0, 0.0, 0, 1, 2) > 0.0


def test_bad_mode_indices_are_refused():
    M, g = photonic_molecule(0.5, 1.0)
    # -1 used to read mode 1 at the angle -phi
    with pytest.raises(ValueError, match="mode index"):
        output_quadrature_variance(M, 0.5, 0.0, -1, 2, phi=0.4)
    with pytest.raises(ValueError, match="mode index"):
        output_quadrature_variance(M, 0.5, 0.0, 2, 2)
    with pytest.raises(ValueError, match="mode index"):
        output_quadrature_variance(M, 0.5, 0.0, 0.5, 2)
    with pytest.raises(ValueError, match="n_modes"):
        output_quadrature_variance(M, 0.5, 0.0, 0, 3)
    with pytest.raises(ValueError, match="n_modes"):
        output_covariance_xxpp(M, 0.5, 0.0, 1)
    # the same mode twice used to return twice the single-mode variance
    with pytest.raises(ValueError, match="differ"):
        output_quadrature_variance(TWIN, 0.5, 0.0, 1, 2, mode_index_b=1)
    for bad in (-1, 2, [0, -1]):
        with pytest.raises(ValueError, match="mode indices"):
            output_variance_ports(M, g, 0.5, bad, 0.0)
        with pytest.raises(ValueError, match="mode indices"):
            classical_noise_variance_ports(M, g, 0.5, bad, 0.0,
                                           [np.ones(4)], [1.0])
    with pytest.raises(ValueError, match="mode index"):
        classical_noise_variance(M, 0.5, 0.0, [np.ones(4)], [1.0], -1)
    with pytest.raises(ValueError, match="agree|disagree"):
        output_variance_ports(M, [1.0, 1.0, 1.0], 0.5, 0, 0.0)


def test_ports_accept_a_marginal_matrix_only_when_told():
    # the parametric mode exactly at threshold has eigenvalues 0 and -2
    M = single_mode_parametric(1.0)
    with pytest.raises(ValueError, match="marginally stable"):
        output_variance_ports(M, [1.0], 0.6, 0, 0.5)
    for w in (0.5, 2.0):
        for phi in (0.0, np.pi / 2, 0.9):
            a = output_variance_ports(M, [1.0], 0.6, 0, w, phi=phi,
                                      allow_marginal=True)
            b = output_quadrature_variance(M, 0.6, w, 0, 1, phi=phi,
                                           allow_marginal=True)
            assert abs(a - b) < 1e-13
        sq = output_variance_ports(M, [1.0], 0.6, 0, w, phi=np.pi / 2,
                                   allow_marginal=True)
        assert sq == pytest.approx(0.5 * (1 - 4 * 0.6 / (4 + w * w)),
                                   rel=1e-12)
    # a truly unstable matrix stays refused
    with pytest.raises(ValueError, match="unstable"):
        output_variance_ports(single_mode_parametric(1.2), [1.0], 0.6, 0,
                              0.5, allow_marginal=True)


def test_thermal_ports_passive_molecule_is_exactly_thermal():
    rng = np.random.default_rng(3)
    for _ in range(12):
        M, g = photonic_molecule(0.0, rng.uniform(0, 3),
                                 rng.uniform(-2, 2), rng.uniform(-2, 2),
                                 rng.uniform(0.2, 4))
        nb = rng.uniform(0, 3)
        v = output_variance_ports(M, g, rng.uniform(0, 1),
                                  int(rng.integers(0, 2)),
                                  rng.uniform(-4, 4),
                                  phi=rng.uniform(0, np.pi),
                                  n_th_port=nb, n_th_loss=nb)
        assert abs(v - (2 * nb + 1) / 2) < 1e-12


def test_thermal_ports_match_the_single_port_path_and_scaling():
    # J = 0: the main ring alone, a hot loss bath and a colder port,
    # through the two independent code paths
    mu = 0.6
    M, g = photonic_molecule(mu, 0.0, gamma=1.7)
    M1 = single_mode_parametric(mu)
    for eta, w, phi in ((0.3, 0.0, 0.0), (0.8, 1.1, np.pi / 2),
                        (0.55, 2.0, 0.7)):
        a = output_variance_ports(M, g, eta, 0, w, phi=phi,
                                  n_th_port=0.4, n_th_loss=2.5)
        b = output_quadrature_variance(M1, eta, w, 0, 1, phi=phi,
                                       n_th_port=0.4, n_th_loss=2.5)
        assert abs(a - b) < 1e-12
    # resonant molecule: every bath at n multiplies the vacuum spectrum
    # by (2 n + 1) (the quadrature sectors are scalar channels)
    M, g = photonic_molecule(1.5, 1.0, gamma=1.0)
    for w in (0.0, 0.9):
        for phi in (0.0, np.pi / 2):
            vac = output_variance_ports(M, g, 0.9, 1, w, phi=phi)
            hot = output_variance_ports(M, g, 0.9, 1, w, phi=phi,
                                        n_th_port=1.3, n_th_loss=1.3)
            assert hot == pytest.approx(3.6 * vac, rel=1e-12)
    with pytest.raises(ValueError, match="non-negative"):
        output_variance_ports(M, g, 0.9, 1, 0.0, n_th_loss=-0.1)
