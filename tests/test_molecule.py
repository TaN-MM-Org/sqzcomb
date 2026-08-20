import math

import numpy as np
import pytest

from sqzcomb import (molecule_threshold, output_variance_ports,
                     photonic_molecule, squeezing_db)
from sqzcomb.linearize import is_stable


def test_decoupled_molecule_reduces_to_single_mode_closed_form():
    # J = 0: port on the Kerr ring reproduces the degenerate
    # parametric-oscillator spectrum exactly
    for mu in (0.3, 0.9):
        M, g = photonic_molecule(mu, J=0.0)
        for eta in (0.3, 0.5, 1.0):
            for omega in (0.0, 0.7, 2.5):
                v = output_variance_ports(M, g, eta, port_mode=0,
                                          omega=omega, phi=np.pi / 2)
                analytic = 0.5 * (1 - 4 * eta * mu
                                  / ((1 + mu) ** 2 + omega ** 2))
                assert np.isclose(v, analytic, atol=1e-12), (mu, eta, omega)


def test_passive_molecule_returns_exact_vacuum_everywhere():
    # mu = 0: any J, gamma, detunings, eta, port, frequency, quadrature
    rng = np.random.default_rng(7)
    for _ in range(25):
        M, g = photonic_molecule(0.0, J=rng.uniform(0.0, 3.0),
                                 delta_a=rng.uniform(-2, 2),
                                 delta_b=rng.uniform(-2, 2),
                                 gamma=rng.uniform(0.2, 5.0))
        v = output_variance_ports(M, g, eta=rng.uniform(0.0, 1.0),
                                  port_mode=int(rng.integers(0, 2)),
                                  omega=rng.uniform(-3, 3),
                                  phi=rng.uniform(0, np.pi))
        assert np.isclose(v, 0.5, atol=1e-12)


def test_passive_supermodes_split_by_two_J():
    J = 0.8
    M, _ = photonic_molecule(0.0, J=J, gamma=1.0)
    ev = np.linalg.eigvals(M[:2, :2])
    assert np.allclose(np.sort(ev.imag), [-J, J], atol=1e-12)
    assert np.allclose(ev.real, -1.0, atol=1e-12)


def test_threshold_static_and_hopf_branches():
    # static branch mu_th = 1 + J^2/gamma for J <= gamma, Hopf 1 + gamma
    # beyond; both checked directly against the eigenvalues
    for J, gamma in [(0.6, 1.0), (0.3, 0.5), (2.0, 1.0), (1.5, 0.8)]:
        th = molecule_threshold(J, gamma)
        expect = min(1.0 + J * J / gamma, 1.0 + gamma)
        assert np.isclose(th, expect, atol=1e-14)
        M_lo, _ = photonic_molecule(0.999 * th, J, gamma=gamma)
        M_hi, _ = photonic_molecule(1.001 * th, J, gamma=gamma)
        assert is_stable(M_lo) and not is_stable(M_hi)


def test_hop_rotates_squeezed_quadrature_by_quarter_turn():
    # squeezed at phi = pi/2 in the Kerr ring, phi = 0 at the auxiliary
    # port
    M, g = photonic_molecule(0.8, J=1.0, gamma=1.0)
    v_sq = output_variance_ports(M, g, 1.0, port_mode=1, omega=0.0, phi=0.0)
    v_anti = output_variance_ports(M, g, 1.0, port_mode=1, omega=0.0,
                                   phi=np.pi / 2)
    assert v_sq < 0.5 < v_anti


def test_auxiliary_port_at_zero_frequency_is_exact_effective_single_mode():
    # Omega = 0: molecule detected through ring b equals a single
    # parametric mode with total decay 1 + J^2/gamma and escape
    # efficiency eta_b (J^2/gamma)/(1 + J^2/gamma), exactly
    for J, gamma, eta_b, mu_frac in [(1.2, 1.0, 1.0, 0.9),
                                     (2.0, 4.0, 0.8, 0.95),
                                     (0.9, 1.5, 1.0, 0.99),
                                     (1.0, 1.0, 0.6, 0.5)]:
        k_ad = J * J / gamma
        mu = mu_frac * molecule_threshold(J, gamma)
        M, g = photonic_molecule(mu, J, gamma=gamma)
        v = output_variance_ports(M, g, eta_b, port_mode=1, omega=0.0,
                                  phi=0.0)
        mu_eff = mu / (1.0 + k_ad)
        eta_eff = eta_b * k_ad / (1.0 + k_ad)
        analytic = 0.5 * (1 - 4 * eta_eff * mu_eff / (1 + mu_eff) ** 2)
        assert np.isclose(v, analytic, atol=1e-12), (J, gamma, eta_b)


def test_molecule_passes_three_db_with_unported_kerr_ring():
    # J^2/gamma = 3, lossless-extraction auxiliary ring: effective escape
    # efficiency 3/4 > 1/2, so the detected squeezing passes the 3 dB
    # value associated with a critically coupled single ring, although
    # the Kerr ring itself has no extraction port. At mu = 0.99 mu_th the
    # closed form gives exactly 0.25 * (vacuum) * (1 + tiny); we assert
    # the -6 dB neighborhood from the machinery.
    J, gamma = math.sqrt(27.0), 9.0            # J^2/gamma = 3, J < gamma
    th = molecule_threshold(J, gamma)
    assert np.isclose(th, 4.0, atol=1e-12)
    M, g = photonic_molecule(0.99 * th, J, gamma=gamma)
    v = output_variance_ports(M, g, 1.0, port_mode=1, omega=0.0, phi=0.0)
    db = squeezing_db(v)
    assert db < -3.0                           # past the 3 dB value
    assert -6.03 < db < -6.01                  # the closed-form -6.02 dB


def test_rejects_unstable_state_and_bad_arguments():
    M, g = photonic_molecule(1.5, J=0.0)       # above single-ring threshold
    with pytest.raises(ValueError):
        output_variance_ports(M, g, 0.5, port_mode=0, omega=0.0)
    with pytest.raises(ValueError):
        photonic_molecule(0.5, J=1.0, gamma=0.0)
    M2, g2 = photonic_molecule(0.5, J=1.0)
    with pytest.raises(ValueError):
        output_variance_ports(M2, g2, 1.5, port_mode=0, omega=0.0)
    with pytest.raises(ValueError):
        molecule_threshold(1.0, -1.0)
