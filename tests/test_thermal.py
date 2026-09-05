"""Thermal input noise against exact statements: the thermal fixed
point of a passive cavity, the exact (2 n_bar + 1) scaling of the
parametric oscillator's spectra, the intracavity occupation, and the
Bose function's exact crossing points (SI-exact constants)."""
import numpy as np
import pytest

from sqzcomb import (intracavity_covariance, output_quadrature_variance,
                     single_mode_parametric, thermal_occupation)
from sqzcomb.thermal import H_PLANCK, K_BOLTZMANN


def test_passive_cavity_thermal_fixed_point():
    """All baths at n_bar: every output quadrature, at every frequency,
    coupling and phase, sits at exactly (2 n_bar + 1) / 2 -- the
    thermal generalization of the vacuum fixed point."""
    M = single_mode_parametric(0.0)
    nb = 0.7
    for eta in (0.3, 0.5, 1.0):
        for om in (0.0, 0.7, 2.3):
            for phi in (0.0, 0.9):
                v = output_quadrature_variance(M, eta, om, 0, 1, phi=phi,
                                               n_th_port=nb, n_th_loss=nb)
                assert abs(v - (2.0 * nb + 1.0) / 2.0) < 1e-12


def test_parametric_oscillator_thermal_scaling_is_exact():
    """The quadrature sectors of the degenerate parametric oscillator
    are scalar channels, so a uniform bath occupation multiplies the
    whole output spectrum by exactly (2 n_bar + 1)."""
    M = single_mode_parametric(0.6)
    nb = 0.45
    for eta in (0.4, 0.85):
        for om in (0.0, 0.5, 1.5):
            for phi in (0.0, np.pi / 2):
                v0 = output_quadrature_variance(M, eta, om, 0, 1, phi=phi)
                vt = output_quadrature_variance(M, eta, om, 0, 1, phi=phi,
                                                n_th_port=nb,
                                                n_th_loss=nb)
                assert abs(vt - (2.0 * nb + 1.0) * v0) < 1e-12


def test_hot_loss_cold_port_matches_the_hand_derived_mix():
    """Single passive mode, cold port and hot intrinsic loss: the
    output is the hand-computable mixture
    v_ex |2 eta G - 1|^2 / 2 ... with G = 1/(1 - i omega)."""
    M = single_mode_parametric(0.0)
    nb_loss, eta, om = 1.3, 0.6, 0.8
    v = output_quadrature_variance(M, eta, om, 0, 1,
                                   n_th_port=0.0, n_th_loss=nb_loss)
    G = 1.0 / (1.0 - 1j * om)
    w_ex = abs(2.0 * eta * G - 1.0) ** 2
    w_0 = 4.0 * eta * (1.0 - eta) * abs(G) ** 2
    expected = 0.5 * w_ex + 0.5 * (2.0 * nb_loss + 1.0) * w_0
    assert abs(v - expected) < 1e-12


def test_intracavity_occupation_equals_the_bath_occupation():
    M = single_mode_parametric(0.0)
    nb = 0.7
    V = intracavity_covariance(M, np.array([1.0]), n_th=nb)
    assert abs(V[1, 1].real - nb) < 1e-12      # <a^dag a> block
    assert abs(V[0, 0].real - (nb + 1.0)) < 1e-12


def test_bose_function_exact_points_and_limits():
    """n_bar = 1 exactly at hbar omega = k_B T ln 2; the constants are
    the exact SI values, so the crossing is exact arithmetic."""
    T = 0.1
    f = K_BOLTZMANN * T * np.log(2.0) / H_PLANCK
    assert abs(thermal_occupation(f, T) - 1.0) < 1e-12
    assert thermal_occupation(1e14, 0.0) == 0.0
    # optical frequency (193 THz, 1550 nm) at room temperature:
    # x = h f / k_B T ~ 31, so n_bar = e^{-31} ~ 4e-14 -- negligible
    # against any squeezing measurement, which is why vacuum inputs
    # are the right default in the optical domain
    nb_opt = thermal_occupation(193e12, 300.0)
    assert 1e-15 < nb_opt < 1e-12
    with pytest.raises(ValueError):
        thermal_occupation(-1.0, 300.0)
    with pytest.raises(ValueError):
        thermal_occupation(1e14, -1.0)


def test_negative_occupations_are_refused():
    M = single_mode_parametric(0.0)
    with pytest.raises(ValueError):
        output_quadrature_variance(M, 0.5, 0.0, 0, 1, n_th_port=-0.1)
    with pytest.raises(ValueError):
        intracavity_covariance(M, np.array([1.0]), n_th=-0.1)
