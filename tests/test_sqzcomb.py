import numpy as np
import pytest

from sqzcomb import (lle_evolve, homogeneous_steady_states,
                     fluctuation_matrix, single_mode_parametric,
                     output_quadrature_variance, squeezing_db)


def test_free_decay():
    n = 64
    psi0 = np.ones(n, dtype=complex)
    psi = lle_evolve(psi0, F=0.0, alpha=0.0, t_end=3.0, dt=0.005)
    assert np.allclose(np.abs(psi), np.exp(-3.0), rtol=1e-3)


def test_flat_state_matches_cubic():
    F, alpha = 1.1, 0.5
    rho = homogeneous_steady_states(F, alpha)
    # evolve from small seed to the (single) flat state
    psi = lle_evolve(np.full(32, 0.1 + 0j), F=F, alpha=alpha,
                     t_end=200.0, dt=0.01)
    assert np.allclose(np.abs(psi) ** 2, rho[0], rtol=1e-4)
    # residual of the cubic at the solver's intensity
    r = np.abs(psi[0]) ** 2
    assert np.isclose(r * (1 + (alpha - r) ** 2), F * F, rtol=1e-3)


def test_vacuum_passes_through_passive_cavity():
    M = np.diag([-1.0 + 0.3j, -1.0 - 0.3j]).astype(complex)
    rng = np.random.default_rng(0)
    for _ in range(10):
        eta = rng.uniform(0.05, 1.0)
        omega = rng.uniform(-3, 3)
        phi = rng.uniform(0, np.pi)
        v = output_quadrature_variance(M, eta, omega, mode_index=0,
                                       n_modes=1, phi=phi)
        assert np.isclose(v, 0.5, atol=1e-12)


def test_matches_parametric_oscillator_closed_form():
    # S_Y(Omega) = 1 - 4 eta mu / ((1 + mu)^2 + Omega^2), vacuum = 1
    for mu in (0.3, 0.9):
        M = single_mode_parametric(mu)
        for eta in (0.3, 0.5, 1.0):
            for omega in (0.0, 0.7, 2.5):
                v = output_quadrature_variance(M, eta, omega, mode_index=0,
                                               n_modes=1, phi=np.pi / 2)
                analytic = 0.5 * (1 - 4 * eta * mu / ((1 + mu) ** 2 + omega ** 2))
                assert np.isclose(v, analytic, rtol=1e-10), (mu, eta, omega)


def test_three_db_extraction_limit_at_critical_coupling():
    # eta = 1/2: as mu -> 1 the detectable squeezing saturates at 3 dB
    M = single_mode_parametric(0.9999)
    v = output_quadrature_variance(M, 0.5, 0.0, 0, 1, phi=np.pi / 2)
    assert squeezing_db(v) > -3.011
    assert squeezing_db(v) < -2.99
    # full extraction breaks the limit (checked away from the mu -> 1
    # numerical degeneracy: at mu = 0.99, eta = 1 the closed form gives
    # about -46 dB)
    M99 = single_mode_parametric(0.99)
    v_full = output_quadrature_variance(M99, 1.0, 0.0, 0, 1, phi=np.pi / 2)
    assert squeezing_db(v_full) < -20.0


def test_above_threshold_rejected():
    M = single_mode_parametric(1.2)
    with pytest.raises(ValueError):
        output_quadrature_variance(M, 0.5, 0.0, 0, 1)


def test_lle_linearization_stable_below_threshold():
    F, alpha = 0.8, 0.2
    psi = lle_evolve(np.full(32, 0.05 + 0j), F=F, alpha=alpha,
                     t_end=200.0, dt=0.01)
    M, modes = fluctuation_matrix(psi, alpha)
    assert np.max(np.linalg.eigvals(M).real) < 0.0
    # and the machinery produces a finite spectrum on the pumped mode
    i0 = int(np.where(modes == 0)[0][0])
    v = output_quadrature_variance(M, 0.5, 0.0, i0, modes.size)
    assert np.isfinite(v) and v > 0.0
