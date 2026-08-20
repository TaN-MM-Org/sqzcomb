import numpy as np
import pytest

from sqzcomb import (fluctuation_matrix, lle_evolve,
                     molecule_fluctuation_matrix, output_quadrature_variance,
                     output_variance_ports, photonic_molecule)

ALPHA, F, D2 = 1.2, 1.05, -0.15


@pytest.fixture(scope="module")
def flat_state():
    # above the bare MI threshold of the +-3 pair (the molecule's added
    # loss is what stabilizes it in the tests below; the J = 0 stability
    # guard on this state is asserted in test_multimode_guards)
    psi = lle_evolve(np.full(64, 0.1 + 0j), F=F, alpha=ALPHA,
                     dispersion=(D2,), t_end=400.0, dt=0.005)
    assert np.ptp(np.abs(psi)) < 1e-8      # genuinely flat
    return psi


@pytest.fixture(scope="module")
def quiet_state():
    # below every pair threshold: usable at J = 0
    psi = lle_evolve(np.full(64, 0.1 + 0j), F=0.9, alpha=0.5,
                     dispersion=(D2,), t_end=300.0, dt=0.005)
    assert np.ptp(np.abs(psi)) < 1e-8
    return psi


def test_single_line_equals_released_two_ring(flat_state):
    # at one retained line the multimode builder must equal the v0.2
    # two-ring matrix with mu = i psi^2 and delta_a = 2|psi|^2 - alpha
    psi0 = flat_state[0]
    M_multi, gammas, _ = molecule_fluctuation_matrix(
        flat_state, ALPHA, J=0.7, gamma_b=1.3, dispersion=(D2,),
        aux_delta=0.4, modes=[0])
    M_two, gammas_two = photonic_molecule(
        1j * psi0 ** 2, 0.7, delta_a=2 * abs(psi0) ** 2 - ALPHA,
        delta_b=0.4, gamma=1.3)
    assert np.allclose(M_multi, M_two, atol=1e-14)
    assert np.allclose(gammas, gammas_two, atol=1e-15)


def test_decoupled_multimode_reduces_to_plain_fluctuations(quiet_state):
    modes = [-3, 0, 3]
    m = len(modes)
    M_multi, gammas, _ = molecule_fluctuation_matrix(
        quiet_state, 0.5, J=0.0, gamma_b=2.0, dispersion=(D2,),
        modes=modes)
    M_ref, _ = fluctuation_matrix(quiet_state, 0.5, (D2,), modes)
    idx = np.r_[0:m, 2 * m:3 * m]
    assert np.allclose(M_multi[np.ix_(idx, idx)], M_ref, atol=1e-14)
    # detected spectra agree with the v0.1 machinery, single and joint
    for eta in (0.4, 1.0):
        for omega in (0.0, 0.9):
            v_new = output_variance_ports(M_multi, gammas, eta,
                                          port_mode=1, omega=omega, phi=0.3)
            v_old = output_quadrature_variance(M_ref, eta, omega,
                                               mode_index=1, n_modes=m,
                                               phi=0.3)
            assert np.isclose(v_new, v_old, atol=1e-12)
    v_new = output_variance_ports(M_multi, gammas, 0.7, port_mode=[0, 2],
                                  omega=0.5, phi=1.0, mode_index=[0, 2])
    v_old = output_quadrature_variance(M_ref, 0.7, 0.5, mode_index=0,
                                       n_modes=m, phi=1.0, mode_index_b=2)
    assert np.isclose(v_new, v_old, atol=1e-12)


def test_passive_multimode_molecule_returns_exact_vacuum():
    psi_vac = np.zeros(16, dtype=complex)
    rng = np.random.default_rng(3)
    for _ in range(10):
        M, gammas, _ = molecule_fluctuation_matrix(
            psi_vac, rng.uniform(-1, 1), J=rng.uniform(0, 2),
            gamma_b=rng.uniform(0.3, 4), aux_delta=rng.uniform(-1, 1),
            modes=[-2, 0, 2])
        ports = [3, 5]                     # auxiliary lines of -2 and +2
        v = output_variance_ports(M, gammas, rng.uniform(0, 1),
                                  port_mode=ports,
                                  omega=rng.uniform(-2, 2),
                                  phi=rng.uniform(0, np.pi),
                                  mode_index=ports)
        assert np.isclose(v, 0.5, atol=1e-12)


def test_zero_frequency_schur_equivalence(flat_state):
    # resonant auxiliary ring at Omega = 0 is exactly a single ring with
    # extra loss J^2/gamma_b per line, escape efficiency
    # eta (J^2/gamma_b)/(1 + J^2/gamma_b), and the quarter-turn
    # quadrature rotation of the -iJ hop
    modes = [-3, 0, 3]
    m = len(modes)
    J, gamma_b, eta = 1.0, 1.0, 1.0
    k_ad = J * J / gamma_b
    k_tot = 1.0 + k_ad
    M_mol, gammas, _ = molecule_fluctuation_matrix(
        flat_state, ALPHA, J=J, gamma_b=gamma_b, dispersion=(D2,),
        aux_delta=0.0, modes=modes)
    M_ref, _ = fluctuation_matrix(flat_state, ALPHA, (D2,), modes)
    M_red = (M_ref - k_ad * np.eye(2 * m)) / k_tot
    eta_eff = eta * k_ad / k_tot
    for phi in (0.0, 0.7, 3 * np.pi / 4):
        v_aux = output_variance_ports(M_mol, gammas, eta, port_mode=[3, 5],
                                      omega=0.0, phi=phi, mode_index=[3, 5])
        v_red = output_quadrature_variance(M_red, eta_eff, 0.0,
                                           mode_index=0, n_modes=m,
                                           phi=phi + np.pi / 2,
                                           mode_index_b=2)
        assert np.isclose(v_aux, v_red, atol=1e-12), phi


def test_aux_extraction_beats_main_when_adiabatic_rate_dominates(flat_state):
    # J^2/gamma_b = 3: the auxiliary bus carries 3/4 of each line's total
    # decay, the main-ring bus at most 1/4, so the twin-beam squeezing
    # detected through the auxiliary ring is strictly deeper
    modes = [-3, 0, 3]
    J, gamma_b = np.sqrt(3.0), 1.0
    M_mol, gammas, _ = molecule_fluctuation_matrix(
        flat_state, ALPHA, J=J, gamma_b=gamma_b, dispersion=(D2,),
        aux_delta=0.0, modes=modes)
    phis = np.linspace(0.0, np.pi, 91)
    v_aux = min(output_variance_ports(M_mol, gammas, 1.0, port_mode=[3, 5],
                                      omega=0.0, phi=p, mode_index=[3, 5])
                for p in phis)
    v_main = min(output_variance_ports(M_mol, gammas, 1.0, port_mode=[0, 2],
                                       omega=0.0, phi=p, mode_index=[0, 2])
                 for p in phis)
    assert v_aux < 0.5 and v_main < 0.5
    assert v_aux < v_main


def test_multimode_guards(flat_state):
    with pytest.raises(ValueError):
        molecule_fluctuation_matrix(flat_state, ALPHA, J=1.0, gamma_b=0.0)
    M, gammas, _ = molecule_fluctuation_matrix(
        flat_state, ALPHA, J=np.sqrt(3.0), gamma_b=1.0, dispersion=(D2,),
        modes=[-3, 0, 3])
    with pytest.raises(ValueError):
        # detection on a mode the bus does not carry
        output_variance_ports(M, gammas, 0.5, port_mode=[3, 5], omega=0.0,
                              mode_index=[0, 2])
    # the fixture state is above the bare MI threshold of the +-3 pair:
    # with J = 0 the stability guard must refuse it
    M0, g0, _ = molecule_fluctuation_matrix(
        flat_state, ALPHA, J=0.0, gamma_b=1.0, dispersion=(D2,),
        modes=[-3, 0, 3])
    with pytest.raises(ValueError):
        output_variance_ports(M0, g0, 0.5, port_mode=0, omega=0.0)
