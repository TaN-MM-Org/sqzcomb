"""Comb molecule with mismatched line spacings (Vernier pairing)."""
import numpy as np
import pytest

from sqzcomb import lle_evolve, molecule_fluctuation_matrix, \
    output_variance_ports
from sqzcomb.molecule import (ring_line_frequencies,
                              vernier_molecule_fluctuation_matrix)

ALPHA, D2 = 0.5, -0.15


@pytest.fixture(scope="module")
def quiet_state():
    psi = lle_evolve(np.full(64, 0.1 + 0j), F=0.9, alpha=ALPHA,
                     dispersion=(D2,), t_end=300.0, dt=0.005)
    assert np.ptp(np.abs(psi)) < 1e-8
    return psi


def test_matched_spacing_reduces_exactly(quiet_state):
    # auxiliary lines exactly one per main line, same spacing: the Vernier
    # builder must return the matched-spacing molecule bit for bit
    fsr, modes = 800.0, [-3, -1, 0, 1, 3]
    dk = np.array([0.4, -0.2, 0.1, 0.3, -0.5])
    nu_b = fsr * np.array(modes) - dk
    Mv, gv, mv, aux = vernier_molecule_fluctuation_matrix(
        quiet_state, ALPHA, 0.7, 1.3, fsr, nu_b, (D2,), modes=modes)
    Mm, gm, mm = molecule_fluctuation_matrix(
        quiet_state, ALPHA, 0.7, 1.3, (D2,), aux_delta=dk, modes=modes)
    assert np.array_equal(mv, mm)
    assert np.allclose(aux["aux_delta"], dk, atol=1e-12)
    assert np.allclose(Mv, Mm, atol=1e-12)
    assert np.array_equal(gv, gm)


def _vernier_setup():
    fsr = 600.0
    modes = np.arange(-6, 7)
    # auxiliary ring 7 % larger spacing, its own dispersion and offset
    ell = np.arange(-8, 9)
    nu_b = ring_line_frequencies(ell, offset=0.8, fsr=fsr * 1.07,
                                 dispersion=(0.3,))
    return fsr, modes, nu_b


def test_lab_frame_identity_and_exact_passive_model():
    # Passive rings (psi = 0). In the laboratory frame the full model,
    # every auxiliary line coupled to every main line, is time
    # independent; the paired (rotating-wave) model put back into the
    # lab frame must (a) have exactly the physical line frequencies on
    # its diagonal and (b) share the exact model's eigenvalues up to the
    # J^2 / fsr shifts of the dropped couplings.
    fsr, modes, nu_b = _vernier_setup()
    J, gb = 0.9, 1.4
    psi0 = np.zeros(64, complex)
    M, g, mv, aux = vernier_molecule_fluctuation_matrix(
        psi0, ALPHA, J, gb, fsr, nu_b, (D2,), modes=modes)
    m, p = mv.size, aux["index"].size
    A = M[:m + p, :m + p]
    frame = np.concatenate([mv * fsr, aux["partner"] * fsr])
    A_lab = A - 1j * np.diag(frame)
    nu_a = ALPHA + fsr * mv - D2 / 2 * mv ** 2   # main lines from the pump
    assert np.allclose(np.diag(A_lab)[:m], -1 - 1j * nu_a, atol=1e-9)
    assert np.allclose(np.diag(A_lab)[m:], -gb - 1j * nu_b[aux["index"]],
                       atol=1e-9)
    # exact all-pairs model on the same lines
    full = np.diag(np.concatenate([-1 - 1j * nu_a,
                                   -gb - 1j * nu_b[aux["index"]]]))
    full[:m, m:] = -1j * J
    full[m:, :m] = -1j * J
    ev_full = np.linalg.eigvals(full)
    ev_rwa = np.linalg.eigvals(A_lab)
    err = max(np.min(np.abs(ev_full - e)) for e in ev_rwa)
    assert err < 4 * J ** 2 / (fsr * 0.4)    # dropped couplings only
    assert err < 1e-2


def test_passive_vernier_molecule_gives_vacuum():
    fsr, modes, nu_b = _vernier_setup()
    M, g, mv, aux = vernier_molecule_fluctuation_matrix(
        np.zeros(64, complex), ALPHA, 0.9, 1.4, fsr, nu_b, (D2,),
        modes=modes)
    m = mv.size
    for port in (0, m, m + 1):
        for om in (0.0, 1.7):
            for phi in (0.0, 1.1):
                assert output_variance_ports(M, g, 0.6, port, om,
                                             phi=phi) == \
                    pytest.approx(0.5, abs=1e-12)


def test_pairing_and_window():
    fsr, modes, nu_b = _vernier_setup()
    psi0 = np.zeros(64, complex)
    _, _, _, aux = vernier_molecule_fluctuation_matrix(
        psi0, ALPHA, 0.9, 1.4, fsr, nu_b, (D2,), modes=modes)
    # every kept auxiliary line is paired with its nearest main line
    for j, k, d in zip(aux["index"], aux["partner"], aux["aux_delta"]):
        assert k == int(np.argmin(np.abs(fsr * modes - nu_b[j])) + modes[0])
        assert d == pytest.approx(k * fsr - nu_b[j])
        assert abs(d) <= fsr / 2
    # the Vernier walk-off: detunings grow away from the centre
    assert np.abs(aux["aux_delta"]).max() > 100
    _, _, _, aux_w = vernier_molecule_fluctuation_matrix(
        psi0, ALPHA, 0.9, 1.4, fsr, nu_b, (D2,), modes=modes, window=50)
    assert np.all(np.abs(aux_w["aux_delta"]) <= 50)
    assert aux_w["index"].size < aux["index"].size


def test_refusals():
    psi0 = np.zeros(64, complex)
    # an auxiliary line almost midway between two main lines that are
    # not far apart compared with the coupling (margin 15.5 < 20 * 1.4)
    with pytest.raises(ValueError, match="midway"):
        vernier_molecule_fluctuation_matrix(psi0, ALPHA, 0.9, 1.4, 30.0,
                                            [14.5], modes=[0, 1])
    with pytest.raises(ValueError):
        vernier_molecule_fluctuation_matrix(psi0, ALPHA, 0.9, 0.0, 100.0,
                                            [1.0], modes=[0])
    with pytest.raises(ValueError):
        vernier_molecule_fluctuation_matrix(psi0, ALPHA, 0.9, 1.0, -1.0,
                                            [1.0], modes=[0])
