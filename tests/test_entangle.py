"""Entanglement module: closed forms, independent cross-checks, invariance,
and the package's own photonic molecule."""
import numpy as np
import pytest

from sqzcomb import covariance_xxpp, intracavity_covariance, photonic_molecule
from sqzcomb.entangle import (
    duan_epr_sum,
    entanglement_report,
    logarithmic_negativity,
    ppt_eigenvalue_direct,
    ppt_symplectic_eigenvalue,
    two_mode_reduction,
)


def tmsv_xxpp(r):
    """Two-mode squeezed vacuum covariance, xxpp ordering, hbar = 2."""
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    sig = np.diag([c, c, c, c]).astype(float)
    sig[0, 1] = sig[1, 0] = s
    sig[2, 3] = sig[3, 2] = -s
    return sig


def test_tmsv_closed_forms():
    """E_N = 2r and Duan sum = hbar e^{-2r} for the two-mode squeezed
    vacuum, at machine precision."""
    for r in (0.3, 1.0, 2.0):
        sig = tmsv_xxpp(r)
        assert np.isclose(logarithmic_negativity(sig), 2 * r, rtol=1e-12)
        duan, bound = duan_epr_sum(sig)
        assert np.isclose(duan, 2.0 * np.exp(-2 * r), rtol=1e-12)
        assert bound == 2.0
        rep = entanglement_report(sig)
        assert rep["entangled"] and rep["duan_violated"]
        assert np.isclose(rep["log_negativity_ebits"], 2 * r / np.log(2))


def test_invariant_formula_matches_explicit_partial_transpose():
    """The Simon-invariant nu equals the value from explicitly
    partial-transposing and taking the Williamson spectrum."""
    rng = np.random.default_rng(0)
    for _ in range(10):
        # random physical state: random symplectic on a thermal state
        r = rng.uniform(0.1, 1.2, 2)
        th = rng.uniform(0, np.pi, 2)
        nbar = rng.uniform(0.0, 1.0, 2)
        sig = np.diag(np.repeat(2 * nbar + 1, 2)[[0, 2, 1, 3]])
        S = np.eye(4)
        for k, (rr, t) in enumerate(zip(r, th)):
            R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
            block = R @ np.diag([np.exp(-rr), np.exp(rr)]) @ R.T
            S[np.ix_([k, 2 + k], [k, 2 + k])] = block
        # entangle with a beam-splitter-like symplectic in xxpp
        c, s = np.cos(0.6), np.sin(0.6)
        BS = np.array([[c, s, 0, 0], [-s, c, 0, 0],
                       [0, 0, c, s], [0, 0, -s, c]])
        full = BS @ S @ sig @ S.T @ BS.T
        four = two_mode_reduction(full, 0, 1)
        nu_inv = ppt_symplectic_eigenvalue(four)
        nu_dir = ppt_eigenvalue_direct(four)
        assert np.isclose(nu_inv, nu_dir, rtol=1e-9)


def test_separable_states_are_certified_separable():
    sig = np.diag([3.0, 5.0, 3.0, 5.0])       # two thermal modes
    assert logarithmic_negativity(sig) == 0.0
    duan, bound = duan_epr_sum(sig)
    assert duan >= bound
    assert not entanglement_report(sig)["entangled"]
    # vacuum sits exactly at the Duan bound
    duan_vac, bound_vac = duan_epr_sum(np.eye(4))
    assert np.isclose(duan_vac, bound_vac)


def test_local_symplectic_invariance():
    """E_N is invariant under local rotations and local squeezing."""
    M, gammas = photonic_molecule(mu=0.8, J=1.0)
    sig = covariance_xxpp(intracavity_covariance(M, gammas))
    e0 = logarithmic_negativity(sig)
    R = np.array([[np.cos(0.7), np.sin(0.7)], [-np.sin(0.7), np.cos(0.7)]])
    Sq = np.diag([np.exp(-0.4), np.exp(0.4)])
    S = np.eye(4)
    S[np.ix_([0, 2], [0, 2])] = R
    S[np.ix_([1, 3], [1, 3])] = Sq
    assert np.isclose(logarithmic_negativity(S @ sig @ S.T), e0, rtol=1e-10)


def test_molecule_rings_are_entangled_below_threshold():
    """The two rings of the driven molecule are PPT-entangled below
    threshold, and here the symmetric Duan sum does NOT flag it: the
    sharper PPT criterion is required. Undriven (mu = 0) the state is
    vacuum and separable."""
    for mu in (0.2, 0.5, 0.8):
        M, gammas = photonic_molecule(mu=mu, J=1.0)
        rep = entanglement_report(
            covariance_xxpp(intracavity_covariance(M, gammas)))
        assert rep["entangled"]
        assert rep["log_negativity"] > 0.0
        assert not rep["duan_violated"]        # Duan misses this state
    M, gammas = photonic_molecule(mu=0.0, J=1.0)
    rep = entanglement_report(
        covariance_xxpp(intracavity_covariance(M, gammas)))
    assert not rep["entangled"]
    assert np.isclose(rep["duan_sum"], rep["duan_bound"])


def test_input_validation():
    with pytest.raises(ValueError):
        two_mode_reduction(np.eye(4), 0, 0)
    with pytest.raises(ValueError):
        two_mode_reduction(np.eye(5), 0, 1)
    with pytest.raises(ValueError):
        ppt_symplectic_eigenvalue(np.eye(3))
