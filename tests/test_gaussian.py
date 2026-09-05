"""Tests for the Gaussian-interop module.

Every claim is anchored to a closed form or to exact agreement with the
already-released machinery. The QuTiP adapter tests are skipped when
qutip is not installed (install the `interop` extra to run them).
"""
import numpy as np
import pytest

from sqzcomb.gaussian import (covariance_xxpp, intracavity_covariance,
                              symplectic_eigenvalues)
from sqzcomb.linearize import single_mode_parametric
from sqzcomb.molecule import photonic_molecule


def test_vacuum_covariance_is_identity():
    # passive single mode: V = [[1,0],[0,0]], sigma = I at hbar = 2
    M = single_mode_parametric(0.0)
    V = intracavity_covariance(M, np.array([1.0]))
    assert np.allclose(V, np.array([[1.0, 0.0], [0.0, 0.0]]), atol=1e-12)
    sigma = covariance_xxpp(V, hbar=2.0)
    assert np.allclose(sigma, np.eye(2), atol=1e-12)
    assert np.allclose(covariance_xxpp(V, hbar=1.0), 0.5 * np.eye(2),
                       atol=1e-12)


@pytest.mark.parametrize("mu", [0.1, 0.5, 0.9])
def test_parametric_covariance_closed_form(mu):
    # decoupled quadrature sectors of the DPO: dx/dt = -(1 - mu) x + in,
    # steady variances 1/(2(1 -+ mu)) at hbar = 1, i.e. 1/(1 -+ mu) at
    # hbar = 2 in the (x, p) ordering with real mu > 0 squeezing p.
    M = single_mode_parametric(mu)
    V = intracavity_covariance(M, np.array([1.0]))
    sigma = covariance_xxpp(V, hbar=2.0)
    expect = np.diag([1.0 / (1.0 - mu), 1.0 / (1.0 + mu)])
    assert np.allclose(sigma, expect, atol=1e-10)


def test_parametric_symplectic_eigenvalue():
    # nu = sqrt(det sigma) for one mode: hbar/2 / sqrt(1 - mu^2) >= hbar/2,
    # a mixed state because the loss port carries information away.
    mu = 0.6
    M = single_mode_parametric(mu)
    sigma = covariance_xxpp(intracavity_covariance(M, np.array([1.0])))
    nu = symplectic_eigenvalues(sigma, hbar=2.0)
    assert nu.size == 1
    assert np.isclose(nu[0], 1.0 / np.sqrt(1.0 - mu ** 2), atol=1e-9)
    assert nu[0] >= 1.0 - 1e-12


def test_passive_molecule_covariance_is_vacuum():
    M, gammas = photonic_molecule(mu=0.0, J=0.7, delta_a=0.3,
                                  delta_b=-0.2, gamma=1.6)
    sigma = covariance_xxpp(intracavity_covariance(M, gammas), hbar=2.0)
    assert np.allclose(sigma, np.eye(4), atol=1e-12)
    nu = symplectic_eigenvalues(sigma, hbar=2.0)
    assert np.allclose(nu, np.ones(2), atol=1e-9)


def test_unstable_matrix_refused():
    with pytest.raises(ValueError):
        intracavity_covariance(single_mode_parametric(1.5), np.array([1.0]))


def test_heisenberg_bound_molecule():
    M, gammas = photonic_molecule(mu=0.8, J=1.2, gamma=2.0)
    sigma = covariance_xxpp(intracavity_covariance(M, gammas), hbar=2.0)
    nu = symplectic_eigenvalues(sigma, hbar=2.0)
    assert np.all(nu >= 1.0 - 1e-9)


# ---------------------------------------------------------------- qutip


def _qutip():
    return pytest.importorskip("qutip")


def test_qutip_reproduces_parametric():
    qutip = _qutip()
    mu = 0.4
    a = qutip.destroy(12)
    H = 0.5j * mu * (a.dag() ** 2 - a ** 2)
    from sqzcomb.gaussian import drift_from_qutip
    M, _ = drift_from_qutip(H, np.array([1.0]))
    assert np.allclose(M, single_mode_parametric(mu), atol=1e-9)


def test_qutip_reproduces_released_molecule():
    qutip = _qutip()
    mu, J, da, db, gb = 0.35, 0.8, 0.25, -0.4, 1.7
    dims = (10, 10)
    a = qutip.tensor(qutip.destroy(dims[0]), qutip.qeye(dims[1]))
    b = qutip.tensor(qutip.qeye(dims[0]), qutip.destroy(dims[1]))
    # rotating-frame sign: photonic_molecule(delta) carries +i delta on
    # the drift diagonal, which is the Hamiltonian -delta a^dag a
    H = (-da * a.dag() * a - db * b.dag() * b
         + J * (a.dag() * b + b.dag() * a)
         + 0.5j * mu * (a.dag() ** 2 - a ** 2))
    from sqzcomb.gaussian import drift_from_qutip
    M_q, gam = drift_from_qutip(H, np.array([1.0, gb]))
    M_ref, gam_ref = photonic_molecule(mu, J, delta_a=da, delta_b=db,
                                       gamma=gb)
    assert np.allclose(M_q, M_ref, atol=1e-9)
    assert np.allclose(gam, gam_ref)


def test_qutip_rejects_nonquadratic():
    qutip = _qutip()
    a = qutip.destroy(8)
    from sqzcomb.gaussian import drift_from_qutip
    with pytest.raises(ValueError):
        drift_from_qutip(a.dag() ** 2 * a ** 2, np.array([1.0]))


# ----------------------------------------------------------------------
# supermode decomposition (principal quadratures)

def test_principal_quadratures_of_the_two_mode_squeezed_vacuum():
    """Exact closed form: variances (hbar/2) e^{-/+ 2r}, each twice,
    with the EPR combinations as supermodes."""
    from sqzcomb import principal_quadratures
    r = 0.6
    ch, sh = np.cosh(2 * r), np.sinh(2 * r)
    sigma = np.zeros((4, 4))
    sigma[0, 0] = sigma[1, 1] = sigma[2, 2] = sigma[3, 3] = ch
    sigma[0, 1] = sigma[1, 0] = sh          # <x1 x2>
    sigma[2, 3] = sigma[3, 2] = -sh         # <p1 p2>
    w, v = principal_quadratures(sigma)
    expect = np.sort([np.exp(-2 * r)] * 2 + [np.exp(2 * r)] * 2)
    assert np.abs(np.sort(w) - expect).max() < 1e-12
    # the most-squeezed supermode is an EPR combination: equal and
    # opposite weights on x1, x2 (or p1, p2)
    u = v[:, 0]
    pair = np.sort(np.abs(u))
    assert np.abs(pair[:2]).max() < 1e-10
    assert np.abs(pair[2:] - 1.0 / np.sqrt(2.0)).max() < 1e-10


def test_principal_quadratures_of_vacuum_are_flat():
    from sqzcomb import principal_quadratures
    w, _ = principal_quadratures(np.eye(6))
    assert np.abs(w - 1.0).max() < 1e-12


def test_principal_variance_lower_bounds_every_tested_quadrature():
    """On the photonic molecule: the smallest eigenvalue of sigma is,
    by construction, at or below the variance of any generalized
    quadrature -- checked against a scan of two-mode quadratures."""
    from sqzcomb import (intracavity_covariance, photonic_molecule,
                         principal_quadratures)
    M, gammas = photonic_molecule(mu=0.8, J=1.0)
    sigma = covariance_xxpp(intracavity_covariance(M, gammas))
    w, _ = principal_quadratures(sigma)
    n = sigma.shape[0]
    rng = np.random.default_rng(5)
    for _ in range(50):
        u = rng.normal(size=n)
        u /= np.linalg.norm(u)
        assert u @ sigma @ u >= w[0] - 1e-10


def test_principal_quadratures_refuse_non_covariances():
    import pytest
    from sqzcomb import principal_quadratures
    with pytest.raises(ValueError):
        principal_quadratures(np.ones((3, 3)))       # odd dimension
    bad = -np.eye(4)
    with pytest.raises(ValueError):
        principal_quadratures(bad)                   # not PSD
