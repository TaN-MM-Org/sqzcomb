"""Gaussian-state interop: covariance matrices, standard conventions, QuTiP.

This module turns the package's linearized machinery into ecosystem
infrastructure. Everything else in sqzcomb builds a doubled drift matrix M
for a specific device (an LLE comb, a photonic molecule); this module goes
the other way and speaks the two formats the open quantum-optics stack
exchanges:

* the steady-state covariance matrix of the intracavity Gaussian state, in
  the real quadrature (xxpp) ordering with an explicit hbar convention, so
  the state can be handed to any continuous-variable toolkit that consumes
  covariance matrices; and
* a drift matrix built from a quadratic QuTiP Hamiltonian, so a model
  written against the established stack can be pushed through this
  package's detected-spectra machinery unchanged.

Conventions, stated once and tested rather than assumed:

* Doubled basis z = (a_1..a_n, a*_1..a*_n); dz/dt = M z + inputs, with
  mode j carrying amplitude decay gamma_j (total input coupling
  sqrt(2 gamma_j), as everywhere in this package).
* Bath input: <z_in(t) z_in(t')^dag> = N delta(t - t') with N =
  [[(n_th + 1) I, 0], [0, n_th I]] (vacuum: n_th = 0); hence the
  diffusion matrix D = [[2 Gamma (n_th + 1), 0], [0, 2 Gamma n_th]].
* Steady covariance V = <z z^dag> solves M V + V M^dag + D = 0 (solved
  here with numpy alone via the Kronecker-vectorized linear system).
* Quadratures x = (a + a^dag)/sqrt(2), p = -i (a - a^dag)/sqrt(2);
  symmetrized covariance sigma with vacuum sigma = (hbar/2) I. The default
  hbar = 2.0 gives vacuum covariance exactly the identity, the convention
  used by the continuous-variable toolchain around covariance matrices;
  hbar = 1.0 recovers this package's internal vacuum variance 1/2.
* Hamiltonian sign: dz/dt includes -i [z, H] terms, so a physical
  detuning +delta a^dag a in H appears as -i delta on the drift diagonal.
  The package's device builders write the rotating-frame detuning with
  the opposite sign (+i delta_a for `photonic_molecule(delta_a)`); the
  equivalence, including that sign, is asserted in the test suite.
"""
from __future__ import annotations

import numpy as np


def _diffusion(gammas, n_th=0.0):
    gammas = np.asarray(gammas, dtype=float)
    n = gammas.size
    nb = np.broadcast_to(np.asarray(n_th, dtype=float), (n,))
    if np.any(nb < 0.0):
        raise ValueError("thermal occupations must be non-negative")
    D = np.zeros((2 * n, 2 * n), dtype=complex)
    D[:n, :n] = 2.0 * np.diag(gammas * (nb + 1.0))
    D[n:, n:] = 2.0 * np.diag(gammas * nb)
    return D


def intracavity_covariance(M, gammas, n_th=0.0):
    """Steady-state complex covariance V = <z z^dag> of the doubled basis.

    Solves M V + V M^dag + D = 0 with the bath diffusion matrix
    D = diag(2 gamma (n_th + 1), 2 gamma n_th) -- vacuum baths for the
    default n_th = 0, a Bose occupation per bath otherwise (scalar or
    one value per mode; see `thermal_occupation` for the physical
    number). A passive mode then holds exactly <a^dag a> = n_th, which
    the tests assert. Requires a stable M (all drift eigenvalues in
    the open left half-plane); raises ValueError otherwise, for the same
    reason the spectra do: the linearized state does not exist above
    threshold.
    """
    M = np.asarray(M, dtype=complex)
    m2 = M.shape[0]
    if M.shape != (m2, m2) or m2 % 2:
        raise ValueError("M must be a square doubled (2n x 2n) matrix")
    if np.max(np.linalg.eigvals(M).real) >= 0.0:
        raise ValueError("drift matrix is unstable (above threshold); "
                         "no steady covariance exists there")
    n = m2 // 2
    gammas = np.asarray(gammas, dtype=float)
    if gammas.shape != (n,):
        raise ValueError("gammas must have one decay rate per mode")
    D = _diffusion(gammas, n_th)
    ident = np.eye(m2, dtype=complex)
    # row-major vec: vec(M V) = (M kron I) vec(V); vec(V M^dag) =
    # (I kron conj(M)) vec(V)
    L = np.kron(M, ident) + np.kron(ident, np.conj(M))
    V = np.linalg.solve(L, -D.reshape(-1)).reshape(m2, m2)
    return 0.5 * (V + V.conj().T)


def covariance_xxpp(V, hbar=2.0):
    """Symmetrized quadrature covariance in xxpp ordering.

    Takes the complex covariance V = <z z^dag> from
    `intracavity_covariance` and returns the real symmetric matrix
    sigma_ij = (hbar/2) <{r_i, r_j}> / <vacuum scale>, ordered
    (x_1..x_n, p_1..p_n), with vacuum exactly (hbar/2) I. The default
    hbar = 2.0 makes vacuum the identity.
    """
    V = np.asarray(V, dtype=complex)
    m2 = V.shape[0]
    n = m2 // 2
    eye = np.eye(n)
    # symmetrization: V_S = V - K/2 with K = [z_i, z_j^dag] = diag(I, -I)
    K = np.block([[eye, np.zeros((n, n))], [np.zeros((n, n)), -eye]])
    V_S = V - 0.5 * K.astype(complex)
    S = np.block([[eye, eye], [-1j * eye, 1j * eye]]) / np.sqrt(2.0)
    W = S @ V_S @ S.conj().T
    if np.max(np.abs(W.imag)) > 1e-10 * max(1.0, np.max(np.abs(W))):
        raise ValueError("covariance did not come out real; V is not a "
                         "valid symmetrized covariance in the (a, a*) basis")
    sigma = W.real
    return float(hbar) * 0.5 * (sigma + sigma.T)


def symplectic_eigenvalues(sigma, hbar=2.0):
    """Williamson symplectic spectrum of an xxpp covariance matrix.

    Returns the n symplectic eigenvalues, each >= hbar/2 for a physical
    state (equality for a pure mode). Computed as the positive
    eigenvalues of i Omega sigma with Omega the xxpp symplectic form.
    """
    sigma = np.asarray(sigma, dtype=float)
    m2 = sigma.shape[0]
    n = m2 // 2
    eye = np.eye(n)
    omega = np.block([[np.zeros((n, n)), eye], [-eye, np.zeros((n, n))]])
    ev = np.linalg.eigvals(1j * omega @ sigma)
    nu = np.sort(np.abs(ev.real[np.abs(ev.imag) < 1e-9 * max(1.0, np.max(np.abs(ev)))]))
    # eigenvalues come in +/- pairs; keep one of each
    return nu[nu > 0][:n] if nu[nu > 0].size >= n else np.sort(np.abs(ev))[m2 - n:]


def drift_from_qutip(H, gammas):
    """Doubled drift matrix from a quadratic QuTiP Hamiltonian.

    H : qutip.Qobj on n modes (tensor structure, each dimension >= 3),
        quadratic in the mode operators:
        H = E0 + sum F_ij a_i^dag a_j
               + (1/2) sum (G_ij a_i^dag a_j^dag + h.c.),
        F Hermitian, G symmetric. The coefficients are read off from
        matrix elements in the low Fock sector, H is reconstructed from
        them, and a ValueError is raised if the reconstruction does not
        match: a Kerr term or any other non-quadratic content is
        rejected rather than silently linearized.
    gammas : per-mode amplitude decay rates, as everywhere in sqzcomb.

    Returns (M, gammas_array) ready for `output_variance_ports`,
    `intracavity_covariance` and the rest of the machinery. Requires
    qutip (install the `interop` extra); imported only here.
    """
    try:
        import qutip
    except ImportError as exc:                     # pragma: no cover
        raise ImportError("drift_from_qutip requires qutip; install "
                          "sqzcomb[interop]") from exc

    dims = H.dims[0]
    n = len(dims)
    if any(d < 3 for d in dims):
        raise ValueError("each mode needs Fock dimension >= 3 to read "
                         "off the squeezing coefficients")
    gammas = np.asarray(gammas, dtype=float)
    if gammas.shape != (n,):
        raise ValueError("gammas must have one decay rate per mode")

    def fock(occ):
        return qutip.tensor(*[qutip.basis(d, k) for d, k in zip(dims, occ)])

    vac = fock([0] * n)
    E0 = complex(H.matrix_element(vac, vac))

    F = np.zeros((n, n), dtype=complex)
    for i in range(n):
        occ_i = [0] * n
        occ_i[i] = 1
        for j in range(n):
            occ_j = [0] * n
            occ_j[j] = 1
            F[i, j] = complex(H.matrix_element(fock(occ_i), fock(occ_j)))
            if i == j:
                F[i, j] -= E0

    G = np.zeros((n, n), dtype=complex)
    for i in range(n):
        for j in range(i, n):
            occ = [0] * n
            if i == j:
                occ[i] = 2
                G[i, j] = np.sqrt(2.0) * complex(H.matrix_element(fock(occ), vac))
            else:
                occ[i] = 1
                occ[j] = 1
                G[i, j] = complex(H.matrix_element(fock(occ), vac))
                G[j, i] = G[i, j]

    # reject anything the quadratic form does not reproduce
    ops = [qutip.tensor(*[qutip.destroy(d) if k == m else qutip.qeye(d)
                          for k, d in enumerate(dims)]) for m in range(n)]
    H_rec = E0 * qutip.tensor(*[qutip.qeye(d) for d in dims])
    for i in range(n):
        for j in range(n):
            H_rec += F[i, j] * ops[i].dag() * ops[j]
            H_rec += 0.5 * G[i, j] * ops[i].dag() * ops[j].dag()
            H_rec += 0.5 * np.conj(G[i, j]) * ops[i] * ops[j]
    if not np.allclose(H.full(), H_rec.full(), atol=1e-9):
        raise ValueError("H is not quadratic in the mode operators; "
                         "refusing to linearize it silently")

    Gam = np.diag(gammas).astype(complex)
    A = -Gam - 1j * F
    B = -1j * G
    M = np.block([[A, B], [np.conj(B), np.conj(A)]])
    return M, gammas


def principal_quadratures(sigma, hbar=2.0):
    """Supermode decomposition of a multimode covariance matrix: the
    principal quadratures and their variances.

    For any real unit vector u, the generalized quadrature u . r
    (r = (x_1..x_n, p_1..p_n)) has variance u^T sigma u, so the
    eigendecomposition of sigma answers, exactly and completely, the
    question "what is the most squeezed collective quadrature this
    state contains, and along which mode combination does it lie" --
    the smallest eigenvalue is the deepest squeezing any generalized
    quadrature attains, its eigenvector the supermode that carries it.
    That statement is linear algebra, not approximation, and the tests
    pin it: the two-mode squeezed vacuum yields the exact pairs
    (hbar/2) e^{-2r} and (hbar/2) e^{+2r} with the EPR combinations
    (x_1 -/+ x_2)/sqrt(2), (p_1 +/- p_2)/sqrt(2) as supermodes, vacuum
    yields hbar/2 in every direction, and on the photonic molecule the
    principal variance is verified to lower-bound every tested
    twin-beam quadrature.

    Returns (variances, vectors): eigenvalues ascending, vectors[:, i]
    the unit xxpp vector of principal quadrature i.

    This is the orthogonal decomposition of the noise ellipsoid --
    deliberately distinct from the *symplectic* (Williamson)
    decomposition `symplectic_eigenvalues`, which measures mixedness:
    a pure squeezed state has all symplectic eigenvalues at hbar/2
    while its principal variances split as e^{-/+ 2r}. Both views are
    exported because they answer different questions.
    """
    sigma = np.asarray(sigma, dtype=float)
    m2 = sigma.shape[0]
    if sigma.shape != (m2, m2) or m2 % 2:
        raise ValueError("sigma must be a square (2n, 2n) xxpp matrix")
    if np.abs(sigma - sigma.T).max() > 1e-9 * max(1.0, np.abs(sigma).max()):
        raise ValueError("sigma must be symmetric")
    w, v = np.linalg.eigh(0.5 * (sigma + sigma.T))
    if w[0] < -1e-12 * max(1.0, abs(w[-1])):
        raise ValueError("sigma is not positive semidefinite; not a "
                         "covariance matrix")
    return w, v
