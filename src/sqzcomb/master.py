"""Beyond Gaussian: the exact Fock-space master equation (new in 0.13).

Everything else in sqzcomb linearizes: it follows small Gaussian
fluctuations around a classical state. That is accurate for the large
photon numbers of a comb, but it cannot describe what happens when a
single photon's Kerr shift is not negligible -- near threshold, in the
switching between the two phase states of a parametric oscillator, or
wherever the Wigner function goes negative (a non-Gaussian state).
This module solves the full quantum problem instead, for a few modes in
a truncated photon-number (Fock) basis, with no linearization and no
Gaussian assumption.

Conventions match the rest of the package (normalized units: time in
units of 2/kappa, amplitude decay 1 per mode):

* the Lindblad master equation
      d rho/dt = -i [H, rho] + sum_c D[C_c] rho,
      D[C] rho = C rho C^dag - (1/2){C^dag C, rho};
  a mode with amplitude decay gamma into a bath of Bose occupation n
  has C = sqrt(2 gamma (n+1)) a and sqrt(2 gamma n) a^dag, so that
  d<a>/dt = ... - gamma <a>, as in every drift matrix of the package;
* superoperators use column stacking, vec(A X B) = (B^T kron A) vec(X);
* the output field of a port with escape fraction eta is
  b_out = sqrt(2 eta) a - b_in (the package's input-output relation),
  and its quadrature X_phi = (b e^{-i phi} + b^dag e^{i phi})/sqrt(2)
  has noise spectrum, vacuum level 1/2,

      S(omega) = 1/2 + 2 eta Int dtau e^{i omega tau}
                       <:dX_phi(tau) dX_phi(0):>_T

  (time- and normally ordered intracavity correlation, fluctuation part
  only), evaluated exactly with the quantum regression theorem as a
  resolvent: Int_0^inf dtau e^{i omega tau} Tr[A e^{L tau} X]
  = Tr[A (-i omega - L)^-1 X] for traceless X.

What the tests hold this module to, rather than state:

* for a purely quadratic Hamiltonian (chi = 0 below threshold) the
  exact master-equation spectrum equals the package's linearized
  spectrum -- the closed form 1/2 [1 -/+ 4 eta mu / ((1 +/- mu)^2 +
  omega^2)] -- and the steady covariance equals `intracavity_covariance`,
  to the truncation error, which the solver bounds;
* the steady state agrees with QuTiP's independent solver (optional
  dependency, test skipped without it);
* the Wigner function equals QuTiP's, integrates to 1, gives exactly
  -1/pi at the origin for one photon, and its x-marginal equals the
  photon-number wavefunction probability;
* above threshold with a small Kerr shift, the exact steady state
  splits into the two phase lobes +/- alpha of `sqzcomb.kerrpo`, its
  spectrum approaches the linearized one as chi shrinks, and the extra
  low-frequency noise from switching between the lobes has the width of
  the slowest nonzero decay rate of the Liouvillian;
* `master_evolve` (the exact action of the Liouvillian's exponential)
  reproduces lossless Kerr evolution element by element, turning a
  coherent state into a cat state with a negative Wigner function, and
  relaxes to `steady_state`.

Honest limits: the Fock basis must hold the state. Given the mode
dimensions `dims` (which `kerr_parametric_master` returns), the solvers
check the population of the top `guard` photon-number levels of every
mode and refuse (with the number) when it exceeds `trunc_tol`;
`master_output_spectrum` always checks the detected mode. Raise
`cutoff` or lower the photon number. The cost grows as cutoff^(2 n)
for n modes, so this is a tool for one or two modes, not for a comb.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.special import eval_genlaguerre, gammaln

__all__ = ["fock_operators", "liouvillian", "steady_state",
           "master_evolve", "fock_state", "coherent_state",
           "kerr_parametric_master", "master_moments",
           "master_output_spectrum", "wigner", "wigner_negativity"]


def fock_operators(dims):
    """Annihilation operators (sparse CSR) of each mode on the tensor
    product of truncated Fock spaces with the given dimensions."""
    dims = [int(d) for d in np.atleast_1d(dims)]
    if any(d < 2 for d in dims):
        raise ValueError("each mode needs at least 2 Fock levels")
    ops = []
    for j, d in enumerate(dims):
        a = sp.diags(np.sqrt(np.arange(1, d, dtype=float)), 1,
                     shape=(d, d), format="csr", dtype=complex)
        full = None
        for k, dk in enumerate(dims):
            f = a if k == j else sp.identity(dk, format="csr",
                                             dtype=complex)
            full = f if full is None else sp.kron(full, f, format="csr")
        ops.append(full.tocsr())
    return ops


def liouvillian(H, c_ops):
    """Sparse Lindblad superoperator (column stacking) of H and the
    collapse operators c_ops."""
    H = sp.csr_matrix(H, dtype=complex)
    n = H.shape[0]
    eye = sp.identity(n, format="csr", dtype=complex)
    L = -1j * (sp.kron(eye, H) - sp.kron(H.T, eye))
    for C in c_ops:
        C = sp.csr_matrix(C, dtype=complex)
        CdC = (C.conj().T @ C).tocsr()
        L = L + sp.kron(C.conj(), C) - 0.5 * sp.kron(eye, CdC) \
            - 0.5 * sp.kron(CdC.T, eye)
    return L.tocsr()


def _trace_row(n):
    idx = np.arange(n) * (n + 1)
    row = np.zeros(n * n, dtype=complex)
    row[idx] = 1.0
    return row


def _replace_row0(A, n):
    A = sp.lil_matrix(A)
    A[0, :] = _trace_row(n)
    return A.tocsc()


def steady_state(H, c_ops, dims=None, trunc_tol=1e-6, guard=2):
    """Unique steady state rho of the master equation.

    Solves L vec(rho) = 0 with Tr rho = 1 by a sparse direct solve (one
    redundant row of L replaced by the trace condition). With `dims`
    (the Fock dimension of each mode) the truncation is checked: the
    population of the top `guard` levels of any mode above `trunc_tol`
    raises ValueError. Returns the (dense) density matrix.
    """
    H = sp.csr_matrix(H, dtype=complex)
    n = H.shape[0]
    L = liouvillian(H, c_ops)
    A = _replace_row0(L, n)
    b = np.zeros(n * n, dtype=complex)
    b[0] = 1.0
    x = spla.spsolve(A, b)
    rho = x.reshape((n, n), order="F")
    rho = 0.5 * (rho + rho.conj().T)
    rho = rho / np.trace(rho).real
    res = np.abs(L @ rho.reshape(-1, order="F")).max()
    if not np.isfinite(res) or res > 1e-8 * max(1.0, np.abs(rho).max()):
        raise RuntimeError(f"steady-state residual {res:.3e}: the "
                           "steady state is not unique or the solve "
                           "failed")
    if dims is not None:
        _check_truncation(rho, dims, trunc_tol, guard)
    return rho


def master_evolve(H, c_ops, rho0, times, dims=None, trunc_tol=1e-6,
                  guard=2):
    """Density matrices rho(t) at the given times (t >= 0, normalized
    units) from rho0 under the time-independent master equation, by the
    exact action of the matrix exponential of the Liouvillian
    (scipy.sparse.linalg.expm_multiply). A pump that changes in time can
    be followed by calling this piecewise. With `dims` every returned
    state's truncation is checked as in `steady_state`. Returns an array
    of shape (len(times), N, N).
    """
    H = sp.csr_matrix(H, dtype=complex)
    n = H.shape[0]
    rho0 = np.asarray(rho0, dtype=complex)
    if rho0.shape != (n, n):
        raise ValueError("rho0 must match the Hilbert-space dimension")
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if np.any(t < 0) or not np.all(np.isfinite(t)) or np.any(np.diff(t) < 0):
        raise ValueError("times must be finite, >= 0 and increasing")
    L = liouvillian(H, c_ops).tocsc()
    out = []
    v = rho0.reshape(-1, order="F")
    t_prev = 0.0
    for tk in t:
        if tk > t_prev:
            v = spla.expm_multiply(L * (tk - t_prev), v)
            t_prev = tk
        r = v.reshape((n, n), order="F")
        r = 0.5 * (r + r.conj().T)
        if dims is not None:
            _check_truncation(r, dims, trunc_tol, guard)
        out.append(r)
    return np.array(out)


def fock_state(n, cutoff):
    """Density matrix of the photon-number state |n> in a Fock space of
    dimension cutoff."""
    n, cutoff = int(n), int(cutoff)
    if not (0 <= n < cutoff):
        raise ValueError("need 0 <= n < cutoff")
    rho = np.zeros((cutoff, cutoff), dtype=complex)
    rho[n, n] = 1.0
    return rho


def coherent_state(alpha, cutoff, tol=1e-10):
    """Density matrix of the coherent state |alpha> truncated to cutoff
    levels; refuses when the truncation loses more than tol of the
    population."""
    alpha = complex(alpha)
    k = np.arange(int(cutoff))
    logc = -0.5 * abs(alpha) ** 2 - 0.5 * gammaln(k + 1)
    c = np.exp(logc) * alpha ** k
    lost = 1.0 - float(np.sum(np.abs(c) ** 2))
    if lost > tol:
        raise ValueError(f"cutoff {int(cutoff)} loses population "
                         f"{lost:.2e} of |alpha = {alpha}>; raise it")
    return np.outer(c, c.conj())


def _check_mode_truncation(rho, a, trunc_tol, guard):
    """Population of the top `guard` photon numbers of the mode whose
    annihilation operator is `a` (works for any tensor-product layout:
    a^dag a is diagonal in the Fock product basis)."""
    a = sp.csr_matrix(a)
    num = np.real((a.conj().T @ a).diagonal())
    top = num >= num.max() - (int(guard) - 1) - 1e-9
    pop = float(np.real(np.diag(rho))[top].sum())
    if pop > trunc_tol:
        raise ValueError(
            f"Fock truncation too small for the detected mode: its top "
            f"{int(guard)} levels hold population {pop:.2e} > trunc_tol "
            f"= {trunc_tol:g}. Raise the cutoff (or lower the photon "
            "number); the answer would otherwise depend on the cutoff")


def _check_truncation(rho, dims, trunc_tol, guard):
    dims = [int(d) for d in np.atleast_1d(dims)]
    p = np.real(np.diag(rho)).reshape(dims)
    for j, d in enumerate(dims):
        axes = tuple(k for k in range(len(dims)) if k != j)
        pj = p.sum(axis=axes) if axes else p
        top = float(pj[d - int(guard):].sum())
        if top > trunc_tol:
            raise ValueError(
                f"Fock truncation too small for mode {j}: the top "
                f"{int(guard)} levels hold population {top:.2e} > "
                f"trunc_tol = {trunc_tol:g}. Raise the cutoff (or "
                "lower the photon number); the answer would otherwise "
                "depend on the cutoff")


def kerr_parametric_master(mu, delta=0.0, chi=0.0, cutoff=30, n_th=0.0):
    """Hamiltonian and collapse operators of the single-mode Kerr
    parametric oscillator of `sqzcomb.kerrpo`:

        H = -delta a^dag a + (i/2)(mu a^dag^2 - mu* a^2)
            - (chi/2) a^dag^2 a^2,
        C = sqrt(2 (n_th + 1)) a,  sqrt(2 n_th) a^dag   (if n_th > 0).

    chi = 0 is the quadratic (linear-optics) model whose drift is
    `single_mode_parametric(mu, delta)`. Returns (H, c_ops, a, dims).
    """
    if int(cutoff) < 4:
        raise ValueError("cutoff must be at least 4")
    if float(n_th) < 0.0:
        raise ValueError("n_th must be non-negative")
    a = fock_operators([int(cutoff)])[0]
    ad = a.conj().T.tocsr()
    mu = complex(mu)
    H = (-float(delta)) * (ad @ a) \
        + 0.5j * (mu * (ad @ ad) - np.conj(mu) * (a @ a)) \
        - 0.5 * float(chi) * (ad @ ad @ a @ a)
    c_ops = [np.sqrt(2.0 * (float(n_th) + 1.0)) * a]
    if float(n_th) > 0.0:
        c_ops.append(np.sqrt(2.0 * float(n_th)) * ad)
    return H.tocsr(), c_ops, a, [int(cutoff)]


def master_moments(rho, a):
    """Intracavity moments of mode `a` in state rho: dict with alpha =
    <a>, n = <a^dag a>, and the symmetrized quadrature covariance
    (x = (a + a^dag)/sqrt(2), p = -i (a - a^dag)/sqrt(2), vacuum 1/2 I)
    of the fluctuations, `cov_xp` (2x2), comparable with
    `covariance_xxpp(intracavity_covariance(...), hbar=1)`."""
    rho = np.asarray(rho)
    a = sp.csr_matrix(a)
    ad = a.conj().T

    def ev(op):
        return complex(np.trace((op @ rho) if sp.issparse(op)
                                else op @ rho))

    x = (a + ad) / np.sqrt(2.0)
    p = -1j * (a - ad) / np.sqrt(2.0)
    mx, mp = ev(x).real, ev(p).real
    vxx = ev(x @ x).real - mx * mx
    vpp = ev(p @ p).real - mp * mp
    vxp = 0.5 * ev(x @ p + p @ x).real - mx * mp
    return {"alpha": ev(a), "n": ev(ad @ a).real,
            "cov_xp": np.array([[vxx, vxp], [vxp, vpp]])}


def master_output_spectrum(H, c_ops, a, eta, omegas, phi=0.0, dims=None,
                           trunc_tol=1e-6, guard=2):
    """Exact output quadrature noise spectrum (vacuum = 1/2) of the port
    that carries the fraction `eta` of mode `a`'s decay, from the full
    master equation by the quantum regression theorem.

    H, c_ops : the model (e.g. from `kerr_parametric_master`); the baths
        must be vacuum for the normally-ordered input-output relation
        used here (a thermal collapse operator a^dag is refused).
    a : the annihilation operator of the detected mode.
    eta : escape fraction of that mode's (amplitude-decay-1) loss into
        the detected port, in (0, 1].
    omegas : analysis frequencies (normalized, units of kappa/2).
    phi : quadrature angle, X_phi = (b e^{-i phi} + b^dag e^{i phi})/sqrt(2).
    dims, trunc_tol, guard : the truncation check of `steady_state`.
        The detected mode is always checked (its top `guard` photon
        numbers must hold at most `trunc_tol` of the population); pass
        `dims` to check every mode.

    Returns an array of S(omega). Includes everything the linearized
    theory drops (the Kerr shift of single photons, switching between
    phase states, non-Gaussian correlations).
    """
    eta = float(eta)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    a = sp.csr_matrix(a, dtype=complex)
    ad = a.conj().T.tocsr()
    for C in c_ops:
        # annihilation-type operators are strictly upper triangular in
        # the Fock ordering; any strictly lower part (a^dag content)
        # means a thermal bath, which this relation does not cover
        if sp.tril(sp.csr_matrix(C), -1).count_nonzero():
            raise ValueError(
                "master_output_spectrum assumes vacuum baths; a "
                "creation-type (thermal) collapse operator was given")
    H = sp.csr_matrix(H, dtype=complex)
    n = H.shape[0]
    rho = steady_state(H, c_ops, dims, trunc_tol, guard)
    _check_mode_truncation(rho, a, trunc_tol, guard)
    L = liouvillian(H, c_ops)
    tr_row = _trace_row(n)
    # sources X (traceless after subtracting the stationary part)
    srcs = {"a*rho": a @ rho, "rho*ad": rho @ ad}
    X = {k: (v - np.trace(v) * rho).reshape(-1, order="F")
         for k, v in srcs.items()}
    # trace functionals Tr[A Y] = vec(A^T) . vec(Y)
    fa = np.asarray(a.T.todense()).reshape(-1, order="F")
    fad = np.asarray(ad.T.todense()).reshape(-1, order="F")
    out = []
    eye = sp.identity(n * n, format="csr", dtype=complex)
    for w in np.atleast_1d(np.asarray(omegas, dtype=float)):
        Msys = sp.lil_matrix(-1j * w * eye - L)
        Msys[0, :] = tr_row
        Msys = Msys.tocsc()
        lu = spla.splu(Msys)
        res = {}
        for k, v in X.items():
            rhs = v.copy()
            rhs[0] = 0.0
            res[k] = lu.solve(rhs)
        # Laplace transforms of the fluctuation correlations
        c_aa = fa @ res["a*rho"]            # <a(t) a(0)>
        c_ada = fad @ res["a*rho"]          # <a^dag(t) a(0)>
        c_ad_a = fa @ res["rho*ad"]         # <a^dag(0) a(t)>
        c_ad_ad = fad @ res["rho*ad"]       # <a^dag(0) a^dag(t)>
        G = 0.5 * (np.exp(-2j * phi) * c_aa + c_ada + c_ad_a
                   + np.exp(2j * phi) * c_ad_ad)
        out.append(0.5 + 4.0 * eta * float(np.real(G)))
    return np.array(out)


def wigner(rho, x, p):
    """Wigner function W(x, p) of a single-mode density matrix in the
    package's quadrature convention x = (a + a^dag)/sqrt(2),
    p = -i (a - a^dag)/sqrt(2) (vacuum: W = exp(-x^2 - p^2)/pi,
    normalized to 1 over dx dp). Returns an array of shape
    (len(p), len(x)). Built from the exact Fock-element Wigner
    functions (Laguerre polynomials); checked against QuTiP.
    """
    rho = np.asarray(rho, dtype=complex)
    N = rho.shape[0]
    X, P = np.meshgrid(np.asarray(x, float), np.asarray(p, float))
    al = (X + 1j * P) / np.sqrt(2.0)
    r2 = np.abs(al) ** 2
    W = np.zeros(X.shape, dtype=complex)
    g = np.exp(-2.0 * r2)
    for m in range(N):
        for k in range(0, m + 1):
            nn = m - k
            coef = ((-1) ** nn / np.pi) * np.exp(
                0.5 * (gammaln(nn + 1) - gammaln(m + 1)))
            w = coef * (2.0 * np.conj(al)) ** k * g \
                * eval_genlaguerre(nn, k, 4.0 * r2)
            W += rho[m, nn] * w
            if k:
                W += rho[nn, m] * np.conj(w)
    return W.real


def wigner_negativity(rho, x, p):
    """Volume of the negative part of the Wigner function,
    Int (|W| - W)/2 dx dp, on the given (uniform) grid -- zero for every
    Gaussian state, positive only for non-Gaussian ones."""
    x = np.asarray(x, float)
    p = np.asarray(p, float)
    W = wigner(rho, x, p)
    dx = float(x[1] - x[0])
    dp = float(p[1] - p[0])
    return float(0.5 * np.sum(np.abs(W) - W) * dx * dp)
