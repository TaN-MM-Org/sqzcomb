"""Two-mode Gaussian entanglement of the comb: negativity and EPR sums.

Twin-beam entanglement between sideband pairs is the resource a squeezing
comb ultimately delivers to continuous-variable information processing;
this module quantifies it directly from the covariance matrices the rest
of the package already produces.

For a two-mode reduction of an xxpp covariance matrix (vacuum =
(hbar/2) I, the package convention) it computes

* the partial-transpose symplectic eigenvalue nu_tilde_minus through the
  local-symplectic invariants (Simon, Phys. Rev. Lett. 84, 2726 (2000)):
  with sigma = [[A, C], [C^T, B]] in per-mode (x, p) blocks and
  Delta_tilde = det A + det B - 2 det C,
      2 nu_tilde_minus^2 = Delta_tilde - sqrt(Delta_tilde^2 - 4 det sigma);
  for 1x1-mode Gaussian states PPT is necessary and sufficient, so
  nu_tilde_minus < hbar/2 if and only if the pair is entangled;
* the logarithmic negativity E_N = max(0, -ln(2 nu_tilde_minus / hbar))
  (Vidal and Werner, Phys. Rev. A 65, 032314 (2002)), an entanglement
  monotone and upper bound on distillable entanglement;
* the symmetric Duan-Simon EPR sum Var[(x_i - x_j)/sqrt(2)] +
  Var[(p_i + p_j)/sqrt(2)], which every separable state keeps at or
  above hbar (Duan, Giedke, Cirac and Zoller, Phys. Rev. Lett. 84, 2722
  (2000)); this symmetric form is sufficient but not necessary for
  entanglement, and the PPT value above is the sharper statement.

The invariant formula is cross-checked in the test suite against an
independent computation (explicit partial transposition followed by a
Williamson symplectic spectrum), against the closed-form two-mode
squeezed vacuum, and on the package's own photonic molecule.
"""
from __future__ import annotations

import numpy as np

from .gaussian import symplectic_eigenvalues


def two_mode_reduction(sigma, i: int, j: int):
    """The 4x4 covariance of modes (i, j) in (x_i, p_i, x_j, p_j) order,
    extracted from an n-mode xxpp covariance matrix."""
    sigma = np.asarray(sigma, dtype=float)
    m = sigma.shape[0]
    if sigma.shape != (m, m) or m % 2:
        raise ValueError("sigma must be a 2n x 2n xxpp covariance matrix")
    n = m // 2
    if not (0 <= i < n and 0 <= j < n) or i == j:
        raise ValueError("need two distinct mode indices below n")
    idx = [i, n + i, j, n + j]
    return sigma[np.ix_(idx, idx)]


def ppt_symplectic_eigenvalue(sigma4, hbar: float = 2.0) -> float:
    """nu_tilde_minus of a two-mode covariance (x1, p1, x2, p2 order),
    from the local-symplectic invariants. Entangled iff the return value
    is below hbar/2 (PPT, necessary and sufficient for 1x1 Gaussian)."""
    s = np.asarray(sigma4, dtype=float) / (0.5 * float(hbar))
    if s.shape != (4, 4):
        raise ValueError("sigma4 must be 4x4 in (x1, p1, x2, p2) order")
    A, B, C = s[:2, :2], s[2:, 2:], s[:2, 2:]
    delta_t = np.linalg.det(A) + np.linalg.det(B) - 2.0 * np.linalg.det(C)
    disc = delta_t ** 2 - 4.0 * np.linalg.det(s)
    if disc < -1e-9 * max(1.0, delta_t ** 2):
        raise ValueError("invalid covariance matrix (negative discriminant)")
    nu2 = 0.5 * (delta_t - np.sqrt(max(disc, 0.0)))
    if nu2 < 0.0:
        if nu2 < -1e-9:
            raise ValueError("invalid covariance matrix (nu^2 < 0)")
        nu2 = 0.0
    return float(0.5 * hbar * np.sqrt(nu2))


def logarithmic_negativity(sigma, i: int = 0, j: int = 1,
                           hbar: float = 2.0) -> float:
    """E_N = max(0, -ln(2 nu_tilde_minus / hbar)) between modes i and j
    of an n-mode xxpp covariance matrix. Zero for every separable pair;
    2r for a pure two-mode squeezed vacuum of squeezing parameter r."""
    nu = ppt_symplectic_eigenvalue(two_mode_reduction(sigma, i, j), hbar)
    return float(max(0.0, -np.log(2.0 * nu / hbar)))


def duan_epr_sum(sigma, i: int = 0, j: int = 1, hbar: float = 2.0):
    """Symmetric Duan-Simon EPR sum and its separability bound.

    Returns (value, bound) with value = Var[(x_i - x_j)/sqrt(2)] +
    Var[(p_i + p_j)/sqrt(2)] and bound = hbar; value < bound certifies
    entanglement (sufficient, not necessary). Vacuum sits exactly at the
    bound; a pure two-mode squeezed vacuum gives hbar e^{-2r}.
    """
    s = two_mode_reduction(sigma, i, j)
    var_u = 0.5 * (s[0, 0] + s[2, 2] - 2.0 * s[0, 2])
    var_v = 0.5 * (s[1, 1] + s[3, 3] + 2.0 * s[1, 3])
    return float(var_u + var_v), float(hbar)


def entanglement_report(sigma, i: int = 0, j: int = 1, hbar: float = 2.0):
    """All of the above for one pair: nu_tilde_minus, E_N (nats and
    ebits), the Duan sum with its bound, and the PPT verdict."""
    nu = ppt_symplectic_eigenvalue(two_mode_reduction(sigma, i, j), hbar)
    e_n = float(max(0.0, -np.log(2.0 * nu / hbar)))
    duan, bound = duan_epr_sum(sigma, i, j, hbar)
    return dict(nu_tilde_minus=nu, log_negativity=e_n,
                log_negativity_ebits=e_n / np.log(2.0),
                duan_sum=duan, duan_bound=bound,
                entangled=bool(nu < 0.5 * hbar - 1e-12),
                duan_violated=bool(duan < bound - 1e-12))


def ppt_eigenvalue_direct(sigma4, hbar: float = 2.0) -> float:
    """Independent cross-check of :func:`ppt_symplectic_eigenvalue`:
    explicitly partial-transpose mode 2 (p2 -> -p2) and take the smaller
    Williamson symplectic eigenvalue. Used by the test suite."""
    s = np.asarray(sigma4, dtype=float)
    L = np.diag([1.0, 1.0, 1.0, -1.0])
    s_pt = L @ s @ L
    # reorder (x1, p1, x2, p2) -> xxpp for the package's Williamson routine
    P = np.zeros((4, 4))
    for row, col in enumerate((0, 2, 1, 3)):     # x1, x2, p1, p2
        P[row, col] = 1.0
    nus = symplectic_eigenvalues(P @ s_pt @ P.T, hbar=hbar)
    return float(np.min(nus))
