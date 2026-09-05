"""Localized steady states of the LLE: solitons, soliton crystals,
and parameter continuation.

`newton_state` solves the stationary Lugiato-Lefever equation

    0 = -(1 + i alpha) psi + i |psi|^2 psi + i D psi + F

by Newton's method in Fourier space -- and its Jacobian is the same
linearization the package's `fluctuation_matrix` builds around a
steady state, evaluated at the current iterate: the operator that
damps a perturbation of a steady state and the operator Newton inverts
to find that steady state are one and the same.  (The solver keeps the
grid's wrapped convolution entries that the physically mode-truncated
`fluctuation_matrix` sets to zero -- the exact Jacobian of the
*discrete* equations, needed for machine-precision convergence.  The
wrapped entries only transfer amplitude across the band edge, so on
the resolved soliton's own modes the two operators act identically:
the test suite asserts that both annihilate the translation mode to
1e-9.)

Exact facts the test suite asserts, rather than states:

* the converged residual is below the requested tolerance (the solver
  refuses to return otherwise), and the converged state, handed to the
  independent split-step evolver `lle_evolve`, stays put;
* from a flat seed, Newton lands on the exact root of the homogeneous
  cubic;
* an N-fold soliton crystal on the 2 pi ring equals, to machine
  precision, the single soliton of the same equation with dispersion
  d_m scaled by N^m, resampled onto the crystal grid -- the exact
  rescaling theta -> N theta of the periodic domain;
* the linearization around any localized steady state carries an
  *exact* zero eigenvalue whose eigenvector is the translation mode
  d psi / d theta (Goldstone mode of the broken translation symmetry);
  the drift matrix is therefore marginally, not asymptotically, stable,
  and the spectra machinery refuses it unless told that the marginal
  direction is understood (`allow_marginal`, see `spectra`).

The sech seed uses the standard soliton asymptotics of the anomalous
LLE (amplitude ~ sqrt(2 alpha), width ~ sqrt(|d2| / alpha), pump phase
cos phi_0 = sqrt(8 alpha) / (pi F); T. Herr et al., Nature Photonics 8,
145 (2014)); the *converged* answers never depend on the seed quality,
only Newton's basin does.
"""
from __future__ import annotations

import numpy as np

from .lle import _linear_symbol, homogeneous_steady_states

__all__ = ["newton_state", "soliton_seed", "continuation"]


def _residual_k(psi, F, alpha, dispersion):
    """Stationary-LLE residual in Fourier space (FFT convention)."""
    n = psi.size
    k = np.fft.fftfreq(n, d=1.0 / n)
    L = _linear_symbol(alpha, dispersion, k)
    Rk = L * np.fft.fft(psi) + np.fft.fft(1j * np.abs(psi) ** 2 * psi)
    Rk[0] += F * n
    return Rk


def _jacobian(psi, alpha, dispersion):
    """Exact Jacobian of the discrete (pseudospectral) residual, in the
    doubled Fourier basis.

    Identical to `fluctuation_matrix` except that the products
    2 |psi|^2 delta and psi^2 conj(delta) are the *grid's* circular
    convolutions (mode transfers taken modulo n): the exact Jacobian
    of the discrete pseudospectral residual, which Newton needs to
    converge to machine precision.  The entries where the two differ
    are exactly the wrapped transfers across the band edge; on
    perturbations resolved inside the band the two operators act
    identically, which the test suite asserts on the soliton's
    translation mode.
    """
    psi = np.asarray(psi, dtype=complex)
    n = psi.size
    kgrid = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    f_abs2 = np.fft.fft(np.abs(psi) ** 2) / n
    f_sq = np.fft.fft(psi ** 2) / n
    L = _linear_symbol(alpha, dispersion, kgrid.astype(float))
    # FFT layout: the component of mode value v sits at slot v mod n
    A = 2j * f_abs2[(kgrid[:, None] - kgrid[None, :]) % n]
    B = 1j * f_sq[(kgrid[:, None] + kgrid[None, :]) % n]
    A[np.arange(n), np.arange(n)] += L
    return np.block([[A, B], [np.conj(B), np.conj(A)]])


def newton_state(psi0, F, alpha, dispersion=(0.0,), tol=1e-12,
                 maxiter=60):
    """Newton solution of the stationary LLE from the seed psi0.

    Returns (psi, info) with info = {"residual", "iterations"}; the
    residual is the max-norm of the stationary equation on the theta
    grid.  Raises RuntimeError if the tolerance is not reached -- a
    state is never returned unverified.  The Jacobian at each iterate
    is the exact discrete form of `fluctuation_matrix(psi, alpha,
    dispersion)` (see `_jacobian`).
    """
    psi = np.asarray(psi0, dtype=complex).copy()
    n = psi.size
    res = np.inf
    for it in range(int(maxiter)):
        Rk = _residual_k(psi, F, alpha, dispersion)
        res = float(np.abs(np.fft.ifft(Rk)).max())
        if res < tol:
            return psi, {"residual": res, "iterations": it}
        J = _jacobian(psi, alpha, dispersion)
        rhs = np.concatenate([Rk / n, np.conj(Rk) / n])
        # minimal-norm least-squares step: at a localized state the
        # Jacobian is exactly singular along the translation (Goldstone)
        # mode, and the minimal-norm solution is the Newton step that
        # does not slide along the soliton position
        delta = np.linalg.lstsq(J, -rhs, rcond=1e-9)[0]
        psi = psi + np.fft.ifft(delta[:n] * n)
    raise RuntimeError(
        f"Newton did not reach tol = {tol:g} in {maxiter} iterations "
        f"(residual {res:.3e}); refusing to return an unconverged "
        "state")


def soliton_seed(n, F, alpha, dispersion, n_pulses=1):
    """Sech ansatz seed for `newton_state`: the lower flat state plus
    ``n_pulses`` equally spaced sech pulses with the standard
    asymptotic amplitude sqrt(2 alpha), width sqrt(|d2| / (2 alpha))
    and pump phase cos phi_0 = sqrt(8 alpha) / (pi F) (Herr et al.,
    Nature Photonics 8, 145 (2014)).  Anomalous dispersion (d2 < 0 in
    this package's sign convention) required.
    """
    d2 = float(dispersion[0])
    if d2 >= 0.0:
        raise ValueError("bright-soliton seeds need anomalous "
                         "dispersion (d2 < 0 in this convention)")
    roots = homogeneous_steady_states(F, alpha)
    rho_low = float(roots[0])
    psi_low = F / (1.0 + 1j * (alpha - rho_low))
    arg = np.sqrt(8.0 * alpha) / (np.pi * F)
    phi0 = np.arccos(min(arg, 1.0))
    theta = 2.0 * np.pi * np.arange(n) / n
    width = np.sqrt(abs(d2) / (2.0 * alpha))
    psi = np.full(n, psi_low, dtype=complex)
    for p in range(int(n_pulses)):
        center = 2.0 * np.pi * (p + 0.5) / n_pulses
        d = np.angle(np.exp(1j * (theta - center)))   # wrapped distance
        psi += np.sqrt(2.0 * alpha) * np.exp(1j * phi0) / np.cosh(d / width)
    return psi


def continuation(psi0, F, alphas, dispersion=(0.0,), tol=1e-12,
                 maxiter=60):
    """Sweep the detuning: Newton-solve at each alpha in ``alphas``,
    seeding each step with the previous converged state.

    Returns (states, infos) as lists.  Every step is verified to the
    tolerance or the sweep raises (naming the failing alpha), so a
    returned branch contains no unconverged member.  Sweeping F at
    fixed alpha works by transposing the roles: call in a loop, this
    function stays deliberately simple.
    """
    psi = np.asarray(psi0, dtype=complex).copy()
    states, infos = [], []
    for a in alphas:
        try:
            psi, info = newton_state(psi, F, float(a), dispersion,
                                     tol=tol, maxiter=maxiter)
        except RuntimeError as exc:
            raise RuntimeError(
                f"continuation failed at alpha = {a}: {exc}") from exc
        states.append(psi.copy())
        infos.append(info)
    return states, infos
