"""Output quadrature-noise spectra via input-output theory.

The cavity couples to the extraction port with rate fraction eta = kappa_ex
/ kappa and to intrinsic loss with 1 - eta (total damping normalized to 1
in the doubled drift matrix M, i.e. Re eigenvalues -> -1 far from any
parametric process). With vacuum at both inputs,

    z(Omega)      = G(Omega) [ sqrt(2 eta) z_ex + sqrt(2 (1-eta)) z_0 ]
    z_out(Omega)  = sqrt(2 eta) z(Omega) - z_ex(Omega)
    G(Omega)      = (-i Omega I - M)^{-1}

and the symmetrized output covariance follows from the vacuum input
covariance. Quadrature X_phi = (a e^{-i phi} + a^dagger e^{i phi}) /
sqrt(2), vacuum variance 1/2.
"""
from __future__ import annotations

import numpy as np


def _bath_covariance(m2, n_bar):
    """<z_in z_in^dagger> of a thermal bath in the doubled ordering:
    <a a^dagger> = n_bar + 1, <a^dagger a> = n_bar (vacuum: n_bar = 0)."""
    half = m2 // 2
    N = np.zeros((m2, m2), dtype=complex)
    N[:half, :half] = (float(n_bar) + 1.0) * np.eye(half)
    N[half:, half:] = float(n_bar) * np.eye(half)
    return N


def _check_spectra_stability(M, allow_marginal, tol=1e-6):
    lam = float(np.max(np.linalg.eigvals(np.asarray(M)).real))
    if lam < -tol:
        return
    if allow_marginal and lam < tol:
        return
    if lam < tol:
        raise ValueError(
            "drift matrix is marginally stable (an eigenvalue's real "
            "part is numerically zero). Around a localized steady "
            "state this is the exact translation (Goldstone) mode of "
            "the soliton; spectra at omega != 0 remain finite, so pass "
            "allow_marginal=True if that marginal direction is "
            "understood. A genuinely positive growth rate is still "
            "refused.")
    raise ValueError("drift matrix is unstable (above threshold); "
                     "linearized spectra are meaningless there")


def _output_covariance(M, eta, omega, n_th_port=0.0, n_th_loss=0.0):
    m2 = M.shape[0]
    ident = np.eye(m2, dtype=complex)
    G = np.linalg.inv(-1j * omega * ident - M)
    T_ex = 2.0 * eta * G - ident
    T_0 = 2.0 * np.sqrt(eta * (1.0 - eta)) * G
    S = T_ex @ _bath_covariance(m2, n_th_port) @ T_ex.conj().T \
        + T_0 @ _bath_covariance(m2, n_th_loss) @ T_0.conj().T
    return S


def output_quadrature_variance(M, eta, omega, mode_index, n_modes,
                               phi=0.0, mode_index_b=None,
                               n_th_port=0.0, n_th_loss=0.0,
                               allow_marginal=False):
    """Symmetrized variance of an output quadrature at frequency omega.

    mode_index : index of the mode (within the retained mode list) whose
        quadrature is detected. If mode_index_b is given, the joint
        two-mode quadrature (a + b)/sqrt(2) rotated by phi is used, the
        natural variable for twin-beam squeezing.
    n_th_port, n_th_loss : Bose occupations of the extraction-port
        input and of the intrinsic-loss bath (default vacuum, 0; see
        `thermal_occupation` for the physical number). A passive cavity
        with both baths at n_bar emits exactly (2 n_bar + 1)/2 at every
        frequency, coupling and phase -- asserted in the tests.
    allow_marginal : accept a drift matrix whose largest eigenvalue
        real part is numerically zero (the soliton's exact translation
        Goldstone mode); spectra at omega != 0 stay finite. Genuinely
        unstable matrices are refused regardless.
    Vacuum level is 0.5. Requires a stable M.
    """
    if float(n_th_port) < 0.0 or float(n_th_loss) < 0.0:
        raise ValueError("thermal occupations must be non-negative")
    _check_spectra_stability(M, allow_marginal)
    S = _output_covariance(M, eta, omega, n_th_port, n_th_loss)
    u = np.zeros(2 * n_modes, dtype=complex)
    if mode_index_b is None:
        u[mode_index] = np.exp(-1j * phi) / np.sqrt(2.0)
        u[n_modes + mode_index] = np.exp(1j * phi) / np.sqrt(2.0)
    else:
        for idx, w in ((mode_index, 0.5), (mode_index_b, 0.5)):
            u[idx] += np.exp(-1j * phi) * w
            u[n_modes + idx] += np.exp(1j * phi) * w
    # With N giving <a a^dagger> = 1 and zero elsewhere, u^dag S u is the
    # output quadrature spectrum with vacuum level exactly 1/2; the
    # passive-cavity identity (T_ex N T_ex^dag + T_0 N T_0^dag = N for any
    # eta, Omega) is verified in the test-suite.
    var = np.real(u.conj() @ S @ u)
    return float(var)


def squeezing_db(variance):
    """Convert a quadrature variance to dB relative to vacuum (0.5)."""
    return 10.0 * np.log10(variance / 0.5)
