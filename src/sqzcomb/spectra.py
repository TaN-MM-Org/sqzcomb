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

from .linearize import is_stable


def _output_covariance(M, eta, omega):
    m2 = M.shape[0]
    ident = np.eye(m2, dtype=complex)
    G = np.linalg.inv(-1j * omega * ident - M)
    T_ex = 2.0 * eta * G - ident
    T_0 = 2.0 * np.sqrt(eta * (1.0 - eta)) * G
    # vacuum covariance <z_in z_in^dagger> in the (a, a*) doubled ordering:
    # <a a^dagger> = 1, all other blocks zero
    half = m2 // 2
    N = np.zeros((m2, m2), dtype=complex)
    N[:half, :half] = np.eye(half)
    S = T_ex @ N @ T_ex.conj().T + T_0 @ N @ T_0.conj().T
    return S


def output_quadrature_variance(M, eta, omega, mode_index, n_modes,
                               phi=0.0, mode_index_b=None):
    """Symmetrized variance of an output quadrature at frequency omega.

    mode_index : index of the mode (within the retained mode list) whose
        quadrature is detected. If mode_index_b is given, the joint
        two-mode quadrature (a + b)/sqrt(2) rotated by phi is used, the
        natural variable for twin-beam squeezing.
    Vacuum level is 0.5. Requires a stable M.
    """
    if not is_stable(M):
        raise ValueError("drift matrix is unstable (above threshold); "
                         "linearized spectra are meaningless there")
    S = _output_covariance(M, eta, omega)
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
