"""Linearized fluctuation dynamics around an LLE steady state.

Writing psi = psi_s + delta and keeping first order, the doubled vector
z = (delta_a(k1..kM), delta_a*(k1..kM)) obeys  dz/dt = M z + inputs, with

    M = [[ A,        B       ],
         [ conj(B),  conj(A) ]]

    A_{kk'} = [-(1 + i alpha) + i D(k)] delta_{kk'} + 2 i (|psi_s|^2)^(k - k')
    (D(k) includes the pulse-drift term d1 k when d1 != 0)
    B_{kk'} = i (psi_s^2)^(k + k')  (hat = Fourier component)

Below threshold every eigenvalue of M has a negative real part; the module
checks this and refuses to build spectra on an unstable state.
"""
from __future__ import annotations

import numpy as np

from .lle import _linear_symbol


def fluctuation_matrix(psi_s, alpha: float, dispersion=(0.0,), modes=None,
                       d1: float = 0.0) -> tuple:
    """Doubled-basis drift matrix M around steady state psi_s.

    modes : iterable of integer mode numbers to keep (default: all grid
        modes). d1 : pump-pulse drift term of `lle_evolve` (0 for a
        continuous-wave pump); the pump itself never enters M.
    Returns (M, modes_array).
    """
    psi_s = np.asarray(psi_s, dtype=complex)
    n = psi_s.size
    kgrid = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    if modes is None:
        modes = kgrid.copy()
    modes = np.asarray(list(modes), dtype=int)
    m = modes.size

    f_abs2 = np.fft.fft(np.abs(psi_s) ** 2) / n     # components of |psi_s|^2
    f_sq = np.fft.fft(psi_s ** 2) / n               # components of psi_s^2

    L = _linear_symbol(alpha, dispersion, modes.astype(float), d1)
    # A_ij = L_i delta_ij + 2i (|psi|^2)^(k_i - k_j), B_ij = i (psi^2)^(k_i
    # + k_j), with a component that falls outside the grid's mode range
    # taken as zero (no wrap-around). Built with index arrays since
    # 0.16; the arithmetic is the same as the earlier double loop, entry
    # by entry, so the matrix is bit-identical (tested).
    A = np.zeros((m, m), dtype=complex)
    A[np.arange(m), np.arange(m)] = L
    A += 2j * _grid_components(f_abs2, kgrid, modes[:, None] - modes[None, :])
    B = np.zeros((m, m), dtype=complex)
    B += 1j * _grid_components(f_sq, kgrid, modes[:, None] + modes[None, :])
    M = np.block([[A, B], [np.conj(B), np.conj(A)]])
    return M, modes


def _grid_components(f, kgrid, k):
    """The Fourier component f of each mode number in the integer array
    k (FFT layout: mode v sits at slot v mod n), and zero for a mode
    number outside the grid's range [min(kgrid), max(kgrid)]."""
    n = f.size
    inside = (k >= kgrid.min()) & (k <= kgrid.max())
    out = np.zeros(k.shape, dtype=complex)
    out[inside] = f[np.mod(k[inside], n)]
    return out


def single_mode_parametric(mu, delta=0.0):
    """Drift matrix of a single-mode below-threshold parametric process.

    dz/dt = M z with M = [[-1 + i delta, mu], [conj(mu), -1 - i delta]].
    Threshold at |mu| = 1 (for delta = 0). For real mu > 0 the phi = pi/2
    output quadrature is squeezed and phi = 0 anti-squeezed. Used for
    validation against the closed-form spectra of the degenerate
    parametric oscillator.
    """
    return np.array([[-1.0 + 1j * delta, mu],
                     [np.conj(mu), -1.0 - 1j * delta]], dtype=complex)


def is_stable(M):
    return bool(np.max(np.linalg.eigvals(M).real) < 0.0)
