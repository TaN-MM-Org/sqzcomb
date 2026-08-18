"""Linearized fluctuation dynamics around an LLE steady state.

Writing psi = psi_s + delta and keeping first order, the doubled vector
z = (delta_a(k1..kM), delta_a*(k1..kM)) obeys  dz/dt = M z + inputs, with

    M = [[ A,        B       ],
         [ conj(B),  conj(A) ]]

    A_{kk'} = [-(1 + i alpha) + i D(k)] delta_{kk'} + 2 i (|psi_s|^2)^(k - k')
    B_{kk'} = i (psi_s^2)^(k + k')  (hat = Fourier component)

Below threshold every eigenvalue of M has a negative real part; the module
checks this and refuses to build spectra on an unstable state.
"""
from __future__ import annotations

import numpy as np

from .lle import _linear_symbol


def fluctuation_matrix(psi_s, alpha, dispersion=(0.0,), modes=None):
    """Doubled-basis drift matrix M around steady state psi_s.

    modes : iterable of integer mode numbers to keep (default: all grid
        modes). Returns (M, modes_array).
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
    comp_abs2 = dict(zip(kgrid, f_abs2))
    comp_sq = dict(zip(kgrid, f_sq))

    L = _linear_symbol(alpha, dispersion, modes.astype(float))
    A = np.zeros((m, m), dtype=complex)
    B = np.zeros((m, m), dtype=complex)
    for i, ki in enumerate(modes):
        for j, kj in enumerate(modes):
            if i == j:
                A[i, j] += L[i]
            A[i, j] += 2j * comp_abs2.get(ki - kj, 0.0)
            B[i, j] += 1j * comp_sq.get(ki + kj, 0.0)
    M = np.block([[A, B], [np.conj(B), np.conj(A)]])
    return M, modes


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
