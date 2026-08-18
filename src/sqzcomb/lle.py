"""Lugiato-Lefever solver (normalized units).

    d psi / dt = -(1 + i alpha) psi + i |psi|^2 psi + i sum_m (d_m/m!) (-i d/dtheta)^m psi + F

with time in units of the photon lifetime 2/kappa, alpha the normalized
detuning, F the normalized pump, and d_m the normalized dispersion
coefficients (d_2 < 0 anomalous in this sign convention: D(k) = d2/2 k^2 + ...).

Integrator: Strang-split step. The Kerr phase step is exact (|psi|^2 is
invariant under it); the linear-plus-pump step is exact in the Fourier
domain (first-order linear ODE with constant inhomogeneity).
"""
from __future__ import annotations

import math

import numpy as np


def _linear_symbol(alpha, dispersion, k):
    D = np.zeros_like(k, dtype=float)
    for m, dm in enumerate(dispersion, start=2):
        D += dm / math.factorial(m) * k ** m
    return -(1.0 + 1j * alpha) + 1j * D


def lle_evolve(psi0, F, alpha, dispersion=(0.0,), t_end=50.0, dt=0.01):
    """Evolve the LLE from psi0 to t_end; returns the final field.

    psi0 : complex array over theta in [0, 2 pi) (uniform grid).
    F : scalar pump amplitude (homogeneous pump).
    dispersion : (d2, d3, ...) normalized coefficients.
    """
    psi = np.asarray(psi0, dtype=complex).copy()
    n = psi.size
    k = np.fft.fftfreq(n, d=1.0 / n)  # integer mode numbers
    L = _linear_symbol(alpha, dispersion, k)
    eL = np.exp(L * dt)
    Fk = np.zeros(n, dtype=complex)
    Fk[0] = F * n  # FFT convention: homogeneous term lives in k = 0
    # exact linear-with-pump propagator: psi_k -> eL psi_k + (eL - 1)/L * Fk
    with np.errstate(divide="ignore", invalid="ignore"):
        pump_prop = np.where(np.abs(L) > 1e-14, (eL - 1.0) / L, dt)
    steps = int(round(t_end / dt))
    for _ in range(steps):
        psi *= np.exp(1j * np.abs(psi) ** 2 * (0.5 * dt))   # half Kerr
        pk = np.fft.fft(psi)
        pk = eL * pk + pump_prop * Fk                        # exact linear
        psi = np.fft.ifft(pk)
        psi *= np.exp(1j * np.abs(psi) ** 2 * (0.5 * dt))   # half Kerr
    return psi


def homogeneous_steady_states(F, alpha):
    """Real intensities rho = |psi|^2 of the flat steady states.

    Roots of  rho * (1 + (alpha - rho)^2) = F^2.  Returns the real,
    non-negative roots in ascending order.
    """
    # rho^3 - 2 alpha rho^2 + (1 + alpha^2) rho - F^2 = 0
    roots = np.roots([1.0, -2.0 * alpha, 1.0 + alpha * alpha, -float(F) ** 2])
    real = roots[np.abs(roots.imag) < 1e-9].real
    return np.sort(real[real >= 0.0])
