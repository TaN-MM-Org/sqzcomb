"""Lugiato-Lefever solver (normalized units).

    d psi / dt = -(1 + i alpha) psi + i |psi|^2 psi + i sum_m (d_m/m!) (-i d/dtheta)^m psi + F

with time in units of the photon lifetime 2/kappa, alpha the normalized
detuning, F the normalized pump, and d_m the normalized dispersion
coefficients (d_2 < 0 anomalous in this sign convention: D(k) = d2/2 k^2 + ...).

Integrator: Strang-split step. The Kerr phase step is exact (|psi|^2 is
invariant under it); the linear-plus-pump step is exact in the Fourier
domain (first-order linear ODE with constant inhomogeneity).

Pumps other than a steady continuous wave (new in 0.13):

* F may be an array over the theta grid: a pump whose amplitude varies
  around the ring, which is how a train of pump pulses repeating once
  per round trip (synchronous pulsed pumping) enters this equation.
  The linearization (`fluctuation_matrix`) does not contain F at all,
  so every spectrum of the package applies unchanged around such a
  steady state.
* d1 adds the term d1 * d psi / d theta: the drift between the pump
  pulses and the ring's own round trip when their repetition rates
  differ. For pulses repeating at f_rep on a ring with free spectral
  range f_FSR (at the pumped line) and loaded linewidth kappa (rad/s),
  written in the frame that moves with the pump pulses,
  d1 = 4 pi (f_rep - f_FSR) / kappa. (Derivation: in the frame turning
  with the ring's round trip the injection point moves at angular
  speed 2 pi (f_rep - f_FSR); moving to the pump's frame adds that
  speed times d psi / d theta, and time is measured in units of
  2/kappa.) With a uniform pump, d1 only moves the pattern along the
  ring at speed d1, which the tests check. (Solitons driven by pulses
  were demonstrated by E. Obrzud, S. Lecomte and T. Herr, Nature
  Photonics 11, 600 (2017).)
  A pump profile breaks the ring's translation symmetry, so a soliton
  is pinned: the zero (Goldstone) eigenvalue of the uniform pump
  moves into the left half-plane and the spectra need no
  `allow_marginal`. Where it pins is not always the pump's peak; the
  tests find a case where the on-peak soliton is unstable and the
  stable one sits beside the peak, the spontaneous symmetry breaking
  described by I. Hendry et al., Phys. Rev. A 97, 053834 (2018). Too
  large a d1 leaves no steady state at all (the soliton keeps
  drifting), and `newton_state` then cannot converge and refuses. A
  refusal alone does not prove that, though: a starting guess too far
  from the state fails the same way, so let `lle_evolve` settle the
  field first.
* F may be a function F(t) of the normalized time (returning a scalar
  or a theta array): a pump whose power changes over many round trips.
  `lle_evolve` then holds F at its value at the middle of each step,
  which keeps the scheme second-order accurate.
"""
from __future__ import annotations

import math

import numpy as np


def _linear_symbol(alpha, dispersion, k, d1=0.0):
    D = np.zeros_like(k, dtype=float)
    for m, dm in enumerate(dispersion, start=2):
        D += dm / math.factorial(m) * k ** m
    if d1:
        D = D + float(d1) * k
    return -(1.0 + 1j * alpha) + 1j * D


def pump_spectrum(F, n):
    """Fourier components (FFT convention, length n) of the pump term:
    a scalar F lives in k = 0 only; an array is F(theta) on the grid."""
    if np.ndim(F) == 0:
        Fk = np.zeros(n, dtype=complex)
        Fk[0] = complex(F) * n  # FFT convention: homogeneous term in k = 0
        return Fk
    Fa = np.asarray(F, dtype=complex)
    if Fa.shape != (n,):
        raise ValueError("a pump profile F(theta) must have the same "
                         "length as the field grid")
    if not np.all(np.isfinite(Fa)):
        raise ValueError("the pump profile must be finite")
    return np.fft.fft(Fa)


def lle_evolve(psi0, F, alpha, dispersion=(0.0,), t_end=50.0, dt=0.01,
               d1=0.0, t0=0.0):
    """Evolve the LLE from psi0 to t_end; returns the final field.

    psi0 : complex array over theta in [0, 2 pi) (uniform grid).
    F : the pump -- a scalar (steady, uniform pump), an array over the
        theta grid (pump profile, e.g. synchronous pulse pumping), or a
        callable F(t) of the normalized time returning either.
    dispersion : (d2, d3, ...) normalized coefficients.
    d1 : drift between pump pulses and the ring's round trip (see the
        module docstring); 0 for a continuous-wave pump.
    t0 : starting time, used only to evaluate a callable F(t).
    """
    psi = np.asarray(psi0, dtype=complex).copy()
    n = psi.size
    k = np.fft.fftfreq(n, d=1.0 / n)  # integer mode numbers
    L = _linear_symbol(alpha, dispersion, k, d1)
    eL = np.exp(L * dt)
    # exact linear-with-pump propagator: psi_k -> eL psi_k + (eL - 1)/L * Fk
    with np.errstate(divide="ignore", invalid="ignore"):
        pump_prop = np.where(np.abs(L) > 1e-14, (eL - 1.0) / L, dt)
    timed = callable(F)
    if not timed:
        Fk = pump_spectrum(F, n)
    steps = int(round(t_end / dt))
    for step in range(steps):
        if timed:   # pump held at its mid-step value
            Fk = pump_spectrum(F(float(t0) + (step + 0.5) * dt), n)
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
