"""Thermal input noise: Bose-Einstein occupations for the baths.

Everywhere else the package's baths are vacuum.  Physically each bath is
a thermal state with mean occupation n_bar = 1 / (exp(hbar omega /
k_B T) - 1); at optical frequencies and room temperature n_bar is
astronomically small (which is why vacuum is the right default), but in
the microwave domain, or for hot spurious baths, it is not.  The
spectra and covariance machinery accept per-bath occupations; this
module supplies the number itself.

Constants are the exact SI values (2019 redefinition): h =
6.62607015e-34 J s and k_B = 1.380649e-23 J/K are exact by definition
(BIPM, The International System of Units, 9th edition, 2019).

Exact facts the test suite asserts, rather than states:

* a passive cavity whose baths all sit at occupation n_bar emits a
  thermal line: every output quadrature, at every frequency, coupling
  and phase, has variance exactly (2 n_bar + 1) / 2 -- the thermal
  generalization of the vacuum fixed point;
* the intracavity occupation of a passive mode equals the bath's n_bar
  exactly;
* the below-threshold parametric oscillator with thermal inputs emits
  exactly (2 n_bar + 1) times its vacuum-input spectrum, at every
  frequency, coupling and quadrature (the quadrature sectors are
  scalar channels, so the input variance factors out);
* n_bar(omega, T) crosses 1 exactly at hbar omega = k_B T ln 2.
"""
from __future__ import annotations

import numpy as np

H_PLANCK = 6.62607015e-34   # J s, exact (SI 2019)
K_BOLTZMANN = 1.380649e-23  # J / K, exact (SI 2019)


def thermal_occupation(frequency_hz, temperature_k):
    """Bose-Einstein mean occupation n_bar = 1/(exp(h f / k_B T) - 1).

    frequency_hz : ordinary frequency f in Hz (scalar or array).
    temperature_k : temperature in kelvin (> 0; zero returns 0 exactly).
    """
    f = np.asarray(frequency_hz, dtype=float)
    T = float(temperature_k)
    if np.any(f <= 0.0):
        raise ValueError("frequency must be positive")
    if T < 0.0:
        raise ValueError("temperature must be non-negative")
    if T == 0.0:
        out = np.zeros_like(f)
        return float(out) if np.isscalar(frequency_hz) else out
    x = H_PLANCK * f / (K_BOLTZMANN * T)
    with np.errstate(over="ignore"):
        out = 1.0 / np.expm1(x)
    return float(out) if np.isscalar(frequency_hz) else out
