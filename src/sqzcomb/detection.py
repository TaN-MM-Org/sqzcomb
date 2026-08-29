"""Imperfect detection: what the photodiodes report, not what the port emits.

Every spectrum elsewhere in this package is the noise at the extraction
port. A homodyne detector then degrades it three ways: optical loss
between port and diodes, sub-unity quantum efficiency of the diodes,
and electronic (dark) noise of the receiver. This module applies those
degradations, in both directions the package speaks: as a scalar map on
quadrature variances (vacuum variance 1/2, the package convention) and
as a Gaussian channel on xxpp covariance matrices from `gaussian`.

The model is the standard beamsplitter picture of inefficiency: a
detector of efficiency eta is a perfect detector behind a beamsplitter
of transmissivity eta whose open port injects vacuum (U. Leonhardt,
Measuring the Quantum State of Light, Cambridge University Press,
1997). On a variance with vacuum at 1/2,

    V_detected = eta * V + (1 - eta) / 2 + V_dark,

and on an xxpp covariance matrix it is the lossy Gaussian channel
sigma -> X sigma X^T + Y with X = sqrt(eta) I and Y = (1 - eta)
(hbar/2) I per mode (C. Weedbrook et al., Rev. Mod. Phys. 84, 621
(2012)). Distinct loss stages compose by multiplying efficiencies, and
the test suite asserts that composition exactly rather than assuming
it.

Electronic noise enters as an additive variance V_dark on the
shot-noise-normalized signal; `dark_from_clearance_db` converts the
number a datasheet or a spectrum analyzer actually gives (dark
clearance below shot noise, in dB) into that variance.

Exact facts the test suite asserts, rather than states:

* Vacuum is a fixed point of pure loss at every efficiency.
* Loss eta1 followed by eta2 equals loss eta1*eta2, scalar and matrix.
* The channel keeps every symplectic eigenvalue at or above hbar/2.
* Scalar and matrix forms agree on a squeezed single mode.
* 3 dB of squeezing detected at eta = 1/2 reads 10 log10(3/4) dB.
"""
from __future__ import annotations

import numpy as np

VACUUM_VARIANCE = 0.5


def _check_eta(eta):
    eta = float(eta)
    if not 0.0 < eta <= 1.0:
        raise ValueError("efficiency must be in (0, 1]")
    return eta


def detected_variance(variance, efficiency=1.0, dark_noise=0.0):
    """Quadrature variance after loss and electronic noise.

    variance : port quadrature variance(s), vacuum = 0.5 (scalar or
        array, e.g. a spectrum over frequencies).
    efficiency : total optical-plus-quantum efficiency in (0, 1]:
        the product of path transmission and diode quantum efficiency.
    dark_noise : additive electronic-noise variance in vacuum units
        (see `dark_from_clearance_db`); must be >= 0.

    Returns eta * V + (1 - eta)/2 + V_dark, elementwise.
    """
    eta = _check_eta(efficiency)
    dark = float(dark_noise)
    if dark < 0.0:
        raise ValueError("dark_noise must be non-negative")
    V = np.asarray(variance, dtype=float)
    if np.any(V < 0.0):
        raise ValueError("a quadrature variance cannot be negative")
    out = eta * V + (1.0 - eta) * VACUUM_VARIANCE + dark
    return float(out) if np.isscalar(variance) else out


def detected_squeezing_db(variance, efficiency=1.0, dark_noise=0.0):
    """Detected squeezing in dB relative to vacuum, after degradation."""
    V = detected_variance(variance, efficiency, dark_noise)
    return 10.0 * np.log10(np.asarray(V, dtype=float) / VACUUM_VARIANCE)


def dark_from_clearance_db(clearance_db):
    """Electronic-noise variance from dark clearance below shot noise.

    A receiver whose dark noise sits `clearance_db` dB below the shot
    noise level contributes V_dark = 0.5 * 10^(-clearance_db / 10) in
    vacuum units. 10 dB of clearance is V_dark = 0.05: enough to turn
    10 dB of otherwise perfectly detected squeezing into about 7 dB,
    which is why the number matters.
    """
    c = float(clearance_db)
    if c < 0.0:
        raise ValueError("clearance is measured below shot noise and "
                         "must be non-negative dB")
    return VACUUM_VARIANCE * 10.0 ** (-c / 10.0)


def lossy_channel_xxpp(sigma, efficiency, hbar=2.0):
    """Apply per-mode loss to an xxpp covariance matrix.

    sigma : (2n, 2n) covariance in xxpp ordering with vacuum =
        (hbar/2) I, as produced by `covariance_xxpp`.
    efficiency : scalar efficiency for all modes, or a length-n array
        of per-mode efficiencies, each in (0, 1].

    Returns X sigma X^T + Y with X = diag(sqrt(eta)) (x and p of a mode
    scaled together) and Y = (hbar/2) diag(1 - eta): the single-mode
    lossy channel of the Gaussian-state formalism, applied modewise.
    """
    sigma = np.asarray(sigma, dtype=float)
    m2 = sigma.shape[0]
    if sigma.shape != (m2, m2) or m2 % 2:
        raise ValueError("sigma must be a square (2n, 2n) matrix")
    n = m2 // 2
    etas = np.asarray(efficiency, dtype=float)
    if etas.ndim == 0:
        etas = np.full(n, float(etas))
    if etas.shape != (n,):
        raise ValueError("efficiency must be a scalar or one value per "
                         "mode")
    if np.any(etas <= 0.0) or np.any(etas > 1.0):
        raise ValueError("every efficiency must be in (0, 1]")
    scale = np.concatenate([np.sqrt(etas), np.sqrt(etas)])
    Y = 0.5 * float(hbar) * np.concatenate([1.0 - etas, 1.0 - etas])
    out = sigma * scale[:, None] * scale[None, :] + np.diag(Y)
    return 0.5 * (out + out.T)


def required_efficiency(target_variance, source_variance):
    """Minimum efficiency that still delivers a target variance.

    Inverts V_det = eta V + (1 - eta)/2 (loss only, no dark noise) for
    eta, given a squeezed source (V_source < 1/2) and a target
    (V_source <= V_target < 1/2). The answer,

        eta = (1/2 - V_target) / (1/2 - V_source),

    is the loss budget of a squeezing experiment in one line.
    """
    Vt = float(target_variance)
    Vs = float(source_variance)
    if not 0.0 <= Vs < VACUUM_VARIANCE:
        raise ValueError("source must be squeezed: 0 <= V < 0.5")
    if not Vs <= Vt < VACUUM_VARIANCE:
        raise ValueError("target must satisfy V_source <= V_target "
                         "< 0.5; loss cannot improve squeezing")
    return (VACUUM_VARIANCE - Vt) / (VACUUM_VARIANCE - Vs)


def required_efficiency_db(target_db, source_db):
    """`required_efficiency` with both levels given in dB (negative
    numbers for squeezing, e.g. target_db=-3.0, source_db=-10.0)."""
    Vt = VACUUM_VARIANCE * 10.0 ** (float(target_db) / 10.0)
    Vs = VACUUM_VARIANCE * 10.0 ** (float(source_db) / 10.0)
    return required_efficiency(Vt, Vs)
