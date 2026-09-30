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
clearance below shot noise, in dB) into that variance. Mind which
clearance you have: by default it is measured from the pure shot-noise
level; the gap between an analyzer's shot-noise trace (which also
carries the dark noise) and its dark trace needs `trace_gap=True`.

Which shot-noise level the result is quoted against matters once
there is dark noise (new in 0.14, `dark_in_reference`). By default the
signal carries V_dark and the reference is the pure vacuum level 1/2,
the convention of every earlier version. In a laboratory, though, the
shot-noise reference is itself a trace taken through the same
receiver, so it carries the dark noise too, and the ratio of the two
traces is

    V_meas = (1/2) (V_det + V_dark) / (1/2 + V_dark),
    V_det = eta V + (1 - eta)/2.

That is again a loss: V_meas = eta_e V_det + (1 - eta_e)/2 with
eta_e = (1/2) / (1/2 + V_dark) (`dark_equivalent_efficiency`), the
equivalence of electronic noise and optical loss shown by J. Appel,
D. Hoffman, E. Figueroa and A. I. Lvovsky, Phys. Rev. A 75, 035802
(2007), for a detector calibrated on the vacuum noise. (The formula
for eta_e above is derived here from the two-trace ratio and checked
in the tests.) Pass `dark_in_reference=True` to get that number; the
default reads slightly LESS squeezing (for 10 dB of source squeezing
seen with eta = 0.7 and a 15 dB dark clearance: -3.962 dB by default,
-4.097 dB against a shot trace that carries the dark noise).

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


def _check_dark(dark_noise):
    dark = float(dark_noise)
    if not (np.isfinite(dark) and dark >= 0.0):
        raise ValueError("dark_noise must be finite and non-negative")
    return dark


def dark_equivalent_efficiency(dark_noise):
    """The loss that dark noise is equivalent to when the result is
    quoted against a shot-noise trace taken through the same receiver:
    eta_e = (1/2) / (1/2 + V_dark) (see the module docstring; Appel et
    al., Phys. Rev. A 75, 035802 (2007)). V_dark in vacuum units, e.g.
    from `dark_from_clearance_db`."""
    return VACUUM_VARIANCE / (VACUUM_VARIANCE + _check_dark(dark_noise))


def detected_variance(variance, efficiency=1.0, dark_noise=0.0,
                      dark_in_reference=False):
    """Quadrature variance after loss and electronic noise.

    variance : port quadrature variance(s), vacuum = 0.5 (scalar or
        array, e.g. a spectrum over frequencies).
    efficiency : total optical-plus-quantum efficiency in (0, 1]:
        the product of path transmission and diode quantum efficiency.
    dark_noise : additive electronic-noise variance in vacuum units
        (see `dark_from_clearance_db`); must be >= 0.
    dark_in_reference : False (default, as before): returns
        eta * V + (1 - eta)/2 + V_dark, the signal against the pure
        vacuum level. True: the signal as a spectrum analyzer reports
        it against a shot-noise trace that carries the same dark
        noise, (1/2) (eta V + (1 - eta)/2 + V_dark) / (1/2 + V_dark),
        which is the loss `dark_equivalent_efficiency` applied after
        eta (see the module docstring).

    Elementwise on arrays.
    """
    eta = _check_eta(efficiency)
    dark = _check_dark(dark_noise)
    V = np.asarray(variance, dtype=float)
    if np.any(V < 0.0):
        raise ValueError("a quadrature variance cannot be negative")
    out = eta * V + (1.0 - eta) * VACUUM_VARIANCE + dark
    if dark_in_reference:
        out = VACUUM_VARIANCE * out / (VACUUM_VARIANCE + dark)
    return float(out) if np.isscalar(variance) else out


def detected_squeezing_db(variance, efficiency=1.0, dark_noise=0.0,
                          dark_in_reference=False):
    """Detected squeezing in dB relative to shot noise, after
    degradation (see `detected_variance` for `dark_in_reference`)."""
    V = detected_variance(variance, efficiency, dark_noise,
                          dark_in_reference)
    return 10.0 * np.log10(np.asarray(V, dtype=float) / VACUUM_VARIANCE)


def dark_from_clearance_db(clearance_db, trace_gap=False):
    """Electronic-noise variance from dark clearance below shot noise.

    Two ways of reading "clearance" are in use, and they differ:

    trace_gap=False (default, as in every earlier version): the dark
        noise sits `clearance_db` dB below the PURE shot-noise level
        (the shot noise alone, without the receiver's own noise), so
        V_dark = 0.5 * 10^(-clearance_db / 10) in vacuum units. 10 dB
        of clearance is V_dark = 0.05: enough to turn 10 dB of
        otherwise perfectly detected squeezing into about 7 dB, which
        is why the number matters.
    trace_gap=True (new in 0.14): `clearance_db` is the gap an analyzer
        shows between the shot-noise trace and the dark trace. The shot
        trace carries the dark noise too, so the gap is 10 log10((0.5 +
        V_dark) / V_dark) and V_dark = 0.5 / (10^(clearance_db / 10) -
        1). This is the convention of `lab.shot_noise_normalize`, whose
        `dark_variance` it reproduces (tested). A 15 dB trace gap gives
        V_dark = 0.01633; the default reading of 15 dB gives 0.01581.
        The gap must be positive.
    """
    c = float(clearance_db)
    if np.isnan(c):
        raise ValueError("clearance must be a number of dB, not NaN")
    if c < 0.0:
        raise ValueError("clearance is measured below shot noise and "
                         "must be non-negative dB")
    if trace_gap:
        if c <= 0.0:
            raise ValueError("a trace gap of 0 dB means the dark noise "
                             "is all there is; it must be positive")
        return VACUUM_VARIANCE / np.expm1(c * np.log(10.0) / 10.0)
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


def required_efficiency(target_variance, source_variance, dark_noise=0.0,
                        v_antisqueezed=None, theta_rms=0.0,
                        dark_in_reference=False):
    """Minimum efficiency that still delivers a target variance.

    Inverts V_det = eta V + (1 - eta)/2 (loss only, no dark noise) for
    eta, given a squeezed source (V_source < 1/2) and a target
    (V_source <= V_target < 1/2). The answer,

        eta = (1/2 - V_target) / (1/2 - V_source),

    is the loss budget of a squeezing experiment in one line.

    The whole budget (new in 0.14, all optional): with dark noise
    V_dark and LO phase jitter theta_rms (which needs the antisqueezed
    variance v_antisqueezed of the source), the detected variance is
    eta V_j + (1 - eta)/2 + V_dark with V_j = `phase_noise_variance`
    (V_source, v_antisqueezed, theta_rms) (jitter and loss commute),
    still linear in eta, so

        eta = (1/2 + V_dark - V_target) / (1/2 - V_j)

    exactly; with `dark_in_reference=True` (see `detected_variance`)
    the dark noise is the extra loss eta_e = 1/2 / (1/2 + V_dark) and
    eta = (1/2 - V_target) / ((1/2 - V_j) eta_e). A target that even
    eta = 1 cannot reach with this dark noise and jitter is refused,
    with the best value that is reachable. The defaults give the
    formula above unchanged.
    """
    Vt = float(target_variance)
    Vs = float(source_variance)
    dark = _check_dark(dark_noise)
    th = float(theta_rms)
    if not (np.isfinite(th) and th >= 0.0):
        raise ValueError("theta_rms must be finite and >= 0")
    if not 0.0 <= Vs < VACUUM_VARIANCE:
        raise ValueError("source must be squeezed: 0 <= V < 0.5")
    if th > 0.0:
        if v_antisqueezed is None:
            raise ValueError("phase jitter mixes in the antisqueezed "
                             "quadrature: give v_antisqueezed")
        Va = float(v_antisqueezed)
        if not Va > Vs:
            raise ValueError("need v_antisqueezed > source_variance")
        Vj = phase_noise_variance(Vs, Va, th)
        if not Vj < VACUUM_VARIANCE:
            raise ValueError(
                f"with {th:g} rad of jitter the source reads {Vj:.4g} "
                ">= 0.5 before any loss: no efficiency gives squeezing")
    else:
        Vj = Vs
    if dark == 0.0 and th == 0.0:
        if not Vs <= Vt < VACUUM_VARIANCE:
            raise ValueError("target must satisfy V_source <= V_target "
                             "< 0.5; loss cannot improve squeezing")
        return (VACUUM_VARIANCE - Vt) / (VACUUM_VARIANCE - Vs)
    best = detected_variance(Vj, 1.0, dark, dark_in_reference)
    if not best <= Vt < VACUUM_VARIANCE:
        raise ValueError(
            f"target {Vt:.6g} must satisfy {best:.6g} <= V_target < 0.5: "
            f"{best:.6g} is what this source gives at efficiency 1 with "
            "this dark noise and jitter, and loss cannot improve on it")
    if dark_in_reference:
        eta = (VACUUM_VARIANCE - Vt) / ((VACUUM_VARIANCE - Vj)
                                        * dark_equivalent_efficiency(dark))
    else:
        eta = (VACUUM_VARIANCE + dark - Vt) / (VACUUM_VARIANCE - Vj)
    return float(min(eta, 1.0))


def required_efficiency_db(target_db, source_db, dark_noise=0.0,
                           anti_db=None, theta_rms=0.0,
                           dark_in_reference=False):
    """`required_efficiency` with the levels given in dB (negative
    numbers for squeezing, e.g. target_db=-3.0, source_db=-10.0;
    anti_db is the source's antisqueezing, needed with theta_rms)."""
    Vt = VACUUM_VARIANCE * 10.0 ** (float(target_db) / 10.0)
    Vs = VACUUM_VARIANCE * 10.0 ** (float(source_db) / 10.0)
    Va = None if anti_db is None else \
        VACUUM_VARIANCE * 10.0 ** (float(anti_db) / 10.0)
    return required_efficiency(Vt, Vs, dark_noise, Va, theta_rms,
                               dark_in_reference)


# ------------------------------------------------------------------
# Homodyne phase noise (new in v0.10): the limit that dominates once
# loss is tamed.
#
# A homodyne detector projects onto a quadrature set by the local-
# oscillator phase. A static error theta mixes the antisqueezed
# quadrature into the measurement,
#
#     V(theta) = V_sq cos^2(theta) + V_anti sin^2(theta),
#
# exactly (the rotation law of the quadrature variances; S. Dwyer et
# al., Opt. Express 21, 19047 (2013); E. Oelker et al., Optica 3, 682
# (2016)). For Gaussian phase jitter of RMS size sigma the average has
# the closed form
#
#     <cos^2 theta> = (1 + exp(-2 sigma^2)) / 2,
#
# from the Gaussian characteristic function E[cos 2 theta] =
# exp(-2 sigma^2) -- anchored in the tests against direct numerical
# integration over the Gaussian, not trusted. Because V_anti of a
# near-pure source is at least the inverse of V_sq, phase noise sets a
# floor on detectable squeezing that tightens as the source improves;
# the 2026 variance-budget analysis of integrated squeezers reaches
# the same expression and identifies it as a dominant practical limit
# (D. J. Dean et al., "Practical limits on integrated squeezers",
# npj Nanophotonics (2026), doi:10.1038/s44310-026-00125-5).


def phase_noise_variance(v_squeezed, v_antisqueezed, theta_rms,
                         static_offset=0.0):
    """Detected quadrature variance under local-oscillator phase noise.

    v_squeezed, v_antisqueezed : the two quadrature variances at the
        detector input (vacuum = 0.5; scalars or arrays, e.g. spectra
        over frequencies).
    theta_rms : RMS Gaussian phase jitter (radians).
    static_offset : optional deterministic phase error (radians),
        applied on top of the jitter.

    Returns the jitter-averaged measured variance
    V = V_sq <cos^2(theta0 + theta)> + V_anti <sin^2(theta0 + theta)>
    with the exact Gaussian average
    <cos 2(theta0 + theta)> = cos(2 theta0) exp(-2 sigma^2).
    """
    vs = np.asarray(v_squeezed, dtype=float)
    va = np.asarray(v_antisqueezed, dtype=float)
    s = float(theta_rms)
    t0 = float(static_offset)
    if s < 0.0 or not np.isfinite(s):
        raise ValueError("theta_rms must be finite and >= 0")
    mean_cos2 = np.cos(2.0 * t0) * np.exp(-2.0 * s * s)
    c2 = 0.5 * (1.0 + mean_cos2)          # <cos^2>
    out = vs * c2 + va * (1.0 - c2)
    return float(out) if np.ndim(out) == 0 else out


def phase_noise_squeezing_db(sq_db, anti_db, theta_rms,
                             static_offset=0.0):
    """`phase_noise_variance` with source levels in dB relative to
    vacuum (squeezing negative, antisqueezing positive); returns the
    detected level in dB."""
    vs = VACUUM_VARIANCE * 10.0 ** (float(sq_db) / 10.0)
    va = VACUUM_VARIANCE * 10.0 ** (float(anti_db) / 10.0)
    v = phase_noise_variance(vs, va, theta_rms, static_offset)
    return 10.0 * np.log10(v / VACUUM_VARIANCE)


def max_phase_noise(target_variance, v_squeezed, v_antisqueezed,
                    efficiency=1.0, dark_noise=0.0,
                    dark_in_reference=False):
    """Largest RMS phase jitter that still delivers a target variance.

    Inverts the Gaussian-averaged mixing formula for sigma (no static
    offset): the planning number of a homodyne experiment, the phase-
    noise counterpart of `required_efficiency`. Refuses a target the
    source cannot reach at any jitter: below V_sq (phase noise cannot
    improve squeezing) or at/above the sigma -> infinity limit
    (V_sq + V_anti)/2, where the measurement no longer distinguishes
    the quadratures.

    efficiency, dark_noise, dark_in_reference (new in 0.14): the
    target is then the detected variance after the jitter, the loss
    and the dark noise, as `detected_variance(phase_noise_variance(
    ...), efficiency, dark_noise, dark_in_reference)` computes it. Both
    maps are affine, so the target is first carried back through the
    loss and dark noise exactly and the formula above applies. The
    refusals then refer to that carried-back target.
    """
    Vs = float(v_squeezed)
    Va = float(v_antisqueezed)
    if not (Va > Vs >= 0.0):
        raise ValueError("need v_antisqueezed > v_squeezed >= 0")
    eta = _check_eta(efficiency)
    dark = _check_dark(dark_noise)
    Vt = float(target_variance)
    if dark_in_reference:
        eta_e = dark_equivalent_efficiency(dark)
        Vt = (Vt - (1.0 - eta_e) * VACUUM_VARIANCE) / eta_e
        dark = 0.0
    Vt = (Vt - (1.0 - eta) * VACUUM_VARIANCE - dark) / eta
    mid = 0.5 * (Vs + Va)
    if Vt < Vs:
        raise ValueError("phase noise cannot improve squeezing: the "
                         "target lies below the source variance"
                         + ("" if (eta == 1.0 and float(dark_noise) == 0.0)
                            else " (after this loss and dark noise)"))
    if Vt >= mid:
        raise ValueError(
            f"target {Vt:.4g} is not phase-noise-limited: even "
            f"infinite jitter only degrades to (V_sq + V_anti)/2 = "
            f"{mid:.4g}. Check the loss budget instead")
    # Vt = Vs c2 + Va (1 - c2), c2 = (1 + e^{-2 s^2})/2
    e = (Vs + Va - 2.0 * Vt) / (Va - Vs)      # = exp(-2 sigma^2)
    return float(np.sqrt(-0.5 * np.log(e)))
