"""Bridge between laboratory ring parameters and the package's
normalized units.

Everything in this package runs in the standard normalized LLE frame
(module docstring of `sqzcomb.lle`): time in units of the photon
lifetime 2/kappa, field normalized so the Kerr term is i|psi|^2 psi,
and every rate in the drift matrices in units of kappa/2 (which is why
`single_mode_parametric` has decay -1 and threshold mu = 1).  An
experiment, however, is specified in laboratory quantities: a loaded
linewidth in Hz, an escape efficiency, a dispersion D2, a single-photon
Kerr shift g0, a pump power in watts.  This module is the exact
dictionary between the two, with the conventions written down once and
checked by identities rather than trusted.

The mapping (derived by direct substitution into the physical LLE
da/dt = -(kappa/2)(1 + i alpha) a + i g0 |a|^2 a + sqrt(kappa_ext
P / (hbar omega_p)), then t -> t kappa/2 and psi = sqrt(2 g0/kappa) a):

    F      = sqrt( 8 g0 kappa_ext P / (kappa^3 hbar omega_p) )
    alpha  = 2 delta_omega / kappa       (delta_omega = omega_cav - omega_pump)
    d2     = -2 D2 / kappa               (anomalous D2 > 0  ->  d2 < 0,
                                          matching `sqzcomb.lle`'s stated
                                          sign convention -- an independent
                                          cross-check of the derivation)
    omega_norm = omega_phys / (kappa/2)  (spectra frequency axis)
    N_photons  = rho kappa / (2 g0)      (rho = |psi|^2)

All internal angular frequencies are rad/s; the public interface takes
ordinary laboratory Hz (linewidths as FWHM kappa/2pi, D2/2pi per
mode^2, g0/2pi) and converts once.  hbar and c come from
`scipy.constants` -- no constant in this module is typed by hand.

Anchors asserted in the tests rather than stated: every conversion
round-trips exactly; the scaling structure of F (proportional to
sqrt(P), to kappa^{-3/2} at fixed couplings) holds numerically; and
`threshold_power` is cross-validated through the package's own cubic:
at the returned power, `homogeneous_steady_states` contains the
rho = 1 root to machine precision, because F_th^2 = 1 + (alpha - 1)^2
is an identity of the flat-state cubic, not a number typed here.

Provenance is mandatory: a RingSpec without a `reference` naming where
its numbers come from (your own calibration, a paper) is refused --
the same rule every package in this organization applies to physical
parameters.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.constants import c as _C_LIGHT
from scipy.constants import hbar as _HBAR

__all__ = ["RingSpec", "normalized_pump", "threshold_power",
           "normalized_detuning", "normalized_dispersion",
           "normalized_frequency", "physical_frequency",
           "intracavity_photons"]


@dataclasses.dataclass(frozen=True)
class RingSpec:
    """Laboratory parameters of one Kerr microresonator.

    kappa_hz : loaded (total) linewidth, FWHM, in ordinary Hz
        (kappa = 2 pi kappa_hz rad/s).
    eta_esc : escape efficiency kappa_ext / kappa, in (0, 1].
    g0_hz : single-photon Kerr shift g0 / 2 pi in Hz (supply it from
        your platform's characterization or literature; this module
        ships no material numbers).
    lambda_pump_m : pump wavelength in metres.
    reference : where the numbers come from.  Required, on purpose.
    """

    kappa_hz: float
    eta_esc: float
    g0_hz: float
    lambda_pump_m: float
    reference: str

    def __post_init__(self):
        for name in ("kappa_hz", "g0_hz", "lambda_pump_m"):
            v = float(getattr(self, name))
            if not (np.isfinite(v) and v > 0.0):
                raise ValueError(f"{name} must be finite and positive")
        if not (0.0 < float(self.eta_esc) <= 1.0):
            raise ValueError("eta_esc must lie in (0, 1]")
        if not isinstance(self.reference, str) or not self.reference.strip():
            raise ValueError(
                "RingSpec requires a non-empty `reference` naming the "
                "source of these numbers (your calibration or a paper)")

    # -- internal angular quantities --------------------------------
    @property
    def kappa(self) -> float:
        return 2.0 * np.pi * float(self.kappa_hz)

    @property
    def g0(self) -> float:
        return 2.0 * np.pi * float(self.g0_hz)

    @property
    def omega_pump(self) -> float:
        return 2.0 * np.pi * _C_LIGHT / float(self.lambda_pump_m)


def normalized_pump(spec: RingSpec, power_w) -> float:
    """Normalized pump amplitude F from the launched power (W) reaching
    the coupler: F = sqrt(8 g0 kappa_ext P / (kappa^3 hbar omega_p))."""
    P = np.asarray(power_w, dtype=float)
    if np.any(P < 0.0) or not np.all(np.isfinite(P)):
        raise ValueError("power must be finite and non-negative")
    kap = spec.kappa
    F2 = 8.0 * spec.g0 * (spec.eta_esc * kap) * P / (
        kap ** 3 * _HBAR * spec.omega_pump)
    out = np.sqrt(F2)
    return float(out) if out.ndim == 0 else out


def threshold_power(spec: RingSpec, alpha: float = 1.0) -> float:
    """Pump power (W) at which the flat-state cubic
    rho [1 + (alpha - rho)^2] = F^2 has the root rho = 1 -- i.e.
    F_th^2 = 1 + (alpha - 1)^2, an identity of the cubic.  At
    alpha = 1 this is the minimum comb threshold F_th = 1."""
    F2 = 1.0 + (float(alpha) - 1.0) ** 2
    kap = spec.kappa
    return float(F2 * kap ** 3 * _HBAR * spec.omega_pump
                 / (8.0 * spec.g0 * (spec.eta_esc * kap)))


def normalized_detuning(spec: RingSpec, delta_f_hz) -> float:
    """alpha = 2 delta_omega / kappa with delta_f = f_cav - f_pump in
    Hz (positive: pump red of resonance)."""
    return float(2.0 * (2.0 * np.pi * float(delta_f_hz)) / spec.kappa)


def normalized_dispersion(spec: RingSpec, D2_hz) -> float:
    """d2 = -2 D2 / kappa with D2/2pi in Hz per mode^2 (anomalous
    D2 > 0 gives d2 < 0, the package's sign convention)."""
    return float(-2.0 * (2.0 * np.pi * float(D2_hz)) / spec.kappa)


def normalized_frequency(spec: RingSpec, f_hz):
    """Spectrum axis: omega_norm = omega_phys / (kappa/2) = 4 pi f
    / kappa -- the omega argument of the `sqzcomb.spectra` functions."""
    f = np.asarray(f_hz, dtype=float)
    out = 4.0 * np.pi * f / spec.kappa
    return float(out) if out.ndim == 0 else out


def physical_frequency(spec: RingSpec, omega_norm):
    """Inverse of `normalized_frequency` (returns ordinary Hz)."""
    w = np.asarray(omega_norm, dtype=float)
    out = w * spec.kappa / (4.0 * np.pi)
    return float(out) if out.ndim == 0 else out


def intracavity_photons(spec: RingSpec, rho) -> float:
    """Intracavity photon number N = rho kappa / (2 g0) for a
    normalized intensity rho = |psi|^2."""
    r = float(rho)
    if r < 0.0:
        raise ValueError("rho must be non-negative")
    return float(r * spec.kappa / (2.0 * spec.g0))
