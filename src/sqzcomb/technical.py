"""Technical noise of the resonator and the pump (new in 0.13).

The quantum-noise spectra elsewhere in the package assume a perfectly
quiet resonator and pump. Real devices add classical noise on top:

* the resonance frequency wanders -- thermorefractive noise (thermal
  fluctuations of the refractive index), fluctuations driven by Raman
  or Brillouin scattering, mechanical or thermal drift;
* the pump laser has intensity noise and phase (frequency) noise.

How large these are is material and laser physics with its own
modelling choices, so the package still ships no material numbers. What
it does now do, exactly, is carry any such noise -- given as a power
spectral density you measured or modelled, with its source -- through
the same linearized resonator and input-output relations as the
quantum noise, and add it to the detected spectrum. That is the part
that is the same for every platform.

How a classical noise enters. A classical fluctuation eps(t) adds a
term b * eps(t) to the linearized equations dz/dt = M z + (quantum
inputs) + b eps(t), with b a fixed vector in the doubled basis
z = (d a_1..d a_n, d a_1*..d a_n*). At the output port (escape fraction
eta) it adds sqrt(2 eta) G(omega) b eps(omega) with
G = (-i omega - M)^-1, and hence to the quadrature spectrum

    Delta S(omega) = | u^dag sqrt(2 eta) G(omega) b |^2  S_eps(omega),

u the same quadrature vector as `output_quadrature_variance` and
S_eps the two-sided power spectral density of eps in the package's
normalized time (units of 2/kappa). Independent sources add. The
vectors b of the common cases:

* resonance-frequency noise, eps = delta alpha (a shift of the
  normalized detuning, positive when the resonance moves up):
  b = (-i c, +i c*) with c the mean intracavity field of each mode in
  photon-amplitude units (|c|^2 photons) -- `resonance_noise_drive`;
  The same eps shifts every listed mode together (a common shift of
  the whole resonance comb, which is what a temperature change does to
  first order); give only one mode a non-zero c to model a shift of
  that mode alone;
* pump relative-amplitude noise, F -> F (1 + eps): b = F_ph on the
  pumped mode; pump phase noise, F -> F e^{i eps}: b = i F_ph --
  `pump_noise_drive`, with F_ph the pump term in photon units;
* noise of a parametric gain mu (the pump of `single_mode_parametric`
  and of `sqzcomb.kerrpo`), mu -> mu (1 + eps): b = (mu c*, mu* c);
  mu -> mu e^{i eps}: b = (i mu c*, -i mu* c) -- `gain_noise_drive`.
  Around a vacuum-centred state (c = 0) such noise does nothing to
  first order, as it should: it only matters above threshold.

Phase noise is always relative to the reference of the detection.
If the local oscillator is derived from the pump laser, the part of
the pump phase noise that the local oscillator shares cancels; give
the PSD of the phase difference that the detection actually sees.
A pump-laser frequency wobble is the same as a resonance wobble of
opposite sign, so it can also be entered with `resonance_noise_drive`
(a PSD does not care about the sign).

The field must be in photon units because a classical noise is
compared with vacuum noise, which is one quantum: the same fractional
wobble matters more the more photons it moves. `lle_mode_amplitudes`
converts a Lugiato-Lefever steady state (field normalized so the Kerr
term is i|psi|^2 psi) with psi = sqrt(chi) a, chi = 2 g0 / kappa
(`sqzcomb.physical`).

Laboratory spectra to normalized ones (`normalized_psd`): for a
one-sided laboratory PSD S1(f) (per Hz) of a dimensionless noise
(relative amplitude, phase in rad) the normalized two-sided PSD is
S(omega) = kappa S1(f) / 4 at omega = 4 pi f / kappa; for resonance-
frequency noise given in Hz^2/Hz it is S(omega) = 4 pi^2 S1(f) / kappa
(kappa the loaded linewidth in rad/s). Both follow from rescaling time
by kappa/2; the tests check them with Parseval's theorem (the variance
is the same in both units).

What the tests hold this module to: a closed form (a passive mode with
a bright mean field, whose phase quadrature picks up exactly
4 eta |c|^2 S_eps / (1 + omega^2)); a time-domain Monte-Carlo
simulation of the same linear equations with coloured classical noise;
zero added noise when S_eps = 0; and the Parseval unit checks.
"""
from __future__ import annotations

import numpy as np

from .spectra import _check_spectra_stability

__all__ = ["classical_noise_variance", "classical_noise_variance_ports",
           "resonance_noise_drive", "pump_noise_drive", "gain_noise_drive",
           "lle_mode_amplitudes", "normalized_psd"]


def _quadrature_vector(m2, mode_index, phi, mode_index_b=None):
    n = m2 // 2
    u = np.zeros(m2, dtype=complex)
    idxs = [(mode_index, 1.0 / np.sqrt(2.0))] if mode_index_b is None \
        else [(mode_index, 0.5), (mode_index_b, 0.5)]
    for idx, w in idxs:
        u[idx] += np.exp(1j * phi) * w
        u[n + idx] += np.exp(-1j * phi) * w
    return u


def _check_sources(m2, drives, psds):
    drives = [np.asarray(b, dtype=complex).ravel() for b in drives]
    psds = [float(s) for s in np.atleast_1d(psds)]
    if len(drives) != len(psds):
        raise ValueError("need one PSD value per drive vector")
    for b in drives:
        if b.shape != (m2,):
            raise ValueError("each drive vector must have the doubled "
                             "dimension of the drift matrix")
    for s in psds:
        if not (np.isfinite(s) and s >= 0.0):
            raise ValueError("PSD values must be finite and >= 0")
    return drives, psds


def classical_noise_variance(M, eta, omega, drives, psds, mode_index,
                             phi=0.0, mode_index_b=None,
                             allow_marginal=False):
    """Added quadrature variance at the extraction port (the geometry of
    `output_quadrature_variance`: every mode decays at rate 1 with the
    fraction eta into the port) from independent classical noises.

    drives : list of doubled-basis vectors b (see module docstring).
    psds : list of the two-sided normalized PSDs S_eps(omega) of each
        source at this analysis frequency.
    Returns sum_j |u^dag sqrt(2 eta) G b_j|^2 S_j, in the package's
    variance units (vacuum 1/2): add it to the quantum spectrum.
    """
    M = np.asarray(M, dtype=complex)
    m2 = M.shape[0]
    eta = float(eta)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    _check_spectra_stability(M, allow_marginal)
    drives, psds = _check_sources(m2, drives, psds)
    G = np.linalg.inv(-1j * float(omega) * np.eye(m2) - M)
    u = _quadrature_vector(m2, mode_index, float(phi), mode_index_b)
    total = 0.0
    for b, s in zip(drives, psds):
        amp = u.conj() @ (np.sqrt(2.0 * eta) * (G @ b))
        total += float(abs(amp) ** 2) * s
    return total


def classical_noise_variance_ports(M, gammas, eta, port_mode, omega,
                                   drives, psds, phi=0.0,
                                   mode_index=None, allow_marginal=False):
    """Same as `classical_noise_variance` for the per-mode-decay port
    geometry of `output_variance_ports` (photonic molecules): the port
    extracts the fraction eta of each listed mode's own decay
    gammas[j]."""
    M = np.asarray(M, dtype=complex)
    gammas = np.asarray(gammas, dtype=float)
    n = gammas.size
    m2 = 2 * n
    if M.shape != (m2, m2):
        raise ValueError("M and gammas disagree on the number of modes")
    if not (0.0 <= float(eta) <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    _check_spectra_stability(M, allow_marginal)
    drives, psds = _check_sources(m2, drives, psds)
    ports = np.atleast_1d(np.asarray(port_mode, dtype=int))
    reads = ports if mode_index is None else \
        np.atleast_1d(np.asarray(mode_index, dtype=int))
    if not np.all(np.isin(reads, ports)):
        raise ValueError("mode_index must be one of the monitored port "
                         "modes")
    amp_port = np.zeros(n)
    amp_port[ports] = np.sqrt(2.0 * float(eta) * gammas[ports])
    C = np.diag(np.concatenate([amp_port, amp_port]).astype(complex))
    G = np.linalg.inv(-1j * float(omega) * np.eye(m2) - M)
    u = np.zeros(m2, dtype=complex)
    w = 1.0 / np.sqrt(2.0 * reads.size)
    for idx in reads:
        u[idx] += np.exp(1j * phi) * w
        u[n + idx] += np.exp(-1j * phi) * w
    total = 0.0
    for b, s in zip(drives, psds):
        total += float(abs(u.conj() @ (C @ (G @ b))) ** 2) * s
    return total


def resonance_noise_drive(mean_field):
    """Drive vector of resonance-frequency (detuning) noise.

    mean_field : complex mean intracavity amplitude of each retained
        mode, in photon-amplitude units (|c|^2 = photon number), in the
        same mode order as the drift matrix (e.g. from
        `lle_mode_amplitudes`, or the `alpha` of a Kerr parametric
        state). Returns b = (-i c, +i c*).
    """
    c = np.atleast_1d(np.asarray(mean_field, dtype=complex))
    return np.concatenate([-1j * c, 1j * np.conj(c)])


def pump_noise_drive(pump_photon, n_modes, pumped_index=0,
                     kind="amplitude"):
    """Drive vector of pump noise.

    pump_photon : the pump term in photon units -- a number for a pump
        on one mode (`pumped_index`; for the LLE with a steady pump,
        F / sqrt(chi) on mode 0), or an array with one value per
        retained mode, in drift-matrix order, for a pump that feeds
        several lines (a pulse train: the Fourier components of the
        pump profile, `sqzcomb.lle.pump_spectrum(F, n)[k] / n /
        sqrt(chi)` for each retained mode number k). The whole pump is
        assumed to fluctuate together.
    kind : "amplitude" (relative amplitude noise, F -> F(1 + eps)) or
        "phase" (F -> F e^{i eps}).
    """
    n = int(n_modes)
    if np.ndim(pump_photon) == 0:
        Fv = np.zeros(n, dtype=complex)
        Fv[int(pumped_index)] = complex(pump_photon)
    else:
        Fv = np.asarray(pump_photon, dtype=complex).ravel()
        if Fv.shape != (n,):
            raise ValueError("an array pump needs one value per mode")
    if kind == "amplitude":
        f = Fv
    elif kind == "phase":
        f = 1j * Fv
    else:
        raise ValueError("kind must be 'amplitude' or 'phase'")
    return np.concatenate([f, np.conj(f)])


def gain_noise_drive(mean_field, mu, kind="amplitude"):
    """Drive vector of noise on a parametric gain mu (single mode).

    mean_field : the mean intracavity amplitude c in photon units (for
        `sqzcomb.kerrpo`, the `alpha` of a bright state).
    mu : the parametric gain of the model.
    kind : "amplitude" (mu -> mu (1 + eps)) or "phase"
        (mu -> mu e^{i eps}).
    Returns b = (f, conj(f)) with f = mu c* (amplitude) or i mu c*
    (phase). It is zero for c = 0.
    """
    c = complex(mean_field)
    mu = complex(mu)
    if kind == "amplitude":
        f = mu * np.conj(c)
    elif kind == "phase":
        f = 1j * mu * np.conj(c)
    else:
        raise ValueError("kind must be 'amplitude' or 'phase'")
    return np.array([f, np.conj(f)], dtype=complex)


def lle_mode_amplitudes(psi_s, modes, chi):
    """Mean field of the retained comb lines in photon-amplitude units.

    psi_s : LLE steady state on the theta grid (Kerr term i|psi|^2 psi).
    modes : the retained mode numbers, in drift-matrix order (the
        `modes` array returned by `fluctuation_matrix`).
    chi : Kerr shift per photon in units of kappa/2, chi = 2 g0 / kappa
        (from a `RingSpec`: 2 * spec.g0 / spec.kappa).
    Returns c_k = hat(psi)_k / sqrt(chi), with hat(psi)_k the Fourier
    component convention of `fluctuation_matrix`.
    """
    chi = float(chi)
    if not (np.isfinite(chi) and chi > 0.0):
        raise ValueError("chi must be finite and positive")
    psi_s = np.asarray(psi_s, dtype=complex)
    n = psi_s.size
    kgrid = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    comp = dict(zip(kgrid, np.fft.fft(psi_s) / n))
    return np.array([comp.get(int(k), 0.0) for k in np.asarray(modes)],
                    dtype=complex) / np.sqrt(chi)


def normalized_psd(spec, f_hz, psd_one_sided, kind="dimensionless"):
    """Convert a laboratory one-sided PSD to the normalized two-sided
    PSD S_eps(omega) the functions above take.

    spec : a `RingSpec` (only its linewidth is used).
    f_hz : analysis frequencies (ordinary Hz, > 0).
    psd_one_sided : the measured one-sided PSD at f_hz: per Hz for
        kind="dimensionless" (relative amplitude, or phase in rad^2/Hz),
        in Hz^2/Hz for kind="frequency" (resonance-frequency noise).
    Returns (omega, S) with omega = 4 pi f / kappa the normalized
    analysis frequency and S the normalized two-sided PSD (of eps, or of
    the normalized detuning shift for kind="frequency").
    """
    f = np.asarray(f_hz, dtype=float)
    S1 = np.asarray(psd_one_sided, dtype=float)
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("f_hz must be finite and positive")
    if np.any(S1 < 0.0) or not np.all(np.isfinite(S1)):
        raise ValueError("the PSD must be finite and >= 0")
    kappa = spec.kappa
    omega = 4.0 * np.pi * f / kappa
    if kind == "dimensionless":
        S = kappa * S1 / 4.0
    elif kind == "frequency":
        S = 4.0 * np.pi ** 2 * S1 / kappa
    else:
        raise ValueError("kind must be 'dimensionless' or 'frequency'")
    return omega, S
