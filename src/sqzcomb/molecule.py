"""Photonic molecule: two coupled rings, and multi-port output spectra.

Model (normalized units of the main ring's amplitude decay rate, i.e. the
main ring's half-linewidth is 1, time in units of its photon lifetime / 2):

    da/dt = (-1     + i delta_a) a - i J b + mu a*
    db/dt = (-gamma + i delta_b) b - i J a

Mode a is the Kerr ring carrying a below-threshold degenerate parametric
process of strength mu (real mu > 0 squeezes the phi = pi/2 quadrature, as
in `single_mode_parametric`). Mode b is a passive auxiliary ring with
amplitude decay gamma (its half-linewidth over the main ring's) and
coherent coupling J from the standard hopping Hamiltonian J (a^dag b +
b^dag a).

Exact facts this module's test suite asserts, rather than states:

* J = 0 reduces exactly to the single-mode machinery.
* A passive molecule (mu = 0) returns exact vacuum at every port,
  frequency, quadrature and coupling: the input-output bookkeeping
  conserves vacuum.
* At delta_a = delta_b = 0 and equal decay, the passive supermodes split
  by exactly 2 J.
* At delta_a = delta_b = 0 and J < gamma the instability threshold is
  mu = 1 + J^2 / gamma: the auxiliary ring adds the adiabatic loss
  J^2 / gamma to the parametric mode.
* The -i J hop rotates the leaked field by a quarter turn, so the
  quadrature squeezed at phi = pi/2 in the Kerr ring appears at the
  auxiliary port at phi = 0.
* At Omega = 0 the molecule detected through the auxiliary ring equals,
  exactly, a single parametric mode with total decay 1 + J^2 / gamma and
  escape efficiency eta_b (J^2/gamma) / (1 + J^2/gamma): the auxiliary
  ring acts as an extraction channel. With J^2/gamma > 1 that effective
  efficiency exceeds the critical-coupling value 1/2 although the Kerr
  ring itself has no extraction port at all, which is the mechanism by
  which a photonic molecule can pass the single-ring 3 dB detected
  squeezing associated with critical coupling (the test suite drives a
  J^2/gamma = 3 molecule to 6 dB detected).

What this module does not do: it is a two-mode model, not the full
multimode comb molecule of the accompanying manuscript; the paper
repository reproduces that study.

Rings with different line spacings (new in 0.13,
`vernier_molecule_fluctuation_matrix`). When the auxiliary ring's free
spectral range differs from the main ring's, its lines slide past the
main ring's comb (a Vernier pattern): a main line may have no auxiliary
line near it, one, or several, and an auxiliary line is near at most
one main line. Written in the frame where main line k sits at
frequency k * fsr (fsr = the main ring's free spectral range in units
of kappa/2), an auxiliary line at frequency nu_b is stationary only
when referred to the main line it is paired with; its detuning is
then aux_delta = k * fsr - nu_b (the same sign as `aux_delta` and
`delta_b` above: positive when the auxiliary line sits below the
frame). Its coupling to every other main line oscillates at least
(fsr - |aux_delta|) faster than the coupling J and is dropped (the
rotating-wave approximation); the builder refuses when that margin is
not large (see `rwa_ratio`), rather than silently keeping a wrong
model. The tests hold the construction to the exact (all-pairs,
no-approximation) model of a passive pair of rings, which is
time-independent in the laboratory frame.
"""
from __future__ import annotations

import math

import numpy as np

from .linearize import fluctuation_matrix, is_stable


def photonic_molecule(mu, J, delta_a=0.0, delta_b=0.0, gamma=1.0):
    """Doubled drift matrix of the two-ring molecule, basis (a, b, a*, b*).

    Returns (M, gammas) with gammas = (1.0, gamma), the per-mode amplitude
    decay rates that `output_variance_ports` needs to normalize the input
    couplings.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    A = np.array([[-1.0 + 1j * delta_a, -1j * J],
                  [-1j * J, -float(gamma) + 1j * delta_b]], dtype=complex)
    B = np.array([[mu, 0.0], [0.0, 0.0]], dtype=complex)
    M = np.block([[A, B], [np.conj(B), np.conj(A)]])
    return M, np.array([1.0, float(gamma)])


def output_variance_ports(M, gammas, eta, port_mode, omega, phi=0.0,
                          mode_index=None):
    """Detected quadrature variance with per-mode decay rates.

    The monitored bus couples to `port_mode` (an int, or a sequence of
    ints for a bus that extracts several spectrally resolved lines, as
    in twin-beam detection); each listed mode contributes the fraction
    `eta` of its own decay `gammas[j]` to the bus. Every other decay
    channel (the remaining 1 - eta of each monitored mode, and the full
    decay of every other mode) is an independent vacuum bath. Vacuum
    level is exactly 0.5; the passive case is asserted in the test
    suite, not assumed.

    mode_index selects whose output quadrature is read (default: the
    monitored modes). An int reads that mode's quadrature; a sequence
    reads the joint equal-weight quadrature of those modes (for two
    modes, the twin-beam variable (a + b)/sqrt(2) rotated by phi).
    Detection must be on monitored modes: a detector on the bus cannot
    see light that never enters the bus.
    """
    gammas = np.asarray(gammas, dtype=float)
    n = gammas.size
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    if not is_stable(M):
        raise ValueError("drift matrix is unstable (above threshold); "
                         "linearized spectra are meaningless there")
    ports = np.atleast_1d(np.asarray(port_mode, dtype=int))
    if mode_index is None:
        mode_index = port_mode
    reads = np.atleast_1d(np.asarray(mode_index, dtype=int))
    if not np.all(np.isin(reads, ports)):
        raise ValueError("mode_index must be one of the monitored port "
                         "modes; a detector on the bus cannot see light "
                         "that never enters the bus")

    m2 = 2 * n
    ident = np.eye(m2, dtype=complex)
    G = np.linalg.inv(-1j * omega * ident - M)

    def doubled_diag(amps):
        return np.diag(np.concatenate([amps, amps]).astype(complex))

    # monitored bus: amplitude sqrt(2 eta gamma_j) on each port mode
    amp_port = np.zeros(n)
    amp_port[ports] = np.sqrt(2.0 * eta * gammas[ports])
    C_port = doubled_diag(amp_port)

    # loss baths: remainder of the monitored modes, full decay of the rest
    amp_loss = np.sqrt(2.0 * gammas)
    amp_loss[ports] = np.sqrt(2.0 * (1.0 - eta) * gammas[ports])
    C_loss = doubled_diag(amp_loss)

    T_port = C_port @ G @ C_port - ident
    T_loss = C_port @ G @ C_loss

    N = np.zeros((m2, m2), dtype=complex)   # vacuum <z z^dagger>
    N[:n, :n] = np.eye(n)
    S = T_port @ N @ T_port.conj().T + T_loss @ N @ T_loss.conj().T

    u = np.zeros(m2, dtype=complex)
    w = 1.0 / np.sqrt(2.0 * reads.size)
    for idx in reads:
        # u^dag z = X_phi = cos(phi) x + sin(phi) p (see spectra.py)
        u[idx] += np.exp(1j * phi) * w
        u[n + idx] += np.exp(-1j * phi) * w
    return float(np.real(u.conj() @ S @ u))


def molecule_fluctuation_matrix(psi_s, alpha, J, gamma_b, dispersion=(0.0,),
                                aux_delta=0.0, modes=None, d1=0.0):
    """Multimode comb molecule: LLE fluctuations + per-line auxiliary modes.

    Every retained comb line k of the main ring (linearized around the
    steady state psi_s exactly as in `fluctuation_matrix`) is coupled
    with rate J to a matching passive mode of an auxiliary ring with
    amplitude decay gamma_b and detuning aux_delta (scalar, or an array
    over the retained modes to encode the auxiliary ring's own detuning
    and dispersion). Valid when the auxiliary ring's free spectral range
    matches the main ring's, so line k couples only to auxiliary mode k.

    Basis ordering: (a_k1..a_kM, b_k1..b_kM, a*_k1.., b*_k1..), i.e. the
    joint mode list is the main lines followed by the auxiliary lines,
    then the conjugates, which is the ordering `output_variance_ports`
    expects. Returns (M, gammas, modes) with gammas = (1, ..., 1,
    gamma_b, ..., gamma_b) and modes the retained mode numbers; the
    auxiliary line for main index i sits at joint index M + i.

    J may be a scalar or an array over the retained modes (per-line
    coupling). At a single retained line this construction reduces
    exactly to `photonic_molecule` with mu = i psi_s^2, which the test
    suite asserts against the released two-ring code. For rings whose
    line spacings differ, use `vernier_molecule_fluctuation_matrix`.
    """
    if gamma_b <= 0.0:
        raise ValueError("gamma_b must be positive")
    M_main, modes = fluctuation_matrix(psi_s, alpha, dispersion, modes,
                                       d1=d1)
    m = modes.size
    A_main = M_main[:m, :m]
    B_main = M_main[:m, m:]

    Jk = np.broadcast_to(np.asarray(J, dtype=float), (m,))
    dk = np.broadcast_to(np.asarray(aux_delta, dtype=float), (m,))
    A_aux = np.diag(-float(gamma_b) + 1j * dk)
    C = np.diag(-1j * Jk).astype(complex)

    zero = np.zeros((m, m), dtype=complex)
    A_joint = np.block([[A_main, C], [C, A_aux]])
    B_joint = np.block([[B_main, zero], [zero, zero]])
    M = np.block([[A_joint, B_joint],
                  [np.conj(B_joint), np.conj(A_joint)]])
    gammas = np.concatenate([np.ones(m), np.full(m, float(gamma_b))])
    return M, gammas, modes


def ring_line_frequencies(mode_numbers, offset, fsr, dispersion=()):
    """Line frequencies of a ring, in units of the main ring's kappa/2,
    measured from the pump:

        nu(l) = offset + fsr * l + sum_m (D_m / m!) l^m   (m = 2, 3, ...)

    offset : frequency of line l = 0; fsr : line spacing; dispersion :
    (D_2, D_3, ...) of this ring in the same units, with the physical
    sign (D_2 > 0 for anomalous dispersion: the lines spread apart).
    Convert laboratory numbers with `sqzcomb.physical.normalized_frequency`
    (a frequency in Hz divided by kappa/2 of the main ring).
    """
    ell = np.asarray(mode_numbers, dtype=float)
    nu = float(offset) + float(fsr) * ell
    for m, Dm in enumerate(dispersion, start=2):
        nu = nu + float(Dm) / math.factorial(m) * ell ** m
    return nu


def vernier_molecule_fluctuation_matrix(psi_s, alpha, J, gamma_b, fsr,
                                        aux_frequencies, dispersion=(0.0,),
                                        modes=None, d1=0.0, window=None,
                                        rwa_ratio=20.0):
    """Comb molecule whose auxiliary ring has its own line spacing.

    psi_s, alpha, dispersion, modes, d1 : the main ring, exactly as in
        `fluctuation_matrix`.
    fsr : the main ring's free spectral range in units of kappa/2
        (FSR / (kappa / 2), with both in the same units; a large number
        for a real ring).
    aux_frequencies : frequencies nu_b of the auxiliary ring's lines in
        the same units, measured from the pump (e.g. from
        `ring_line_frequencies`).
    J : coupling rate, a scalar or one value per auxiliary line;
    gamma_b : amplitude decay of the auxiliary lines (units of the main
        ring's kappa/2).
    window : keep only auxiliary lines with |aux_delta| <= window
        (default: all whose partner is retained). Lines far from any main
        line matter only through shifts of order J^2 / |aux_delta|.
    rwa_ratio : refuse when an auxiliary line's detuning from the
        second-nearest main line, fsr - |aux_delta|, is below
        rwa_ratio * max(J, gamma_b, 1).

    Each auxiliary line is paired with its nearest main line k (rounding
    nu_b / fsr) and enters with aux_delta = k * fsr - nu_b. Returns
    (M, gammas, modes, aux) with the joint ordering (main lines, then
    the kept auxiliary lines, then conjugates) that
    `output_variance_ports` expects; aux is a dict of arrays: index
    (position in aux_frequencies), partner (main mode number),
    aux_delta and J of every kept auxiliary line, in basis order.
    """
    if gamma_b <= 0.0:
        raise ValueError("gamma_b must be positive")
    fsr = float(fsr)
    if not (np.isfinite(fsr) and fsr > 0.0):
        raise ValueError("fsr must be finite and positive")
    nu_b = np.atleast_1d(np.asarray(aux_frequencies, dtype=float))
    if not np.all(np.isfinite(nu_b)):
        raise ValueError("aux_frequencies must be finite")
    Jall = np.broadcast_to(np.asarray(J, dtype=float), nu_b.shape)
    M_main, modes = fluctuation_matrix(psi_s, alpha, dispersion, modes,
                                       d1=d1)
    m = modes.size
    A_main = M_main[:m, :m]
    B_main = M_main[:m, m:]

    partner = np.rint(nu_b / fsr).astype(int)
    delta = partner * fsr - nu_b
    keep = np.isin(partner, modes)
    if window is not None:
        keep &= np.abs(delta) <= float(window)
    idx = np.nonzero(keep)[0]
    margin = fsr - np.abs(delta[idx])
    scale = np.maximum(np.maximum(np.abs(Jall[idx]), float(gamma_b)), 1.0)
    bad = margin < float(rwa_ratio) * scale
    if np.any(bad):
        j = int(idx[np.argmax(bad)])
        raise ValueError(
            f"auxiliary line {j} (nu_b = {nu_b[j]:g}) is only "
            f"{fsr - abs(delta[j]):g} from its second-nearest main line, "
            "not far enough for the rotating-wave pairing; the lines are "
            "too close to midway between two main lines (or the rings "
            "too strongly coupled) for this model")
    p = idx.size
    pos = {int(k): i for i, k in enumerate(modes)}
    C_ab = np.zeros((m, p), dtype=complex)
    for col, j in enumerate(idx):
        C_ab[pos[int(partner[j])], col] = -1j * Jall[j]
    A_aux = np.diag(-float(gamma_b) + 1j * delta[idx])
    A_joint = np.block([[A_main, C_ab], [C_ab.T, A_aux]])
    B_joint = np.zeros((m + p, m + p), dtype=complex)
    B_joint[:m, :m] = B_main
    M = np.block([[A_joint, B_joint],
                  [np.conj(B_joint), np.conj(A_joint)]])
    gammas = np.concatenate([np.ones(m), np.full(p, float(gamma_b))])
    aux = {"index": idx, "partner": partner[idx], "aux_delta": delta[idx],
           "J": np.array(Jall[idx])}
    return M, gammas, modes, aux


def molecule_threshold(J, gamma):
    """Exact instability threshold mu_th of the resonant molecule.

    For delta_a = delta_b = 0 the quadrature sectors decouple and the
    squeezed sector's zero-eigenvalue condition gives
    mu_th = 1 + J^2 / gamma, valid while that (static) instability
    precedes the oscillatory one, i.e. for J <= gamma; beyond that the
    Hopf boundary mu_th = 1 + gamma takes over. Both branches follow
    from the 2x2 sector matrix [[-1 + mu, J], [-J, -gamma]] and are
    asserted against `is_stable` in the test suite.
    """
    if gamma <= 0.0:
        raise ValueError("gamma must be positive")
    static = 1.0 + J * J / gamma
    hopf = 1.0 + gamma
    return min(static, hopf)
