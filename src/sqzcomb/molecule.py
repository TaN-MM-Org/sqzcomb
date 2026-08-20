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
"""
from __future__ import annotations

import numpy as np

from .linearize import is_stable


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

    One physical port is monitored: it couples to mode `port_mode` and
    carries the fraction `eta` of that mode's decay `gammas[port_mode]`.
    Every other decay channel (the remaining 1 - eta of the monitored
    mode, and the full decay of every other mode) is an independent
    vacuum bath. Vacuum level is exactly 0.5; the passive case is
    asserted in the test suite, not assumed.

    mode_index selects which mode's output quadrature is read (default:
    the monitored port's mode, the only one a detector on that port can
    see; the parameter exists for constructing joint quadratures in
    future multimode work).
    """
    gammas = np.asarray(gammas, dtype=float)
    n = gammas.size
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    if not is_stable(M):
        raise ValueError("drift matrix is unstable (above threshold); "
                         "linearized spectra are meaningless there")
    if mode_index is None:
        mode_index = port_mode

    m2 = 2 * n
    ident = np.eye(m2, dtype=complex)
    G = np.linalg.inv(-1j * omega * ident - M)

    def doubled_diag(amps):
        return np.diag(np.concatenate([amps, amps]).astype(complex))

    # monitored port: amplitude sqrt(2 eta gamma_j) on port_mode only
    amp_port = np.zeros(n)
    amp_port[port_mode] = np.sqrt(2.0 * eta * gammas[port_mode])
    C_port = doubled_diag(amp_port)

    # loss baths: remainder of the monitored mode, full decay of the rest
    amp_loss = np.sqrt(2.0 * gammas)
    amp_loss[port_mode] = np.sqrt(2.0 * (1.0 - eta) * gammas[port_mode])
    C_loss = doubled_diag(amp_loss)

    T_port = C_port @ G @ C_port - ident
    T_loss = C_port @ G @ C_loss

    N = np.zeros((m2, m2), dtype=complex)   # vacuum <z z^dagger>
    N[:n, :n] = np.eye(n)
    S = T_port @ N @ T_port.conj().T + T_loss @ N @ T_loss.conj().T

    u = np.zeros(m2, dtype=complex)
    u[mode_index] = np.exp(-1j * phi) / np.sqrt(2.0)
    u[n + mode_index] = np.exp(1j * phi) / np.sqrt(2.0)
    return float(np.real(u.conj() @ S @ u))


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
