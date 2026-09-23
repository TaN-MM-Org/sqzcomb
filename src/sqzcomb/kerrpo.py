"""Above threshold: the Kerr-saturated parametric oscillator (new in 0.13).

`single_mode_parametric` is the textbook below-threshold model: a
parametric gain mu against a loss of 1, with nothing to stop the growth
once |mu| > 1, which is why the package refuses its drift matrix there.
In a Kerr resonator the growth *is* stopped: the intracavity field shifts
its own resonance (self-phase modulation) until the gain no longer
wins. Adding that Kerr term gives a single-mode model with genuine
bright steady states above threshold, around which the same
linearization and input-output machinery apply.

Model, in the package's normalized units (time in units of 2/kappa,
amplitude decay 1, photon-number amplitude alpha = <a>):

    d alpha/dt = (-1 + i delta) alpha + mu alpha* + i chi |alpha|^2 alpha

delta and mu follow `single_mode_parametric` exactly (same signs, same
threshold |mu| = 1 at delta = 0); chi is the Kerr shift per photon in
units of kappa/2 (chi = 2 g0 / kappa for a ring with single-photon Kerr
shift g0), with the same sign as the Lugiato-Lefever term
+i |psi|^2 psi of `sqzcomb.lle`. The quantum model behind it is the
Hamiltonian (hbar = 1, normalized units)

    H = -delta a^dag a + (i/2)(mu a^dag^2 - mu* a^2) - (chi/2) a^dag^2 a^2

with damping 2 D[a], built exactly in `sqzcomb.master`.

Steady states (algebra, not iteration). Besides alpha = 0, a bright
state needs (-1 + i Delta) alpha = -mu alpha* with the Kerr-shifted
detuning Delta = delta + chi n, n = |alpha|^2. Taking the modulus,
1 + Delta^2 = |mu|^2, so

    Delta = +/- sqrt(|mu|^2 - 1),   n = (Delta - delta) / chi  (> 0),
    alpha^2 = mu n / (1 - i Delta),

each giving the pair +/- alpha (the two phase states of a degenerate
parametric oscillator). Linearizing alpha -> alpha + d alpha,

    d(d alpha)/dt = (-1 + i (delta + 2 chi n)) d alpha
                    + (mu + i chi alpha^2) d alpha*,

a 2x2 doubled drift matrix in exactly the form `single_mode_parametric`
returns, so every spectra, covariance and detection routine of the
package applies unchanged. States whose drift has an eigenvalue with a
non-negative real part are reported as unstable, not used.

Validity, stated plainly: linearization around a bright state assumes
many photons (chi n >> chi, i.e. n >> 1) and ignores the slow quantum
switching between the two phase states +/- alpha, which the exact
Fock-space model in `sqzcomb.master` contains (and which shows up there
as extra low-frequency noise). The tests compare the two.
"""
from __future__ import annotations

import numpy as np

__all__ = ["kerr_parametric_states", "kerr_parametric_drift"]


def _check(mu, delta, chi, allow_zero=False):
    mu = complex(mu)
    delta = float(delta)
    chi = float(chi)
    if not (np.isfinite(mu) and np.isfinite(delta) and np.isfinite(chi)):
        raise ValueError("mu, delta and chi must be finite")
    if chi == 0.0 and not allow_zero:
        raise ValueError(
            "chi = 0 has no bright steady state (nothing stops the "
            "growth above threshold); use single_mode_parametric for "
            "the below-threshold linear model")
    return mu, delta, chi


def kerr_parametric_drift(alpha, mu, delta, chi):
    """Doubled 2x2 drift matrix of the fluctuations around the classical
    state `alpha` (use alpha = 0 for the vacuum-centred state):

        [[-1 + i (delta + 2 chi |alpha|^2),  mu + i chi alpha^2],
         [conj(...),                         conj(...)]]

    At alpha = 0 this is exactly `single_mode_parametric(mu, delta)`
    (for any chi, including 0).
    """
    mu, delta, chi = _check(mu, delta, chi, allow_zero=True)
    a = complex(alpha)
    n = abs(a) ** 2
    d = -1.0 + 1j * (delta + 2.0 * chi * n)
    o = mu + 1j * chi * a * a
    return np.array([[d, o], [np.conj(o), np.conj(d)]], dtype=complex)


def kerr_parametric_states(mu, delta, chi):
    """All classical steady states of the Kerr parametric oscillator.

    Returns a list of dicts, one per fixed point, with keys
    alpha (complex amplitude, photon-number units), n (= |alpha|^2),
    stable (bool: every drift eigenvalue has negative real part),
    growth (largest real part of the drift eigenvalues) and drift (the
    2x2 matrix of `kerr_parametric_drift`). The vacuum-centred state
    alpha = 0 is always listed first; bright states come in +/- pairs.
    The residual of the classical equation at each returned state is
    checked to 1e-9 relative, or the function raises.
    """
    mu, delta, chi = _check(mu, delta, chi)
    states = []

    def add(alpha):
        M = kerr_parametric_drift(alpha, mu, delta, chi)
        growth = float(np.max(np.linalg.eigvals(M).real))
        states.append({"alpha": complex(alpha), "n": abs(alpha) ** 2,
                       "stable": growth < 0.0, "growth": growth,
                       "drift": M})

    add(0.0)
    m2 = abs(mu) ** 2
    if m2 > 1.0:
        s = np.sqrt(m2 - 1.0)
        for Delta in (s, -s):
            n = (Delta - delta) / chi
            if n <= 0.0:
                continue
            a2 = mu * n / (1.0 - 1j * Delta)
            a = np.sqrt(a2)
            for alpha in (a, -a):
                res = ((-1.0 + 1j * delta) * alpha + mu * np.conj(alpha)
                       + 1j * chi * abs(alpha) ** 2 * alpha)
                scale = max(1.0, abs(mu) * abs(alpha))
                if abs(res) > 1e-9 * scale:
                    raise RuntimeError(
                        "steady-state residual check failed "
                        f"({abs(res):.3e}); please report this")
                add(alpha)
    return states
