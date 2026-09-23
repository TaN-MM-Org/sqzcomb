"""Pumps that change in time: the pulse-driven LLE (pump profile F(theta),
pulse drift d1, time-dependent F(t)) and the time-domain Gaussian
covariance with temporal-mode output squeezing."""
import numpy as np
import pytest
from scipy.integrate import quad

from sqzcomb import (fluctuation_matrix, intracavity_covariance,
                     lle_evolve, newton_state, output_quadrature_variance,
                     single_mode_parametric, soliton_seed)
from sqzcomb.pulsed import (covariance_evolution, parametric_pulse_drift,
                            temporal_mode_variance)
from sqzcomb.soliton import _residual_k


def _shift(psi, s):
    """psi(theta + s) by an exact Fourier shift."""
    k = np.fft.fftfreq(psi.size, d=1.0 / psi.size)
    return np.fft.ifft(np.fft.fft(psi) * np.exp(1j * k * s))


def _psi0(n):
    th = 2 * np.pi * np.arange(n) / n
    return th, 0.3 + 0.2 * np.exp(1j * th) + 0.1j * np.exp(-3j * th)


# --- pulse-driven Lugiato-Lefever equation -------------------------------

def test_uniform_profile_equals_scalar_pump():
    th, psi0 = _psi0(64)
    a = lle_evolve(psi0, 1.3, 1.0, (-0.02,), t_end=2.0, dt=0.005)
    b = lle_evolve(psi0, np.full(64, 1.3 + 0j), 1.0, (-0.02,), t_end=2.0,
                   dt=0.005)
    c = lle_evolve(psi0, lambda t: 1.3, 1.0, (-0.02,), t_end=2.0, dt=0.005)
    assert np.abs(a - b).max() < 1e-13
    assert np.abs(a - c).max() < 1e-13


def test_d1_with_uniform_pump_is_a_pure_translation():
    th, psi0 = _psi0(128)
    T, d1 = 5.0, 0.37
    a = lle_evolve(psi0, 1.3, 1.0, (-0.02,), t_end=T, dt=0.005)
    c = lle_evolve(psi0, 1.3, 1.0, (-0.02,), t_end=T, dt=0.005, d1=d1)
    assert np.abs(c - _shift(a, d1 * T)).max() < 1e-12


def test_d1_is_the_frame_of_a_moving_pump():
    # The derivation of d1: a pump profile that moves around the ring at
    # angular speed Om (callable F(t)), seen from the frame moving with
    # it, is a fixed profile plus the term d1 d/dtheta with d1 = Om.
    th, psi0 = _psi0(256)
    P = lambda x: 0.5 + 2.0 * np.exp(-(np.angle(np.exp(1j * x)) / 0.4) ** 2)
    Om, T = 0.3, 5.0
    ring = lle_evolve(psi0, lambda t: P(th - Om * t), 1.0, (-0.02,),
                      t_end=T, dt=0.002)
    pump = lle_evolve(psi0, P(th), 1.0, (-0.02,), t_end=T, dt=0.002,
                      d1=Om)
    assert np.abs(_shift(ring, Om * T) - pump).max() < 2e-5


N, D2, ALPHA, F0 = 256, -0.25, 3.0, 1.9


@pytest.fixture(scope="module")
def pinned():
    th = 2 * np.pi * np.arange(N) / N
    prof = 1.6 + 0.5 * np.exp(-(np.angle(np.exp(1j * (th - np.pi)))
                                / 0.8) ** 2)
    psi, _ = newton_state(soliton_seed(N, F0, ALPHA, (D2,)), F0, ALPHA,
                          (D2,))
    for s in np.linspace(0.1, 1.0, 10):      # morph the pump to the pulse
        psi, _ = newton_state(psi, F0 + s * (prof - F0), ALPHA, (D2,))
    return th, prof, psi


def test_pulse_driven_soliton_on_the_peak_is_unstable(pinned):
    # On the pump peak the soliton is a steady state but not a stable
    # one: the translation mode has a positive growth rate, and the
    # split-step evolver carries a nudged soliton off the peak. (A
    # soliton sitting beside, not on, the peak of a pulsed drive is the
    # spontaneous symmetry breaking reported by Hendry et al., Phys.
    # Rev. A 97, 053834 (2018).)
    th, prof, psi = pinned
    assert th[np.argmax(np.abs(psi))] == pytest.approx(np.pi)
    M, _ = fluctuation_matrix(psi, ALPHA, (D2,))
    ev, vec = np.linalg.eig(M)
    i = int(np.argmax(ev.real))
    assert ev[i].real > 0.01
    # the unstable direction is the translation d psi / d theta
    k = np.fft.fftfreq(N, d=1.0 / N)
    dth = np.fft.ifft(1j * k * np.fft.fft(psi))
    mode = np.fft.ifft(vec[:N, i] * N)
    overlap = abs(np.vdot(dth, mode)) / (np.linalg.norm(dth)
                                         * np.linalg.norm(mode))
    assert overlap > 0.99


@pytest.fixture(scope="module")
def off_peak(pinned):
    th, prof, psi = pinned
    e = lle_evolve(np.roll(psi, -8), prof, ALPHA, (D2,), t_end=1500.0,
                   dt=0.004)
    return newton_state(e, prof, ALPHA, (D2,))


def test_off_peak_soliton_is_pinned_and_stable(pinned, off_peak):
    th, prof, _ = pinned
    sol, info = off_peak
    assert info["residual"] < 1e-12
    pos = th[np.argmax(np.abs(sol))]
    assert abs(pos - np.pi) > 0.3            # beside the peak
    # the pump profile breaks translation symmetry: no zero mode is
    # left, every drift eigenvalue is strictly negative, and the
    # spectra need no allow_marginal
    M, modes = fluctuation_matrix(sol, ALPHA, (D2,))
    assert np.max(np.linalg.eigvals(M).real) < -0.01
    v = output_quadrature_variance(M, 0.5, 0.3, int(np.where(modes == 1)[0][0]),
                                   len(modes), phi=0.0)
    assert np.isfinite(v) and v > 0
    # the mirror image about the peak is an exact steady state too
    mir = np.roll(sol[::-1], 1)
    res = np.abs(np.fft.ifft(_residual_k(mir, prof, ALPHA, (D2,)))).max()
    assert res < 1e-11


def test_newton_with_d1_is_confirmed_by_the_evolver(pinned, off_peak):
    # A small mismatch of repetition rate and FSR shifts where the
    # soliton locks; the evolver's settled state and Newton's root agree.
    # (A large mismatch, d1 = 0.02 for this pump, has no steady state:
    # the soliton keeps drifting round the ring, and Newton refuses.)
    th, prof, _ = pinned
    d1 = 0.005
    e = lle_evolve(off_peak[0], prof, ALPHA, (D2,), t_end=400.0, dt=0.004,
                   d1=d1)
    sol, info = newton_state(e, prof, ALPHA, (D2,), d1=d1)
    assert info["residual"] < 1e-12
    assert np.abs(e - sol).max() < 5e-3     # split-step error, as in test_soliton
    assert th[np.argmax(np.abs(sol))] != th[np.argmax(np.abs(off_peak[0]))]
    M, _ = fluctuation_matrix(sol, ALPHA, (D2,), d1=d1)
    assert np.max(np.linalg.eigvals(M).real) < 0


def test_pump_profile_refusals():
    with pytest.raises(ValueError):
        lle_evolve(np.ones(16, complex), np.ones(8), 1.0)
    with pytest.raises(ValueError):
        lle_evolve(np.ones(16, complex), np.full(16, np.nan), 1.0)


# --- time-dependent Gaussian covariance ------------------------------------

@pytest.mark.parametrize("eta", [1.0, 0.4])
def test_passive_cavity_output_pulse_is_vacuum(eta):
    f = lambda t: ((2 / np.pi) ** 0.25 * np.exp(-(t - 5) ** 2)
                   * np.exp(0.7j * t))
    r = temporal_mode_variance(-np.eye(2, dtype=complex), (0, 10), f,
                               eta=eta, phi=0.3)
    assert r["variance"] == pytest.approx(0.5, abs=1e-12)
    assert r["norm"] == pytest.approx(1.0, abs=1e-9)
    assert abs(r["max_photons"]) < 1e-12


def test_covariance_relaxes_to_the_steady_state():
    M = single_mode_parametric(0.5 * np.exp(0.4j), 0.3)
    t, V = covariance_evolution(M, (0, 30))
    assert np.abs(V[-1] - intracavity_covariance(M, [1.0])).max() < 1e-9
    # thermal bath, passive mode: <a^dag a> -> n_th
    t, V = covariance_evolution(-np.eye(2, dtype=complex), (0, 30),
                                n_th=0.7, V0=np.diag([1.0, 0.0]))
    assert V[-1][1, 1].real == pytest.approx(0.7, abs=1e-9)


@pytest.mark.parametrize("Om,phi", [(0.0, 0.3), (1.3, 1.1)])
def test_long_pulse_matches_the_steady_spectrum(Om, phi):
    # constant pump, steady state at t = 0: a Gaussian temporal mode
    # (at baseband, or on the two sidebands +-Om) must give
    # int |f~(w)|^2 S(w) dw / 2 pi with the package's steady spectrum
    M = single_mode_parametric(0.5 * np.exp(0.4j), 0.3)
    eta, sig, tc = 0.8, 6.0, 40.0
    g = lambda t: (1 / (np.pi * sig ** 2)) ** 0.25 * np.exp(
        -(t - tc) ** 2 / (2 * sig ** 2))
    if Om == 0.0:
        f = g
        F2 = lambda w: 2 * np.sqrt(np.pi) * sig * np.exp(-w ** 2 * sig ** 2)
    else:
        f = lambda t: np.sqrt(2) * g(t) * np.cos(Om * t)
        F2 = lambda w: np.sqrt(np.pi) * sig * (
            np.exp(-(w - Om) ** 2 * sig ** 2) + np.exp(-(w + Om) ** 2 * sig ** 2)
            + 2 * np.exp(-(w ** 2 + Om ** 2) * sig ** 2) * np.cos(2 * Om * tc))
    r = temporal_mode_variance(M, (0, 80), f, eta=eta, phi=phi,
                               V0=intracavity_covariance(M, [1.0]),
                               max_step=0.1)
    pred = quad(lambda w: F2(w) * output_quadrature_variance(
        M, eta, w, 0, 1, phi=phi), -10, 10, limit=500,
        points=[-Om, Om])[0] / (2 * np.pi)
    assert r["variance"] == pytest.approx(pred, rel=1e-8)


def test_pulse_above_threshold_matches_monte_carlo():
    # Seeded simulation of the equivalent classical (Wigner) equations,
    # with a pump pulse that goes briefly above threshold (|mu| = 1.6).
    mu = lambda t: 1.6 * np.exp(0.5j) * np.exp(-((t - 5) / 1.5) ** 2)
    dl, eta, phi, T = 0.2, 0.85, 0.0, 14.0
    f = lambda t: ((2 / np.pi / 1.5 ** 2) ** 0.25
                   * np.exp(-((t - 7) / 1.5) ** 2) * np.exp(0.3j * t))
    r = temporal_mode_variance(parametric_pulse_drift(mu, dl), (0, T), f,
                               eta=eta, phi=phi, max_step=0.05)
    assert r["max_photons"] > 1.0      # really did go above threshold
    rng = np.random.default_rng(7)
    K, dt = 12000, 4e-3
    a = np.zeros(K, complex)
    X = np.zeros(K)
    for s in range(int(round(T / dt))):
        t = (s + 0.5) * dt
        dBex = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) \
            * np.sqrt(dt / 4)
        dBl = (rng.standard_normal(K) + 1j * rng.standard_normal(K)) \
            * np.sqrt(dt / 4)
        aout = np.sqrt(2 * eta) * a * dt - dBex
        X += np.sqrt(2) * np.real(np.conj(f(t)) * np.exp(-1j * phi) * aout)
        a = a + ((-1 + 1j * dl) * a + mu(t) * np.conj(a)) * dt \
            + np.sqrt(2 * eta) * dBex + np.sqrt(2 * (1 - eta)) * dBl
    v = X.var()
    assert v == pytest.approx(r["variance"], rel=4 * np.sqrt(2 / K))
    assert r["variance"] < 0.5         # a squeezed pulse


def test_pulsed_refusals():
    M = -np.eye(2, dtype=complex)
    f = lambda t: 1.0
    with pytest.raises(ValueError):
        temporal_mode_variance(M, (1, 0), f)
    with pytest.raises(ValueError):
        temporal_mode_variance(M, (0, 1), f, eta=1.5)
    with pytest.raises(ValueError):
        temporal_mode_variance(M, (0, 1), lambda t: 0.0)
    with pytest.raises(ValueError):
        temporal_mode_variance(M, (0, 1), f, mode_index=2)
    with pytest.raises(ValueError):
        parametric_pulse_drift(0.5)
    with pytest.raises(ValueError):
        covariance_evolution(lambda t: np.eye(3), (0, 1))
