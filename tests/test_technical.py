"""Technical (classical) noise carried through the linearized resonator."""
import numpy as np
import pytest
from scipy import signal
from scipy.integrate import quad
from scipy.linalg import expm, solve_continuous_lyapunov

from sqzcomb.kerrpo import kerr_parametric_states
from sqzcomb.physical import RingSpec
from sqzcomb.technical import (classical_noise_variance,
                               classical_noise_variance_ports,
                               gain_noise_drive, lle_mode_amplitudes,
                               normalized_psd, pump_noise_drive,
                               resonance_noise_drive)


def _two_mode_drift():
    A = np.array([[-1 + 0.3j, 0.1j], [0.1j, -1 - 0.2j]])
    B = np.array([[0.5, 0.1], [0.1, 0.4]], dtype=complex)
    return np.block([[A, B], [B.conj(), A.conj()]])


@pytest.mark.parametrize("omega", [0.0, 0.5, 2.0])
def test_passive_mode_closed_form(omega):
    # passive mode, mean field c: resonance noise goes entirely into the
    # phase quadrature, 4 eta |c|^2 S / (1 + omega^2); none into amplitude
    M = -np.eye(2, dtype=complex)
    c, eta, S = 3.0 * np.exp(0.7j), 0.8, 0.01
    b = resonance_noise_drive(c)
    ph = classical_noise_variance(M, eta, omega, [b], [S], 0,
                                  phi=np.angle(c) + np.pi / 2)
    am = classical_noise_variance(M, eta, omega, [b], [S], 0,
                                  phi=np.angle(c))
    assert ph == pytest.approx(4 * eta * abs(c) ** 2 * S / (1 + omega ** 2),
                               rel=1e-12)
    assert am < 1e-25
    # pump amplitude noise on the same mode (mean field c = F): the
    # mirror image, entirely in the amplitude quadrature
    bp = pump_noise_drive(c, 1, kind="amplitude")
    am = classical_noise_variance(M, eta, omega, [bp], [S], 0,
                                  phi=np.angle(c))
    ph = classical_noise_variance(M, eta, omega, [bp], [S], 0,
                                  phi=np.angle(c) + np.pi / 2)
    assert am == pytest.approx(4 * eta * abs(c) ** 2 * S / (1 + omega ** 2),
                               rel=1e-12)
    assert ph < 1e-25


def test_zero_psd_and_independent_sources_add():
    M = _two_mode_drift()
    b1 = resonance_noise_drive([1.5 + 0.5j, -0.7j])
    b2 = pump_noise_drive(2.0, 2, pumped_index=1, kind="phase")
    assert classical_noise_variance(M, 0.7, 0.4, [b1], [0.0], 1) == 0.0
    v1 = classical_noise_variance(M, 0.7, 0.4, [b1], [0.3], 1, phi=0.2)
    v2 = classical_noise_variance(M, 0.7, 0.4, [b2], [0.1], 1, phi=0.2)
    v12 = classical_noise_variance(M, 0.7, 0.4, [b1, b2], [0.3, 0.1], 1,
                                   phi=0.2)
    assert v12 == pytest.approx(v1 + v2, rel=1e-12)


def test_time_domain_lyapunov_agrees_with_spectrum():
    # Independent time-domain calculation: append an Ornstein-Uhlenbeck
    # noise eps (d eps = -g eps dt + sqrt(2 D) dW, PSD 2D/(g^2+w^2)) to
    # the state, solve the stationary Lyapunov equation, and compare the
    # output variance with the integral of the added spectrum.
    M = _two_mode_drift()
    b = resonance_noise_drive([1.5 + 0.5j, -0.7j])
    g, D, eta, phi = 0.8, 0.05, 0.7, 0.9
    A = np.zeros((5, 5), dtype=complex)
    A[:4, :4] = M
    A[:4, 4] = b
    A[4, 4] = -g
    Q = np.zeros((5, 5))
    Q[4, 4] = 2 * D
    Sig = solve_continuous_lyapunov(A, -Q)
    r = np.zeros(5, dtype=complex)
    r[1] = np.sqrt(2 * eta) * np.exp(-1j * phi) / np.sqrt(2)
    r[3] = np.sqrt(2 * eta) * np.exp(1j * phi) / np.sqrt(2)
    var_time = float((r @ Sig @ r.conj()).real)
    var_freq = quad(lambda w: classical_noise_variance(
        M, eta, w, [b], [2 * D / (g * g + w * w)], 1, phi=phi),
        -np.inf, np.inf, limit=400)[0] / (2 * np.pi)
    assert var_freq == pytest.approx(var_time, rel=1e-7)


def test_monte_carlo_spectrum():
    # Seeded simulation of the same linear equations driven by a
    # coloured (OU) classical noise; Welch spectrum vs the formula.
    M = _two_mode_drift()
    b = resonance_noise_drive([1.5 + 0.5j, -0.7j])
    g, D, eta, phi, dt = 0.8, 0.05, 0.7, 0.9, 0.02
    # real representation (Re a1, Re a2, Im a1, Im a2)
    T = np.zeros((4, 4), dtype=complex)
    T[0, 0] = T[0, 2] = T[1, 1] = T[1, 3] = 0.5
    T[2, 0] = T[3, 1] = -0.5j
    T[2, 2] = T[3, 3] = 0.5j
    Ti = np.linalg.inv(T)
    P = expm(M * dt)
    Qv = np.linalg.solve(M, P - np.eye(4)) @ b      # zero-order hold
    Pr = (T @ P @ Ti).real
    Qr = (T @ Qv).real
    u = np.zeros(4, dtype=complex)
    u[1] = np.exp(1j * phi) / np.sqrt(2)
    u[3] = np.exp(-1j * phi) / np.sqrt(2)
    Cr = (np.sqrt(2 * eta) * u.conj() @ Ti).real
    num, den = signal.ss2tf(Pr, Qr[:, None], Cr[None, :], [[0.0]])
    rng = np.random.default_rng(12345)
    a = np.exp(-g * dt)
    w = rng.standard_normal(12_000_000) * np.sqrt(D / g * (1 - a * a))
    eps = signal.lfilter([1.0], [1.0, -a], w)
    # state after the update of step k uses eps_k: y_k = C z_{k+1}
    y = signal.lfilter(num[0], den, eps)[1000:]
    f, Sy = signal.welch(y, fs=1 / dt, nperseg=2 ** 14,
                         return_onesided=False, scaling="density")
    # (a) the exact spectrum of the discretized system (z-transform of
    # the simulated recursion) agrees with the continuous formula
    for om in (0.5, 1.0, 2.0):
        zz = np.exp(1j * om * dt)
        H = np.polyval(num[0], zz) / np.polyval(den, zz)
        S_disc = abs(H) ** 2 * dt * (D / g * (1 - a * a)) / abs(1 - a / zz) ** 2
        pred = classical_noise_variance(M, eta, om, [b],
                                        [2 * D / (g * g + om * om)], 1,
                                        phi=phi)
        assert S_disc == pytest.approx(pred, rel=1e-3)
    # (b) the simulated record's spectrum, averaged over a band of
    # Welch bins (statistical error about 1 %), against the formula
    # averaged over the same bins
    for om in (0.5, 1.0, 2.0):
        sel = np.abs(2 * np.pi * f - om) < 0.2
        pred = np.mean([classical_noise_variance(
            M, eta, w, [b], [2 * D / (g * g + w * w)], 1, phi=phi)
            for w in 2 * np.pi * f[sel]])
        assert Sy[sel].mean() == pytest.approx(pred, rel=0.04)


def test_ports_geometry_reduces_to_single_port():
    M = _two_mode_drift()
    b = resonance_noise_drive([1.5 + 0.5j, -0.7j])
    for om in (0.0, 0.7):
        a = classical_noise_variance(M, 0.6, om, [b], [0.2], 1, phi=0.4)
        p = classical_noise_variance_ports(M, [1.0, 1.0], 0.6, 1, om, [b],
                                           [0.2], phi=0.4)
        assert p == pytest.approx(a, rel=1e-12)


def test_static_response_matches_kerr_steady_state_shift():
    # Zero-frequency check against the exact nonlinear steady state: a
    # small constant eps moves the bright state by -M^-1 b eps.
    mu, delta, chi, h = 1.5 * np.exp(0.3j), 0.2, 0.05, 1e-6

    def bright(mu, delta):
        st = [s for s in kerr_parametric_states(mu, delta, chi)
              if s["n"] > 0 and s["stable"]]
        return st

    ref = bright(mu, delta)[0]
    Minv = np.linalg.inv(ref["drift"])

    def closest(states):
        return min(states, key=lambda s: abs(s["alpha"] - ref["alpha"]))

    # resonance moves up by eps: delta -> delta - eps in this convention
    moved = closest(bright(mu, delta - h))["alpha"]
    pred = -(Minv @ resonance_noise_drive(ref["alpha"]))[0] * h
    assert moved - ref["alpha"] == pytest.approx(pred, rel=1e-4)
    for kind, fac in (("amplitude", 1 + h), ("phase", np.exp(1j * h))):
        moved = closest(bright(mu * fac, delta))["alpha"]
        pred = -(Minv @ gain_noise_drive(ref["alpha"], mu, kind))[0] * h
        assert moved - ref["alpha"] == pytest.approx(pred, rel=1e-4)
    assert np.all(gain_noise_drive(0.0, mu) == 0)


def test_lle_mode_amplitudes():
    chi = 2e-4
    psi = 0.8 - 0.6j
    grid = np.full(64, psi)
    c = lle_mode_amplitudes(grid, [0, 1, -1], chi)
    assert c[0] == pytest.approx(complex(psi) / np.sqrt(chi), rel=1e-12)
    assert np.all(np.abs(c[1:]) < 1e-12)
    # photon number: sum over all lines = mean |psi|^2 / chi
    theta = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    field = 1.0 + 0.3 * np.exp(1j * theta) + 0.2j * np.exp(-2j * theta)
    ks = np.fft.fftfreq(128, d=1 / 128).astype(int)
    c = lle_mode_amplitudes(field, ks, chi)
    assert np.sum(np.abs(c) ** 2) == pytest.approx(
        np.mean(np.abs(field) ** 2) / chi, rel=1e-12)
    assert c[list(ks).index(1)] == pytest.approx(0.3 / np.sqrt(chi))


def _spec():
    return RingSpec(kappa_hz=100e6, eta_esc=0.8, g0_hz=1.0,
                    lambda_pump_m=1.55e-6,
                    reference="made-up test numbers")


def test_normalized_psd_parseval():
    spec = _spec()
    # Lorentzian lab PSDs; the variance must be the same in both units
    f = np.logspace(-1, 13, 40001)
    for kind, scale in (("dimensionless", 1.0),
                        ("frequency", (4 * np.pi / spec.kappa) ** 2)):
        S1 = 1e-9 / (1 + (f / 3e6) ** 2)
        om, S = normalized_psd(spec, f, S1, kind=kind)
        var_lab = np.sum(0.5 * (S1[1:] + S1[:-1]) * np.diff(f))
        var_norm = 2 * np.sum(0.5 * (S[1:] + S[:-1]) * np.diff(om)) \
            / (2 * np.pi)
        assert var_norm == pytest.approx(scale * var_lab, rel=1e-9)


def test_refusals():
    M = _two_mode_drift()
    b = resonance_noise_drive([1.0, 1.0])
    with pytest.raises(ValueError):
        classical_noise_variance(M, 0.5, 0.0, [b], [0.1, 0.2], 0)
    with pytest.raises(ValueError):
        classical_noise_variance(M, 0.5, 0.0, [b[:2]], [0.1], 0)
    with pytest.raises(ValueError):
        classical_noise_variance(M, 0.5, 0.0, [b], [-0.1], 0)
    with pytest.raises(ValueError):
        classical_noise_variance(np.array([[0.1, 0], [0, 0.1]]), 0.5, 0.0,
                                 [np.ones(2)], [0.1], 0)
    with pytest.raises(ValueError):
        pump_noise_drive(1.0, 1, kind="frequency")
    with pytest.raises(ValueError):
        normalized_psd(_spec(), [0.0, 1.0], [1.0, 1.0])
    with pytest.raises(ValueError):
        lle_mode_amplitudes(np.ones(8), [0], 0.0)


def test_pump_drive_array_form_and_marginal_ports():
    b1 = pump_noise_drive(2.0 - 1.0j, 3, pumped_index=1, kind="phase")
    b2 = pump_noise_drive([0.0, 2.0 - 1.0j, 0.0], 3, kind="phase")
    assert np.array_equal(b1, b2)
    with pytest.raises(ValueError):
        pump_noise_drive([1.0, 2.0], 3)
    # a marginal drift (zero eigenvalue) is refused unless allowed
    M = np.diag([0.0, -1.0, 0.0, -1.0]).astype(complex)
    b = resonance_noise_drive([0.0, 1.0])
    with pytest.raises(ValueError, match="marginal"):
        classical_noise_variance_ports(M, [1.0, 1.0], 0.5, 1, 0.5, [b],
                                       [0.1])
    v = classical_noise_variance_ports(M, [1.0, 1.0], 0.5, 1, 0.5, [b],
                                       [0.1], allow_marginal=True)
    assert v == pytest.approx(classical_noise_variance(
        M, 0.5, 0.5, [b], [0.1], 1, allow_marginal=True), rel=1e-12)
