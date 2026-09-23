"""Above threshold and beyond Gaussian: the Kerr parametric oscillator
(classical states + linearization) and the exact Fock-space master
equation, each held to something independent."""
import numpy as np
import pytest

from sqzcomb import (covariance_xxpp, intracavity_covariance,
                     output_quadrature_variance, single_mode_parametric)
from sqzcomb.kerrpo import kerr_parametric_drift, kerr_parametric_states
from sqzcomb.master import (coherent_state, fock_state,
                            kerr_parametric_master, master_evolve,
                            master_moments, master_output_spectrum,
                            steady_state, wigner, wigner_negativity)

# --- kerrpo -------------------------------------------------------------


def test_drift_at_vacuum_is_the_linear_model():
    for mu, d in ((0.5, 0.0), (0.7 * np.exp(0.4j), -0.3)):
        assert np.array_equal(kerr_parametric_drift(0.0, mu, d, 0.1),
                              single_mode_parametric(mu, d))


def test_below_threshold_only_vacuum():
    st = kerr_parametric_states(0.8, 0.0, 0.05)
    assert len(st) == 1 and st[0]["n"] == 0.0 and st[0]["stable"]


def _rhs(al, mu, d, chi):
    return (-1 + 1j * d) * al + mu * np.conj(al) + 1j * chi * abs(al) ** 2 * al


@pytest.mark.parametrize("mu,d,chi", [(1.5, 0.0, 0.1),
                                      (1.3 * np.exp(0.6j), 0.4, 0.02),
                                      (2.0, -0.5, -0.05)])
def test_bright_states_solve_the_equation_and_drift_is_its_jacobian(mu, d,
                                                                    chi):
    st = kerr_parametric_states(mu, d, chi)
    bright = [s for s in st if s["n"] > 0]
    assert bright and len(bright) % 2 == 0
    for s in bright:
        al = s["alpha"]
        assert abs(_rhs(al, mu, d, chi)) < 1e-9 * max(1, abs(al))
        assert (d + chi * s["n"]) ** 2 == pytest.approx(abs(mu) ** 2 - 1)
        # the drift is the Jacobian of the classical equation (finite
        # differences in the doubled basis), an independent derivation
        h = 1e-6
        Jn = np.zeros((2, 2), complex)
        for col, e in enumerate((1.0, 1j)):
            f1 = _rhs(al + h * e, mu, d, chi)
            f0 = _rhs(al - h * e, mu, d, chi)
            df = (f1 - f0) / (2 * h)
            Jn[0, col], Jn[1, col] = df, np.conj(df)
        # convert from (Re, Im) perturbations to (d alpha, d alpha*)
        T = np.array([[1, 1j], [1, -1j]])
        J_doubled = Jn @ np.linalg.inv(T)
        assert np.allclose(J_doubled, s["drift"], atol=1e-6)
    # the pair +alpha, -alpha share one drift
    a, b = bright[0], bright[1]
    assert np.allclose(a["drift"], b["drift"])
    assert a["alpha"] == pytest.approx(-b["alpha"])


def test_zero_kerr_refused():
    with pytest.raises(ValueError, match="chi = 0"):
        kerr_parametric_states(1.5, 0.0, 0.0)


# --- master: quadratic limit equals the linear theory --------------------

@pytest.mark.parametrize("mu,d", [(0.5, 0.0), (0.4 * np.exp(0.7j), 0.3)])
def test_quadratic_master_equals_linear_spectra(mu, d):
    H, c, a, dims = kerr_parametric_master(mu, d, 0.0, cutoff=40)
    rho = steady_state(H, c, dims)
    M = single_mode_parametric(mu, d)
    cov = covariance_xxpp(intracavity_covariance(M, [1.0]), hbar=1.0)
    assert np.allclose(master_moments(rho, a)["cov_xp"], cov, atol=1e-8)
    om = [0.0, 0.8, 2.0]
    for phi in (0.0, 0.3, 1.2, np.pi / 2):
        exact = master_output_spectrum(H, c, a, 0.7, om, phi)
        lin = [output_quadrature_variance(M, 0.7, w, 0, 1, phi=phi)
               for w in om]
        assert np.allclose(exact, lin, atol=1e-8)


def test_closed_form_degenerate_parametric():
    mu, eta = 0.5, 0.7
    H, c, a, dims = kerr_parametric_master(mu, 0.0, 0.0, cutoff=40)
    om = np.array([0.0, 0.8, 2.0])
    sq = master_output_spectrum(H, c, a, eta, om, np.pi / 2)
    an = master_output_spectrum(H, c, a, eta, om, 0.0)
    assert np.allclose(sq, 0.5 * (1 - 4 * eta * mu / ((1 + mu) ** 2 + om ** 2)),
                       atol=1e-9)
    assert np.allclose(an, 0.5 * (1 + 4 * eta * mu / ((1 - mu) ** 2 + om ** 2)),
                       atol=1e-9)


def test_phi_convention_scalar_equals_covariance_path():
    # the 0.13 fix: the scalar quadrature routine and the xxpp
    # covariance must describe the same X_phi = cos phi x + sin phi p
    from sqzcomb import output_covariance_xxpp
    M = single_mode_parametric(0.4 * np.exp(0.7j), 0.3)
    for om in (0.0, 0.9):
        C = output_covariance_xxpp(M, 0.8, om, 1)
        for phi in (0.3, 1.2, 2.5):
            v = np.array([np.cos(phi), np.sin(phi)])
            assert output_quadrature_variance(M, 0.8, om, 0, 1, phi=phi) \
                == pytest.approx(float(v @ C @ v), rel=1e-10)
    # molecule ports (same fix in `output_variance_ports`)
    from sqzcomb import output_variance_ports
    for phi in (0.3, 1.2):
        a = output_variance_ports(M, [1.0], 0.8, 0, 0.5, phi=phi)
        b = output_quadrature_variance(M, 0.8, 0.5, 0, 1, phi=phi)
        assert a == pytest.approx(b, rel=1e-12)


# --- master: Kerr, above threshold ---------------------------------------

def test_truncation_refused():
    H, c, a, dims = kerr_parametric_master(1.5, 0.0, 0.05, cutoff=12)
    with pytest.raises(ValueError, match="truncation"):
        steady_state(H, c, dims)


def test_above_threshold_lobes_and_convergence_to_linear():
    mu, om, gaps = 1.5, 2.0, []
    for chi, cut in ((0.2, 50), (0.05, 110)):
        st = [s for s in kerr_parametric_states(mu, 0.0, chi)
              if s["n"] > 0 and s["stable"]]
        al = st[0]["alpha"]
        H, c, a, dims = kerr_parametric_master(mu, 0.0, chi, cutoff=cut)
        rho = steady_state(H, c, dims)
        mom = master_moments(rho, a)
        assert abs(mom["alpha"]) < 1e-8          # the +-alpha mixture
        assert mom["n"] == pytest.approx(st[0]["n"], rel=0.12)
        # two lobes at +-alpha in the Wigner function
        x = np.array([-np.sqrt(2) * al.real, 0.0, np.sqrt(2) * al.real])
        p = np.array([-np.sqrt(2) * al.imag, 0.0, np.sqrt(2) * al.imag])
        W = wigner(rho, x, p)
        assert W[2, 2] == pytest.approx(W[0, 0], rel=1e-6)
        assert W[2, 2] > 5 * abs(W[1, 1])        # lobes, not a single hump
        # phase quadrature (perpendicular to alpha): exact vs linearized
        phi = np.angle(al) + np.pi / 2
        exact = master_output_spectrum(H, c, a, 1.0, [om], phi)[0]
        lin = output_quadrature_variance(st[0]["drift"], 1.0, om, 0, 1,
                                         phi=phi)
        gaps.append(abs(exact - lin))
    assert gaps[1] < gaps[0]                     # converges as chi -> 0


# --- master: Wigner function ---------------------------------------------

def test_wigner_fock_one_and_normalization():
    x = np.linspace(-6, 6, 241)
    W = wigner(fock_state(1, 6), x, x)
    assert W[120, 120] == pytest.approx(-1 / np.pi, abs=1e-14)
    dx = x[1] - x[0]
    assert W.sum() * dx * dx == pytest.approx(1.0, abs=1e-9)
    # x-marginal = |psi_1(x)|^2 = 2 x^2 exp(-x^2) / sqrt(pi)
    marg = W.sum(axis=0) * dx
    assert np.allclose(marg, 2 * x ** 2 * np.exp(-x ** 2) / np.sqrt(np.pi),
                       atol=1e-9)
    # negative volume of |1>: 2 exp(-1/2) - 1 exactly
    xf = np.linspace(-4, 4, 801)
    assert wigner_negativity(fock_state(1, 6), xf, xf) == pytest.approx(
        2 * np.exp(-0.5) - 1, rel=1e-4)
    # a Gaussian (vacuum) has none
    assert wigner_negativity(fock_state(0, 6), xf, xf) == 0.0


def test_wigner_matches_qutip():
    qt = pytest.importorskip("qutip")
    H, c, a, dims = kerr_parametric_master(1.2 * np.exp(0.3j), 0.2, 0.3,
                                           cutoff=30)
    rho = steady_state(H, c, dims)
    x = np.linspace(-4, 4, 41)
    Wq = qt.wigner(qt.Qobj(rho), x, x, g=np.sqrt(2))
    assert np.allclose(wigner(rho, x, x), Wq, atol=1e-10)


def test_steady_state_matches_qutip():
    qt = pytest.importorskip("qutip")
    mu, d, chi, N = 1.2 * np.exp(0.3j), 0.2, 0.3, 30
    H, c, a, dims = kerr_parametric_master(mu, d, chi, cutoff=N)
    rho = steady_state(H, c, dims)
    aq = qt.destroy(N)
    Hq = (-d * aq.dag() * aq
          + 0.5j * (mu * aq.dag() ** 2 - np.conj(mu) * aq ** 2)
          - 0.5 * chi * aq.dag() ** 2 * aq ** 2)
    rq = qt.steadystate(Hq, [np.sqrt(2) * aq])
    assert np.allclose(rho, rq.full(), atol=1e-8)


# --- master: time evolution and a non-Gaussian state ---------------------

def test_lossless_kerr_evolution_is_exact_and_makes_a_cat():
    # H = -(chi/2) a^dag^2 a^2: rho_mn(t) = rho_mn(0)
    # exp(i chi t [m(m-1) - n(n-1)] / 2), exactly
    chi, N, al = 1.0, 40, 2.0
    H, _, a, dims = kerr_parametric_master(0.0, 0.0, chi, cutoff=N)
    rho0 = coherent_state(al, N)
    t = np.pi / chi
    rho_t = master_evolve(H, [], rho0, [t])[0]
    k = np.arange(N)
    ph = np.exp(0.5j * chi * t * (k * (k - 1)))
    assert np.allclose(rho_t, rho0 * np.outer(ph, ph.conj()), atol=1e-10)
    # a coherent state is Gaussian (no negativity); the Kerr cat is not
    x = np.linspace(-6, 6, 241)
    assert wigner_negativity(rho0, x, x) < 1e-12
    assert wigner_negativity(rho_t, x, x) > 0.2


def test_evolution_relaxes_to_steady_state():
    H, c, a, dims = kerr_parametric_master(0.8, 0.1, 0.2, cutoff=30)
    rho_inf = steady_state(H, c, dims)
    rho = master_evolve(H, c, fock_state(0, 30), [0.0, 40.0], dims)
    assert np.allclose(rho[0], fock_state(0, 30))
    assert np.abs(rho[1] - rho_inf).max() < 1e-10


def test_master_refusals():
    H, c, a, dims = kerr_parametric_master(0.5, cutoff=10, n_th=0.2)
    with pytest.raises(ValueError, match="vacuum"):
        master_output_spectrum(H, c, a, 0.5, [0.0])
    with pytest.raises(ValueError):
        master_output_spectrum(H, c[:1], a, 0.0, [0.0])
    with pytest.raises(ValueError):
        coherent_state(5.0, 10)
    with pytest.raises(ValueError):
        master_evolve(H, c, fock_state(0, 10), [1.0, 0.5])


def test_switching_noise_has_the_width_of_the_slowest_rate():
    # Above threshold the exact low-frequency excess over the linearized
    # spectrum is a Lorentzian whose half-width is the slowest nonzero
    # decay rate of the master equation (the switching rate between the
    # two phase states): S_ex(0) / S_ex(w) = 1 + (w / rate)^2.
    import scipy.sparse.linalg as spla
    from sqzcomb.master import liouvillian
    mu, chi = 1.5, 0.2
    st = [s for s in kerr_parametric_states(mu, 0.0, chi)
          if s["n"] > 0 and s["stable"]][0]
    H, c, a, dims = kerr_parametric_master(mu, 0.0, chi, cutoff=50)
    ev = spla.eigs(liouvillian(H, c).tocsc(), k=4, sigma=-1e-3,
                   which="LM", return_eigenvectors=False)
    re = np.sort(-ev.real[ev.real < -1e-10])
    rate = re[0]
    assert 0 < rate < 0.05                       # slow, as switching is
    phi = np.angle(st["alpha"])
    w1 = rate
    ex = master_output_spectrum(H, c, a, 1.0, [0.0, w1], phi)
    lin = [output_quadrature_variance(st["drift"], 1.0, w, 0, 1, phi=phi)
           for w in (0.0, w1)]
    ratio = (ex[0] - lin[0]) / (ex[1] - lin[1])
    assert ratio == pytest.approx(2.0, rel=0.02)


def test_output_spectrum_refuses_a_small_basis():
    # cutoff 20 is far too small for mu = 1.5, chi = 0.1 (about 11
    # photons); without the check it returned 1.10 instead of 0.508
    H, c, a, dims = kerr_parametric_master(1.5, 0.0, 0.1, cutoff=20)
    with pytest.raises(ValueError, match="truncation"):
        master_output_spectrum(H, c, a, 1.0, [2.0], 0.0)
    with pytest.raises(ValueError, match="truncation"):
        master_output_spectrum(H, c, a, 1.0, [2.0], 0.0, dims=dims)


def test_drift_accepts_zero_kerr():
    assert np.array_equal(kerr_parametric_drift(0.0, 0.5, 0.2, 0.0),
                          single_mode_parametric(0.5, 0.2))
