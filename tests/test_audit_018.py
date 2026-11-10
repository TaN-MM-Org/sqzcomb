"""0.18 regression tests: the wrong answers found by the fourth audit.

Each test fails on 0.17.0 and passes now. The references are
independent of the code under test: closed forms (the vacuum level, a
parametric mode with its rates rescaled, exponential decay, the
intracavity variances of the single-mode squeezer), or the same
computation run on a smaller, valid input.
"""
import numpy as np
import pytest

from sqzcomb import (covariance_evolution, covariance_xxpp,
                     fluctuation_matrix, intracavity_covariance, lle_evolve,
                     max_phase_noise, molecule_fluctuation_matrix,
                     newton_state, optimal_quadrature, output_covariance_xxpp,
                     output_quadrature_variance, output_variance_ports,
                     photonic_molecule, single_mode_parametric,
                     soliton_seed, vernier_molecule_fluctuation_matrix)


# ----------------------------------------------------------------------
# 1. The damping in M must be the decay the input-output relations use


def test_single_ring_functions_refuse_a_damping_other_than_one():
    """A passive mode damped at 0.5 instead of 1 is not what the
    single-ring formulas describe (their bath couplings are sqrt(2)):
    0.17.0 returned 4.5 at omega = 0 and eta = 1 for it, where any
    passive mode emits the vacuum level 0.5."""
    M = np.diag([-0.5, -0.5]).astype(complex)
    with pytest.raises(ValueError, match="rate 1"):
        output_quadrature_variance(M, 1.0, 0.0, 0, 1)
    with pytest.raises(ValueError, match="rate 1"):
        optimal_quadrature(M, 1.0, [0.0, 1.0], 0, 1)
    with pytest.raises(ValueError, match="rate 1"):
        output_covariance_xxpp(M, 1.0, 0.0, 1)
    # the same mode through the port functions, with its decay rate
    # stated, is vacuum (closed form), to rounding
    for w in (0.0, 0.3, 2.0):
        for phi in (0.0, 0.7, np.pi / 2):
            v = output_variance_ports(M, [0.5], 1.0, 0, w, phi=phi)
            assert v == pytest.approx(0.5, abs=1e-14)


@pytest.mark.parametrize("gam", [0.5, 3.0])
def test_port_functions_take_a_mode_with_its_own_decay(gam):
    """M = [[-g, mu], [mu, -g]] is single_mode_parametric(mu / g) with
    time measured in units of 1/g, so through the port functions it must
    give the closed form 0.5 (1 -/+ 4 eta m / ((1 +/- m)^2 + (w/g)^2)),
    m = mu / g. This is the call the damping check sends users to."""
    mu, eta = 0.6 * gam, 0.8
    M = np.array([[-gam, mu], [mu, -gam]], dtype=complex)
    m = mu / gam
    for w in (0.0, 0.4, 3.0):
        x2 = (w / gam) ** 2
        sq = output_variance_ports(M, [gam], eta, 0, w, phi=np.pi / 2)
        an = output_variance_ports(M, [gam], eta, 0, w, phi=0.0)
        assert sq == pytest.approx(
            0.5 * (1 - 4 * eta * m / ((1 + m) ** 2 + x2)), rel=1e-12)
        assert an == pytest.approx(
            0.5 * (1 + 4 * eta * m / ((1 - m) ** 2 + x2)), rel=1e-12)


def test_port_functions_refuse_gammas_that_do_not_match_m():
    """photonic_molecule(0, 1, gamma=9) is a passive pair of rings, so
    every port emits vacuum. Read with gammas=[1, 1] (the decay of the
    second ring left at 1), 0.17.0 gave 0.34 at omega = 0, an apparent
    -1.67 dB of squeezing from no pump at all; a negative rate gave
    NaN."""
    M, g = photonic_molecule(0.0, 1.0, gamma=9.0)
    assert output_variance_ports(M, g, 1.0, 1, 0.0) == pytest.approx(
        0.5, abs=1e-14)
    with pytest.raises(ValueError, match="gammas do not match"):
        output_variance_ports(M, [1.0, 1.0], 1.0, 1, 0.0)
    with pytest.raises(ValueError, match="finite and >= 0"):
        output_variance_ports(M, [1.0, -9.0], 1.0, 1, 0.0)
    with pytest.raises(ValueError, match="finite and >= 0"):
        output_variance_ports(M, [1.0, np.nan], 1.0, 1, 0.0)


def test_intracavity_and_pulsed_refuse_mismatched_gammas():
    """With gammas=[1, 1] for the gamma = 9 molecule, 0.17.0 returned
    <a a^dag> = 0.92 and 0.12 for a passive (vacuum) state, i.e.
    negative photon numbers; the right gammas give exactly 1."""
    M, g = photonic_molecule(0.0, 1.0, gamma=9.0)
    V = intracavity_covariance(M, g)
    assert np.allclose(np.diag(V).real, [1, 1, 0, 0], atol=1e-13)
    with pytest.raises(ValueError, match="gammas do not match"):
        intracavity_covariance(M, [1.0, 1.0])
    with pytest.raises(ValueError, match="gammas do not match"):
        covariance_evolution(M, (0.0, 1.0), gammas=[1.0, 1.0])
    # the default gammas (all 1) against a pulse drift is fine
    t, V = covariance_evolution(single_mode_parametric(0.5), (0.0, 1.0))
    assert np.all(np.isfinite(V))


# ----------------------------------------------------------------------
# 2. The intracavity covariance of a marginal (soliton) drift matrix


def _resolved_soliton():
    """A soliton resolved on a 48-point grid: its translation mode comes
    out with a growth rate of about -5e-14 (rounding of an exact 0)."""
    F, alpha, d2 = 1.9, 3.0, -4.0
    sol, _ = newton_state(soliton_seed(48, F, alpha, (d2,)), F, alpha,
                          (d2,))
    return fluctuation_matrix(sol, alpha, (d2,))


def test_intracavity_covariance_refuses_a_marginal_drift_matrix():
    """Along the soliton's free translation the steady covariance is
    infinite. Where rounding put the zero growth rate at -5e-14, 0.17.0
    solved the near-singular system anyway and returned entries of
    1.5e14 (with NumPy 2.5 / OpenBLAS here); where it came out slightly
    positive (NumPy 1.22) 0.17.0 refused it as above threshold. It is
    now refused as marginal either way."""
    M, modes = _resolved_soliton()
    assert abs(np.max(np.linalg.eigvals(M).real)) < 1e-8
    with pytest.raises(ValueError, match="marginally stable"):
        intracavity_covariance(M, np.ones(modes.size))
    # at threshold (a growth rate of 0 up to rounding of either sign)
    with pytest.raises(ValueError, match="marginally stable"):
        intracavity_covariance(single_mode_parametric(1.0), [1.0])


def test_intracavity_covariance_near_threshold_with_allow_marginal():
    """A genuinely stable state this close to threshold still works when
    asked for, and matches the closed-form intracavity variances of the
    single-mode squeezer, 1 / (2 (1 -/+ mu)) (vacuum 1/2)."""
    mu = 1.0 - 5e-7
    M = single_mode_parametric(mu)
    with pytest.raises(ValueError, match="marginally stable"):
        intracavity_covariance(M, [1.0])
    s = covariance_xxpp(intracavity_covariance(M, [1.0],
                                               allow_marginal=True),
                        hbar=1.0)
    assert s[0, 0] == pytest.approx(1.0 / (2.0 * (1.0 - mu)), rel=1e-8)
    assert s[1, 1] == pytest.approx(1.0 / (2.0 * (1.0 + mu)), rel=1e-8)
    # unstable stays refused, flag or no flag
    with pytest.raises(ValueError, match="unstable"):
        intracavity_covariance(single_mode_parametric(1.0 + 1e-3), [1.0],
                               allow_marginal=True)


# ----------------------------------------------------------------------
# 3. lle_evolve must end at t_end


@pytest.mark.parametrize("t_end,dt", [(0.25, 0.1), (0.04, 0.1),
                                      (1.0, 0.3), (2.0, 0.01)])
def test_lle_evolve_ends_at_t_end(t_end, dt):
    """Without pump the flat field decays as |psi| = |psi0| e^{-t}
    exactly (the Kerr step only turns the phase, and the linear step of
    the k = 0 line is the exact exponential). 0.17.0 ran round(t_end /
    dt) steps of dt: t_end = 0.25, dt = 0.1 stopped at 0.2, and
    t_end = 0.04, dt = 0.1 returned psi0 unchanged."""
    psi0 = np.full(8, 1e-3 * np.exp(0.3j))
    out = lle_evolve(psi0, 0.0, 0.3, t_end=t_end, dt=dt)
    assert np.allclose(np.abs(out), 1e-3 * np.exp(-t_end), rtol=1e-12,
                       atol=0)


def test_lle_evolve_whole_steps_unchanged_and_refusals():
    """A t_end that is a whole number of steps keeps dt exactly (so
    results from before 0.18 are unchanged): two half runs equal one
    full run bit for bit, the same arithmetic repeated."""
    rng = np.random.default_rng(0)
    psi0 = 0.3 * (rng.standard_normal(32) + 1j * rng.standard_normal(32))
    one = lle_evolve(psi0, 1.2, 1.5, (-0.1,), t_end=1.0, dt=0.01)
    half = lle_evolve(psi0, 1.2, 1.5, (-0.1,), t_end=0.5, dt=0.01)
    two = lle_evolve(half, 1.2, 1.5, (-0.1,), t_end=0.5, dt=0.01)
    assert np.array_equal(one, two)
    assert np.array_equal(lle_evolve(psi0, 1.0, 0.0, t_end=0.0), psi0)
    for kw in (dict(dt=-0.01), dict(dt=0.0), dict(dt=np.nan),
               dict(t_end=-1.0), dict(t_end=np.inf)):
        with pytest.raises(ValueError):
            lle_evolve(psi0, 1.0, 0.0, **kw)


# ----------------------------------------------------------------------
# 4. Mode lists with a repeated or fractional mode number


def test_fluctuation_matrix_refuses_repeated_and_fractional_modes():
    """modes=[0, 0] kept two coupled copies of the pumped line, which
    doubled its Kerr coupling: for the flat state of README example 3
    at eta = 0.8 it read -1.279 dB at the best angle, where modes=[0]
    gives -4.194 dB. modes=[0, 0.6] was cut to [0, 0], the same wrong
    answer."""
    psi = lle_evolve(np.full(16, 0.05 + 0j), F=0.9, alpha=0.3,
                     t_end=100.0)
    M, m = fluctuation_matrix(psi, 0.3, modes=[0])
    assert optimal_quadrature(M, 0.8, 0.0, 0, 1)["squeezing_db"] == \
        pytest.approx(-4.1936, abs=1e-4)
    for modes in ([0, 0], [0, 0.6], [-1, 0, 1, 0], []):
        with pytest.raises(ValueError):
            fluctuation_matrix(psi, 0.3, modes=modes)
    with pytest.raises(ValueError, match="more than once"):
        molecule_fluctuation_matrix(psi, 0.3, 1.0, 1.0, modes=[1, 1])
    with pytest.raises(ValueError, match="integers"):
        vernier_molecule_fluctuation_matrix(psi, 0.3, 1.0, 1.0, 600.0,
                                            [0.3], modes=[0.5])
    # numpy integer arrays and ranges work as before
    M1, m1 = fluctuation_matrix(psi, 0.3, modes=range(-2, 3))
    M2, m2 = fluctuation_matrix(psi, 0.3, modes=np.arange(-2, 3))
    assert np.array_equal(M1, M2) and m1.tolist() == [-2, -1, 0, 1, 2]


# ----------------------------------------------------------------------
# 5. max_phase_noise with a NaN target


def test_max_phase_noise_refuses_a_nan_target():
    """A NaN target passed both range checks and came back as a NaN
    jitter in 0.17.0."""
    assert max_phase_noise(0.1, 0.05, 6.3) == pytest.approx(0.0898,
                                                            abs=1e-4)
    with pytest.raises(ValueError, match="finite"):
        max_phase_noise(np.nan, 0.05, 6.3)
