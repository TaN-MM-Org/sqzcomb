"""Newton steady states against exact statements: the flat seed lands
on the exact cubic root, a converged state is verified below tolerance
and confirmed by the independent split-step evolver, the soliton
carries an exact translation (Goldstone) zero mode, an N-fold crystal
is exactly the rescaled single soliton, and continuation returns only
verified states."""
import numpy as np
import pytest

from sqzcomb import (continuation, fluctuation_matrix,
                     homogeneous_steady_states, lle_evolve, newton_state,
                     output_quadrature_variance, soliton_seed)

F, ALPHA, D2, N = 1.9, 3.0, -0.25, 256


def _soliton():
    seed = soliton_seed(N, F, ALPHA, (D2,))
    return newton_state(seed, F, ALPHA, (D2,))


def test_flat_seed_lands_on_the_exact_cubic_root():
    psi, info = newton_state(np.full(64, 0.05 + 0j), 1.2, 0.8, (-0.02,))
    assert info["residual"] < 1e-12
    assert np.abs(psi - psi[0]).max() < 1e-12          # stays flat
    roots = homogeneous_steady_states(1.2, 0.8)
    assert abs(np.abs(psi[0]) ** 2 - roots[0]) < 1e-10


def test_soliton_converges_and_the_independent_evolver_confirms_it():
    """Two code paths, one state: the Newton root of the stationary
    equation barely moves under the split-step time evolver."""
    sol, info = _soliton()
    assert info["residual"] < 1e-12
    ev = lle_evolve(sol, F, ALPHA, (D2,), t_end=20.0, dt=0.005)
    assert np.abs(ev - sol).max() < 5e-3
    # and it is a localized state on the lower background
    peak = float(np.abs(sol).max() ** 2)
    ipk = int(np.argmax(np.abs(sol)))
    bg = float(np.abs(sol[(ipk + N // 2) % N]) ** 2)
    roots = homogeneous_steady_states(F, ALPHA)
    assert peak > 5.0 * roots[0]
    assert abs(bg - roots[0]) < 1e-2


def test_soliton_translation_mode_is_an_exact_zero_mode():
    """Broken translation symmetry: d psi / d theta is annihilated by
    the linearization, so the drift matrix is exactly marginal."""
    sol, _ = _soliton()
    M, modes = fluctuation_matrix(sol, ALPHA, (D2,))
    lam = np.linalg.eigvals(M).real.max()
    assert abs(lam) < 1e-8                     # marginal, not unstable
    k = np.fft.fftfreq(N, d=1.0 / N)
    dpsi = np.fft.ifft(1j * k * np.fft.fft(sol))
    z = np.concatenate([np.fft.fft(dpsi) / N,
                        np.conj(np.fft.fft(dpsi)) / N])
    assert np.abs(M @ z).max() / np.abs(z).max() < 1e-9


def test_crystal_equals_the_rescaled_single_soliton():
    """theta -> N theta maps the 2-pulse crystal at dispersion d2 onto
    the single soliton at dispersion 4 d2, grid point for grid point."""
    n_p = 2
    seed_c = soliton_seed(N, F, ALPHA, (D2,), n_pulses=n_p)
    crystal, info_c = newton_state(seed_c, F, ALPHA, (D2,), tol=1e-11)
    assert info_c["residual"] < 1e-11
    # exact discrete periodicity: only modes divisible by n_p survive
    ck = np.fft.fft(crystal) / N
    kgrid = np.fft.fftfreq(N, d=1.0 / N).astype(int)
    assert np.abs(ck[kgrid % n_p != 0]).max() < 1e-10
    # and it is the single soliton of the rescaled equation, resampled
    seed_1 = soliton_seed(N, F, ALPHA, (D2 * n_p ** 2,))
    single, _ = newton_state(seed_1, F, ALPHA, (D2 * n_p ** 2,),
                             tol=1e-11)
    resampled = single[(n_p * np.arange(N)) % N]
    # align the free positions before comparing
    pk_c = int(np.argmax(np.abs(crystal)))
    pk_r = int(np.argmax(np.abs(resampled)))
    aligned = np.roll(resampled, pk_c - pk_r)
    assert np.abs(np.abs(aligned) - np.abs(crystal)).max() < 1e-8


def test_spectra_refuse_the_marginal_soliton_unless_told():
    sol, _ = _soliton()
    M, modes = fluctuation_matrix(sol, ALPHA, (D2,))
    i0 = int(np.where(modes == 0)[0][0])
    with pytest.raises(ValueError, match="marginal"):
        output_quadrature_variance(M, 0.6, 1.0, i0, modes.size)
    v = output_quadrature_variance(M, 0.6, 1.0, i0, modes.size,
                                   allow_marginal=True)
    assert np.isfinite(v) and v > 0.0
    # a genuinely unstable matrix stays refused, flag or no flag
    bad = M + 0.5 * np.eye(M.shape[0])
    with pytest.raises(ValueError, match="unstable"):
        output_quadrature_variance(bad, 0.6, 1.0, i0, modes.size,
                                   allow_marginal=True)


def test_soliton_shows_sideband_squeezing_below_vacuum():
    """Physics smoke test on top of the exact anchors: a truncated,
    strictly stable mode set around the soliton squeezes some phase of
    the pump line below vacuum."""
    sol, _ = _soliton()
    M, modes = fluctuation_matrix(sol, ALPHA, (D2,), modes=range(-8, 9))
    assert np.linalg.eigvals(M).real.max() < 0.0
    i0 = int(np.where(modes == 0)[0][0])
    best = min(output_quadrature_variance(M, 0.9, 0.5, i0, modes.size,
                                          phi=p)
               for p in np.linspace(0.0, np.pi, 40))
    assert best < 0.5


def test_continuation_returns_only_verified_states():
    sol, _ = _soliton()
    alphas = [3.0, 3.05, 3.1, 3.15]
    states, infos = continuation(sol, F, alphas, (D2,))
    assert len(states) == len(alphas)
    assert all(i["residual"] < 1e-12 for i in infos)
    # branch continuity: successive states stay close
    for a, b in zip(states[:-1], states[1:]):
        assert np.abs(b - a).max() < 1.0


def test_normal_dispersion_seed_is_refused():
    with pytest.raises(ValueError):
        soliton_seed(N, F, ALPHA, (0.25,))


def test_unconverged_newton_refuses_to_return():
    with pytest.raises(RuntimeError):
        newton_state(soliton_seed(N, F, ALPHA, (D2,)), F, ALPHA, (D2,),
                     maxiter=1)
