"""`gain_for_squeezing(..., delta=)` (new in 0.19): the gain a target
squeezing needs when the squeezed mode is detuned.

Independent check: the output noise of the detuned single mode written
out here in real quadratures (x, p), a 2 x 2 calculation that does not
use the package's complex doubled form:

    d/dt (x, p) = A (x, p) + noise,
    A = [[-(1 - mu), -delta], [delta, -(1 + mu)]]

(from da/dt = (-1 + i delta) a + mu a^dag, the convention of
`single_mode_parametric`), G = (-i omega - A)^-1, and the output matrix
S = (1/2) Re[I - 2 eta (G + G^H) + 4 eta G G^H], whose eigenvalues are
the smallest and largest variance over the angle.
"""
import numpy as np
import pytest

import sqzcomb as sc


def _quadrature_extremes_2x2(mu, delta, eta, omega):
    A = np.array([[-(1.0 - mu), -delta], [delta, -(1.0 + mu)]])
    G = np.linalg.inv(-1j * omega * np.eye(2) - A)
    S = 0.5 * (np.eye(2) - 2.0 * eta * (G + G.conj().T)
               + 4.0 * eta * G @ G.conj().T)
    return np.linalg.eigvalsh(S.real)


def _detected_db(mu, delta, eta, omega, theta, dark=0.0, dir_=False):
    v_min, v_max = _quadrature_extremes_2x2(mu, delta, eta, omega)
    c = np.exp(-2.0 * theta ** 2)
    s = 0.5 * (v_min + v_max) - 0.5 * (v_max - v_min) * c
    a = 0.5 * (v_min + v_max) + 0.5 * (v_max - v_min) * c
    out = []
    for v in (s, a):
        v = v + dark
        if dir_:
            v = 0.5 * v / (0.5 + dark)
        out.append(10.0 * np.log10(v / 0.5))
    return out


CASES = [(-4.0, 0.72, 0.2, 0.0, 0.3, 0.0, False),
         (-4.0, 0.72, 0.2, 0.02, -0.3, 0.0, False),
         (-3.0, 0.6, 1.0, 0.03, 1.0, 0.01, False),
         (-3.0, 0.6, 0.0, 0.01, 0.5, 0.01, True)]


@pytest.mark.parametrize("target, eta, omega, theta, delta, dark, dir_",
                         CASES)
def test_detuned_answer_against_the_real_quadrature_model(
        target, eta, omega, theta, delta, dark, dir_):
    r = sc.gain_for_squeezing(target, eta, omega, theta_rms=theta,
                              dark_noise=dark, dark_in_reference=dir_,
                              delta=delta)
    assert r["mu_threshold"] == pytest.approx(np.sqrt(1.0 + delta ** 2),
                                              rel=1e-15)
    sq, an = _detected_db(r["mu"], delta, eta, omega, theta, dark, dir_)
    assert sq == pytest.approx(target, abs=1e-9)        # measured 3e-15
    assert r["detected_db"] == pytest.approx(target, abs=1e-9)  # measured 2e-15
    assert r["anti_db"] == pytest.approx(an, abs=1e-9)  # measured 4e-15
    best, _ = _detected_db(r["mu_best"], delta, eta, omega, theta, dark,
                           dir_)
    assert r["best_db"] == pytest.approx(best, abs=1e-9)  # measured 9e-14
    # a scan of 4001 gains up to mu_max by the 2 x 2 model: nothing
    # below the reported best, and the first gain reaching the target
    # sits just above the answer
    mu_th = np.sqrt(1.0 + delta ** 2)
    grid = np.linspace(0.0, (1.0 - 1e-9) * mu_th, 4001)[1:]
    sq_grid = np.array([_detected_db(m, delta, eta, omega, theta, dark,
                                     dir_)[0] for m in grid])
    assert np.min(sq_grid) >= r["best_db"] - 1e-9
    first = grid[np.argmax(sq_grid <= target)]
    assert r["mu"] <= first <= r["mu"] + (grid[1] - grid[0])


def test_threshold_moves_to_sqrt_one_plus_delta_squared():
    for delta in (0.3, 1.0, 2.0):
        mu_th = np.sqrt(1.0 + delta ** 2)
        lam = np.linalg.eigvals(sc.single_mode_parametric(mu_th, delta))
        assert np.max(lam.real) == pytest.approx(0.0, abs=1e-12)
        lam = np.linalg.eigvals(sc.single_mode_parametric(0.999 * mu_th,
                                                          delta))
        assert np.max(lam.real) < 0.0
        r = sc.gain_for_squeezing(-3.0, 0.8, 0.5, delta=delta)
        if delta < 1.5:
            # without jitter it keeps improving up to the threshold
            assert r["best_at_limit"]
            assert r["mu_best"] == (1.0 - 1e-9) * mu_th
        else:
            # a large detuning moves the squeezing to a sideband: at
            # omega = 0.5 the deepest value comes before threshold
            # (checked against the 2 x 2 model in the next lines)
            assert not r["best_at_limit"]
        best, _ = _detected_db(r["mu_best"], delta, 0.8, 0.5, 0.0)
        grid = np.linspace(0.5, 1.0 - 1e-9, 2001) * mu_th
        scan = [_detected_db(m, delta, 0.8, 0.5, 0.0)[0] for m in grid]
        assert r["best_db"] == pytest.approx(best, abs=1e-9)  # measured 7e-14
        # measured: 2e-14 above the scan's own value at mu_max (rounding),
        # 6e-8 dB below the scan for delta = 2
        assert r["best_db"] <= min(scan) + 1e-12
    with pytest.raises(ValueError, match="mu_max"):
        sc.gain_for_squeezing(-3.0, 0.8, 0.5, delta=1.0, mu_max=1.5)


def test_pure_output_gives_the_same_detected_pairs_for_any_detuning():
    """Without loss the output of the single mode is pure at every
    analysis frequency (V_min V_max = 1/4), so the detected pairs lie
    on one curve whatever the detuning: the same target needs the same
    antisqueezing, and with jitter the deepest squeezing is the same.
    Only the gain that gets there changes."""
    for mu, delta, omega in ((0.5, 0.3, 0.2), (0.9, 1.0, 1.5),
                             (1.2, 2.0, 0.0)):
        v = _quadrature_extremes_2x2(mu, delta, 1.0, omega)
        assert v[0] * v[1] == pytest.approx(0.25, rel=1e-12)  # measured 7e-16
    base = sc.gain_for_squeezing(-4.0, 0.72, 0.2, theta_rms=0.02)
    for delta in (0.3, 1.0):
        r = sc.gain_for_squeezing(-4.0, 0.72, 0.2, theta_rms=0.02,
                                  delta=delta)
        assert r["anti_db"] == pytest.approx(base["anti_db"],
                                             abs=1e-9)  # measured 4e-15
        assert r["best_db"] == pytest.approx(base["best_db"],
                                             abs=1e-9)  # measured 2e-15
        assert r["mu"] > base["mu"]


def test_zero_detuning_is_unchanged_and_small_detuning_is_continuous():
    for theta in (0.0, 0.02):
        a = sc.gain_for_squeezing(-4.0, 0.72, 0.2, theta_rms=theta)
        b = sc.gain_for_squeezing(-4.0, 0.72, 0.2, theta_rms=theta,
                                  delta=0.0)
        assert a == b
        assert "mu_threshold" not in a
        c = sc.gain_for_squeezing(-4.0, 0.72, 0.2, theta_rms=theta,
                                  delta=1e-6)
        for key in ("mu", "anti_db", "best_db"):
            assert c[key] == pytest.approx(a[key], rel=1e-9)  # measured 7e-13


def test_detuned_refusals():
    with pytest.raises(ValueError, match="delta"):
        sc.gain_for_squeezing(-3.0, 0.8, delta=np.nan)
    with pytest.raises(ValueError, match="cannot be reached"):
        sc.gain_for_squeezing(-8.0, 0.8, 0.5, theta_rms=0.05, delta=0.5)
