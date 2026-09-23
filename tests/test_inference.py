"""Source-inference anchors: the closed form is the exact inverse of
the package's own `detected_variance` (two code paths, machine
precision, over a grid of sources and efficiencies); the pure-state
product V_s V_a = 1/4 is restored exactly by the inferred source;
eta = 1 round-trips as lossless; every refusal (no antisqueezing, no
squeezing, sub-vacuum uncertainty product) is pinned; the error bars
of the closed form match seeded Monte Carlo; and the published
Ulanov et al. pair runs to a sane efficiency without any number
being asserted for the paper."""
import numpy as np
import pytest

from sqzcomb import detected_variance, infer_source, squeezing_db

VAC = 0.5


def _pair_db(r, eta):
    v_s, v_a = VAC * np.exp(-2.0 * r), VAC * np.exp(2.0 * r)
    S = detected_variance(v_s, efficiency=eta)
    A = detected_variance(v_a, efficiency=eta)
    return squeezing_db(S), squeezing_db(A)


def test_exact_inverse_of_detected_variance():
    for r in (0.2, 0.8, 1.5, 2.5):
        for eta in (0.15, 0.5, 0.9, 1.0):
            s_db, a_db = _pair_db(r, eta)
            out = infer_source(s_db, a_db)
            assert abs(out["eta"] - eta) < 1e-9
            assert abs(out["v_s"] - VAC * np.exp(-2.0 * r)) < 1e-12
            assert abs(out["v_a"] - VAC * np.exp(2.0 * r)) < 1e-9
            assert abs(out["v_s"] * out["v_a"] - 0.25) < 1e-9
            assert abs(out["sq_db_source"]
                       + out["anti_db_source"]) < 1e-6


def test_refusals_cover_the_algebra():
    with pytest.raises(ValueError, match="antisqueezed"):
        infer_source(-1.0, -0.1)
    with pytest.raises(ValueError, match="below shot noise"):
        infer_source(0.5, 3.0)
    # S*A below the vacuum product: impossible after loss
    with pytest.raises(ValueError, match="uncertainty product"):
        infer_source(-3.0, 1.0)
    with pytest.raises(ValueError, match="finite"):
        infer_source(np.nan, 3.0)
    with pytest.raises(ValueError, match="positive"):
        infer_source(-1.0, 3.0, sigma_db=-0.1)


def test_sigma_matches_monte_carlo():
    s_db, a_db = _pair_db(1.2, 0.6)
    sig = 0.05
    out = infer_source(s_db, a_db, sigma_db=sig)
    rng = np.random.default_rng(9)
    etas, srcs = [], []
    for _ in range(600):
        try:
            o = infer_source(s_db + sig * rng.standard_normal(),
                             a_db + sig * rng.standard_normal())
        except ValueError:
            continue
        etas.append(o["eta"])
        srcs.append(o["sq_db_source"])
    assert np.isclose(np.std(etas, ddof=1), out["eta_sigma"],
                      rtol=0.2)
    assert np.isclose(np.std(srcs, ddof=1),
                      out["sq_db_source_sigma"], rtol=0.2)


def test_published_pair_runs_sanely():
    """The directly measured pair of Ulanov et al., Nat. Commun. 16,
    10791 (2025) -- 1.71 dB squeezing, 5.54 dB antisqueezing -- must
    invert to a physical efficiency and a stronger inferred source.
    No number from the paper's own (different-reference-plane)
    inference is asserted; this checks OUR model runs their
    measurement to a consistent answer."""
    out = infer_source(-1.71, 5.54)
    assert 0.0 < out["eta"] < 1.0
    assert out["sq_db_source"] < -1.71        # source beats detected
    assert abs(out["v_s"] * out["v_a"] - 0.25) < 1e-9


# --- impure sources and phase jitter (0.13) ------------------------------

def _measure(r, P, eta, theta):
    from sqzcomb.detection import phase_noise_variance
    v_s, v_a = VAC / P * np.exp(-2.0 * r), VAC / P * np.exp(2.0 * r)
    S = detected_variance(v_s, efficiency=eta)
    A = detected_variance(v_a, efficiency=eta)
    return (squeezing_db(phase_noise_variance(S, A, theta)),
            squeezing_db(phase_noise_variance(A, S, theta)), v_s, v_a)


def test_pure_default_is_unchanged_digit_for_digit():
    for r, eta in ((0.4, 0.3), (1.2, 0.8)):
        s_db, a_db = _pair_db(r, eta)
        a = infer_source(s_db, a_db)
        b = infer_source(s_db, a_db, purity=1.0, theta_rms=0.0)
        assert a == b


def test_impure_jittered_round_trip():
    # known purity and jitter: the inversion must return the generating
    # efficiency and source exactly (on the right branch when the
    # answer is two-valued, which the function must then flag)
    n_amb = 0
    for r in (0.3, 0.8, 1.5):
        for P in (1.0, 0.9, 0.6, 0.3):
            for eta in (0.2, 0.5, 0.9, 1.0):
                for th in (0.0, 0.05, 0.15):
                    s_db, a_db, v_s, v_a = _measure(r, P, eta, th)
                    if s_db >= 0.0:
                        continue            # no squeezing left to see
                    try:
                        outs = [infer_source(s_db, a_db, purity=P,
                                             theta_rms=th)]
                    except ValueError as exc:
                        assert "two efficiencies" in str(exc)
                        n_amb += 1
                        outs = [infer_source(s_db, a_db, purity=P,
                                             theta_rms=th, branch=b)
                                for b in ("low", "high")]
                    o = min(outs, key=lambda o: abs(o["eta"] - eta))
                    assert o["eta"] == pytest.approx(eta, abs=1e-9)
                    assert o["v_s"] == pytest.approx(v_s, rel=1e-9)
                    assert o["v_s"] * o["v_a"] == pytest.approx(
                        0.25 / P ** 2, rel=1e-9)
    assert n_amb > 0      # the two-valued case is real, not hypothetical


def test_assuming_purity_flatters_the_source():
    # true source: purity 0.7; the pure-state inference of the same
    # measurement reports lower efficiency and stronger squeezing
    s_db, a_db, v_s, _ = _measure(1.0, 0.7, 0.6, 0.0)
    pure = infer_source(s_db, a_db)
    true = infer_source(s_db, a_db, purity=0.7)
    assert true["eta"] == pytest.approx(0.6, abs=1e-9)
    assert pure["eta"] < true["eta"]
    assert pure["sq_db_source"] < true["sq_db_source"] - 1.0


def test_jitter_and_purity_refusals():
    s_db, a_db = _pair_db(1.0, 0.7)
    with pytest.raises(ValueError, match="jitter"):
        infer_source(s_db, a_db, theta_rms=1.0)
    with pytest.raises(ValueError, match="purity"):
        infer_source(s_db, a_db, purity=0.0)
    with pytest.raises(ValueError, match="branch"):
        infer_source(s_db, a_db, branch="middle")
    with pytest.raises(ValueError, match="theta_rms"):
        infer_source(s_db, a_db, theta_rms=-0.1)


def test_purity_near_one_is_continuous():
    # the quadratic must not lose accuracy as the purity approaches 1
    ref = infer_source(-1.71, 5.54)
    for P in (1 - 1e-6, 1 - 1e-12, 1 - 1e-15):
        out = infer_source(-1.71, 5.54, purity=P)
        assert out["eta"] == pytest.approx(ref["eta"], rel=1e-5)
