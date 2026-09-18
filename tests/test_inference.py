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
