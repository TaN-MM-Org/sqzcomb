"""Lab-planning anchors: the planner's predicted error bars match the
fit's reported error bars on the same design (two code paths of one
matrix); the single-quadrature degeneracy is exactly rank-deficient in
the planner AND refused by the fit; the greedy frequency design obeys
its own recomputed rule and never loses to a random subset; the g0
calibration round-trips through `threshold_power` to machine
precision with an exactly propagated error bar; and the spectra CSV
round trip is exact."""
import numpy as np
import pytest

from sqzcomb import (design_noise_frequencies, fit_noise_spectra,
                     load_spectra_csv, noise_spectrum_db,
                     plan_noise_measurement, ring_from_threshold,
                     save_spectra_csv, threshold_power)

MU, ETA, KAPPA = 0.55, 0.62, 12e6
F = np.linspace(0.5e6, 30e6, 25)
SIG = 0.05


def test_plan_matches_fit_covariance():
    """Noise-free synthetic spectra: the fit lands on the truth, and
    its reported sigmas equal the planner's prediction -- both are
    (J^T W J)^-1 at the same point."""
    sq = noise_spectrum_db(MU, ETA, KAPPA, F, "squeezed")
    an = noise_spectrum_db(MU, ETA, KAPPA, F, "anti")
    fit = fit_noise_spectra(F, sq, an, sigma_db=SIG)
    assert abs(fit.mu - MU) < 1e-6
    plan = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG)
    assert plan["identifiable"]
    for name in ("mu", "eta", "kappa_hz"):
        assert abs(plan["sigma"][name] - fit.sigma[name]) \
            < 0.02 * fit.sigma[name]


def test_single_quadrature_degeneracy_is_exact_rank_deficiency():
    """One Lorentzian carries two shape numbers for three unknowns:
    the information matrix is rank-deficient however many frequencies
    are recorded -- the planner reports it, and `fit_noise_spectra`
    refuses the same design after the fact."""
    plan = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG,
                                  quadratures=("squeezed",))
    assert not plan["identifiable"]
    # unit-free (correlation-scaled) singular values: exact functional
    # rank deficiency, not a units artifact
    fi = plan["fisher"]
    d = np.sqrt(np.diag(fi))
    sv = np.linalg.svd(fi / np.outer(d, d), compute_uv=False)
    assert sv[-1] < 1e-8 * sv[0]
    both = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG)
    fb = both["fisher"]
    db = np.sqrt(np.diag(fb))
    svb = np.linalg.svd(fb / np.outer(db, db), compute_uv=False)
    assert svb[-1] > 1e-6 * svb[0]
    sq = noise_spectrum_db(MU, ETA, KAPPA, F, "squeezed")
    with pytest.raises(ValueError, match="single Lorentzian"):
        fit_noise_spectra(F, sq_db=sq)
    # with kappa known independently, the same single trace IS enough:
    # both tools must agree on that too
    plan2 = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG,
                                   quadratures=("squeezed",),
                                   fit_kappa=False)
    assert plan2["identifiable"]
    fit2 = fit_noise_spectra(F, sq_db=sq, sigma_db=SIG, kappa_hz=KAPPA)
    assert abs(fit2.mu - MU) < 1e-5


def test_design_greedy_invariant_and_quality():
    out = design_noise_frequencies(MU, ETA, KAPPA, F, 6, sigma_db=SIG)
    idx = out["indices"]
    assert len(idx) == 6 and len(set(idx)) == 6
    assert out["sigma"] is not None

    def logdet_of(subset):
        plan = plan_noise_measurement(MU, ETA, KAPPA, F[list(subset)],
                                      sigma_db=SIG)
        s, d = np.linalg.slogdet(plan["fisher"])
        return d if s > 0 else -np.inf

    best = logdet_of(idx)
    rng = np.random.default_rng(5)
    for _ in range(30):
        assert best >= logdet_of(rng.choice(F.size, 6, replace=False)) \
            - 1e-9


def test_ring_from_threshold_round_trip_and_sigma():
    """The calibration must invert the package's own `threshold_power`
    exactly, and its error bar is the exact delta-method combination
    of the measured errors (checked against finite differences)."""
    kappa_hz, eta_esc, lam = 25e6, 0.8, 1.55e-6
    p_meas = 3.2e-3
    spec, sig = ring_from_threshold(kappa_hz, eta_esc, p_meas, lam,
                                    "test bench measurement",
                                    sigma_kappa_hz=0.5e6,
                                    sigma_threshold_w=0.1e-3)
    assert abs(threshold_power(spec) - p_meas) < 1e-12 * p_meas
    assert "ring_from_threshold" in spec.reference
    assert "test bench" in spec.reference
    # finite-difference check of the propagated sigma
    g0 = spec.g0_hz
    dk = ring_from_threshold(kappa_hz + 1.0, eta_esc, p_meas, lam,
                             "test bench measurement")[0].g0_hz - g0
    dp = ring_from_threshold(kappa_hz, eta_esc, p_meas + 1e-9, lam,
                             "test bench measurement")[0].g0_hz - g0
    sig_fd = np.hypot(dk * 0.5e6, dp / 1e-9 * 0.1e-3)
    assert abs(sig - sig_fd) < 1e-3 * sig_fd
    # off-minimum threshold: alpha enters only through the exact
    # F_th^2 = 1 + (alpha - 1)^2 identity
    spec2, _ = ring_from_threshold(kappa_hz, eta_esc, p_meas, lam,
                                   "test bench measurement", alpha=2.0)
    assert abs(spec2.g0_hz / g0 - 2.0) < 1e-12
    with pytest.raises(ValueError, match="reference"):
        ring_from_threshold(kappa_hz, eta_esc, p_meas, lam, "  ")


def test_spectra_csv_round_trip_and_refusals(tmp_path):
    sq = noise_spectrum_db(MU, ETA, KAPPA, F, "squeezed")
    an = noise_spectrum_db(MU, ETA, KAPPA, F, "anti")
    sig = np.full(F.size, 0.07)
    path = tmp_path / "spectra.csv"
    save_spectra_csv(path, F, sq, an, sig)
    f2, s2, a2, g2 = load_spectra_csv(path)
    assert np.array_equal(f2, F)
    assert np.array_equal(s2, sq)
    assert np.array_equal(a2, an)
    assert np.array_equal(g2, sig)
    bad = tmp_path / "bad.csv"
    bad.write_text("nope,header\n1,2\n")
    with pytest.raises(ValueError, match="header"):
        load_spectra_csv(bad)
    with pytest.raises(ValueError, match="same length"):
        save_spectra_csv(path, F, sq[:-1], an)


def test_plan_refusals():
    with pytest.raises(ValueError, match="mu"):
        plan_noise_measurement(1.5, ETA, KAPPA, F)
    with pytest.raises(ValueError, match="quadratures"):
        plan_noise_measurement(MU, ETA, KAPPA, F, quadratures=("x",))
    with pytest.raises(ValueError, match="positive"):
        plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=-1.0)
    with pytest.raises(ValueError, match="n_pick"):
        design_noise_frequencies(MU, ETA, KAPPA, F, 0)
