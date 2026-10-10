"""0.15 anchors.

Regression tests for the four inputs that gave silent wrong answers in
0.14.0 (each fails on 0.14.0 and passes now), and the checks of the two
new features: the shot-noise calibration uncertainty (against the
generalized least-squares matrix formula built here from the closed-form
spectra, against closed-form derivatives of the pure-state inversion,
and against seeded Monte Carlo with a common offset) and the best fixed
local-oscillator angle for a band (against the eigenvectors of the
band-averaged output covariance, a separate code path, and against the
per-frequency optimum)."""
import numpy as np
import pytest

from sqzcomb import (classical_noise_variance_ports, fit_noise_spectra,
                     infer_source, logarithmic_negativity,
                     lossy_channel_xxpp, noise_spectrum_db,
                     optimal_band_quadrature, optimal_quadrature,
                     output_covariance_xxpp, output_entanglement,
                     output_quadrature_variance, output_variance_ports,
                     photonic_molecule, plan_noise_measurement,
                     quadrature_extremes, single_mode_parametric,
                     soliton_seed, entanglement_report)

TWIN = np.array([[-1, 0, 0, 0.5], [0, -1, 0.5, 0],
                 [0, 0.5, -1, 0], [0.5, 0, 0, -1]], dtype=complex)
MU, ETA, KAPPA = 0.55, 0.62, 12e6
F = np.linspace(0.5e6, 30e6, 25)
SIG = 0.05


# ----------------------------------------------------------------------
# Fixed in 0.15


def test_repeated_read_mode_is_refused_not_doubled():
    """0.14.0: mode_index=[1, 1] (or port_mode=[1, 1] with the default
    mode_index) weighted mode 1 twice and returned 3.84444 instead of
    the single-mode 1.92222 for the twin beam (eta = 0.8, omega = 0,
    phi = 0.3); the same doubling hit classical_noise_variance_ports."""
    g = [1.0, 1.0]
    single = output_variance_ports(TWIN, g, 0.8, [0, 1], 0.0, phi=0.3,
                                   mode_index=1)
    # the independent single-port path agrees on the single-mode value
    ref = output_quadrature_variance(TWIN, 0.8, 0.0, 1, 2, phi=0.3)
    assert abs(single - ref) < 1e-13
    assert abs(single - 1.9222222222222) < 1e-12
    for kw in ({"port_mode": [0, 1], "mode_index": [1, 1]},
               {"port_mode": [1, 1]}):
        with pytest.raises(ValueError, match="more than once"):
            output_variance_ports(TWIN, g, 0.8, omega=0.0, phi=0.3, **kw)
    # a repeated port with distinct modes read stays allowed (the bus
    # couples to each mode once) and gives the same number as before
    assert output_variance_ports(TWIN, g, 0.8, [1, 1, 0], 0.0, phi=0.3,
                                 mode_index=1) == single
    M, gm = photonic_molecule(0.5, 1.0)
    b = [np.array([1.0, 0.5j, 1.0, -0.5j])]
    with pytest.raises(ValueError, match="more than once"):
        classical_noise_variance_ports(M, gm, 0.7, [1, 1], 0.4, b, [1.0],
                                       phi=0.2)
    one = classical_noise_variance_ports(M, gm, 0.7, 1, 0.4, b, [1.0],
                                         phi=0.2)
    assert classical_noise_variance_ports(
        M, gm, 0.7, [1, 1], 0.4, b, [1.0], phi=0.2, mode_index=1) == one


def test_noise_spectrum_db_refuses_unknown_quadrature():
    """0.14.0: any name but "squeezed" gave the antisqueezed curve, so
    quadrature="Squeezed" returned +8.141 dB at 1 MHz where the
    squeezed trace is -5.210 dB (mu 0.5, eta 0.8, kappa 10 MHz)."""
    f = np.array([1e6, 5e6])
    sq = noise_spectrum_db(0.5, 0.8, 10e6, f, "squeezed")
    om = 2.0 * f / 10e6
    closed = 10 * np.log10(1 - 4 * 0.8 * 0.5 / (1.5 ** 2 + om ** 2))
    assert np.allclose(sq, closed, atol=1e-12)
    assert abs(sq[0] + 5.20986) < 1e-4
    for bad in ("Squeezed", "sq", "x", None):
        with pytest.raises(ValueError, match="quadrature"):
            noise_spectrum_db(0.5, 0.8, 10e6, f, bad)
    an = noise_spectrum_db(0.5, 0.8, 10e6, f, "anti")
    assert np.array_equal(noise_spectrum_db(0.5, 0.8, 10e6, f,
                                            "antisqueezed"), an)
    closed_an = 10 * np.log10(1 + 4 * 0.8 * 0.5 / (0.5 ** 2 + om ** 2))
    assert np.allclose(an, closed_an, atol=1e-12)


def test_nan_efficiency_or_covariance_gives_no_entanglement_verdict():
    """0.14.0: detection_efficiency=nan passed the lossy channel, and
    the PPT formula turned the NaN matrix into E_N = max(0, nan) = 0,
    "not entangled", for a pair whose E_N is 1.2417 (0.6887 at 70 %)."""
    assert abs(output_entanglement(TWIN, 0.8, 0.0, 0, 1, 2)
               - 1.2417131323) < 1e-9
    for eff in (np.nan, [0.7, np.nan]):
        with pytest.raises(ValueError, match="efficiency"):
            output_entanglement(TWIN, 0.8, 0.0, 0, 1, 2,
                                detection_efficiency=eff)
        with pytest.raises(ValueError, match="efficiency"):
            lossy_channel_xxpp(np.eye(4), eff)
    s = np.eye(4)
    s[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        logarithmic_negativity(s)
    with pytest.raises(ValueError, match="non-finite"):
        entanglement_report(s)
    # valid inputs are untouched: vacuum stays separable, E_N = 0
    assert logarithmic_negativity(np.eye(4)) == 0.0


def test_soliton_seed_refuses_nonpositive_detuning():
    """0.14.0: alpha < 0 returned an all-NaN seed (a RuntimeWarning
    only), alpha = 0 raised ZeroDivisionError."""
    for a in (-0.5, 0.0, np.nan):
        with pytest.raises(ValueError, match="positive detuning"):
            soliton_seed(16, 1.5, a, (-0.1,))
    with pytest.raises(ValueError, match="n_pulses"):
        soliton_seed(16, 1.9, 3.0, (-0.25,), n_pulses=0)
    assert np.all(np.isfinite(soliton_seed(16, 1.9, 3.0, (-0.25,))))


# ----------------------------------------------------------------------
# optimal_quadrature: same numbers, fewer matrix inversions


def test_optimal_quadrature_is_bit_identical_to_the_per_angle_path():
    M = single_mode_parametric(0.6 * np.exp(0.4j), delta=0.3)
    ws = np.array([0.0, 0.5, 2.0])
    for kw in ({}, {"n_th_port": 0.2, "n_th_loss": 0.7}):
        ref = quadrature_extremes(lambda p: np.array([
            output_quadrature_variance(M, 0.8, x, 0, 1, phi=p, **kw)
            for x in ws]))
        new = optimal_quadrature(M, 0.8, ws, 0, 1, **kw)
        for k in ("v_min", "v_max", "phi_min", "phi_max", "mean"):
            assert np.array_equal(new[k], ref[k])
    ref = quadrature_extremes(lambda p: np.array([
        output_quadrature_variance(TWIN, 0.8, x, 0, 2, phi=p,
                                   mode_index_b=1) for x in ws]))
    new = optimal_quadrature(TWIN, 0.8, ws, 0, 2, mode_index_b=1)
    assert np.array_equal(new["v_min"], ref["v_min"])


# ----------------------------------------------------------------------
# New: the best fixed angle for a band


def _band_cov_eig(M, eta, ws, wt):
    """Smallest eigenvalue and its angle of the weighted band average of
    the 2x2 (x, p) output covariance from `output_covariance_xxpp`."""
    blocks = [output_covariance_xxpp(M, eta, w, 1) for w in ws]
    avg = sum(c * b for c, b in zip(wt, blocks)) / np.sum(wt)
    val, vec = np.linalg.eigh(avg)
    return val[0], np.mod(np.arctan2(vec[1, 0], vec[0, 0]), np.pi)


def test_band_angle_matches_band_averaged_covariance():
    M = single_mode_parametric(0.6, delta=0.3)
    ws = np.linspace(0.0, 2.0, 21)
    rng = np.random.default_rng(4)
    for wt in (np.ones(ws.size), rng.uniform(0.1, 2.0, ws.size)):
        r = optimal_band_quadrature(M, 0.8, ws, 0, 1, weights=wt)
        v_ref, phi_ref = _band_cov_eig(M, 0.8, ws, wt)
        assert abs(r["v_band"] - v_ref) < 1e-12
        assert abs(r["phi"] - phi_ref) < 1e-9
        # the per-frequency values at phi are the scalar path's
        for x, v in zip(ws, r["v"]):
            assert abs(v - output_quadrature_variance(
                M, 0.8, x, 0, 1, phi=r["phi"])) < 1e-15
        assert abs(np.dot(wt, r["v"]) / wt.sum() - r["v_band"]) < 1e-12
        # holding one angle never beats the per-frequency optimum
        assert np.all(r["penalty_db"] > -1e-12)
        assert r["penalty_db"].max() > 0.05      # and here it costs
        # no angle of a dense scan does better on the band
        scan = [np.dot(wt, [output_quadrature_variance(
            M, 0.8, x, 0, 1, phi=p) for x in ws]) / wt.sum()
            for p in np.linspace(0.0, np.pi, 181)]
        assert min(scan) >= r["v_band"] - 1e-14


def test_band_angle_closed_forms_and_limits():
    # real gain, no detuning: pi/2 is best at every frequency, so the
    # fixed angle costs nothing and equals the closed form
    ws = np.array([0.0, 0.7, 1.5, 3.0])
    r = optimal_band_quadrature(single_mode_parametric(0.5), 0.9, ws, 0, 1)
    assert abs(r["phi"] - np.pi / 2) < 1e-12
    assert np.all(np.abs(r["penalty_db"]) < 1e-12)
    closed = 0.5 * (1 - 4 * 0.9 * 0.5 / (1.5 ** 2 + ws ** 2))
    assert np.allclose(r["v"], closed, rtol=1e-12, atol=0)
    # one frequency, or all weight on one frequency: the band answer is
    # that frequency's optimum
    M = single_mode_parametric(0.6, delta=0.3)
    one = optimal_band_quadrature(M, 0.8, [0.5], 0, 1)
    opt = optimal_quadrature(M, 0.8, 0.5, 0, 1)
    assert abs(one["v_band"] - opt["v_min"]) < 1e-15
    assert abs(one["phi"] - opt["phi_min"]) < 1e-12
    wt = np.zeros(ws.size)
    wt[2] = 3.0
    r = optimal_band_quadrature(M, 0.8, ws, 0, 1, weights=wt)
    opt = optimal_quadrature(M, 0.8, ws[2], 0, 1)
    assert abs(r["v_band"] - opt["v_min"]) < 1e-14
    assert abs(r["penalty_db"][2]) < 1e-12
    # the twin beam with a real gain: the best joint-quadrature angle
    # does not move with frequency, so the band angle loses nothing
    r = optimal_band_quadrature(TWIN, 0.8, ws, 0, 2, mode_index_b=1)
    o = optimal_quadrature(TWIN, 0.8, ws, 0, 2, mode_index_b=1)
    assert np.allclose(o["phi_min"], o["phi_min"][0], atol=1e-12)
    assert np.allclose(r["v"], o["v_min"], rtol=1e-12, atol=0)


def test_band_angle_refusals():
    M = single_mode_parametric(0.6, delta=0.3)
    ws = [0.0, 1.0, 2.0]
    with pytest.raises(ValueError, match="one value per frequency"):
        optimal_band_quadrature(M, 0.8, ws, 0, 1, weights=[1.0, 1.0])
    for bad in ([1.0, -1.0, 1.0], [0.0, 0.0, 0.0], [1.0, np.nan, 1.0]):
        with pytest.raises(ValueError, match="weights"):
            optimal_band_quadrature(M, 0.8, ws, 0, 1, weights=bad)
    for bad in ([], [0.0, np.inf], [[0.0, 1.0]]):
        with pytest.raises(ValueError, match="omegas"):
            optimal_band_quadrature(M, 0.8, bad, 0, 1)
    with pytest.raises(ValueError, match="eta"):
        optimal_band_quadrature(M, 1.2, ws, 0, 1)
    with pytest.raises(ValueError, match="unstable"):
        optimal_band_quadrature(single_mode_parametric(1.3), 0.8, ws, 0, 1)


# ----------------------------------------------------------------------
# New: the shot-noise calibration uncertainty


def _closed_db(p, f):
    """Closed-form single-mode spectra in dB (both traces), written
    here independently of the package's solver."""
    mu, eta, kap = p
    om = 2.0 * f / kap
    vs = 0.5 * (1 - 4 * eta * mu / ((1 + mu) ** 2 + om ** 2))
    va = 0.5 * (1 + 4 * eta * mu / ((1 - mu) ** 2 + om ** 2))
    return np.concatenate([10 * np.log10(vs / 0.5),
                           10 * np.log10(va / 0.5)])


def _gls_sigma(sig, csig):
    """sqrt(diag((J^T C^-1 J)^-1)) with C = sig^2 I + csig^2 (all
    ones), from the closed form and an explicit matrix inverse."""
    p0 = np.array([MU, ETA, KAPPA])
    J = np.empty((2 * F.size, 3))
    for j in range(3):
        h = 1e-6 * p0[j]
        dp = np.zeros(3)
        dp[j] = h
        J[:, j] = (_closed_db(p0 + dp, F) - _closed_db(p0 - dp, F)) / (2 * h)
    n = 2 * F.size
    C = sig ** 2 * np.eye(n) + csig ** 2 * np.ones((n, n))
    cov = np.linalg.inv(J.T @ np.linalg.solve(C, J))
    return np.sqrt(np.diag(cov))


def test_calibrated_fit_equals_the_correlated_error_formula():
    sq = noise_spectrum_db(MU, ETA, KAPPA, F, "squeezed")
    an = noise_spectrum_db(MU, ETA, KAPPA, F, "anti")
    n = 2 * F.size
    for csig in (0.02, 0.2):
        ref = _gls_sigma(SIG, csig)
        fit = fit_noise_spectra(F, sq, an, sigma_db=SIG,
                                calib_sigma_db=csig)
        assert abs(fit.mu - MU) < 1e-6 and abs(fit.eta - ETA) < 1e-6
        assert abs(fit.calib_offset_db) < 1e-6
        plan = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG,
                                      calib_sigma_db=csig)
        for i, name in enumerate(("mu", "eta", "kappa_hz")):
            assert plan["sigma"][name] == pytest.approx(ref[i], rel=1e-4)
            assert fit.sigma[name] == pytest.approx(ref[i], rel=0.02)
        # the constraint on the offset is one datum for one parameter
        assert fit.chi2_dof == n - 3
        # traces shifted by 0.1 dB and with noise: the fitted offset is
        # the closed-form optimum given the fitted parameters, and chi2
        # is r^T C^-1 r with C built here explicitly
        rng = np.random.default_rng(2)
        data = np.concatenate([sq, an]) + 0.1 + rng.normal(0, SIG, n)
        fit = fit_noise_spectra(F, data[:F.size], data[F.size:],
                                sigma_db=SIG, calib_sigma_db=csig)
        r = data - _closed_db([fit.mu, fit.eta, fit.kappa_hz], F)
        b_opt = (r.sum() / SIG ** 2) / (n / SIG ** 2 + 1 / csig ** 2)
        assert fit.calib_offset_db == pytest.approx(b_opt, rel=1e-6)
        C = SIG ** 2 * np.eye(n) + csig ** 2 * np.ones((n, n))
        assert fit.chi2 == pytest.approx(r @ np.linalg.solve(C, r),
                                         rel=1e-6)
    # a free offset (huge calibration error) absorbs a pure offset
    fit = fit_noise_spectra(F, sq + 0.1, an + 0.1, sigma_db=SIG,
                            calib_sigma_db=100.0)
    assert abs(fit.mu - MU) < 1e-6 and abs(fit.eta - ETA) < 1e-6
    assert abs(fit.kappa_hz - KAPPA) / KAPPA < 1e-6
    assert abs(fit.calib_offset_db - 0.1) < 1e-6
    # a vanishing calibration error gives back the plain fit
    plain = fit_noise_spectra(F, sq, an, sigma_db=SIG)
    tiny = fit_noise_spectra(F, sq, an, sigma_db=SIG, calib_sigma_db=1e-7)
    for name in ("mu", "eta", "kappa_hz"):
        assert tiny.sigma[name] == pytest.approx(plain.sigma[name],
                                                 rel=1e-6)
    assert plain.calib_offset_db is None


def test_calibrated_fit_error_bars_match_monte_carlo():
    """Every seeded trial shifts both traces by one common offset
    drawn with the calibration error. The plain fit's scatter is far
    larger than its error bar; the calibrated fit's matches it."""
    sq = noise_spectrum_db(MU, ETA, KAPPA, F, "squeezed")
    an = noise_spectrum_db(MU, ETA, KAPPA, F, "anti")
    csig = 0.2
    rng = np.random.default_rng(5)
    plain, cal = [], []
    for _ in range(30):
        b = rng.normal(0.0, csig)
        s = sq + b + rng.normal(0.0, SIG, F.size)
        a = an + b + rng.normal(0.0, SIG, F.size)
        f0 = fit_noise_spectra(F, s, a, sigma_db=SIG)
        f1 = fit_noise_spectra(F, s, a, sigma_db=SIG, calib_sigma_db=csig)
        plain.append([f0.mu, f0.eta])
        cal.append([f1.mu, f1.eta])
    rep_plain = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG)
    rep_cal = plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG,
                                     calib_sigma_db=csig)
    for i, name in enumerate(("mu", "eta")):
        emp_plain = np.std(np.asarray(plain)[:, i], ddof=1)
        emp_cal = np.std(np.asarray(cal)[:, i], ddof=1)
        assert emp_plain > 3.0 * rep_plain["sigma"][name]
        assert 0.6 * rep_cal["sigma"][name] < emp_cal \
            < 1.6 * rep_cal["sigma"][name]


def test_calibration_refusals():
    sq = noise_spectrum_db(MU, ETA, KAPPA, F, "squeezed")
    an = noise_spectrum_db(MU, ETA, KAPPA, F, "anti")
    with pytest.raises(ValueError, match="needs sigma_db"):
        fit_noise_spectra(F, sq, an, calib_sigma_db=0.1)
    for bad in (0.0, -0.1, np.nan):
        with pytest.raises(ValueError, match="calib_sigma_db"):
            fit_noise_spectra(F, sq, an, sigma_db=SIG, calib_sigma_db=bad)
        with pytest.raises(ValueError, match="calib_sigma_db"):
            plan_noise_measurement(MU, ETA, KAPPA, F, sigma_db=SIG,
                                   calib_sigma_db=bad)
        with pytest.raises(ValueError, match="calib_sigma_db"):
            infer_source(-1.71, 5.54, sigma_db=0.1, calib_sigma_db=bad)


def _closed_infer_grad(s_db, a_db):
    """Closed-form gradients of eta = -2 s a / (s + a) and of the
    source level 10 log10(-s / a) (s = S - 1/2, a = A - 1/2) with
    respect to the two measured dB values."""
    S, A = 0.5 * 10 ** (s_db / 10), 0.5 * 10 ** (a_db / 10)
    s, a = S - 0.5, A - 0.5
    dS, dA = S * np.log(10) / 10, A * np.log(10) / 10
    g_eta = np.array([-2 * a * a / (s + a) ** 2 * dS,
                      -2 * s * s / (s + a) ** 2 * dA])
    g_src = np.array([S / s, -A / a])
    return g_eta, g_src


def test_infer_source_calibration_closed_form_and_monte_carlo():
    s_db, a_db = -1.71, 5.54
    g_eta, g_src = _closed_infer_grad(s_db, a_db)
    for sig, csig in ((0.1, 0.1), (None, 0.2), (0.05, 0.3)):
        out = infer_source(s_db, a_db, sigma_db=sig, calib_sigma_db=csig)
        w = 0.0 if sig is None else sig
        ref_eta = np.sqrt(w ** 2 * g_eta @ g_eta + csig ** 2
                          * g_eta.sum() ** 2)
        ref_src = np.sqrt(w ** 2 * g_src @ g_src + csig ** 2
                          * g_src.sum() ** 2)
        assert out["eta_sigma"] == pytest.approx(ref_eta, rel=1e-6)
        assert out["sq_db_source_sigma"] == pytest.approx(ref_src,
                                                          rel=1e-6)
    # sigma_db alone: the 0.14 numbers digit for digit (README example 6)
    out = infer_source(s_db, a_db, sigma_db=0.1)
    assert round(out["eta_sigma"], 4) == 0.0204
    assert round(out["sq_db_source_sigma"], 2) == 0.25
    assert out["eta_sigma"] == pytest.approx(
        0.1 * np.linalg.norm(g_eta), rel=1e-6)
    # seeded Monte Carlo: one common offset per trial plus independent
    # point errors
    sig, csig = 0.05, 0.1
    ref = infer_source(s_db, a_db, sigma_db=sig, calib_sigma_db=csig)
    rng = np.random.default_rng(11)
    etas, srcs = [], []
    for _ in range(1500):
        b = csig * rng.standard_normal()
        o = infer_source(s_db + b + sig * rng.standard_normal(),
                         a_db + b + sig * rng.standard_normal())
        etas.append(o["eta"])
        srcs.append(o["sq_db_source"])
    assert np.std(etas, ddof=1) == pytest.approx(ref["eta_sigma"],
                                                 rel=0.1)
    assert np.std(srcs, ddof=1) == pytest.approx(
        ref["sq_db_source_sigma"], rel=0.1)
