"""0.17: loss and phase jitter from squeezed / antisqueezed pairs at
several pump levels (`fit_squeezing_pairs`).

References, none of them the fit itself: the package's own forward
path (the output spectrum of `single_mode_parametric` from
`output_quadrature_variance`, then `phase_noise_variance` and
`detected_variance`), the closed-form single-pair inversion
`infer_source`, the dark-noise convention of `detected_variance`, and a
seeded Monte Carlo of noisy pairs for the error bars and chi-square.
"""
import numpy as np
import pytest

from sqzcomb import (detected_squeezing_db, fit_squeezing_pairs,
                     infer_source, output_quadrature_variance,
                     phase_noise_variance, single_mode_parametric)

MUS = (0.2, 0.4, 0.6, 0.75, 0.85, 0.92)


def _pairs(eta_esc, eta_det, theta, omega=0.4, dark=0.0, dir_=False,
           mus=MUS):
    """Detected pairs (dB) through the package's forward path: the
    cavity output at escape efficiency eta_esc, LO jitter, then the
    detection efficiency and dark noise."""
    S, A = [], []
    for mu in mus:
        M = single_mode_parametric(mu)
        vs = output_quadrature_variance(M, eta_esc, omega, 0, 1,
                                        phi=np.pi / 2)
        va = output_quadrature_variance(M, eta_esc, omega, 0, 1, phi=0.0)
        js = phase_noise_variance(vs, va, theta)
        ja = phase_noise_variance(va, vs, theta)
        S.append(detected_squeezing_db(js, eta_det, dark, dir_))
        A.append(detected_squeezing_db(ja, eta_det, dark, dir_))
    return np.array(S), np.array(A)


@pytest.mark.parametrize("eta_esc,eta_det,theta,omega", [
    (0.85, 0.9, 0.03, 0.4), (0.95, 0.97, 0.0, 0.0),
    (0.6, 0.8, 0.08, 1.5)])
def test_noise_free_pairs_return_the_efficiency_and_jitter(
        eta_esc, eta_det, theta, omega):
    """The detected output of the cavity squeezer is a pure source
    behind the efficiency eta_esc * eta_det, at every analysis
    frequency: noise-free pairs give both back to rounding."""
    S, A = _pairs(eta_esc, eta_det, theta, omega)
    r = fit_squeezing_pairs(S, A, sigma_db=0.05)
    assert r["eta"] == pytest.approx(eta_esc * eta_det, abs=1e-8)
    mix = 0.5 * (1.0 - np.exp(-2.0 * theta ** 2))
    assert r["mix"] == pytest.approx(mix, abs=1e-9)
    if theta > 0:
        assert r["theta_rms"] == pytest.approx(theta, rel=1e-6)
    assert r["chi2"] < 1e-12
    assert r["chi2_dof"] == len(MUS) - 2
    assert np.allclose(r["model_sq_db"], S, atol=1e-8)
    assert np.allclose(r["model_anti_db"], A, atol=1e-8)


def test_source_levels_are_the_lossless_cavity_output():
    """Each pair's source is the lossless (eta_esc = 1) squeezer output
    at the same gain and frequency."""
    S, A = _pairs(0.85, 0.9, 0.03, 0.4)
    r = fit_squeezing_pairs(S, A, sigma_db=0.05)
    for mu, src in zip(MUS, r["sq_db_source"]):
        v = output_quadrature_variance(single_mode_parametric(mu), 1.0,
                                       0.4, 0, 1, phi=np.pi / 2)
        assert src == pytest.approx(10 * np.log10(v / 0.5), abs=1e-6)
    assert np.allclose(r["anti_db_source"], -r["sq_db_source"])


def test_dark_noise_conventions():
    """A known dark noise, in either convention of `detected_variance`,
    is taken out exactly."""
    for dir_ in (False, True):
        S, A = _pairs(0.8, 0.9, 0.02, 0.3, dark=0.02, dir_=dir_)
        r = fit_squeezing_pairs(S, A, 0.05, dark_noise=0.02,
                                dark_in_reference=dir_)
        assert r["eta"] == pytest.approx(0.72, abs=1e-8)
        assert r["theta_rms"] == pytest.approx(0.02, rel=1e-5)


def test_one_pair_without_jitter_is_infer_source():
    """Loss only and one pair: the fit is exactly determined and must
    give the closed form of `infer_source`."""
    S, A = _pairs(0.8, 0.9, 0.0, 0.2)
    for s, a in zip(S, A):
        r = fit_squeezing_pairs([s], [a], 0.05, fit_jitter=False)
        ref = infer_source(s, a)
        assert r["eta"] == pytest.approx(ref["eta"], rel=1e-10)
        assert r["sq_db_source"][0] == pytest.approx(ref["sq_db_source"],
                                                     abs=1e-8)
        assert r["mix"] == 0.0 and r["chi2_dof"] == 0
        assert r["p_value"] is None


def test_error_bars_and_chi2_match_monte_carlo():
    """Seeded noisy pairs (0.05 dB): the scatter of eta and of the
    mixing fraction agrees with the reported error bars, and chi-square
    averages to its degrees of freedom."""
    S, A = _pairs(0.85, 0.9, 0.03, 0.4)
    ref = fit_squeezing_pairs(S, A, 0.05)
    rng = np.random.default_rng(17)
    eta, mix, chi2 = [], [], []
    for _ in range(150):
        r = fit_squeezing_pairs(S + 0.05 * rng.standard_normal(S.size),
                                A + 0.05 * rng.standard_normal(A.size),
                                0.05)
        eta.append(r["eta"])
        mix.append(r["mix"])
        chi2.append(r["chi2"])
    assert 0.75 < np.std(eta) / ref["eta_sigma"] < 1.3
    assert 0.75 < np.std(mix) / ref["mix_sigma"] < 1.3
    assert abs(np.mean(eta) - 0.765) < 3 * ref["eta_sigma"] / np.sqrt(150)
    assert abs(np.mean(chi2) - ref["chi2_dof"]) < 0.25 * ref["chi2_dof"]


def test_jitter_error_bar_and_upper_limit():
    """theta_rms_sigma is the first-order error of theta for the error of
    the mixing fraction; with no jitter in the data the fit reports no
    error bar on theta but an upper limit."""
    S, A = _pairs(0.85, 0.9, 0.05, 0.4)
    r = fit_squeezing_pairs(S, A, 0.02)
    assert r["mix"] > 3 * r["mix_sigma"]
    h = 1e-7
    th = lambda m: np.sqrt(-0.5 * np.log(1 - 2 * m))   # noqa: E731
    dth = (th(r["mix"] + h) - th(r["mix"] - h)) / (2 * h)
    assert r["theta_rms_sigma"] == pytest.approx(dth * r["mix_sigma"],
                                                 rel=1e-5)
    assert r["theta_rms_upper"] == pytest.approx(
        th(r["mix"] + 2 * r["mix_sigma"]), rel=1e-12)
    S0, A0 = _pairs(0.85, 0.9, 0.0, 0.4)
    r0 = fit_squeezing_pairs(S0, A0, 0.05)
    assert r0["theta_rms_sigma"] is None
    assert r0["theta_rms"] < 1e-3
    assert r0["theta_rms_upper"] > 0.0


def test_refusals():
    S, A = _pairs(0.85, 0.9, 0.03, 0.4)
    with pytest.raises(ValueError, match="at least 2"):
        fit_squeezing_pairs(S[:1], A[:1], 0.05)
    with pytest.raises(ValueError, match="same length"):
        fit_squeezing_pairs(S, A[:-1], 0.05)
    with pytest.raises(ValueError, match="finite"):
        fit_squeezing_pairs(np.r_[S[:-1], np.nan], A, 0.05)
    with pytest.raises(ValueError, match="below shot noise"):
        fit_squeezing_pairs(np.r_[S[:-1], 0.1], A, 0.05)
    with pytest.raises(ValueError, match="above shot"):
        fit_squeezing_pairs(S, np.r_[A[:-1], -0.1], 0.05)
    with pytest.raises(ValueError, match="positive"):
        fit_squeezing_pairs(S, A, 0.0)
    with pytest.raises(ValueError, match="dark"):
        fit_squeezing_pairs(S, A, 0.05, dark_noise=-0.01)
    # every pair at the same gain: loss and jitter cannot be separated
    with pytest.raises(ValueError, match="do not separate"):
        fit_squeezing_pairs([S[2]] * 4, [A[2]] * 4, 0.05)
