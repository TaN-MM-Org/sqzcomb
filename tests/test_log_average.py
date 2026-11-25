"""Traces from an analyzer that averaged in dB (closed in 0.19 by
showing that `shot_noise_normalize` needs no correction for them).

A log-averaged value of Gaussian noise reads 10 gamma / ln 10 =
2.507 dB below the mean power when many readings are averaged (HP
Application Note 1303, p. 8: 2.51 dB). The factor is the same for every
noise trace, so the ratios the function returns are unaffected. Checks:
the constant from a numerical integral, and seeded simulations of
log-averaged Gaussian noise whose true powers are known.
"""
import numpy as np
import pytest
from scipy.integrate import quad

import sqzcomb as sc

BIAS_DB = 10.0 * np.euler_gamma / np.log(10.0)


def test_bias_constant_from_the_exponential_distribution():
    # E[10 log10 P] - 10 log10 E[P] for an exponential P of mean 1
    e_ln = quad(lambda x: np.log(x) * np.exp(-x), 0.0, np.inf,
                epsabs=1e-14, epsrel=1e-13)[0]
    assert -10.0 * e_ln / np.log(10.0) == pytest.approx(
        BIAS_DB, rel=1e-10)                            # measured 4e-15
    # the 2.51 dB of the application note, to its two decimals
    assert round(BIAS_DB, 2) == 2.51


def _log_traces(rng, powers, n_sweeps, n_avg):
    """Log-averaged sweeps: each value is the mean of n_avg readings of
    10 log10 P, with P exponential of the given mean power (Gaussian
    noise after the resolution filter)."""
    p = np.asarray(powers, dtype=float)
    reads = rng.exponential(1.0, size=(n_sweeps, n_avg, p.size)) * p
    return np.mean(10.0 * np.log10(reads), axis=1)


@pytest.mark.parametrize("n_avg, n_sweeps", [(4000, 20), (4, 4000)])
def test_log_averaged_traces_give_back_the_true_levels(n_avg, n_sweeps):
    # true mean powers (linear, any unit) at three frequencies: shot
    # noise 1, dark noise 15 dB below it, and a signal 4 dB below shot
    # noise after the dark noise is removed
    rng = np.random.default_rng(11)
    shot = np.ones(3)
    dark = 10 ** -1.5 * shot
    sig = 10 ** -0.4 * (shot - dark) + dark
    true_db = 10.0 * np.log10((sig - dark) / (shot - dark))
    true_dark = 0.5 * dark / (shot - dark)
    tr = _log_traces(rng, sig, n_sweeps, n_avg)
    sh = _log_traces(rng, shot, n_sweeps, n_avg)
    dk = _log_traces(rng, dark, n_sweeps, n_avg)
    if n_avg > 1000:
        # each trace alone reads the 2.507 dB low
        assert np.mean(sh) == pytest.approx(-BIAS_DB, abs=0.01)  # measured 3e-4
    out = sc.shot_noise_normalize(tr, sh, dk)
    z = (out["db"] - true_db) / out["sigma_db"]
    assert np.all(np.abs(z) < 4.0)                     # measured |z| up to 1.27
    # the dark trace is subtracted correctly although it reads low: the
    # dark variance 0.5 D / (S - D) agrees with the truth within 4 of
    # its standard errors (first order, from the sweep scatter)
    pw = [10.0 ** (x / 10.0) for x in (sh, dk)]
    S, D = (p.mean(axis=0) for p in pw)
    sS, sD = (p.std(axis=0, ddof=1) / np.sqrt(n_sweeps) for p in pw)
    s_rel = np.hypot(sD * S / (D * (S - D)), sS / (S - D))
    rel = out["dark_variance"] / true_dark - 1.0
    assert np.all(np.abs(rel) < 4.0 * s_rel)           # measured up to 1.04 s_rel
    # the same data with every value raised by the bias: identical
    # ratios (to rounding), which is why no correction is needed
    up = sc.shot_noise_normalize(tr + BIAS_DB, sh + BIAS_DB, dk + BIAS_DB)
    np.testing.assert_allclose(up["db"], out["db"], rtol=0,
                               atol=1e-12)             # measured 3e-14
    np.testing.assert_allclose(up["dark_variance"], out["dark_variance"],
                               rtol=1e-12)             # measured 2e-15
