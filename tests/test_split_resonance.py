"""0.17: a resonance split into a doublet by backscattering
(`split_transmission`, `fit_split_resonance`).

References: the forward transmission derived again from the drift
matrix of `photonic_molecule` (two passive modes coupled by J, the bus
on one of them: the cw and ccw modes of Kippenberg, Spillane and
Vahala, Opt. Lett. 27, 1669 (2002)), the single-dip
`through_transmission` at zero splitting, the algebra of the eta -> 1 -
eta symmetry, noise-free recovery and a seeded Monte Carlo.
"""
import numpy as np
import pytest

from sqzcomb import (fit_resonance_transmission, fit_split_resonance,
                     photonic_molecule, split_transmission,
                     through_transmission)

F = np.linspace(-500e6, 500e6, 201)


def _t_from_molecule(eta, b, x):
    """Forward transmission from the package's drift matrix: modes cw
    and ccw (decay 1, coupled by J = b), the bus input and output on cw
    with escape fraction eta, laser detuning x in units of kappa/2."""
    M, _ = photonic_molecule(0.0, b, gamma=1.0)
    G = np.linalg.inv(-1j * x * np.eye(4) - M)
    return abs(2.0 * eta * G[0, 0] - 1.0) ** 2


def test_model_is_the_coupled_mode_transmission():
    kap = 2.0                                  # then x = f - f0
    for eta in (0.0, 0.2, 0.5, 0.8, 1.0):
        for b in (0.0, 0.3, 1.0, 4.0):
            for x in (-6.0, -1.0, -0.2, 0.0, 0.5, 1.0, 7.0):
                ref = _t_from_molecule(eta, b, x)
                got = split_transmission(x, 0.0, kap, eta, b * kap)
                assert got == pytest.approx(ref, rel=1e-13, abs=1e-15)


def test_zero_splitting_is_the_single_dip():
    for eta in (0.1, 0.5, 0.9):
        assert np.allclose(split_transmission(F, 3e6, 100e6, eta, 0.0, 2.0),
                           through_transmission(F, 3e6, 100e6, eta, 2.0),
                           rtol=1e-14, atol=1e-15)


def test_splitting_breaks_the_coupling_symmetry():
    """|t|^2 is symmetric under eta -> 1 - eta only without splitting:
    then an under- and an over-coupled ring look alike, with splitting
    they do not."""
    a = split_transmission(F, 0.0, 100e6, 0.3, 0.0)
    b = split_transmission(F, 0.0, 100e6, 0.7, 0.0)
    assert np.allclose(a, b, rtol=1e-13, atol=1e-15)
    a = split_transmission(F, 0.0, 100e6, 0.3, 60e6)
    b = split_transmission(F, 0.0, 100e6, 0.7, 60e6)
    assert np.max(np.abs(a - b)) > 0.05
    # on resonance of the doublet's center: 1 - 2 eta / (1 + s^2), with
    # s the splitting over the linewidth
    for eta in (0.3, 0.7):
        assert split_transmission(0.0, 0.0, 100e6, eta, 60e6) == \
            pytest.approx((1 - 2 * eta / (1 + 0.6 ** 2)) ** 2, rel=1e-14)


@pytest.mark.parametrize("eta,branch,other", [(0.8, "over", "under"),
                                               (0.3, "under", "over")])
def test_noise_free_doublet_is_recovered(eta, branch, other):
    T = split_transmission(F, 3e6, 100e6, eta, 60e6, 2.0)
    r = fit_split_resonance(F, T, sigma=0.005)
    d = r[branch]
    assert d["eta_esc"] == pytest.approx(eta, abs=1e-8)
    assert d["kappa_hz"] == pytest.approx(100e6, rel=1e-8)
    assert d["splitting_hz"] == pytest.approx(60e6, rel=1e-8)
    assert d["f0_hz"] == pytest.approx(3e6, abs=1.0)
    assert d["baseline"] == pytest.approx(2.0, rel=1e-9)
    # a resolved doublet decides the coupling regime: the other branch
    # fails the chi-square test
    assert r[other] is None and r["delta_chi2"] is None
    # and the single-Lorentzian fit refuses these data
    with pytest.raises(ValueError, match="single Lorentzian"):
        fit_resonance_transmission(F, T, sigma=0.005)


def test_tilted_baseline_is_fitted():
    """A laser power that drifts linearly during the sweep (2e-10 per Hz
    of the off-resonance level, about 20 % over the scan)."""
    T = split_transmission(F, 3e6, 100e6, 0.8, 60e6, 2.0) \
        * (1.0 + 2e-10 * F)
    r = fit_split_resonance(F, T, sigma=0.005, fit_slope=True)["over"]
    assert r["eta_esc"] == pytest.approx(0.8, abs=1e-8)
    assert r["kappa_hz"] == pytest.approx(100e6, rel=1e-8)
    assert r["splitting_hz"] == pytest.approx(60e6, rel=1e-8)
    assert r["slope_per_hz"] == pytest.approx(2.0 * 2e-10, rel=1e-7)
    assert np.allclose(r["model"], T, rtol=1e-9, atol=0)


def test_small_splitting_leaves_both_branches():
    """At a splitting of a fifth of the linewidth both coupling regimes
    fit with 0.5 % noise: the fit returns both and their chi-square
    difference instead of choosing."""
    rng = np.random.default_rng(5)
    T = split_transmission(F, 3e6, 100e6, 0.8, 20e6, 2.0) \
        + 0.005 * rng.standard_normal(F.size)
    r = fit_split_resonance(F, T, sigma=0.005)
    assert r["under"] is not None and r["over"] is not None
    assert abs(r["delta_chi2"]) < 9.0
    assert r["over"]["eta_esc"] == pytest.approx(
        0.8, abs=3 * r["over"]["sigma"]["eta_esc"])


def test_error_bars_match_monte_carlo():
    T = split_transmission(F, 3e6, 100e6, 0.8, 60e6, 2.0)
    ref = fit_split_resonance(F, T, sigma=0.01)["over"]
    rng = np.random.default_rng(11)
    keys = ("eta_esc", "kappa_hz", "splitting_hz")
    draws = {k: [] for k in keys}
    chi2 = []
    for _ in range(40):
        r = fit_split_resonance(F, T + 0.01 * rng.standard_normal(F.size),
                                sigma=0.01)["over"]
        for k in keys:
            draws[k].append(r[k])
        chi2.append(r["chi2"])
    for k in keys:
        # 40 draws: the sample standard deviation scatters by about 11 %
        ratio = np.std(draws[k], ddof=1) / ref["sigma"][k]
        assert 0.7 < ratio < 1.35, (k, ratio)
    assert abs(np.mean(chi2) - (F.size - 5)) < 3 * np.sqrt(2 * 196 / 40)


def test_refusals():
    rng = np.random.default_rng(2)
    unsplit = through_transmission(F, 3e6, 100e6, 0.8, 2.0) \
        + 0.01 * rng.standard_normal(F.size)
    with pytest.raises(ValueError, match="not resolved"):
        fit_split_resonance(F, unsplit, sigma=0.01)
    with pytest.raises(ValueError, match="sigma is required"):
        fit_split_resonance(F, unsplit, sigma=None)
    with pytest.raises(ValueError, match="positive"):
        fit_split_resonance(F, unsplit, sigma=0.0)
    with pytest.raises(ValueError, match="12 points"):
        fit_split_resonance(F[:10], unsplit[:10], sigma=0.01)
    # a scan that stops inside the doublet
    T = split_transmission(F, 3e6, 100e6, 0.8, 60e6, 2.0)
    part = np.abs(F) < 90e6
    with pytest.raises(ValueError, match="Widen the sweep"):
        fit_split_resonance(F[part], T[part], sigma=0.005)
    # three dips are neither one nor two
    three = T + split_transmission(F, 300e6, 60e6, 0.5, 0.0, 2.0) - 2.0
    with pytest.raises(ValueError, match="does not describe"):
        fit_split_resonance(F, three, sigma=0.005)
    with pytest.raises(ValueError):
        split_transmission(F, 0.0, 100e6, 0.5, -1.0)
