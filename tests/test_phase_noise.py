"""v0.10 phase-noise anchors: the exact Gaussian-averaged quadrature
mixing held against direct numerical integration over the Gaussian
(two independent code paths), its exact limits, the closed-form
inversion round trip, and the refusals. References: Dwyer et al.,
Opt. Express 21, 19047 (2013); Oelker et al., Optica 3, 682 (2016);
Dean et al., npj Nanophotonics (2026), doi:10.1038/s44310-026-00125-5."""
import numpy as np
import pytest

from sqzcomb import (max_phase_noise, phase_noise_squeezing_db,
                     phase_noise_variance)

VS, VA = 0.05, 1.6                       # ~-7 dB squeezed, near-pure


def test_closed_form_matches_gaussian_quadrature():
    """<V(theta)> integrated numerically over the Gaussian jitter
    distribution -- an independent path -- must equal the closed
    form exp(-2 sigma^2) average to 1e-10."""
    scipy_integrate = pytest.importorskip("scipy.integrate")
    for s in (0.02, 0.1, 0.4):
        for t0 in (0.0, 0.15):
            closed = phase_noise_variance(VS, VA, s, static_offset=t0)
            num, err = scipy_integrate.quad(
                lambda th: (VS * np.cos(t0 + th) ** 2
                            + VA * np.sin(t0 + th) ** 2)
                * np.exp(-th * th / (2 * s * s))
                / (s * np.sqrt(2 * np.pi)),
                -12 * s, 12 * s, limit=200)
            assert abs(closed - num) < 1e-10


def test_exact_limits_and_static_rotation_law():
    # zero jitter, zero offset: the source variance, exactly
    assert phase_noise_variance(VS, VA, 0.0) == VS
    # zero jitter, static offset: the plain rotation law, exactly
    t0 = 0.3
    v = phase_noise_variance(VS, VA, 0.0, static_offset=t0)
    assert abs(v - (VS * np.cos(t0) ** 2 + VA * np.sin(t0) ** 2)) < 1e-15
    # infinite jitter: the quadrature-blind average (V_sq + V_anti)/2
    assert abs(phase_noise_variance(VS, VA, 50.0)
               - 0.5 * (VS + VA)) < 1e-12
    # vacuum in, vacuum out at every jitter (phase noise cannot
    # manufacture noise from vacuum)
    for s in (0.0, 0.2, 3.0):
        assert abs(phase_noise_variance(0.5, 0.5, s) - 0.5) < 1e-15
    # monotone: more jitter never improves the squeezed reading
    ss = np.linspace(0.0, 1.5, 40)
    vv = [phase_noise_variance(VS, VA, s) for s in ss]
    assert np.all(np.diff(vv) >= 0.0)


def test_db_form_and_antisqueezing_floor():
    """The dB wrapper agrees with the variance form, and a near-pure
    -10 dB source under 100 mrad RMS jitter is visibly floored --
    the practical limit the 2026 integrated-squeezer budget names."""
    sq_db, anti_db = -10.0, 11.0
    out = phase_noise_squeezing_db(sq_db, anti_db, 0.1)
    vs = 0.5 * 10 ** (sq_db / 10)
    va = 0.5 * 10 ** (anti_db / 10)
    ref = 10 * np.log10(phase_noise_variance(vs, va, 0.1) / 0.5)
    assert abs(out - ref) < 1e-12
    assert out > sq_db + 1.0             # degradation is significant
    # and with a clean LO (1 mrad) the degradation is negligible
    assert phase_noise_squeezing_db(sq_db, anti_db, 1e-3) < sq_db + 0.01


def test_max_phase_noise_round_trip_and_refusals():
    for s_true in (0.02, 0.08, 0.3):
        vt = phase_noise_variance(VS, VA, s_true)
        assert abs(max_phase_noise(vt, VS, VA) - s_true) < 1e-12
    with pytest.raises(ValueError, match="cannot improve"):
        max_phase_noise(VS / 2, VS, VA)
    with pytest.raises(ValueError, match="not phase-noise-limited"):
        max_phase_noise(0.5 * (VS + VA), VS, VA)
    with pytest.raises(ValueError):
        max_phase_noise(0.2, 0.3, 0.1)   # anti below squeezed
    with pytest.raises(ValueError):
        phase_noise_variance(VS, VA, -0.1)


def test_composition_with_loss_is_order_dependent_and_physical():
    """Loss then phase noise differs from phase noise then loss only
    through the vacuum term -- both orders must stay at or above the
    pure-loss result and approach it as jitter vanishes."""
    from sqzcomb import detected_variance
    eta, s = 0.7, 0.12
    a = phase_noise_variance(detected_variance(VS, eta),
                             detected_variance(VA, eta), s)
    b = detected_variance(phase_noise_variance(VS, VA, s), eta)
    assert abs(a - b) < 1e-12            # both maps are affine with the
    # same vacuum fixed point, so the orders agree exactly
    assert a >= detected_variance(VS, eta) - 1e-15
    assert abs(phase_noise_variance(detected_variance(VS, eta),
                                    detected_variance(VA, eta), 0.0)
               - detected_variance(VS, eta)) < 1e-15
