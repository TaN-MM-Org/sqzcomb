"""g0 from n2: the formula, its scaling, and its consistency with the
package's threshold power."""
import numpy as np
import pytest
from scipy.constants import c, hbar

from sqzcomb import RingSpec, threshold_power
from sqzcomb.physical import kerr_shift_from_n2

# made-up numbers for a test, not material data
N2, N0, V, LAM = 2.0e-19, 2.0, 6.0e-16, 1.55e-6


def test_formula_and_scaling():
    g = kerr_shift_from_n2(N2, N0, V, LAM)
    w0 = 2 * np.pi * c / LAM
    assert g == pytest.approx(hbar * w0 ** 2 * c * N2 / (N0 ** 2 * V)
                              / (2 * np.pi), rel=1e-14)
    assert kerr_shift_from_n2(2 * N2, N0, V, LAM) == pytest.approx(2 * g)
    assert kerr_shift_from_n2(N2, 2 * N0, V, LAM) == pytest.approx(g / 4)
    assert kerr_shift_from_n2(N2, N0, 2 * V, LAM) == pytest.approx(g / 2)
    assert kerr_shift_from_n2(N2, N0, V, 2 * LAM) == pytest.approx(g / 4)


def test_threshold_takes_the_textbook_form():
    # with this g0 the package's threshold at alpha = 1 becomes
    # P_th = kappa^2 n0^2 V / (8 eta omega0 c n2): no hbar left
    g = kerr_shift_from_n2(N2, N0, V, LAM)
    spec = RingSpec(kappa_hz=200e6, eta_esc=0.5, g0_hz=g,
                    lambda_pump_m=LAM, reference="test numbers")
    w0 = 2 * np.pi * c / LAM
    expect = spec.kappa ** 2 * N0 ** 2 * V / (8 * 0.5 * w0 * c * N2)
    assert threshold_power(spec) == pytest.approx(expect, rel=1e-12)


def test_refusals():
    for bad in ((0, N0, V, LAM), (N2, -1, V, LAM), (N2, N0, np.inf, LAM),
                (N2, N0, V, 0.0)):
        with pytest.raises(ValueError):
            kerr_shift_from_n2(*bad)
