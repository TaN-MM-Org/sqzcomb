"""0.16 regression tests: three wrong answers found by the second audit,
and the equality checks of the speed changes.

The tests of the three fixes fail on 0.15.1 and pass now. The
references are closed forms (the single-mode output spectrum, the
two-mode squeezed vacuum) or an independent computation written here
(Williamson's symplectic spectrum from the Hermitian matrix
sigma^(1/2) (i Omega) sigma^(1/2)). The speed changes are checked
against the old code, kept here verbatim, for exact equality.
"""
import numpy as np
import pytest

from sqzcomb import (logarithmic_negativity, optimal_band_quadrature,
                     optimal_quadrature, output_entanglement,
                     output_quadrature_variance, output_variance_ports,
                     photonic_molecule, ppt_symplectic_eigenvalue,
                     quadrature_extremes, single_mode_parametric,
                     symplectic_eigenvalues)


def _closed_sq(mu, eta, w):
    """Squeezed (phi = pi/2) output variance of the single-mode
    parametric process, 1/2 [1 - 4 eta mu / ((1 + mu)^2 + w^2)]."""
    return 0.5 * (1.0 - 4.0 * eta * mu / ((1.0 + mu) ** 2 + w * w))


# ----------------------------------------------------------------------
# 1. Output variance close to threshold


@pytest.mark.parametrize("mu,eta,rtol", [
    (0.999, 1.0, 1e-7), (0.9999, 1.0, 1e-6), (0.99999, 1.0, 1e-3),
    (0.9999, 0.99, 1e-10), (0.999999, 0.99, 1e-10)])
def test_near_threshold_variance_is_accurate_and_positive(mu, eta, rtol):
    """0.15.1 formed the output matrix S and then u^dag S u, which
    subtracts numbers of the size of the antisqueezed variance:
    mu = 0.9999, eta = 1, omega = 0 gave -2.107e-8 (negative) instead of
    1.250e-9, and mu = 0.999999, eta = 0.99 gave 4.920e-3 instead of
    5.000e-3."""
    M = single_mode_parametric(mu)
    exact = _closed_sq(mu, eta, 0.0)
    v = output_quadrature_variance(M, eta, 0.0, 0, 1, phi=np.pi / 2)
    assert v > 0.0
    assert v == pytest.approx(exact, rel=rtol)
    opt = optimal_quadrature(M, eta, np.array([0.0, 0.3]), 0, 1)
    assert np.all(opt["v_min"] > 0.0)
    assert opt["v_min"][0] == pytest.approx(exact, rel=rtol)
    assert opt["v_min"][1] == pytest.approx(_closed_sq(mu, eta, 0.3),
                                            rel=1e-9)
    # the antisqueezed one, 1/2 [1 + 4 eta mu / (1 - mu)^2], is easy
    assert opt["v_max"][0] == pytest.approx(
        0.5 * (1.0 + 4.0 * eta * mu / (1.0 - mu) ** 2), rel=1e-8)
    # the same through the per-mode-decay port path (J = 0: the main
    # ring alone) and its scalar angle search
    Mp, g = photonic_molecule(mu, 0.0)
    vp = output_variance_ports(Mp, g, eta, 0, 0.0, phi=np.pi / 2)
    assert vp == pytest.approx(exact, rel=rtol)
    ext = quadrature_extremes(
        lambda p: output_variance_ports(Mp, g, eta, 0, 0.0, phi=p))
    assert ext["v_min"] == pytest.approx(exact, rel=rtol)
    band = optimal_band_quadrature(M, eta, [0.0], 0, 1)
    assert band["v_band"] == pytest.approx(exact, rel=rtol)


def test_variance_unchanged_away_from_threshold():
    """Away from threshold the new summation agrees with 0.15.1's
    matrix reading to rounding (values printed by 0.15.1)."""
    M = single_mode_parametric(0.6 * np.exp(0.4j), delta=0.3)
    ref = {0.0: 3.1945607835166596, 0.5: 1.622726705794609}
    for w, v in ref.items():
        assert output_quadrature_variance(M, 0.8, w, 0, 1, phi=1.1) \
            == pytest.approx(v, rel=1e-13)


# ----------------------------------------------------------------------
# 2. Logarithmic negativity of strongly entangled pairs


def _tmsv(r):
    """Two-mode squeezed vacuum, xxpp order, vacuum = identity."""
    c, s = np.cosh(2 * r), np.sinh(2 * r)
    z = np.zeros((2, 2))
    return np.block([[np.array([[c, s], [s, c]]), z],
                     [z, np.array([[c, -s], [-s, c]])]])


@pytest.mark.parametrize("r", [3.0, 4.0, 5.0, 6.0])
def test_strong_tmsv_negativity_equals_2r(r):
    """E_N = 2r exactly. 0.15.1 took nu^2 = (Delta - sqrt(Delta^2 -
    4 det)) / 2, which cancels: r = 4 gave 7.99931 and r = 5 gave inf
    (with a divide-by-zero warning)."""
    en = logarithmic_negativity(_tmsv(r), 0, 1)
    assert en == pytest.approx(2.0 * r, rel=1e-7)


def test_singular_covariance_is_refused_not_infinite():
    """Beyond double precision (r = 10: cosh 20 ~ 2.4e8, condition
    number ~ e^40) det(sigma) rounds to zero or below; 0.15.1 returned
    E_N = inf, now a ValueError."""
    with pytest.raises(ValueError, match="singular|invalid"):
        logarithmic_negativity(_tmsv(10.0), 0, 1)
    with pytest.raises(ValueError):
        ppt_symplectic_eigenvalue(np.zeros((4, 4)))


@pytest.mark.parametrize("mu", [0.99, 0.999, 0.9999])
def test_output_entanglement_near_threshold(mu):
    """Twin beam at 90 % escape: E_N = -ln(2 V_min) with the closed-form
    V_min. 0.15.1 gave 2.31444 instead of 2.30258 at mu = 0.999 and
    inf at mu = 0.9999."""
    twin = np.array([[-1, 0, 0, mu], [0, -1, mu, 0],
                     [0, mu, -1, 0], [mu, 0, 0, -1]], dtype=complex)
    exact = -np.log(2.0 * _closed_sq(mu, 0.9, 0.0))
    en = output_entanglement(twin, 0.9, 0.0, 0, 1, 2)
    assert en == pytest.approx(exact, rel=1e-6)


# ----------------------------------------------------------------------
# 3. Symplectic eigenvalues of a state with unequal eigenvalues


def _williamson(sigma):
    """Independent route: the eigenvalues of the Hermitian matrix
    sigma^(1/2) (i Omega) sigma^(1/2) are +/- nu_k."""
    n = sigma.shape[0] // 2
    om = np.block([[np.zeros((n, n)), np.eye(n)],
                   [-np.eye(n), np.zeros((n, n))]])
    w, v = np.linalg.eigh(sigma)
    root = v @ np.diag(np.sqrt(w)) @ v.T
    ev = np.linalg.eigvalsh(root @ (1j * om) @ root)
    return np.sort(ev[ev > 0])


def test_symplectic_eigenvalues_are_not_duplicated():
    """0.15.1 returned the smallest eigenvalue twice: diag(1, 3, 1, 3)
    gave [1, 1] instead of [1, 3] (two thermal modes, n_bar = 0 and 1),
    and README example 5 printed [1.0484 1.0484] instead of
    [1.0484 1.2429]."""
    assert np.allclose(symplectic_eigenvalues(np.diag([1., 3., 1., 3.])),
                       [1.0, 3.0], rtol=1e-14)
    assert np.allclose(symplectic_eigenvalues(
        np.diag([2., 5., 7., 2., 5., 7.])), [2.0, 5.0, 7.0], rtol=1e-14)
    from sqzcomb import covariance_xxpp, intracavity_covariance
    M, g = photonic_molecule(mu=0.8, J=1.0)
    s = covariance_xxpp(intracavity_covariance(M, g))
    nu = symplectic_eigenvalues(s)
    assert np.allclose(nu, _williamson(s), rtol=1e-10)
    assert np.round(nu, 4).tolist() == [1.0484, 1.2429]
    # product of the symplectic eigenvalues squared is det(sigma)
    assert np.prod(nu) ** 2 == pytest.approx(np.linalg.det(s), rel=1e-10)
    with pytest.raises(ValueError):
        symplectic_eigenvalues(np.eye(3))
    with pytest.raises(ValueError):
        symplectic_eigenvalues(np.full((2, 2), np.nan))


# ----------------------------------------------------------------------
# Performance changes of 0.16: the old code paths are kept here as the
# references, and the new ones must agree with them exactly.


def _fluctuation_matrix_loop(psi_s, alpha, dispersion=(0.0,), modes=None,
                             d1=0.0):
    """The double loop of `fluctuation_matrix` up to 0.15.1, verbatim."""
    from sqzcomb.lle import _linear_symbol
    psi_s = np.asarray(psi_s, dtype=complex)
    n = psi_s.size
    kgrid = np.fft.fftfreq(n, d=1.0 / n).astype(int)
    if modes is None:
        modes = kgrid.copy()
    modes = np.asarray(list(modes), dtype=int)
    m = modes.size
    f_abs2 = np.fft.fft(np.abs(psi_s) ** 2) / n
    f_sq = np.fft.fft(psi_s ** 2) / n
    comp_abs2 = dict(zip(kgrid, f_abs2))
    comp_sq = dict(zip(kgrid, f_sq))
    L = _linear_symbol(alpha, dispersion, modes.astype(float), d1)
    A = np.zeros((m, m), dtype=complex)
    B = np.zeros((m, m), dtype=complex)
    for i, ki in enumerate(modes):
        for j, kj in enumerate(modes):
            if i == j:
                A[i, j] += L[i]
            A[i, j] += 2j * comp_abs2.get(ki - kj, 0.0)
            B[i, j] += 1j * comp_sq.get(ki + kj, 0.0)
    return np.block([[A, B], [np.conj(B), np.conj(A)]]), modes


@pytest.mark.parametrize("n,modes", [
    (16, None), (17, None), (64, None), (32, [0, 3, -3, 7, -7]),
    (16, [0, 1, -1, 9, -12, 20])])            # 9, -12, 20: off the grid
def test_fluctuation_matrix_is_bit_identical_to_the_loop(n, modes):
    from sqzcomb import fluctuation_matrix
    rng = np.random.default_rng(n)
    psi = rng.normal(size=n) + 1j * rng.normal(size=n)
    for disp, d1 in (((0.0,), 0.0), ((-0.02, 0.003), 0.4)):
        new, mn = fluctuation_matrix(psi, 1.3, disp, modes, d1=d1)
        ref, mr = _fluctuation_matrix_loop(psi, 1.3, disp, modes, d1=d1)
        assert np.array_equal(mn, mr)
        assert np.array_equal(new.view(np.uint64), ref.view(np.uint64))


def test_noise_models_are_bit_identical_to_the_per_call_path():
    from sqzcomb import noise_spectrum_db, parametric_spectra_model
    f = np.linspace(0.5e6, 30e6, 25)
    for q, phi in (("squeezed", np.pi / 2), ("anti", 0.0)):
        M = single_mode_parametric(0.55)
        om = 4.0 * np.pi * f / (2.0 * np.pi * 12e6)
        ref = np.array([output_quadrature_variance(M, 0.62, o, 0, 1,
                                                   phi=phi) for o in om])
        ref = 10.0 * np.log10((ref + 0.01) / 0.5)
        new = noise_spectrum_db(0.55, 0.62, 12e6, f, q, dark=0.01)
        assert np.array_equal(new, ref)
    names, model = parametric_spectra_model(detuned=True, jitter=True)
    p = dict(mu=0.5, eta=0.7, kappa_hz=10e6, delta=0.2, phi_sq=1.4,
             theta_rms=0.05)
    out = model(p, f)
    M = single_mode_parametric(0.5, 0.2)
    om = 4.0 * np.pi * f / (2.0 * np.pi * 10e6)
    v_sq = np.array([output_quadrature_variance(M, 0.7, o, 0, 1, phi=1.4)
                     for o in om])
    v_an = np.array([output_quadrature_variance(
        M, 0.7, o, 0, 1, phi=1.4 + np.pi / 2.0) for o in om])
    from sqzcomb import jitter_average
    assert np.array_equal(out["squeezed"], jitter_average(v_sq, v_an, 0.05))
    assert np.array_equal(out["anti"], jitter_average(v_an, v_sq, 0.05))
