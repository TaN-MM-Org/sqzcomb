"""0.14 anchors: the exact best and worst quadrature angle.

Every quadrature variance is c0 + c1 cos 2phi + c2 sin 2phi, so its
extremes follow from three evaluations. The tests hold them to an
independent path (the eigenvalues and eigenvectors of the 2x2 output
covariance block, from `output_covariance_xxpp`), to closed forms, to
dense angle scans, and to the twin-beam identity E_N = -ln(2 V_min)."""
import numpy as np
import pytest

from sqzcomb import (fluctuation_matrix, lle_evolve, optimal_quadrature,
                     output_covariance_xxpp, output_entanglement,
                     output_quadrature_variance, output_variance_ports,
                     photonic_molecule, quadrature_extremes,
                     single_mode_parametric, squeezing_db)


def _block_eig(sig, k, n):
    blk = sig[np.ix_([k, n + k], [k, n + k])]
    w, v = np.linalg.eigh(blk)
    return w, v


def _same_angle(a, b, tol):
    d = np.mod(a - b + np.pi / 2, np.pi) - np.pi / 2    # modulo pi
    return abs(d) < tol


def test_extremes_equal_the_covariance_eigenvalues_single_mode():
    # a detuned squeezer with a complex gain: the best angle moves with
    # frequency, and the extremes are the eigenvalues of the 2x2 block
    M = single_mode_parametric(0.6 * np.exp(0.4j), 0.3)
    omegas = np.array([0.0, 0.7, 2.0, 5.0])
    for eta in (0.35, 0.8, 1.0):
        r = optimal_quadrature(M, eta, omegas, 0, 1)
        for i, w in enumerate(omegas):
            ev, vec = _block_eig(output_covariance_xxpp(M, eta, w, 1), 0, 1)
            assert abs(r["v_min"][i] - ev[0]) < 1e-12
            assert abs(r["v_max"][i] - ev[1]) < 1e-12
            ang = np.arctan2(vec[1, 0], vec[0, 0])      # X = cos x + sin p
            assert _same_angle(r["phi_min"][i], ang, 1e-9)
            assert abs(output_quadrature_variance(
                M, eta, w, 0, 1, phi=r["phi_min"][i]) - ev[0]) < 1e-12
    assert len(set(np.round(r["phi_min"], 6))) == omegas.size


def test_extremes_on_a_comb_line_beat_or_match_every_scanned_angle():
    F, alpha = 0.9, 0.3                  # the state of README example 3
    psi = lle_evolve(np.full(16, 0.05 + 0j), F=F, alpha=alpha, t_end=100.0)
    M, modes = fluctuation_matrix(psi, alpha)
    i0 = int(np.where(modes == 0)[0][0])
    m = modes.size
    for eta in (0.2, 0.5, 1.0):
        r = optimal_quadrature(M, eta, 0.0, i0, m)
        ev, _ = _block_eig(output_covariance_xxpp(M, eta, 0.0, m), i0, m)
        assert abs(r["v_min"] - ev[0]) < 1e-12
        assert abs(r["v_max"] - ev[1]) < 1e-12
        scan = [output_quadrature_variance(M, eta, 0.0, i0, m, phi=p)
                for p in np.linspace(0.0, np.pi, 721)]
        # a scan can never go below the exact minimum ...
        assert min(scan) >= r["v_min"] - 1e-14
        # ... and misses it by at most 2 r (dphi / 2)^2 (half a step)
        dphi = np.pi / 720
        bound = (r["v_max"] - r["v_min"]) * dphi ** 2 / 4
        assert min(scan) - r["v_min"] <= bound * (1 + 1e-9) + 1e-15
        assert r["squeezing_db"] == pytest.approx(squeezing_db(r["v_min"]))


def test_single_mode_closed_form_through_the_port_path():
    # molecule with J = 0 is a single parametric mode: V_min is the
    # closed form 0.5 (1 - 4 eta mu / ((1 + mu)^2 + w^2)) at phi = pi/2
    mu = 0.7
    M, g = photonic_molecule(mu, 0.0, gamma=2.0)
    for eta in (0.4, 1.0):
        for w in (0.0, 1.3):
            r = quadrature_extremes(
                lambda p: output_variance_ports(M, g, eta, 0, w, phi=p))
            closed = 0.5 * (1 - 4 * eta * mu / ((1 + mu) ** 2 + w ** 2))
            anti = 0.5 * (1 + 4 * eta * mu / ((1 - mu) ** 2 + w ** 2))
            assert r["v_min"] == pytest.approx(closed, rel=1e-12)
            assert r["v_max"] == pytest.approx(anti, rel=1e-12)
            assert _same_angle(r["phi_min"], np.pi / 2, 1e-9)


def test_detuned_molecule_port_against_a_dense_scan():
    M, g = photonic_molecule(1.5, 1.2, delta_a=0.4, delta_b=-0.3, gamma=1.5)
    for w in (0.0, 0.8):
        r = quadrature_extremes(
            lambda p: output_variance_ports(M, g, 0.9, 1, w, phi=p))
        phis = np.linspace(0.0, np.pi, 721)
        v = np.array([output_variance_ports(M, g, 0.9, 1, w, phi=p)
                      for p in phis])
        assert v.min() >= r["v_min"] - 1e-14
        assert v.max() <= r["v_max"] + 1e-14
        # the scan can miss the extremum by at most half a grid step,
        # which costs at most 2 r (dphi / 2)^2 with r = (Vmax - Vmin)/2
        bound = (r["v_max"] - r["v_min"]) * (phis[1] - phis[0]) ** 2 / 4
        assert v.min() - r["v_min"] <= bound * (1 + 1e-9) + 1e-15
        assert r["v_max"] - v.max() <= bound * (1 + 1e-9) + 1e-15


def test_twin_beam_negativity_is_minus_log_of_exact_min():
    # the 0.8.0 test had to allow 1e-6 because of its phase grid
    M = np.array([[-1, 0, 0, 0.5], [0, -1, 0.5, 0],
                  [0, 0.5, -1, 0], [0.5, 0, 0, -1]], dtype=complex)
    for eta, w in ((0.8, 0.3), (0.6, 1.0), (1.0, 0.0)):
        r = optimal_quadrature(M, eta, w, 0, 2, mode_index_b=1)
        EN = output_entanglement(M, eta, w, 0, 1, 2)
        assert abs(EN + np.log(2.0 * r["v_min"])) < 1e-10


def test_phase_insensitive_and_malformed_inputs():
    # a passive cavity: every angle gives vacuum, r = 0
    Mp = np.diag([-1.0 + 0.2j, -1.0 - 0.2j]).astype(complex)
    r = optimal_quadrature(Mp, 0.7, 0.4, 0, 1)
    assert abs(r["v_min"] - 0.5) < 1e-12 and abs(r["v_max"] - 0.5) < 1e-12
    M = single_mode_parametric(0.5)
    # dB instead of variance is not of the harmonic form: refused
    with pytest.raises(ValueError, match="not of the form"):
        quadrature_extremes(lambda p: squeezing_db(
            output_quadrature_variance(M, 0.8, 0.2, 0, 1, phi=p)))
    with pytest.raises(ValueError, match="not of the form"):
        quadrature_extremes(lambda p: 1.0 + 0.3 * np.cos(4 * p))
    with pytest.raises(ValueError):
        quadrature_extremes(lambda p: np.nan)
