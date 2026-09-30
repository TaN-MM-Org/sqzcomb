"""0.14 anchors: dark noise against a shot trace that carries it, the
whole loss + jitter + dark-noise budget inverted exactly, and raw
spectrum-analyzer traces turned into dB relative to shot noise. The
references are the forward maps of `sqzcomb.detection` (round trips),
the equivalent-loss identity, and a seeded Monte Carlo of repeated
sweeps."""
import os
import tempfile

import numpy as np
import pytest

from sqzcomb import (dark_equivalent_efficiency, dark_from_clearance_db,
                     detected_squeezing_db, detected_variance,
                     load_spectra_csv, max_phase_noise, phase_noise_variance,
                     required_efficiency, required_efficiency_db,
                     shot_noise_normalize)


def test_dark_in_reference_is_the_equivalent_loss():
    for V in (0.02, 0.2, 0.5, 3.0):
        for eta in (0.3, 0.9, 1.0):
            for dark in (0.0, 0.01, 0.2):
                ee = dark_equivalent_efficiency(dark)
                a = detected_variance(V, eta, dark, dark_in_reference=True)
                b = detected_variance(V, eta * ee)
                assert abs(a - b) < 1e-15
                # the literal two-trace ratio it stands for
                ratio = (detected_variance(V, eta) + dark) / (0.5 + dark)
                assert abs(a - 0.5 * ratio) < 1e-15
    # the shot trace itself (vacuum in) reads exactly 0 dB this way,
    # but not in the default convention
    d = dark_from_clearance_db(10.0)
    assert detected_squeezing_db(0.5, 0.8, d, dark_in_reference=True) \
        == pytest.approx(0.0, abs=1e-14)
    assert detected_squeezing_db(0.5, 0.8, d) == pytest.approx(
        10 * np.log10(1.1), abs=1e-14)
    # the default is unchanged (README example 4 value)
    d15 = dark_from_clearance_db(15.0)
    assert detected_squeezing_db(0.05, 0.7, d15) == pytest.approx(
        -3.9618166570, abs=1e-9)
    assert detected_squeezing_db(0.05, 0.7, d15, dark_in_reference=True) \
        == pytest.approx(10 * np.log10((0.185 + d15) / (0.5 + d15)),
                         abs=1e-14)


def test_required_efficiency_inverts_the_whole_budget():
    for Vs, Va in ((0.05, 5.0), (0.2, 1.3)):
        for th in (0.0, 0.02, 0.1):
            for dark in (0.0, 0.004, 0.03):
                for dir_ in (False, True):
                    for eta in (0.35, 0.7, 0.97):
                        Vj = phase_noise_variance(Vs, Va, th)
                        Vt = detected_variance(Vj, eta, dark, dir_)
                        if not Vt < 0.5:
                            continue
                        got = required_efficiency(Vt, Vs, dark, Va, th,
                                                  dir_)
                        assert got == pytest.approx(eta, abs=1e-12)
    # the default path is the old formula, bit for bit
    assert required_efficiency(0.2, 0.05) == (0.5 - 0.2) / (0.5 - 0.05)
    assert required_efficiency_db(-3.0, -10.0) == required_efficiency(
        0.5 * 10 ** -0.3, 0.05)
    # dB front end
    Vt = detected_variance(phase_noise_variance(0.05, 5.0, 0.03), 0.8,
                           0.01)
    assert required_efficiency_db(10 * np.log10(Vt / 0.5), -10.0, 0.01,
                                  10.0, 0.03) == pytest.approx(0.8,
                                                               abs=1e-12)


def test_required_efficiency_refusals():
    # dark noise alone puts -9 dB out of reach of a -10 dB source
    with pytest.raises(ValueError, match="efficiency 1"):
        required_efficiency_db(-9.9, -10.0, dark_noise=0.01)
    with pytest.raises(ValueError, match="v_antisqueezed"):
        required_efficiency(0.2, 0.05, theta_rms=0.05)
    with pytest.raises(ValueError, match="no efficiency"):
        required_efficiency(0.45, 0.05, v_antisqueezed=5.0, theta_rms=1.0)
    with pytest.raises(ValueError, match="dark_noise"):
        required_efficiency(0.2, 0.05, dark_noise=-0.01)


def test_max_phase_noise_inverts_jitter_after_loss_and_dark():
    Vs, Va = 0.05, 6.0
    for eta in (1.0, 0.8):
        for dark in (0.0, 0.01):
            for dir_ in (False, True):
                for th in (0.01, 0.05, 0.12):
                    Vt = detected_variance(phase_noise_variance(Vs, Va, th),
                                           eta, dark, dir_)
                    got = max_phase_noise(Vt, Vs, Va, eta, dark, dir_)
                    assert got == pytest.approx(th, rel=1e-9)
    assert max_phase_noise(0.1, 0.05, 6.3) == max_phase_noise(
        0.1, 0.05, 6.3, 1.0, 0.0)
    with pytest.raises(ValueError, match="after this loss"):
        max_phase_noise(detected_variance(Vs, 0.8) - 1e-6, Vs, Va, 0.8)


def _to_dbm(p_mw):
    return 10 * np.log10(p_mw)


def test_shot_noise_normalize_exact_identities():
    # traces as an analyzer records them: gain g times (optical
    # variance + electronic variance), in dBm, noise free
    g = 3.7e-6                                # mW per vacuum unit
    Vs, eta, Vd = 0.08, 0.75, 0.012
    f = 5
    V_opt = detected_variance(np.full(f, Vs), eta)
    V_shot = np.full(f, 0.5)
    trace = _to_dbm(g * (V_opt + Vd))
    shot = _to_dbm(g * (V_shot + Vd))
    dark = _to_dbm(g * np.full(f, Vd))
    raw = shot_noise_normalize(trace, shot)
    assert raw["sigma_db"] is None and raw["dark_variance"] is None
    assert np.allclose(raw["db"], detected_squeezing_db(
        Vs, eta, Vd, dark_in_reference=True), atol=1e-12, rtol=0)
    sub = shot_noise_normalize(trace, shot, dark)
    assert np.allclose(sub["db"], detected_squeezing_db(Vs, eta),
                       atol=1e-12, rtol=0)
    assert np.allclose(sub["dark_variance"], Vd, rtol=1e-12, atol=0)
    # repeated identical sweeps: same mean, zero scatter
    # the analyzer's shot-to-dark gap is the trace_gap clearance
    gap = shot - dark
    assert np.allclose(dark_from_clearance_db(gap[0], trace_gap=True), Vd,
                       rtol=1e-12, atol=0)
    assert np.allclose(sub["dark_variance"],
                       dark_from_clearance_db(gap[0], trace_gap=True),
                       rtol=1e-12, atol=0)
    rep = shot_noise_normalize(np.tile(trace, (4, 1)),
                               np.tile(shot, (4, 1)),
                               np.tile(dark, (3, 1)))
    assert np.allclose(rep["db"], sub["db"], atol=1e-12, rtol=0)
    assert np.all(rep["sigma_db"] < 1e-9)


def test_shot_noise_normalize_error_bars_match_monte_carlo():
    rng = np.random.default_rng(11)
    means = {"t": 0.2, "s": 0.5 + 0.02, "d": 0.02}   # relative powers
    rel, K, reps = 0.05, 8, 400
    dbs, sig2 = [], []
    for _ in range(reps):
        tr = {k: _to_dbm(m * (1 + rel * rng.standard_normal((K, 1))))
              for k, m in means.items()}
        out = shot_noise_normalize(tr["t"], tr["s"], tr["d"])
        dbs.append(out["db"][0])
        sig2.append(out["sigma_db"][0] ** 2)
    scatter = np.std(dbs, ddof=1)
    reported = np.sqrt(np.mean(sig2))
    assert 0.85 < scatter / reported < 1.15
    exact = 10 * np.log10((0.2 - 0.02) / 0.5)
    assert abs(np.mean(dbs) - exact) < 4 * scatter / np.sqrt(reps)


def test_shot_noise_normalize_refusals_and_csv_checks():
    with pytest.raises(ValueError, match="frequency points"):
        shot_noise_normalize([-80.0, -81.0], [-79.0])
    with pytest.raises(ValueError, match="cannot be subtracted"):
        shot_noise_normalize([-80.0], [-79.0], [-79.5])
    with pytest.raises(ValueError, match="non-finite"):
        shot_noise_normalize([np.nan], [-79.0])
    path = os.path.join(tempfile.mkdtemp(), "s.csv")
    with open(path, "w") as fh:
        fh.write("f_hz,sq_db,anti_db\n1e6,nan,3.0\n")
    with pytest.raises(ValueError, match="non-finite"):
        load_spectra_csv(path)
    with open(path, "w") as fh:
        fh.write("f_hz,sq_db,anti_db,sigma_db\n1e6,-1.0,3.0,0\n")
    with pytest.raises(ValueError, match="sigma_db"):
        load_spectra_csv(path)


def test_clearance_conventions():
    # default: dark relative to the pure shot noise (unchanged)
    assert dark_from_clearance_db(10.0) == 0.05
    # trace gap: (0.5 + V) / V = 10^(c/10), checked by the forward map
    for c in (3.0, 10.0, 15.0, 30.0):
        v = dark_from_clearance_db(c, trace_gap=True)
        assert 10 * np.log10((0.5 + v) / v) == pytest.approx(c, abs=1e-12)
        # the two readings agree when the dark noise is negligible
        assert v > dark_from_clearance_db(c)
    assert dark_from_clearance_db(40.0, trace_gap=True) == pytest.approx(
        dark_from_clearance_db(40.0), rel=2e-4)
    with pytest.raises(ValueError, match="positive"):
        dark_from_clearance_db(0.0, trace_gap=True)
    with pytest.raises(ValueError):
        dark_from_clearance_db(np.nan)
