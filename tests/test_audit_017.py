"""0.17 regression tests: the wrong answer found by the third audit,
and the equality check of the speed change.

The error-bar test fails on 0.16.0 and passes now. Its references are
independent of the code under test: seeded Monte Carlo scatter, and
first-order propagation written out here from the covariance the fit
reports. The faster molecule model is checked against the old code
path, kept here verbatim.
"""
import numpy as np
import pytest

from sqzcomb import fit_resonance_transmission, through_transmission

SCAN = np.linspace(-300e6, 300e6, 121)


@pytest.mark.parametrize("eta,branch", [(0.8, "over"), (0.3, "under")])
def test_kappa_ex_and_kappa_0_error_bars(eta, branch):
    """kappa_ex = eta kappa and kappa_0 = (1 - eta) kappa: their error
    bars need d eta / dD = -1 / (4 sqrt(1 - D)) on the over-coupled
    branch and +1 / (4 sqrt(1 - D)) on the under-coupled one. 0.16.0 had
    both signs reversed, which flipped the (large, negative) kappa-D
    correlation term: for this over-coupled ring (kappa = 100 MHz,
    eta = 0.8, 1 % noise) it reported 0.325 MHz for kappa_0 against a
    Monte Carlo scatter of 0.17 MHz, and 0.768 MHz for kappa_ex against
    0.90 MHz; for the under-coupled one (eta = 0.3) 0.427 MHz for
    kappa_ex against 0.19 MHz."""
    sig = 0.01
    T = through_transmission(SCAN, 0.0, 100e6, eta)
    ref = fit_resonance_transmission(SCAN, T, sigma=sig, branch=branch)
    # independent first-order propagation from the reported covariance
    # of (kappa, D), with the derivative of eta(D) by central difference
    C = ref["covariance"][1:3, 1:3]
    D, kap = ref["depth"], ref["kappa_hz"]
    sgn = 1.0 if branch == "over" else -1.0

    def eta_of(d):
        return 0.5 * (1.0 + sgn * np.sqrt(1.0 - d))

    h = 1e-7
    de = (eta_of(D + h) - eta_of(D - h)) / (2 * h)
    for key, g in (("kappa_ex_hz", [eta_of(D), kap * de]),
                   ("kappa_0_hz", [1.0 - eta_of(D), -kap * de])):
        g = np.array(g)
        assert ref[key + "_sigma"] == pytest.approx(np.sqrt(g @ C @ g),
                                                    rel=1e-6)
    # and the scatter of seeded noisy scans
    rng = np.random.default_rng(3)
    k_ex, k_0 = [], []
    for _ in range(200):
        r = fit_resonance_transmission(
            SCAN, T + sig * rng.standard_normal(SCAN.size), sigma=sig,
            branch=branch)
        k_ex.append(r["kappa_ex_hz"])
        k_0.append(r["kappa_0_hz"])
    for key, draws in (("kappa_ex_hz", k_ex), ("kappa_0_hz", k_0)):
        ratio = np.std(draws, ddof=1) / ref[key + "_sigma"]
        assert 0.8 < ratio < 1.25, (key, ratio)


# ----------------------------------------------------------------------
# Performance change of 0.17: the old code path is kept here as the
# reference.


def _molecule_model_per_call(p, f_hz, port, detuned):
    """`molecule_spectra_model`'s model up to 0.16.0, verbatim: one
    full `output_variance_ports` call (checks, stability test,
    resolvent) per frequency and angle."""
    from sqzcomb import output_variance_ports, photonic_molecule
    pm = 0 if port == "main" else 1
    ph0 = np.pi / 2.0 if port == "main" else 0.0
    M, g = photonic_molecule(float(p["mu"]), float(p["J"]),
                             float(p.get("delta_a", 0.0)),
                             float(p.get("delta_b", 0.0)),
                             float(p["gamma"]))
    om = 4.0 * np.pi * np.asarray(f_hz, dtype=float) / (
        2.0 * np.pi * float(p["kappa_hz"]))
    eta = float(p["eta"])
    ph_sq = float(p["phi_sq"]) if detuned else ph0
    v_sq = np.array([output_variance_ports(M, g, eta, pm, o,
                                           phi=ph_sq) for o in om])
    v_an = np.array([output_variance_ports(
        M, g, eta, pm, o, phi=ph_sq + np.pi / 2.0) for o in om])
    return {"squeezed": v_sq, "anti": v_an}


@pytest.mark.parametrize("port", ["main", "aux"])
@pytest.mark.parametrize("detuned", [False, True])
def test_molecule_model_matches_the_per_call_path(port, detuned):
    """The model now runs the checks once and builds the resolvent once
    per frequency for both angles, with the same operations per value as
    `output_variance_ports`. Compared to a relative 1e-13 (the values
    are O(1) variances; a different order of operations would show up at
    the 1e-16 level, and the bound leaves room for any machine)."""
    from sqzcomb import molecule_spectra_model
    f = np.linspace(0.5e6, 40e6, 17)
    names, model = molecule_spectra_model(port, detuned=detuned)
    p = dict(mu=1.5, J=1.2, gamma=0.8, eta=0.85, kappa_hz=10e6)
    if detuned:
        p.update(delta_a=0.2, delta_b=-0.1, phi_sq=0.1)
    new = model(p, f)
    ref = _molecule_model_per_call(p, f, port, detuned)
    for k in ("squeezed", "anti"):
        assert new[k].shape == ref[k].shape
        assert np.allclose(new[k], ref[k], rtol=1e-13, atol=0)
    # an unstable molecule is still refused, as before
    p["mu"] = 5.0
    with pytest.raises(ValueError, match="unstable"):
        model(p, f)
