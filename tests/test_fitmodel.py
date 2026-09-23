"""General spectrum fitting: round trips, identifiability refusals, the
multi-start ambiguity check, and the jitter-average law."""
import numpy as np
import pytest
from scipy.integrate import quad

from sqzcomb import output_quadrature_variance, single_mode_parametric
from sqzcomb.detection import phase_noise_variance
from sqzcomb.fitnoise import (fit_noise_spectra, fit_spectra_model,
                              jitter_average, molecule_spectra_model,
                              parametric_spectra_model)

F = np.linspace(2e6, 300e6, 60)
B = dict(mu=(0.0, 0.99), eta=(0.0, 1.0), kappa_hz=(1e6, 1e9),
         delta=(-2.0, 2.0), phi_sq=(0.0, np.pi), theta_rms=(0.0, 0.5))


def _db(out):
    return {k: 10 * np.log10(v / 0.5) for k, v in out.items()}


def test_jitter_average_is_the_exact_gaussian_average():
    # detuned squeezer, arbitrary LO angle: the closed law must equal a
    # direct numerical average over the Gaussian jitter
    M = single_mode_parametric(0.5 * np.exp(0.3j), 0.4)
    for phi, s, om in ((0.3, 0.1, 0.0), (1.1, 0.25, 0.8)):
        V = lambda a: output_quadrature_variance(M, 0.8, om, 0, 1, phi=a)
        direct = quad(lambda t: V(phi + t) * np.exp(-t * t / (2 * s * s))
                      / np.sqrt(2 * np.pi * s * s), -8 * s, 8 * s)[0]
        law = jitter_average(V(phi), V(phi + np.pi / 2), s)
        assert law == pytest.approx(direct, rel=1e-10)
    # and at the principal axes it is `phase_noise_variance`
    assert jitter_average(0.2, 3.0, 0.1) == pytest.approx(
        phase_noise_variance(0.2, 3.0, 0.1), rel=1e-14)


def test_parametric_model_agrees_with_fit_noise_spectra():
    names, model = parametric_spectra_model()
    true = dict(mu=0.6, eta=0.7, kappa_hz=80e6)
    data = _db(model(true, F))
    fit = fit_spectra_model(F, data, model, names,
                            p0=dict(mu=0.5, eta=0.5, kappa_hz=60e6),
                            bounds=B)
    old = fit_noise_spectra(F, data["squeezed"], data["anti"])
    for k in true:
        assert fit.params[k] == pytest.approx(true[k], rel=1e-7)
        assert fit.params[k] == pytest.approx(getattr(old, k), rel=1e-6)


def test_jitter_model_round_trip_and_error_bars():
    names, model = parametric_spectra_model(jitter=True)
    true = dict(mu=0.6, eta=0.7, kappa_hz=80e6, theta_rms=0.08)
    clean = _db(model(true, F))
    p0 = dict(mu=0.5, eta=0.5, kappa_hz=60e6, theta_rms=0.03)
    fit = fit_spectra_model(F, clean, model, names, p0=p0, bounds=B)
    for k in true:
        assert fit.params[k] == pytest.approx(true[k], rel=1e-6)
    # seeded Monte Carlo: scatter of refits vs reported sigma
    sd, rng = 0.05, np.random.default_rng(3)
    ref = fit_spectra_model(F, {k: v + rng.normal(0, sd, F.size)
                                for k, v in clean.items()}, model, names,
                            p0=true, bounds=B, sigma_db=sd, n_starts=1)
    thetas = []
    for _ in range(30):
        noisy = {k: v + rng.normal(0, sd, F.size) for k, v in clean.items()}
        thetas.append(fit_spectra_model(F, noisy, model, names, p0=true,
                                        bounds=B, sigma_db=sd,
                                        n_starts=1).params["theta_rms"])
    assert np.std(thetas, ddof=1) == pytest.approx(
        ref.sigma["theta_rms"], rel=0.35)
    assert ref.chi2_dof == 2 * F.size - 4


def test_molecule_model_round_trip():
    names, model = molecule_spectra_model("aux")
    true = dict(mu=1.2, J=1.5, gamma=1.2, eta=0.8, kappa_hz=50e6)
    data = _db(model(true, F))
    fit = fit_spectra_model(
        F, data, model, names,
        p0=dict(mu=1.0, J=1.2, gamma=1.0, eta=0.6, kappa_hz=40e6),
        bounds=dict(mu=(0, 5), J=(0, 5), gamma=(0.05, 10), eta=(0, 1),
                    kappa_hz=(1e6, 1e9)))
    for k in true:
        assert fit.params[k] == pytest.approx(true[k], rel=1e-6)


def test_single_trace_is_refused_as_unidentifiable():
    names, model = parametric_spectra_model()
    data = _db(model(dict(mu=0.6, eta=0.7, kappa_hz=80e6), F))
    with pytest.raises(ValueError, match="do not determine"):
        fit_spectra_model(F, {"squeezed": data["squeezed"]}, model, names,
                          p0=dict(mu=0.5, eta=0.5, kappa_hz=60e6),
                          bounds=B)


def test_detuned_pair_is_ambiguous_until_detuning_is_known():
    # With a detuning, two fixed-angle traces are fitted exactly by
    # different parameter sets (for one, delta -> -delta with the LO
    # angle mirrored), and the fit must say so instead of returning
    # whichever it found first.
    names, model = parametric_spectra_model(detuned=True)
    true = dict(mu=0.6, eta=0.7, kappa_hz=80e6, delta=0.3, phi_sq=1.4)
    data = _db(model(true, F))
    p0 = dict(mu=0.5, eta=0.5, kappa_hz=60e6, delta=0.1, phi_sq=1.5)
    with pytest.raises(ValueError, match="equally well"):
        fit_spectra_model(F, data, model, names, p0=p0, bounds=B)
    # the mirror image really is an exact fit
    mirror = dict(true, delta=-0.3, phi_sq=np.pi - 1.4)
    for k, v in _db(model(mirror, F)).items():
        assert np.abs(v - data[k]).max() < 1e-10
    # a detuning known from elsewhere removes the ambiguity
    p0.pop("delta")
    fit = fit_spectra_model(F, data, model, names, p0=p0, bounds=B,
                            fixed=dict(delta=0.3))
    for k in ("mu", "eta", "kappa_hz", "phi_sq"):
        assert fit.params[k] == pytest.approx(true[k], rel=1e-6)


def test_detuning_and_jitter_together_are_refused():
    names, model = parametric_spectra_model(detuned=True, jitter=True)
    true = dict(mu=0.6, eta=0.7, kappa_hz=80e6, delta=0.3, phi_sq=1.4,
                theta_rms=0.08)
    data = _db(model(true, F))
    with pytest.raises(ValueError):
        fit_spectra_model(F, data, model, names,
                          p0={k: v * 1.02 for k, v in true.items()},
                          bounds=B, n_starts=2)


def test_input_refusals():
    names, model = parametric_spectra_model()
    data = _db(model(dict(mu=0.6, eta=0.7, kappa_hz=80e6), F))
    p0 = dict(mu=0.5, eta=0.5, kappa_hz=60e6)
    with pytest.raises(ValueError, match="starting value"):
        fit_spectra_model(F, data, model, names, p0=dict(mu=0.5))
    with pytest.raises(ValueError, match="not parameters"):
        fit_spectra_model(F, data, model, names, p0=dict(p0, foo=1.0))
    with pytest.raises(ValueError, match="match"):
        fit_spectra_model(F, {"squeezed": data["squeezed"][:5]}, model,
                          names, p0=p0)
    with pytest.raises(ValueError):
        molecule_spectra_model("left")
    with pytest.raises(ValueError):
        jitter_average(0.2, 3.0, -0.1)
