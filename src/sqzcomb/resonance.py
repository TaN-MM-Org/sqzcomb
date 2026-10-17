"""Linewidth and escape efficiency from a linear transmission scan
(new in 0.16).

`RingSpec` needs the loaded linewidth kappa and the escape efficiency
eta = kappa_ex / kappa, and every detected-squeezing prediction scales
with eta. In a laboratory both come from the same routine measurement:
a slow, low-power laser sweep across one resonance, recording the power
transmitted past the ring on the bus waveguide (the through port). This
module fits that dip and returns kappa and eta with error bars.

The model is the package's own input-output relation for a passive
mode, applied to a coherent input instead of noise. With the escape
fraction eta, the through-port amplitude at a laser detuning
x = 2 (f - f0) / kappa (in units of the half linewidth) is the mean-field
transfer of `output_quadrature_variance`'s T_ex = 2 eta G - 1 with
G = 1 / (1 - i x),

    t = 1 - 2 eta / (1 - i x),
    |t|^2 = ((1 - 2 eta)^2 + x^2) / (1 + x^2) = 1 - D / (1 + x^2),
    D = 4 eta (1 - eta)   (the depth of the dip),

a Lorentzian dip of full width at half depth kappa (the loaded
linewidth) and on-resonance transmission (1 - 2 eta)^2. The test suite
derives |t|^2 again from the resolvent of the package's drift matrix,
a separate code path. The on-resonance value is the familiar
((kappa_0 - kappa_ex) / (kappa_0 + kappa_ex))^2 of a resonator coupled to
a waveguide, which vanishes at critical coupling (kappa_ex = kappa_0,
eta = 1/2); critical coupling was observed with a fiber taper and a
silica microsphere by M. Cai, O. Painter and K. J. Vahala, Phys. Rev.
Lett. 85, 74 (2000).

The two-fold answer, stated plainly. D is symmetric under
eta -> 1 - eta: an under-coupled ring (eta < 1/2) and an over-coupled
one (eta > 1/2) with the same loaded linewidth give the same
transmission dip. The power transmission alone cannot decide between
them, so the fit returns both, eta_under = (1 - sqrt(1 - D)) / 2 and
eta_over = (1 + sqrt(1 - D)) / 2, and gives `eta_esc` only for the
branch you name. What decides it is outside this scan: the phase of the
transmitted field (the phase response of t winds through 2 pi across an
over-coupled resonance and not across an under-coupled one), or how the
dip changes when the coupling gap is changed. For a squeezing ring,
which is designed to be strongly over-coupled, `branch="over"` is the
usual case, but it is a fact about your device to check, not a default.

What the fit does not model: a resonance split into a doublet by
backscattering, a dip distorted by thermal or Kerr shifts at too high a
power (a triangular shape), and Fano-shaped dips. With measurement
errors `sigma` given, a chi-square test refuses data the single
Lorentzian does not describe. Without them the fit cannot tell, so look
at the residuals (`model` is returned).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import chi2 as _chi2

__all__ = ["through_transmission", "fit_resonance_transmission"]


def through_transmission(f_hz, f0_hz: float, kappa_hz: float,
                         eta_esc: float, baseline: float = 1.0):
    """Power transmission past one resonance on the bus waveguide.

    f_hz : laser frequencies (Hz; any common offset); f0_hz : the
    resonance frequency; kappa_hz : loaded linewidth (FWHM, Hz);
    eta_esc : escape efficiency kappa_ex / kappa in [0, 1]; baseline :
    the off-resonance transmission (any power unit).

    Returns baseline * (1 - 4 eta (1 - eta) / (1 + x^2)) with
    x = 2 (f - f0) / kappa (see the module docstring).
    """
    k = float(kappa_hz)
    e = float(eta_esc)
    if not (np.isfinite(k) and k > 0.0):
        raise ValueError("kappa_hz must be finite and positive")
    if not (0.0 <= e <= 1.0):
        raise ValueError("eta_esc must lie in [0, 1]")
    x = 2.0 * (np.asarray(f_hz, dtype=float) - float(f0_hz)) / k
    return float(baseline) * (1.0 - 4.0 * e * (1.0 - e) / (1.0 + x * x))


def _dip(p, u, fit_slope):
    """Model in scaled units: p = (u0, w, D, b0[, b1]), u the scaled
    frequency, w the scaled linewidth."""
    u0, w, D, b0 = p[:4]
    base = b0 + (p[4] * u if fit_slope else 0.0)
    x = 2.0 * (u - u0) / w
    return base * (1.0 - D / (1.0 + x * x))


def fit_resonance_transmission(f_hz, transmission, sigma=None,
                               branch=None, fit_slope: bool = False,
                               min_p_value: float = 1e-6) -> dict:
    """Fit the loaded linewidth and the escape efficiency to a measured
    through-port transmission dip.

    f_hz : laser frequencies of the scan (Hz; an offset such as the
        detuning from a reference is fine, only differences matter).
    transmission : the transmitted power at each frequency, in any
        linear unit (not dB); the off-resonance level is fitted.
    sigma : optional 1-sigma error of each transmission value (scalar
        or per point, same unit). With it the error bars are the
        weighted least-squares ones and the fit is refused when a
        chi-square test rejects the single-Lorentzian model at
        p < `min_p_value` (a split or thermally distorted resonance).
        Without it the error bars are scaled by the scatter of the
        residuals, as in `fit_noise_spectra`.
    branch : None, "under" or "over": which of the two escape
        efficiencies that fit the dip equally well to report as
        `eta_esc` (see the module docstring; the scan cannot decide).
    fit_slope : also fit a linear tilt of the off-resonance level (a
        laser power that drifts during the sweep).

    Returns a dict:
        f0_hz, kappa_hz : resonance frequency and loaded linewidth
            (FWHM), with `sigma` holding their 1-sigma errors;
        depth : D = 4 eta (1 - eta); on_resonance : 1 - D, the
            transmission at f0 relative to the off-resonance level;
        baseline (and slope_per_hz with fit_slope) : the off-resonance
            level at the center of the scan;
        eta_under, eta_over : the two escape efficiencies, (1 -/+
            sqrt(1 - D)) / 2, with the common error `eta_sigma`
            = sigma_D / (4 sqrt(1 - D)) (first order; None when 1 - D
            is within 3 sigma_D of zero, where the two branches are not
            separated);
        critical_consistent : True when the depth is within 3 sigma_D
            of D = 1 (critical coupling, eta = 1/2);
        eta_esc, eta_esc_sigma, kappa_ex_hz, kappa_0_hz and their
            sigmas : for the named `branch` only (kappa_ex = eta kappa,
            kappa_0 = (1 - eta) kappa, errors propagated with the
            correlation of D and kappa);
        chi2, chi2_dof, p_value : with sigma (None otherwise);
        model : the fitted transmission at f_hz.

    Refused (ValueError): fewer than 8 points; non-finite input or a
    non-positive sigma; a scan that does not reach one linewidth beyond
    the resonance on both sides (the off-resonance level, and with it
    the depth, would be extrapolated); fewer than 5 points inside the
    linewidth (the width is not resolved); a dip not significantly
    deeper than zero (below 5 sigma); and, with sigma, a chi-square
    p-value below `min_p_value`.
    """
    f = np.asarray(f_hz, dtype=float).ravel()
    T = np.asarray(transmission, dtype=float).ravel()
    if f.shape != T.shape:
        raise ValueError("f_hz and transmission must have the same length")
    if branch not in (None, "under", "over"):
        raise ValueError("branch must be None, 'under' or 'over'")
    n_par = 5 if fit_slope else 4
    if f.size < 8:
        raise ValueError("need at least 8 points across the resonance")
    if not (np.all(np.isfinite(f)) and np.all(np.isfinite(T))):
        raise ValueError("frequencies and transmission must be finite")
    if np.unique(f).size != f.size:
        raise ValueError("the scan frequencies must be distinct")
    if sigma is None:
        w = np.ones(f.size)
    else:
        sig = np.broadcast_to(np.asarray(sigma, dtype=float), f.shape)
        if not (np.all(np.isfinite(sig)) and np.all(sig > 0.0)):
            raise ValueError("sigma must be finite and positive")
        w = 1.0 / sig
    pmin = float(min_p_value)
    if not (0.0 <= pmin < 1.0):
        raise ValueError("min_p_value must lie in [0, 1)")

    # scaled coordinates: u in [-1/2, 1/2] over the scan, T / its peak
    fc = 0.5 * (f.min() + f.max())
    span = float(f.max() - f.min())
    u = (f - fc) / span
    tscale = float(np.max(np.abs(T)))
    if not tscale > 0.0:
        raise ValueError("the transmission is zero everywhere")
    y = T / tscale
    ws = w * tscale

    # starting values from the data
    b0 = float(np.percentile(y, 90))
    i0 = int(np.argmin(y))
    D0 = float(np.clip(1.0 - y[i0] / b0, 0.05, 0.95)) if b0 > 0 else 0.5
    below = u[y < b0 * (1.0 - 0.5 * D0)]
    w0 = float(below.max() - below.min()) if below.size > 1 else 0.0
    w0 = max(w0, 3.0 / f.size)
    x0 = [u[i0], w0, D0, b0] + ([0.0] if fit_slope else [])
    lo = [-1.0, 1e-9, 0.0, 0.0] + ([-np.inf] if fit_slope else [])
    hi = [1.0, 10.0, 1.0, np.inf] + ([np.inf] if fit_slope else [])

    def resid(p):
        return ws * (_dip(p, u, fit_slope) - y)

    res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac",
                        xtol=1e-15, ftol=1e-15, gtol=1e-15)
    if not res.success:
        raise RuntimeError(f"transmission fit failed: {res.message}")
    p = res.x
    J = res.jac
    dof = f.size - n_par
    try:
        cov = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        raise ValueError("the scan does not determine the dip (singular "
                         "information matrix)") from None
    rss = float(2.0 * res.cost)
    chi2 = chi2_dof = p_value = None
    if sigma is None:
        cov = cov * (rss / dof)
    else:
        chi2, chi2_dof = rss, dof
        p_value = float(_chi2.sf(rss, dof))
    sd = np.sqrt(np.abs(np.diag(cov)))

    f0 = fc + p[0] * span
    kap = p[1] * span
    D = float(p[2])
    sD = float(sd[2])
    s_f0, s_kap = sd[0] * span, sd[1] * span

    # what the scan can and cannot support
    if not D > 5.0 * sD:
        raise ValueError(
            f"no resolved dip: depth {D:.3g} +/- {sD:.2g} is not "
            "significantly above zero")
    if not (f.min() <= f0 - kap and f.max() >= f0 + kap):
        raise ValueError(
            f"the scan ({f.min():.6g} to {f.max():.6g} Hz) does not reach "
            f"one linewidth ({kap:.4g} Hz) beyond the resonance at "
            f"{f0:.6g} Hz on both sides; the off-resonance level and the "
            "depth would be extrapolated. Widen the sweep")
    inside = int(np.sum(np.abs(f - f0) <= 0.5 * kap))
    if inside < 5:
        raise ValueError(
            f"only {inside} points lie inside the linewidth ({kap:.4g} "
            "Hz); the width is not resolved. Use a finer sweep")
    if p_value is not None and p_value < pmin:
        raise ValueError(
            f"a single Lorentzian dip does not describe these data "
            f"(chi^2 = {chi2:.4g} for {chi2_dof} degrees of freedom, "
            f"p = {p_value:.2g}): a split (backscattering) doublet, a "
            "dip distorted by heating at too high a power, or "
            "underestimated errors")

    root = np.sqrt(max(1.0 - D, 0.0))
    eta_under = 0.5 * (1.0 - root)
    eta_over = 0.5 * (1.0 + root)
    critical = bool(1.0 - D <= 3.0 * sD)
    eta_sigma = None if critical else float(sD / (4.0 * root))
    scale = np.array([span, span, 1.0, tscale] + ([tscale / span]
                                                  if fit_slope else []))
    cov_phys = cov * np.outer(scale, scale)
    out = {"f0_hz": float(f0), "kappa_hz": float(kap), "depth": D,
           "on_resonance": 1.0 - D, "baseline": float(p[3] * tscale),
           "eta_under": float(eta_under), "eta_over": float(eta_over),
           "eta_sigma": eta_sigma, "critical_consistent": critical,
           "sigma": {"f0_hz": float(s_f0), "kappa_hz": float(s_kap),
                     "depth": sD, "baseline": float(sd[3] * tscale)},
           "covariance": cov_phys,
           "chi2": chi2, "chi2_dof": chi2_dof, "p_value": p_value,
           "model": _dip(p, u, fit_slope) * tscale}
    if fit_slope:
        out["slope_per_hz"] = float(p[4] * tscale / span)
        out["sigma"]["slope_per_hz"] = float(sd[4] * tscale / span)
    if branch is not None:
        eta = eta_under if branch == "under" else eta_over
        out["eta_esc"] = float(eta)
        out["eta_esc_sigma"] = eta_sigma
        out["kappa_ex_hz"] = float(eta * kap)
        out["kappa_0_hz"] = float((1.0 - eta) * kap)
        if eta_sigma is None:
            out["kappa_ex_hz_sigma"] = out["kappa_0_hz_sigma"] = None
        else:
            # d eta / dD = -/+ 1 / (4 sqrt(1 - D)) for under / over
            de = (1.0 if branch == "over" else -1.0) / (4.0 * root)
            c_kk, c_kd, c_dd = cov_phys[1, 1], cov_phys[1, 2], \
                cov_phys[2, 2]
            for key, g_k, g_d in (("kappa_ex_hz", eta, kap * de),
                                  ("kappa_0_hz", 1.0 - eta, -kap * de)):
                var = g_k * g_k * c_kk + 2 * g_k * g_d * c_kd \
                    + g_d * g_d * c_dd
                out[key + "_sigma"] = float(np.sqrt(max(var, 0.0)))
    return out
