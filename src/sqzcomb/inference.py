"""Infer the source from a measured squeezing/anti-squeezing pair
(new in v0.12).

The routine characterization of the 2025-2026 on-chip squeezing
literature: measure the squeezed variance S and the antisqueezed
variance A after all losses, then infer what the source made before
the losses ate it (used e.g. by Ulanov et al., Nat. Commun. 16,
10791 (2025); Karnik et al., arXiv:2605.27607, infer their 18 dB in
thin-film lithium niobate with a richer distributed loss-and-squeezing
model that also includes phase noise). Under the standard model -- a
PURE squeezed state
(V_s V_a = 1/4) followed by a beamsplitter loss of transmission
eta -- the measured pair determines both unknowns in closed form:

    S = eta V_s + (1 - eta)/2,   A = eta V_a + (1 - eta)/2,
    V_s V_a = 1/4
    =>  eta = (2(S + A) - 1 - 4 S A) / (2(S + A) - 2),

then V_s = (S - (1-eta)/2)/eta. This is algebra, not fitting, and
the tests hold the inversion as an exact round trip through the
package's own `detected_variance` -- two code paths, one identity.

Honest limits, stated plainly: the pure-state assumption is the
model, and it is what makes "inferred squeezing" a model statement
rather than a measurement -- excess phase noise or thermal
contamination make the source impurer than assumed, in which case
the inferred numbers flatter it. The refusals catch what the
algebra can catch: a measured product S A < 1/4 is impossible for
ANY source after loss (the uncertainty product only grows), an
antisqueezed variance at or below vacuum leaves nothing to infer
from, and an implied efficiency outside (0, 1] is reported as the
inconsistency it is.

Impure sources and phase jitter (new in 0.13). Two further facts can
be supplied when they are known from elsewhere:

* purity P of the source state (P = 1 pure). A single-mode Gaussian
  state with principal variances V_s, V_a has V_s V_a = 1 / (4 P^2);
  excess antisqueezing from an impure source raises that product.
* theta_rms, the RMS Gaussian phase jitter of the detection (the model
  of `phase_noise_variance`), which mixes a fraction of each measured
  quadrature into the other.

Jitter is undone exactly first: the sum S + A is unchanged by it and
the difference shrinks by exp(-2 theta_rms^2). Loss and jitter are both
affine maps with the vacuum as fixed point, so their order does not
matter (asserted in the tests of `detection`). The loss equation then
becomes quadratic in eta,

    (Q - 1/4) eta^2 - ((s + a) / 2) eta - s a = 0,
    s = S0 - 1/2,  a = A0 - 1/2,  Q = 1 / (4 P^2),

(for P = 1 it is linear and gives the closed form above, which the
code keeps as its own branch so that the pure case is unchanged
digit for digit). For an impure source it can have two physical
roots: two different (eta, source) pairs that produce the same
measurement. The function then refuses unless told which one
(`branch="high"` or `"low"` efficiency), because the pair alone
cannot decide.

What the pair can never decide is the purity itself: two measured
numbers cannot determine three unknowns (eta, squeezing, purity). The
tests show which way the error goes: at a fixed measurement, assuming
the source purer than it is gives a LOWER efficiency and a MORE
squeezed source than the truth, i.e. the pure-state inference
flatters the source.
"""
from __future__ import annotations

import numpy as np

__all__ = ["infer_source", "fit_squeezing_pairs"]

_VAC = 0.5


def infer_source(sq_db: float, anti_db: float, sigma_db=None,
                 purity: float = 1.0, theta_rms: float = 0.0,
                 branch=None, calib_sigma_db=None) -> dict:
    """Closed-form source inference from one measured pair.

    sq_db, anti_db : measured squeezed and antisqueezed levels in dB
        relative to shot noise (squeezed negative, antisqueezed
        positive, the spectrum-analyzer convention of `fitnoise`).
    sigma_db : optional 1-sigma uncertainty of each dB value; the
        inferred quantities' error bars follow by exact derivatives
        of the closed form (central differences on the formula),
        validated against seeded Monte Carlo in the tests.
    purity : known purity P in (0, 1] of the source state (default 1,
        the pure-state model). Must come from an independent
        measurement; the pair cannot determine it.
    theta_rms : known RMS phase jitter of the detection in radians
        (default 0).
    branch : "high" or "low": which efficiency to return when an
        impure source makes the answer two-valued (see the module
        docstring); None refuses in that case.

    calib_sigma_db : optional 1-sigma uncertainty (dB) of the
        shot-noise reference both levels were measured against (new in
        0.15). It moves both measured levels by the same number of dB,
        so it is propagated with full correlation: with g the gradient
        of a result with respect to (sq_db, anti_db), the variance is
        sigma_db^2 |g|^2 + calib_sigma_db^2 (g_sq + g_anti)^2. Either
        uncertainty may be given alone.

    Returns dict(eta, v_s, v_a, sq_db_source, anti_db_source) plus,
    with sigma_db or calib_sigma_db, dict entries eta_sigma and
    sq_db_source_sigma (the total 1-sigma error bars).
    """
    s_db, a_db = float(sq_db), float(anti_db)
    if not (np.isfinite(s_db) and np.isfinite(a_db)):
        raise ValueError("the measured dB values must be finite")
    P = float(purity)
    if not (np.isfinite(P) and 0.0 < P <= 1.0):
        raise ValueError("purity must lie in (0, 1]")
    th = float(theta_rms)
    if not (np.isfinite(th) and th >= 0.0):
        raise ValueError("theta_rms must be finite and >= 0")
    if branch not in (None, "high", "low"):
        raise ValueError("branch must be None, 'high' or 'low'")
    sig = c_sig = 0.0
    if sigma_db is not None:
        sig = float(sigma_db)
        if not (np.isfinite(sig) and sig > 0.0):
            raise ValueError("sigma_db must be finite and positive")
    if calib_sigma_db is not None:
        c_sig = float(calib_sigma_db)
        if not (np.isfinite(c_sig) and c_sig > 0.0):
            raise ValueError("calib_sigma_db must be finite and positive")
    out = _invert(s_db, a_db, P, th, branch)
    if sigma_db is not None or calib_sigma_db is not None:
        h = 1e-6
        d_eta = np.zeros(2)
        d_src = np.zeros(2)
        for i, (ds, da) in enumerate(((h, 0.0), (0.0, h))):
            p = _invert(s_db + ds, a_db + da, P, th, out.get("branch"))
            m = _invert(s_db - ds, a_db - da, P, th, out.get("branch"))
            d_eta[i] = (p["eta"] - m["eta"]) / (2.0 * h)
            d_src[i] = (p["sq_db_source"] - m["sq_db_source"]) \
                / (2.0 * h)
        if calib_sigma_db is None:
            # unchanged digit for digit from 0.14 when only sigma_db
            out["eta_sigma"] = float(sig * np.linalg.norm(d_eta))
            out["sq_db_source_sigma"] = float(sig
                                              * np.linalg.norm(d_src))
        else:
            out["eta_sigma"] = float(np.sqrt(
                (sig * np.linalg.norm(d_eta)) ** 2
                + (c_sig * d_eta.sum()) ** 2))
            out["sq_db_source_sigma"] = float(np.sqrt(
                (sig * np.linalg.norm(d_src)) ** 2
                + (c_sig * d_src.sum()) ** 2))
    out.pop("branch", None)
    return out


def _invert(s_db, a_db, P=1.0, theta=0.0, branch=None):
    S = _VAC * 10.0 ** (s_db / 10.0)
    A = _VAC * 10.0 ** (a_db / 10.0)
    if A <= _VAC:
        raise ValueError(
            "the antisqueezed level must lie above shot noise: with "
            "no visible antisqueezing there is nothing to infer the "
            "source from (check the trace assignment)")
    if S >= _VAC:
        raise ValueError(
            "the squeezed level must lie below shot noise; a pair "
            "with no squeezing implies zero detection efficiency "
            "under this model, which is not an inference")
    if theta > 0.0:
        # undo the jitter: S + A is invariant, S - A shrinks by
        # exp(-2 theta^2)
        grow = np.exp(2.0 * theta * theta)
        S, A = (0.5 * ((S + A) + (S - A) * grow),
                0.5 * ((S + A) - (S - A) * grow))
        if S <= 0.0 or S >= _VAC:
            raise ValueError(
                "with the stated phase jitter the pair implies a "
                "squeezed variance before the jitter that is not "
                f"between 0 and shot noise ({S:.4g}): the jitter is "
                "larger than this measurement allows")
    Q = 0.25 / (P * P)
    if S * A < _VAC ** 2 * (1.0 - 1e-12):
        raise ValueError(
            f"the measured uncertainty product S*A = {S * A:.6g} is "
            f"below the vacuum product {_VAC ** 2}: impossible for "
            "any source followed by loss (loss only grows the "
            "product). Check the shot-noise calibration")
    if P == 1.0:
        etas = [(2.0 * (S + A) - 1.0 - 4.0 * S * A)
                / (2.0 * (S + A) - 2.0)]
    else:
        s, a = S - _VAC, A - _VAC
        qa, qb, qc = Q - 0.25, -0.5 * (s + a), -s * a
        disc = qb * qb - 4.0 * qa * qc
        if disc < 0.0:
            etas = []
        else:
            # numerically stable roots (no cancellation as qa -> 0,
            # i.e. purity -> 1, where one root runs off to infinity
            # and the other tends to the pure-state closed form)
            q = -0.5 * (qb + np.copysign(np.sqrt(disc), qb))
            etas = sorted({q / qa, qc / q})
        # physical: 0 < eta <= 1 and a positive source variance
        etas = [e for e in etas
                if 0.0 < e <= 1.0 + 1e-12 and (S - (1 - e) * _VAC) > 0]
        if not etas:
            raise ValueError(
                f"no efficiency in (0, 1] reproduces this pair with a "
                f"source of purity {P:g}: the stated purity is "
                "inconsistent with the measurement")
    if len(etas) == 2:
        if branch is None:
            raise ValueError(
                f"two efficiencies, {etas[0]:.6g} and {etas[1]:.6g}, "
                f"both reproduce this pair with a purity-{P:g} source; "
                "the pair cannot decide between them. Pass branch="
                "'low' or 'high' if you know which applies")
        eta = etas[1] if branch == "high" else etas[0]
    else:
        eta = etas[0]
    if not (0.0 < eta <= 1.0 + 1e-12):
        raise ValueError(
            f"the pair implies detection efficiency {eta:.4g}, "
            "outside (0, 1]: inconsistent with a pure squeezed "
            "source plus loss")
    eta = min(eta, 1.0)
    v_s = (S - (1.0 - eta) * _VAC) / eta
    v_a = (A - (1.0 - eta) * _VAC) / eta
    return {"eta": float(eta), "v_s": float(v_s), "v_a": float(v_a),
            "sq_db_source": float(10.0 * np.log10(v_s / _VAC)),
            "anti_db_source": float(10.0 * np.log10(v_a / _VAC)),
            "branch": branch}


# ----------------------------------------------------------------------
# Loss and phase jitter from pairs at several gains (new in 0.17)


def _pairs_model(eta, mix, g, dark, dark_in_reference):
    """Detected squeezed and antisqueezed variances (vacuum 1/2) of pure
    sources V_s = e^{-g}/2, V_a = e^{g}/2 after a mixing fraction `mix`
    of the other quadrature (phase jitter, mix = <sin^2 theta>), the
    efficiency eta and the dark noise (as `detected_variance`)."""
    v_s = 0.5 * np.exp(-g)
    v_a = 0.5 * np.exp(g)
    s = (1.0 - mix) * v_s + mix * v_a
    a = (1.0 - mix) * v_a + mix * v_s
    S = eta * s + (1.0 - eta) * _VAC + dark
    A = eta * a + (1.0 - eta) * _VAC + dark
    if dark_in_reference:
        S = _VAC * S / (_VAC + dark)
        A = _VAC * A / (_VAC + dark)
    return S, A


def fit_squeezing_pairs(sq_db, anti_db, sigma_db, sigma_anti_db=None,
                        fit_jitter: bool = True, dark_noise: float = 0.0,
                        dark_in_reference: bool = False,
                        cond_max: float = 1e10) -> dict:
    """Total efficiency and phase jitter from squeezed / antisqueezed
    pairs measured at several pump levels (new in 0.17).

    One measured pair cannot separate loss from phase jitter (two
    numbers, three unknowns: see `infer_source`). Pairs taken at
    several pump levels with the same detection chain can: loss pulls
    every pair toward shot noise by the same fraction, while jitter
    mixes in a fraction of the antisqueezed variance, which grows with
    the gain. Squeezing experiments describe their data in this way
    (for example B. Shajilal et al., arXiv:2211.06632 (2022), model
    their squeezing and antisqueezing against pump power with a total
    efficiency and a jitter angle).

    Model, per pair i: a pure squeezed source V_s = e^{-g_i}/2,
    V_a = e^{g_i}/2 (one free g_i per pair, so no pump-to-gain law is
    assumed), a jitter that mixes the fraction m = <sin^2 theta> of the
    other quadrature into each (for Gaussian jitter of RMS theta_rms,
    m = (1 - exp(-2 theta_rms^2)) / 2, the law of
    `phase_noise_variance`), then the efficiency eta and the known dark
    noise as in `detected_variance`. Loss and jitter commute, and the
    output of the package's single-mode squeezer
    (`single_mode_parametric` without detuning) at any analysis
    frequency is exactly such a pure source behind its escape
    efficiency, so eta is the escape efficiency times the detection
    efficiency (both checked in the tests).

    sq_db, anti_db : the N measured pairs, dB relative to shot noise
        (squeezed below 0 dB, antisqueezed above).
    sigma_db : 1-sigma error of each squeezed level (scalar or one per
        pair); sigma_anti_db : the same for the antisqueezed levels
        (default: sigma_db). Required: the fit weighs the two traces
        against each other and tests the model with chi-square.
    fit_jitter : False holds m = 0 (loss only); then one pair is
        enough and reproduces `infer_source`.
    dark_noise, dark_in_reference : a known dark-noise variance (vacuum
        units) and its convention, as in `detected_variance`; leave 0
        when the traces were dark-subtracted.
    cond_max : largest allowed squared condition number of the
        column-scaled Jacobian; beyond it the pairs do not separate the
        parameters (for example all taken at about the same gain) and
        the fit refuses.

    Returns a dict:
        eta, eta_sigma : total efficiency;
        mix, mix_sigma : the mixing fraction m (0 without jitter);
        theta_rms : the Gaussian RMS jitter (rad) for that m;
        theta_rms_sigma : its error (first order), or None when m is
            within 3 mix_sigma of zero (the data are then consistent
            with no jitter and an error bar on theta is not meaningful);
        theta_rms_upper : theta_rms at m + 2 mix_sigma (an upper limit
            at about 97.7 % for a Gaussian error of m);
        sq_db_source, anti_db_source, sq_db_source_sigma : each pair's
            source level before loss and jitter (arrays);
        chi2, chi2_dof, p_value : the model check (dof = N - 2 with
            jitter, N - 1 without; p_value is None at dof 0);
        model_sq_db, model_anti_db : the fitted levels;
        names, covariance : fitted parameter names and their covariance
            (eta, [mix,] g_1..g_N).

    Refused (ValueError): fewer than 2 pairs with fit_jitter (1 without),
    inputs of different lengths, non-finite values, a squeezed level at
    or above shot noise or an antisqueezed level at or below it, a
    non-positive sigma, and pairs that do not determine the parameters.
    """
    from scipy.optimize import least_squares
    from scipy.stats import chi2 as _chi2
    from .detection import _check_dark

    s = np.atleast_1d(np.asarray(sq_db, dtype=float)).ravel()
    a = np.atleast_1d(np.asarray(anti_db, dtype=float)).ravel()
    n = s.size
    if a.shape != s.shape:
        raise ValueError("sq_db and anti_db must have the same length")
    n_min = 2 if fit_jitter else 1
    if n < n_min:
        raise ValueError(
            f"need at least {n_min} pair(s): one pair has two numbers for "
            "three unknowns (efficiency, jitter, source); measure pairs "
            "at several pump levels" if fit_jitter else
            "need at least one pair")
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(a))):
        raise ValueError("the measured dB values must be finite")
    if np.any(s >= 0.0):
        raise ValueError("every squeezed level must lie below shot noise "
                         "(0 dB)")
    if np.any(a <= 0.0):
        raise ValueError("every antisqueezed level must lie above shot "
                         "noise (0 dB)")
    sig_s = np.broadcast_to(np.asarray(sigma_db, dtype=float),
                            s.shape).astype(float)
    sig_a = sig_s if sigma_anti_db is None else np.broadcast_to(
        np.asarray(sigma_anti_db, dtype=float), s.shape).astype(float)
    for sg in (sig_s, sig_a):
        if not (np.all(np.isfinite(sg)) and np.all(sg > 0.0)):
            raise ValueError("sigma_db and sigma_anti_db must be finite "
                             "and positive")
    dark = _check_dark(dark_noise)
    dir_ = bool(dark_in_reference)
    w = np.concatenate([1.0 / sig_s, 1.0 / sig_a])
    data = np.concatenate([s, a])

    def unpack(p):
        if fit_jitter:
            return p[0], p[1], p[2:]
        return p[0], 0.0, p[1:]

    def model_db(p):
        eta, mix, g = unpack(p)
        S, A = _pairs_model(eta, mix, g, dark, dir_)
        return 10.0 * np.log10(np.concatenate([S, A]) / _VAC)

    def resid(p):
        return w * (model_db(p) - data)

    # starting values: the loss-only inversion of each pair, where it
    # exists, gives the efficiency; each g from the antisqueezed level
    etas = []
    for si, ai in zip(s, a):
        try:
            etas.append(_invert(si, ai)["eta"])
        except ValueError:
            pass
    eta0 = float(np.clip(np.median(etas) if etas else 0.5, 0.05, 1.0))

    def g_start(e):
        A = _VAC * 10.0 ** (a / 10.0)
        if dir_:
            A = A * (_VAC + dark) / _VAC
        va = (A - dark - (1.0 - e) * _VAC) / e
        return np.log(2.0 * np.maximum(va, 0.5 * (1.0 + 1e-6)))

    g_hi = 80.0
    lo = [1e-6] + ([0.0] if fit_jitter else []) + [1e-9] * n
    hi = [1.0] + ([0.5 - 1e-9] if fit_jitter else []) + [g_hi] * n
    starts = []
    for m0 in ((0.0, 1e-3, 1e-2, 0.05) if fit_jitter else (None,)):
        x = [eta0] + ([m0] if fit_jitter else []) \
            + list(np.clip(g_start(eta0), 1e-6, g_hi - 1.0))
        starts.append(np.clip(x, lo, hi))
    best = None
    for x0 in starts:
        r = least_squares(resid, x0, bounds=(lo, hi), jac="3-point",
                          x_scale="jac", xtol=1e-15, ftol=1e-15,
                          gtol=1e-15)
        if r.success and (best is None or r.cost < best.cost):
            best = r
    if best is None:
        raise RuntimeError("the pair fit failed from every start")
    p = best.x
    J = best.jac
    norms = np.sqrt(np.sum(J * J, axis=0))
    if np.any(norms == 0.0):
        raise ValueError("the pairs do not determine the parameters "
                         "(a parameter has no effect on the data)")
    sv = np.linalg.svd(J / norms, compute_uv=False)
    if sv[-1] <= 0.0 or (sv[0] / sv[-1]) ** 2 > float(cond_max):
        raise ValueError(
            "the pairs do not separate the efficiency from the jitter: "
            "take them at clearly different pump levels (different "
            "antisqueezing), or fit loss only (fit_jitter=False)")
    cov = np.linalg.inv(J.T @ J)
    sd = np.sqrt(np.diag(cov))
    eta, mix, g = unpack(p)
    k0 = 2 if fit_jitter else 1
    chi2 = float(2.0 * best.cost)
    dof = 2 * n - p.size
    p_value = float(_chi2.sf(chi2, dof)) if dof > 0 else None
    mod = model_db(p)
    out = {"eta": float(eta), "eta_sigma": float(sd[0]),
           "mix": float(mix), "mix_sigma": float(sd[1]) if fit_jitter
           else 0.0}
    m_sig = out["mix_sigma"]

    def theta_of(m):
        m = min(max(m, 0.0), 0.5 - 1e-15)
        return float(np.sqrt(-0.5 * np.log1p(-2.0 * m)))

    out["theta_rms"] = theta_of(mix)
    if fit_jitter and mix > 3.0 * m_sig:
        # d theta / d m = 1 / (2 theta (1 - 2 m))
        out["theta_rms_sigma"] = float(
            m_sig / (2.0 * out["theta_rms"] * (1.0 - 2.0 * mix)))
    else:
        out["theta_rms_sigma"] = None
    out["theta_rms_upper"] = theta_of(mix + 2.0 * m_sig)
    db_per_g = 10.0 / np.log(10.0)
    out["sq_db_source"] = -db_per_g * np.asarray(g, dtype=float)
    out["anti_db_source"] = db_per_g * np.asarray(g, dtype=float)
    out["sq_db_source_sigma"] = db_per_g * sd[k0:]
    out.update({"chi2": chi2, "chi2_dof": int(dof), "p_value": p_value,
                "model_sq_db": mod[:n], "model_anti_db": mod[n:],
                "names": ["eta"] + (["mix"] if fit_jitter else [])
                + [f"g_{i}" for i in range(n)],
                "covariance": cov})
    return out
