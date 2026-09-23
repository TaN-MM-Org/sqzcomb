"""Infer the source from a measured squeezing/anti-squeezing pair
(new in v0.12).

The routine characterization of the 2025-2026 on-chip squeezing
literature: measure the squeezed variance S and the antisqueezed
variance A after all losses, then infer what the source made before
the losses ate it (used e.g. by Ulanov et al., Nat. Commun. 16,
10791 (2025) and by the 18-dB TFLN inference of Karnik et al.,
arXiv:2605.27607). Under the standard model -- a PURE squeezed state
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

__all__ = ["infer_source"]

_VAC = 0.5


def infer_source(sq_db, anti_db, sigma_db=None, purity=1.0,
                 theta_rms=0.0, branch=None):
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

    Returns dict(eta, v_s, v_a, sq_db_source, anti_db_source) plus,
    with sigma_db, dict entries eta_sigma and sq_db_source_sigma.
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
    out = _invert(s_db, a_db, P, th, branch)
    if sigma_db is not None:
        sig = float(sigma_db)
        if not (np.isfinite(sig) and sig > 0.0):
            raise ValueError("sigma_db must be finite and positive")
        h = 1e-6
        d_eta = np.zeros(2)
        d_src = np.zeros(2)
        for i, (ds, da) in enumerate(((h, 0.0), (0.0, h))):
            p = _invert(s_db + ds, a_db + da, P, th, out.get("branch"))
            m = _invert(s_db - ds, a_db - da, P, th, out.get("branch"))
            d_eta[i] = (p["eta"] - m["eta"]) / (2.0 * h)
            d_src[i] = (p["sq_db_source"] - m["sq_db_source"]) \
                / (2.0 * h)
        out["eta_sigma"] = float(sig * np.linalg.norm(d_eta))
        out["sq_db_source_sigma"] = float(sig * np.linalg.norm(d_src))
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
