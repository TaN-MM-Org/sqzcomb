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
"""
from __future__ import annotations

import numpy as np

__all__ = ["infer_source"]

_VAC = 0.5


def infer_source(sq_db, anti_db, sigma_db=None):
    """Closed-form source inference from one measured pair.

    sq_db, anti_db : measured squeezed and antisqueezed levels in dB
        relative to shot noise (squeezed negative, antisqueezed
        positive, the spectrum-analyzer convention of `fitnoise`).
    sigma_db : optional 1-sigma uncertainty of each dB value; the
        inferred quantities' error bars follow by exact derivatives
        of the closed form (central differences on the formula),
        validated against seeded Monte Carlo in the tests.

    Returns dict(eta, v_s, v_a, sq_db_source, anti_db_source) plus,
    with sigma_db, dict entries eta_sigma and sq_db_source_sigma.
    """
    s_db, a_db = float(sq_db), float(anti_db)
    if not (np.isfinite(s_db) and np.isfinite(a_db)):
        raise ValueError("the measured dB values must be finite")
    out = _invert(s_db, a_db)
    if sigma_db is not None:
        sig = float(sigma_db)
        if not (np.isfinite(sig) and sig > 0.0):
            raise ValueError("sigma_db must be finite and positive")
        h = 1e-6
        d_eta = np.zeros(2)
        d_src = np.zeros(2)
        for i, (ds, da) in enumerate(((h, 0.0), (0.0, h))):
            p = _invert(s_db + ds, a_db + da)
            m = _invert(s_db - ds, a_db - da)
            d_eta[i] = (p["eta"] - m["eta"]) / (2.0 * h)
            d_src[i] = (p["sq_db_source"] - m["sq_db_source"]) \
                / (2.0 * h)
        out["eta_sigma"] = float(sig * np.linalg.norm(d_eta))
        out["sq_db_source_sigma"] = float(sig * np.linalg.norm(d_src))
    return out


def _invert(s_db, a_db):
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
    if S * A < _VAC ** 2 * (1.0 - 1e-12):
        raise ValueError(
            f"the measured uncertainty product S*A = {S * A:.6g} is "
            f"below the vacuum product {_VAC ** 2}: impossible for "
            "any source followed by loss (loss only grows the "
            "product). Check the shot-noise calibration")
    eta = (2.0 * (S + A) - 1.0 - 4.0 * S * A) \
        / (2.0 * (S + A) - 2.0)
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
            "anti_db_source": float(10.0 * np.log10(v_a / _VAC))}
