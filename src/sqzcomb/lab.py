"""Adapt sqzcomb to your ring: plan the measurement, calibrate from it.

`sqzcomb.fitnoise` already turns a measured pair of noise spectra into
(mu, eta, kappa) with error bars. This module adds the two steps
around that fit, in the order a lab actually works, plus one
calibration the package could not do before:

1. `plan_noise_measurement`: before connecting the spectrum analyzer,
   ask whether the frequencies you plan to record can determine the
   parameters at all, and how small the error bars would come out.
   The classic trap is caught by exact arithmetic, the same one
   `fit_noise_spectra` refuses after the fact: one quadrature alone is
   a single Lorentzian -- two shape numbers for three unknowns.
2. `design_noise_frequencies`: given candidate analysis frequencies,
   pick the subset that pins the parameters down best (measuring both
   quadratures at each chosen frequency).
3. `ring_from_threshold`: calibrate the single-photon Kerr shift g0 of
   your ring from its measured comb threshold power -- the one number
   in `RingSpec` that is hard to measure directly -- by exact
   inversion of the package's own `threshold_power` identity, and get
   back a ready-to-use `RingSpec` whose `reference` records the
   calibration.

Statistics, stated plainly: predicted error bars are the standard
weighted-least-squares covariance (J^T W J)^-1 -- the same matrix the
fit itself reports -- evaluated before any data exists (the Fisher
information of independent Gaussian measurements; see any statistics
text under "Cramer-Rao bound"). The design tool maximizes its
determinant (D-optimal design; F. Pukelsheim, Optimal Design of
Experiments, SIAM (2006)). The model curves come from
`sqzcomb.fitnoise.noise_spectrum_db`, the package's own solver path.

Measured spectra travel in a plain, checked CSV contract
(`save_spectra_csv` / `load_spectra_csv`) whose round trip is exact.
"""
from __future__ import annotations

import csv

import numpy as np

from .fitnoise import noise_spectrum_db
from .physical import RingSpec

__all__ = ["plan_noise_measurement", "design_noise_frequencies",
           "ring_from_threshold", "save_spectra_csv",
           "load_spectra_csv"]

_COND_MAX = 1e10
_QUADS = ("squeezed", "anti")


def _invert_information(fisher, names):
    """Scale-invariant inversion of an information matrix.

    Parameters carry wildly different units (mu of order one, kappa in
    Hz), so the raw condition number only measures the units. The
    identifiability verdict therefore uses the correlation-scaled
    matrix D^-1 F D^-1 with D = sqrt(diag F), which is invariant under
    reparameterizing any single parameter -- exact rank deficiencies
    (a functional degeneracy of the model) survive the scaling, unit
    mismatches do not. Returns (identifiable, condition_number, sigma).
    """
    d = np.sqrt(np.diag(fisher))
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return False, np.inf, None
    fs = fisher / np.outer(d, d)
    sv = np.linalg.svd(fs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    if not (np.isfinite(cond) and cond <= _COND_MAX):
        return False, cond, None
    cov = np.linalg.inv(fs) / np.outer(d, d)
    err = np.sqrt(np.diag(cov))
    return True, cond, {n: float(s) for n, s in zip(names, err)}


def _check_plan(mu, eta, kappa_hz, f_hz, sigma_db, quadratures,
                fit_kappa, fit_dark):
    if not (0.0 < float(mu) < 1.0):
        raise ValueError("mu must lie in (0, 1): below threshold")
    if not (0.0 < float(eta) <= 1.0):
        raise ValueError("eta must lie in (0, 1]")
    if not (np.isfinite(kappa_hz) and kappa_hz > 0.0):
        raise ValueError("kappa_hz must be finite and positive")
    f = np.asarray(f_hz, dtype=float).ravel()
    if f.size < 1 or np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("analysis frequencies must be positive and "
                         "finite")
    quadratures = tuple(quadratures)
    if not quadratures or any(q not in _QUADS for q in quadratures) \
            or len(set(quadratures)) != len(quadratures):
        raise ValueError(f"quadratures must be a subset of {_QUADS}")
    sig = np.broadcast_to(np.asarray(sigma_db, dtype=float),
                          f.shape).copy()
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigma_db must be finite and positive")
    names = ["mu", "eta"] + (["kappa_hz"] if fit_kappa else []) \
        + (["dark"] if fit_dark else [])
    return f, sig, quadratures, names


def _rows(mu, eta, kappa_hz, f, sig, quadratures, names, dark=0.0):
    """Weighted sensitivity rows d(model dB)/d(parameter), one row per
    (quadrature, frequency), by central differences around the working
    point -- the same model `fit_noise_spectra` fits."""
    x0 = {"mu": float(mu), "eta": float(eta),
          "kappa_hz": float(kappa_hz), "dark": float(dark)}

    def model(x):
        out = []
        for q in quadratures:
            out.append(noise_spectrum_db(x["mu"], x["eta"],
                                         x["kappa_hz"], f, quadrature=q,
                                         dark=x["dark"]))
        return np.concatenate(out)

    w = np.concatenate([1.0 / sig] * len(quadratures))
    jac = np.empty((f.size * len(quadratures), len(names)))
    for j, name in enumerate(names):
        h = 1e-6 * max(abs(x0[name]), 1e-6)
        xp, xm = dict(x0), dict(x0)
        xp[name] += h
        xm[name] -= h
        if name in ("mu", "eta") and xm[name] <= 0.0:
            xm[name] = x0[name]
            jac[:, j] = (model(xp) - model(xm)) / h
        else:
            jac[:, j] = (model(xp) - model(xm)) / (2.0 * h)
    return jac * w[:, None]


def plan_noise_measurement(mu, eta, kappa_hz, f_hz, sigma_db=0.1,
                           quadratures=("squeezed", "anti"),
                           fit_kappa=True, fit_dark=False):
    """Would this measurement determine the squeezer parameters?

    Give your expected operating point (mu, eta, kappa_hz), the
    analysis frequencies you plan to record, the per-point trace
    uncertainty in dB, and which quadratures you will take. Returns
    dict(names, fisher, identifiable, condition_number, sigma):
    `sigma` maps each fitted parameter to the error bar the weighted
    fit would report, or is None when the design cannot tell the
    parameters apart.

    The single-quadrature degeneracy is exact: with kappa unknown, one
    Lorentzian carries only two shape numbers, so the information
    matrix is rank-deficient however many frequencies you record --
    the planner reports it before the beam time, the fit refuses after.
    """
    f, sig, quadratures, names = _check_plan(
        mu, eta, kappa_hz, f_hz, sigma_db, quadratures, fit_kappa,
        fit_dark)
    jac = _rows(mu, eta, kappa_hz, f, sig, quadratures, names)
    fisher = jac.T @ jac
    identifiable, cond, sigma = _invert_information(fisher, names)
    if jac.shape[0] < len(names):
        identifiable, sigma = False, None
    return {"names": names, "fisher": fisher,
            "identifiable": identifiable, "condition_number": cond,
            "sigma": sigma}


def design_noise_frequencies(mu, eta, kappa_hz, candidates_hz, n_pick,
                             sigma_db=0.1, fit_kappa=True,
                             fit_dark=False):
    """Pick the most informative analysis frequencies.

    From the candidate frequencies, greedily choose `n_pick` at which
    to record BOTH quadratures, each pick being the one that most
    increases the determinant of the information matrix. The greedy
    rule is transparent and each step can only add information, but it
    is a good-practice heuristic, not a proof of the globally best
    subset.

    Returns dict(indices, names, fisher, condition_number, sigma) with
    the chosen candidate indices in pick order. Refuses when even the
    full candidate list cannot identify the parameters.
    """
    f, sig, quadratures, names = _check_plan(
        mu, eta, kappa_hz, candidates_hz, sigma_db, _QUADS, fit_kappa,
        fit_dark)
    n, p = f.size, len(names)
    n_pick = int(n_pick)
    if not 1 <= n_pick <= n:
        raise ValueError(f"n_pick must be between 1 and {n}")
    full = plan_noise_measurement(mu, eta, kappa_hz, f, sig, _QUADS,
                                  fit_kappa, fit_dark)
    if not full["identifiable"]:
        raise ValueError("even the full candidate list cannot "
                         "determine the parameters; widen the "
                         "frequency span around the cavity linewidth")
    jac = _rows(mu, eta, kappa_hz, f, sig, _QUADS, names)
    # work in column-scaled (unit-free) parameters: scaling every
    # column identically multiplies every candidate determinant by the
    # same constant, so the greedy choices are unchanged -- but the
    # tiny start-up regularizer becomes meaningful in every direction
    scale = np.sqrt(np.mean(jac * jac, axis=0))
    js = jac / scale
    # rows i and i + n belong to candidate frequency i (two quadratures)
    eps = 1e-12 * float(np.max(np.sum(js * js, axis=1)))
    fs = eps * np.eye(p)
    chosen = []
    for _ in range(n_pick):
        best_j, best_det = -1, -np.inf
        for j in range(n):
            if j in chosen:
                continue
            g = js[[j, j + n], :]
            det = float(np.linalg.slogdet(fs + g.T @ g)[1])
            if det > best_det:
                best_j, best_det = j, det
        g = js[[best_j, best_j + n], :]
        fs = fs + g.T @ g
        chosen.append(best_j)
    fisher = (fs - eps * np.eye(p)) * np.outer(scale, scale)
    _, cond, sigma = _invert_information(fisher, names)
    return {"indices": chosen, "names": names, "fisher": fisher,
            "condition_number": cond, "sigma": sigma}


def ring_from_threshold(kappa_hz, eta_esc, threshold_w, lambda_pump_m,
                        reference, alpha=1.0, sigma_kappa_hz=None,
                        sigma_threshold_w=None):
    """Calibrate g0 from the measured comb threshold power.

    The single-photon Kerr shift g0 is the one `RingSpec` number a lab
    rarely measures directly, but the comb threshold power is routine.
    `threshold_power` is exactly invertible for g0:

        g0 = F_th^2 kappa^3 hbar omega_p / (8 kappa_ext P_th),

    with F_th^2 = 1 + (alpha - 1)^2 the flat-state-cubic identity
    (alpha is the normalized detuning at which the threshold was
    found; the minimum threshold is at alpha = 1).

    Returns (spec, sigma_g0_hz): a ready-to-use `RingSpec` whose
    `reference` records the calibration, and the 1-sigma uncertainty
    of g0_hz propagated exactly from the supplied measurement errors
    (g0 scales as kappa^2 and 1/P, so the relative errors combine as
    sqrt((2 s_kappa/kappa)^2 + (s_P/P)^2)); None when no errors were
    given.

    The round trip is an identity, checked in the tests through the
    package's own code path: `threshold_power` of the returned spec
    equals the measured threshold to machine precision.
    """
    if not (np.isfinite(threshold_w) and threshold_w > 0.0):
        raise ValueError("threshold_w must be finite and positive")
    if not isinstance(reference, str) or not reference.strip():
        raise ValueError("a `reference` naming the measurement is "
                         "required, on purpose")
    # any positive g0 gives the correct kappa, eta and wavelength
    # checks through RingSpec itself; then solve for g0 exactly
    probe = RingSpec(kappa_hz=kappa_hz, eta_esc=eta_esc, g0_hz=1.0,
                     lambda_pump_m=lambda_pump_m,
                     reference="probe (internal)")
    from .physical import threshold_power
    # threshold_power is proportional to 1/g0 at fixed everything else
    p_at_unit_g0 = threshold_power(probe, alpha)
    g0_hz = p_at_unit_g0 / float(threshold_w)
    spec = RingSpec(
        kappa_hz=float(kappa_hz), eta_esc=float(eta_esc),
        g0_hz=float(g0_hz), lambda_pump_m=float(lambda_pump_m),
        reference=(f"g0 calibrated by sqzcomb.lab.ring_from_threshold "
                   f"from the measured threshold power "
                   f"{float(threshold_w):.6g} W at alpha="
                   f"{float(alpha):.6g}; {reference}"))
    sigma_g0 = None
    if sigma_kappa_hz is not None or sigma_threshold_w is not None:
        sk = 0.0 if sigma_kappa_hz is None else float(sigma_kappa_hz)
        sp = 0.0 if sigma_threshold_w is None else \
            float(sigma_threshold_w)
        if sk < 0.0 or sp < 0.0:
            raise ValueError("measurement sigmas must be >= 0")
        rel = np.sqrt((2.0 * sk / float(kappa_hz)) ** 2
                      + (sp / float(threshold_w)) ** 2)
        sigma_g0 = float(g0_hz * rel)
    return spec, sigma_g0


_HEADER = ("f_hz", "sq_db", "anti_db")
_HEADER_S = _HEADER + ("sigma_db",)


def save_spectra_csv(path, f_hz, sq_db, anti_db, sigma_db=None):
    """Write a measured spectrum pair; the exact inverse of
    `load_spectra_csv` (values round-trip bit for bit)."""
    f = np.asarray(f_hz, dtype=float).ravel()
    sq = np.asarray(sq_db, dtype=float).ravel()
    an = np.asarray(anti_db, dtype=float).ravel()
    if not (f.shape == sq.shape == an.shape):
        raise ValueError("f_hz, sq_db and anti_db must have the same "
                         "length")
    sig = None
    if sigma_db is not None:
        sig = np.broadcast_to(np.asarray(sigma_db, dtype=float),
                              f.shape).copy()
    header = _HEADER if sig is None else _HEADER_S
    with open(path, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(header)
        for i in range(f.size):
            row = [repr(float(f[i])), repr(float(sq[i])),
                   repr(float(an[i]))]
            if sig is not None:
                row.append(repr(float(sig[i])))
            wr.writerow(row)


def load_spectra_csv(path):
    """Read a spectrum pair written by `save_spectra_csv`.

    Returns (f_hz, sq_db, anti_db, sigma_db) with sigma_db None when
    the file has no sigma column. The header and every value are
    checked; a malformed file is refused, not guessed at.
    """
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise ValueError("empty spectrum file")
    header = tuple(rows[0])
    if header not in (_HEADER, _HEADER_S):
        raise ValueError(f"spectrum header must be {_HEADER} or "
                         f"{_HEADER_S}; got {header}")
    has_sigma = header == _HEADER_S
    body = rows[1:]
    if not body:
        raise ValueError("spectrum file has no data rows")
    cols = [[] for _ in header]
    for row in body:
        if len(row) != len(header):
            raise ValueError(f"row {row!r} does not match the header")
        try:
            for c, v in zip(cols, row):
                c.append(float(v))
        except ValueError as exc:
            raise ValueError(f"non-numeric value in row {row!r}") \
                from exc
    f = np.array(cols[0])
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("analysis frequencies must be positive and "
                         "finite")
    return (f, np.array(cols[1]), np.array(cols[2]),
            np.array(cols[3]) if has_sigma else None)
