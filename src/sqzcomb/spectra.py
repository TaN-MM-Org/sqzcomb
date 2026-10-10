"""Output quadrature-noise spectra via input-output theory.

The cavity couples to the extraction port with rate fraction eta = kappa_ex
/ kappa and to intrinsic loss with 1 - eta (total damping normalized to 1
in the doubled drift matrix M, i.e. Re eigenvalues -> -1 far from any
parametric process). With vacuum at both inputs,

    z(Omega)      = G(Omega) [ sqrt(2 eta) z_ex + sqrt(2 (1-eta)) z_0 ]
    z_out(Omega)  = sqrt(2 eta) z(Omega) - z_ex(Omega)
    G(Omega)      = (-i Omega I - M)^{-1}

and the symmetrized output covariance follows from the vacuum input
covariance. Quadrature X_phi = (a e^{-i phi} + a^dagger e^{i phi}) /
sqrt(2), vacuum variance 1/2.
"""
from __future__ import annotations

import numpy as np


def _bath_covariance(m2, n_bar):
    """<z_in z_in^dagger> of a thermal bath in the doubled ordering:
    <a a^dagger> = n_bar + 1, <a^dagger a> = n_bar (vacuum: n_bar = 0)."""
    half = m2 // 2
    N = np.zeros((m2, m2), dtype=complex)
    N[:half, :half] = (float(n_bar) + 1.0) * np.eye(half)
    N[half:, half:] = float(n_bar) * np.eye(half)
    return N


def _check_spectra_stability(M, allow_marginal, tol=1e-6):
    lam = float(np.max(np.linalg.eigvals(np.asarray(M)).real))
    if lam < -tol:
        return
    if allow_marginal and lam < tol:
        return
    if lam < tol:
        raise ValueError(
            "drift matrix is marginally stable (an eigenvalue's real "
            "part is numerically zero). Around a localized steady "
            "state this is the exact translation (Goldstone) mode of "
            "the soliton; spectra at omega != 0 remain finite, so pass "
            "allow_marginal=True if that marginal direction is "
            "understood. A genuinely positive growth rate is still "
            "refused.")
    raise ValueError("drift matrix is unstable (above threshold); "
                     "linearized spectra are meaningless there")


def _check_eta_port(eta):
    """Escape fraction of a port: a number in [0, 1] (0: no port, the
    output is the reflected input). Values outside made the square
    roots below NaN in 0.13.0 and earlier, silently."""
    e = float(eta)
    if not (0.0 <= e <= 1.0):
        raise ValueError(f"eta must lie in [0, 1] (got {e:g}): it is "
                         "the fraction of the decay that leaves "
                         "through the port")
    return e


def _check_modes(M, n_modes, indices):
    """The drift matrix must be the doubled 2n x 2n form with n =
    n_modes, and every mode index an integer in [0, n). Python's
    negative indices are refused: in the doubled vector they land on
    the conjugate half and read the quadrature at -phi."""
    M = np.asarray(M)
    n = int(n_modes)
    if M.ndim != 2 or M.shape != (2 * n, 2 * n):
        raise ValueError(f"n_modes = {n} does not match the drift matrix "
                         f"of shape {M.shape} (expected {2 * n} x {2 * n})")
    for idx in indices:
        if idx is None:
            continue
        if int(idx) != idx or not (0 <= int(idx) < n):
            raise ValueError(f"mode index {idx!r} is not an integer in "
                             f"[0, {n})")


def _port_indices(port_mode, mode_index, n):
    """Checked (ports, reads) of the multi-port spectra.

    Every index must be an integer in [0, n). Repeating a port is
    harmless (the bus couples to each listed mode once), so ports are
    de-duplicated. The modes read (mode_index, or the ports when it is
    None) form an equal-weight joint quadrature, so each mode may
    appear only once: in 0.14.0 and earlier a repeated mode got twice
    its weight and the variance came out doubled (fixed in 0.15).
    """
    ports = np.ravel(np.asarray(port_mode))
    reads = ports if mode_index is None else \
        np.ravel(np.asarray(mode_index))
    for arr in (ports, reads):
        if arr.size == 0 \
                or not np.all(np.isfinite(arr.astype(float))) \
                or not np.all(np.equal(np.mod(arr, 1), 0)) \
                or np.any(arr < 0) or np.any(arr >= n):
            raise ValueError(f"mode indices must be integers in [0, {n})")
    reads = reads.astype(int)
    uniq, counts = np.unique(reads, return_counts=True)
    if np.any(counts > 1):
        raise ValueError(
            f"mode {int(uniq[np.argmax(counts > 1)])} is listed more than "
            "once among the modes read; the joint quadrature needs "
            "distinct modes (a repeated mode used to be counted twice, "
            "doubling the variance)")
    ports = np.unique(ports.astype(int))
    if not np.all(np.isin(reads, ports)):
        raise ValueError("mode_index must be one of the monitored port "
                         "modes; a detector on the bus cannot see light "
                         "that never enters the bus")
    return ports, reads


def _output_covariance(M, eta, omega, n_th_port=0.0, n_th_loss=0.0):
    m2 = M.shape[0]
    ident = np.eye(m2, dtype=complex)
    G = np.linalg.inv(-1j * omega * ident - M)
    T_ex = 2.0 * eta * G - ident
    T_0 = 2.0 * np.sqrt(eta * (1.0 - eta)) * G
    S = T_ex @ _bath_covariance(m2, n_th_port) @ T_ex.conj().T \
        + T_0 @ _bath_covariance(m2, n_th_loss) @ T_0.conj().T
    return S


def output_quadrature_variance(M, eta, omega, mode_index, n_modes,
                               phi=0.0, mode_index_b=None,
                               n_th_port=0.0, n_th_loss=0.0,
                               allow_marginal=False):
    """Symmetrized variance of an output quadrature at frequency omega.

    mode_index : index of the mode (within the retained mode list) whose
        quadrature is detected. If mode_index_b is given, the joint
        two-mode quadrature (a + b)/sqrt(2) rotated by phi is used, the
        natural variable for twin-beam squeezing.
    n_th_port, n_th_loss : Bose occupations of the extraction-port
        input and of the intrinsic-loss bath (default vacuum, 0; see
        `thermal_occupation` for the physical number). A passive cavity
        with both baths at n_bar emits exactly (2 n_bar + 1)/2 at every
        frequency, coupling and phase -- asserted in the tests.
    allow_marginal : accept a drift matrix whose largest eigenvalue
        real part is numerically zero (the soliton's exact translation
        Goldstone mode); spectra at omega != 0 stay finite. Genuinely
        unstable matrices are refused regardless.
    Vacuum level is 0.5. Requires a stable M.

    Refused (ValueError, new in 0.14): eta outside [0, 1] (0.13.0
    returned NaN), n_modes that does not match M, a mode index that is
    not in [0, n_modes) (a negative index used to read the quadrature
    at -phi of another slot), and mode_index_b equal to mode_index
    (which used to return twice the single-mode variance). The best
    and worst angle need no scan: see `optimal_quadrature`.
    """
    eta = _check_quadrature_inputs(M, eta, mode_index, n_modes,
                                   mode_index_b, n_th_port, n_th_loss,
                                   allow_marginal)
    S = _output_covariance(M, eta, omega, n_th_port, n_th_loss)
    return _read_quadrature(S, n_modes, mode_index, phi, mode_index_b)


def _check_quadrature_inputs(M, eta, mode_index, n_modes, mode_index_b,
                             n_th_port, n_th_loss, allow_marginal):
    """The input checks of `output_quadrature_variance`; returns eta."""
    if float(n_th_port) < 0.0 or float(n_th_loss) < 0.0:
        raise ValueError("thermal occupations must be non-negative")
    eta = _check_eta_port(eta)
    _check_modes(M, n_modes, (mode_index, mode_index_b))
    if mode_index_b is not None and int(mode_index_b) == int(mode_index):
        raise ValueError("mode_index_b must differ from mode_index: the "
                         "joint quadrature needs two different modes")
    _check_spectra_stability(M, allow_marginal)
    return eta


def _read_quadrature(S, n_modes, mode_index, phi, mode_index_b=None):
    """u^dag S u for the quadrature X_phi of one mode, or of the joint
    mode (a + b)/sqrt(2), from the output matrix S of
    `_output_covariance` (indices already checked)."""
    n_modes = int(n_modes)
    mode_index = int(mode_index)
    # u^dag z = (a e^{-i phi} + a^dag e^{i phi}) / sqrt(2) = X_phi
    # = cos(phi) x + sin(phi) p, the documented quadrature (sqzcomb
    # 0.12.1 and earlier built the conjugate vector here, i.e. read
    # X_{-phi}; see the 0.13.0 changelog)
    u = np.zeros(2 * n_modes, dtype=complex)
    if mode_index_b is None:
        u[mode_index] = np.exp(1j * phi) / np.sqrt(2.0)
        u[n_modes + mode_index] = np.exp(-1j * phi) / np.sqrt(2.0)
    else:
        for idx, w in ((mode_index, 0.5), (int(mode_index_b), 0.5)):
            u[idx] += np.exp(1j * phi) * w
            u[n_modes + idx] += np.exp(-1j * phi) * w
    # With N giving <a a^dagger> = 1 and zero elsewhere, u^dag S u is the
    # output quadrature spectrum with vacuum level exactly 1/2; the
    # passive-cavity identity (T_ex N T_ex^dag + T_0 N T_0^dag = N for any
    # eta, Omega) is verified in the test-suite.
    var = np.real(u.conj() @ S @ u)
    return float(var)


def squeezing_db(variance):
    """Convert a quadrature variance to dB relative to vacuum (0.5)."""
    return 10.0 * np.log10(variance / 0.5)


def output_covariance_xxpp(M, eta, omega, n_modes, n_th_port=0.0,
                           n_th_loss=0.0, allow_marginal=False):
    """Full symmetrized xxpp covariance of the OUTPUT field at analysis
    frequency omega, vacuum = 0.5 I (hbar = 1, the same normalization
    as `output_quadrature_variance`).

    This is the matrix a bank of homodyne detectors measures: every
    single- or joint-quadrature variance read off it equals the
    `output_quadrature_variance` scalar path exactly (a two-code-path
    identity asserted in the tests), a passive cavity with both baths
    at n_bar returns exactly ((2 n_bar + 1)/2) I at every frequency,
    coupling and omega, and in the omega -> infinity limit the output
    is the reflected input bath exactly.
    """
    if float(n_th_port) < 0.0 or float(n_th_loss) < 0.0:
        raise ValueError("thermal occupations must be non-negative")
    eta = _check_eta_port(eta)
    _check_modes(M, n_modes, ())
    _check_spectra_stability(M, allow_marginal)
    S = _output_covariance(M, eta, omega, n_th_port, n_th_loss)
    n = n_modes
    eye = np.eye(n)
    K = np.block([[eye, np.zeros((n, n))], [np.zeros((n, n)), -eye]])
    V_S = S - 0.5 * K.astype(complex)
    T = np.block([[eye, eye], [-1j * eye, 1j * eye]]) / np.sqrt(2.0)
    W = T @ V_S @ T.conj().T
    if np.max(np.abs(W.imag)) > 1e-9 * max(1.0, np.max(np.abs(W))):
        raise ValueError("output covariance did not come out real")
    out = W.real
    return 0.5 * (out + out.T)


def output_entanglement(M, eta, omega, i, j, n_modes, n_th_port=0.0,
                        n_th_loss=0.0, allow_marginal=False,
                        detection_efficiency=None):
    """Logarithmic negativity between output modes i and j at analysis
    frequency omega -- the entanglement a two-line homodyne pair would
    certify at the detector, not inside the cavity.

    detection_efficiency : optional scalar or per-mode efficiencies,
        applied to the output covariance through the exact Gaussian
        lossy channel of `sqzcomb.detection` before the PPT test
        (entanglement is monotone under this local channel, asserted).

    Anchors in the tests rather than the docstring: separable passive
    outputs give exactly zero at every frequency; for the symmetric
    twin-beam parametric process E_N(omega) equals
    -ln(2 V_EPR(omega)) with V_EPR the minimal joint quadrature
    variance computed by the independent `output_quadrature_variance`
    path; and E_N -> 0 as omega -> infinity.
    """
    from .detection import lossy_channel_xxpp
    from .entangle import logarithmic_negativity
    sig = output_covariance_xxpp(M, eta, omega, n_modes, n_th_port,
                                 n_th_loss, allow_marginal)
    if detection_efficiency is not None:
        sig = lossy_channel_xxpp(sig, detection_efficiency, hbar=1.0)
    return logarithmic_negativity(sig, i, j, hbar=1.0)


def output_entanglement_spectrum(M, eta, omegas, i, j, n_modes,
                                 n_th_port=0.0, n_th_loss=0.0,
                                 allow_marginal=False,
                                 detection_efficiency=None):
    """E_N(omega) over an array of analysis frequencies."""
    return np.array([
        output_entanglement(M, eta, w, i, j, n_modes, n_th_port,
                            n_th_loss, allow_marginal,
                            detection_efficiency)
        for w in np.asarray(omegas, dtype=float)])


def quadrature_extremes(variance_at, rtol=1e-8):
    """Exact smallest and largest variance over the quadrature angle.

    variance_at : callable phi -> variance of the quadrature X_phi
        (scalar, or an array such as a spectrum over frequencies), for
        example ``lambda p: output_variance_ports(M, g, eta, 1, w,
        phi=p)``.

    Any second moment of X_phi = (b e^{-i phi} + b^dag e^{i phi}) /
    sqrt(2) contains only the angle harmonics 0 and 2 phi, so

        V(phi) = c0 + c1 cos 2 phi + c2 sin 2 phi

    exactly -- for the quantum spectra, the port spectra, added
    classical noise, jitter averages and the exact master-equation
    spectrum alike. Three evaluations (phi = 0, pi/4, pi/2) fix c0, c1
    and c2, and then

        V_min = c0 - r,  V_max = c0 + r,  r = sqrt(c1^2 + c2^2),

    at phi_max = atan2(c2, c1) / 2 and phi_min = phi_max + pi/2 (both
    returned in [0, pi)). A fourth evaluation at phi = pi/8 checks the
    form and refuses a function that does not have it (for example one
    that returns dB), to a relative `rtol` of c0 + r. When r = 0 (a
    phase-insensitive state) every angle is equally good and the angles
    returned carry no meaning.

    Returns dict(v_min, v_max, phi_min, phi_max, mean=c0), each a float
    or an array shaped like the values returned by variance_at.
    """
    def ev(p):
        return np.asarray(variance_at(float(p)), dtype=float)

    v0, v45, v90, v22 = (ev(p) for p in (0.0, np.pi / 4, np.pi / 2,
                                          np.pi / 8))
    if not all(np.all(np.isfinite(v)) for v in (v0, v45, v90, v22)):
        raise ValueError("variance_at returned a non-finite value")
    c0 = 0.5 * (v0 + v90)
    c1 = 0.5 * (v0 - v90)
    c2 = v45 - c0
    r = np.hypot(c1, c2)
    pred = c0 + (c1 + c2) * np.sqrt(0.5)          # V(pi/8)
    scale = np.abs(c0) + r
    if np.any(np.abs(v22 - pred) > float(rtol) * np.maximum(scale, 1e-300)):
        raise ValueError(
            "variance_at(phi) is not of the form c0 + c1 cos 2phi + c2 "
            "sin 2phi, which every quadrature variance has; pass the "
            "variance itself (not dB), as a function of the angle")
    phi_max = np.mod(0.5 * np.arctan2(c2, c1), np.pi)
    phi_min = np.mod(phi_max + 0.5 * np.pi, np.pi)
    out = {"v_min": c0 - r, "v_max": c0 + r, "phi_min": phi_min,
           "phi_max": phi_max, "mean": c0}
    if np.ndim(v0) == 0:
        out = {k: float(v) for k, v in out.items()}
    return out


def optimal_quadrature(M, eta, omega, mode_index, n_modes,
                       mode_index_b=None, n_th_port=0.0, n_th_loss=0.0,
                       allow_marginal=False):
    """The most squeezed and most antisqueezed output quadrature, exactly.

    Same arguments as `output_quadrature_variance`, except that omega
    may be an array (a spectrum) and there is no phi: the angle is
    optimized at each frequency by `quadrature_extremes`, with no scan
    and no grid error. With a detuning the best angle changes with
    frequency; `phi_min` gives it.

    Returns dict(v_min, v_max, phi_min, phi_max, mean, squeezing_db,
    antisqueezing_db); arrays over omega when omega is an array.

    Since 0.15 the inputs are checked and the output matrix is built
    once per frequency, instead of once per frequency and angle; the
    numbers are identical to the earlier per-angle calls (tested for
    exact equality).
    """
    w = np.asarray(omega, dtype=float)
    ws = np.atleast_1d(w)
    var = _angle_function(M, eta, ws, mode_index, n_modes, mode_index_b,
                          n_th_port, n_th_loss, allow_marginal)
    out = quadrature_extremes(var)
    if w.ndim == 0:
        out = {k: float(v[0]) for k, v in out.items()}
    out["squeezing_db"] = squeezing_db(out["v_min"])
    out["antisqueezing_db"] = squeezing_db(out["v_max"])
    return out


def _angle_function(M, eta, ws, mode_index, n_modes, mode_index_b,
                    n_th_port, n_th_loss, allow_marginal):
    """phi -> array of output variances over the frequencies ws, each
    exactly what `output_quadrature_variance` returns, with the checks
    run and the output matrix S(omega) built once."""
    eta = _check_quadrature_inputs(M, eta, mode_index, n_modes,
                                   mode_index_b, n_th_port, n_th_loss,
                                   allow_marginal)
    Ss = [_output_covariance(M, eta, x, n_th_port, n_th_loss) for x in ws]

    def var(p):
        return np.array([_read_quadrature(S, n_modes, mode_index, p,
                                          mode_index_b) for S in Ss])
    return var


def optimal_band_quadrature(M, eta, omegas, mode_index: int, n_modes: int,
                            weights=None, mode_index_b=None,
                            n_th_port: float = 0.0, n_th_loss: float = 0.0,
                            allow_marginal: bool = False) -> dict:
    """The best single local-oscillator angle for a band of frequencies.

    In an experiment the local oscillator usually sits at one angle
    while the analyzer records many analysis frequencies (or a detector
    integrates over a band). With a detuning the best angle changes with
    frequency (see `optimal_quadrature`), so no fixed angle is best
    everywhere. This function finds, exactly, the fixed angle phi that
    minimizes the weighted band average

        V_band(phi) = sum_k w_k V(phi, omega_k) / sum_k w_k,

    which is the noise power a detector integrating over the band would
    report (take w_k as the width of the frequency bin around omega_k)
    or the plain average of the trace (equal weights, the default).
    Every V(phi, omega_k) has the form c0 + c1 cos 2phi + c2 sin 2phi,
    so the weighted average has it too, and `quadrature_extremes` gives
    the minimum and its angle with no scan.

    M, eta, mode_index, n_modes, mode_index_b, n_th_port, n_th_loss,
    allow_marginal : as in `output_quadrature_variance`.
    omegas : the analysis frequencies of the band (units of kappa/2;
        convert Hz with `normalized_frequency`); 1-D, finite.
    weights : None (equal weights) or one finite value >= 0 per
        frequency, not all zero.

    Returns a dict:
        phi : the best fixed angle, in [0, pi);
        v_band, squeezing_db_band : the band average at phi (variance,
            vacuum 0.5) and in dB;
        v, squeezing_db : the variance and dB at phi at every frequency;
        v_opt, phi_opt : the per-frequency optimum of
            `optimal_quadrature`, for comparison;
        penalty_db : squeezing_db - 10 log10(v_opt / 0.5) at every
            frequency (>= 0): what holding the angle fixed costs there.
    The angle is meaningless when the band average does not depend on
    it (phase-insensitive light); `quadrature_extremes` notes the same.

    For the photonic molecule's ports, the same answer comes from
    ``quadrature_extremes(lambda p: np.average([output_variance_ports(
    M, g, eta, port, w, phi=p) for w in omegas], weights=weights))``.
    """
    ws = np.asarray(omegas, dtype=float)
    if ws.ndim != 1 or ws.size == 0 or not np.all(np.isfinite(ws)):
        raise ValueError("omegas must be a non-empty 1-D array of finite "
                         "analysis frequencies")
    if weights is None:
        wt = np.ones(ws.size)
    else:
        wt = np.asarray(weights, dtype=float)
        if wt.shape != ws.shape:
            raise ValueError(f"weights must have one value per frequency "
                             f"({ws.size}), got shape {wt.shape}")
        if not np.all(np.isfinite(wt)) or np.any(wt < 0.0) \
                or not np.sum(wt) > 0.0:
            raise ValueError("weights must be finite, >= 0 and not all "
                             "zero")
    var = _angle_function(M, eta, ws, mode_index, n_modes, mode_index_b,
                          n_th_port, n_th_loss, allow_marginal)
    cache = {}

    def cached(p):
        if p not in cache:
            cache[p] = var(p)
        return cache[p]

    per = quadrature_extremes(cached)
    band = quadrature_extremes(
        lambda p: float(np.dot(wt, cached(p)) / np.sum(wt)))
    phi = band["phi_min"]
    v = var(phi)
    return {"phi": phi, "v_band": band["v_min"],
            "squeezing_db_band": float(squeezing_db(band["v_min"])),
            "v": v, "squeezing_db": squeezing_db(v),
            "v_opt": per["v_min"], "phi_opt": per["phi_min"],
            "penalty_db": squeezing_db(v) - squeezing_db(per["v_min"])}
