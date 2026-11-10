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


def _check_damping(M, gammas=None):
    """Each mode's damping in M must equal its decay rate (new in 0.18).

    The input-output relations of this package couple every mode j to
    its port and loss baths with amplitudes sqrt(2 eta gamma_j) and
    sqrt(2 (1 - eta) gamma_j), a total decay rate gamma_j, and they
    conserve vacuum only when M damps that mode at the same rate: the
    Hermitian part of each diagonal block of the doubled M must be
    -diag(gamma),
    gamma = 1 for the single-ring functions (rates in units of kappa/2)
    or `gammas` for the port functions. A coherent coupling (-i J), a
    detuning or a Kerr term is anti-Hermitian and does not enter. Up to
    0.17.0 a mismatch gave a wrong number without an error: a passive
    mode with damping 0.5 emitted 4.5 instead of the vacuum level 0.5,
    and a passive molecule with gamma = 9 read through `gammas=[1, 1]`
    gave 0.34, an apparent -1.67 dB of squeezing from no pump at all.
    """
    M = np.asarray(M)
    n = M.shape[0] // 2
    g = np.ones(n) if gammas is None else np.asarray(gammas, dtype=float)
    if not np.all(np.isfinite(M)):
        raise ValueError("the drift matrix holds a non-finite value")
    if not (np.all(np.isfinite(g)) and np.all(g >= 0.0)):
        raise ValueError("decay rates (gammas) must be finite and >= 0")
    scale = max(1.0, float(np.max(np.abs(M))))
    for blk in (M[:n, :n], M[n:, n:]):
        herm = 0.5 * (blk + blk.conj().T)
        err = float(np.max(np.abs(herm + np.diag(g))))
        if not err <= 1e-9 * scale:
            if gammas is None:
                raise ValueError(
                    "the drift matrix does not damp every mode at rate 1 "
                    "(the Hermitian part of its diagonal blocks is not "
                    "-I): these formulas take each mode's decay as 1 in "
                    "units of kappa/2, and with any other damping a "
                    "passive mode would not even emit vacuum. For modes "
                    "with other decay rates use output_variance_ports "
                    "with their gammas")
            raise ValueError(
                "gammas do not match the damping of the drift matrix (the "
                "Hermitian part of its diagonal blocks must be "
                "-diag(gammas)): the bath couplings are built from "
                "gammas, so with a mismatch even a passive system would "
                "not emit vacuum. Pass the gammas the model builder "
                "returned")


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


def _resolvent(M, omega):
    """G(omega) = (-i omega I - M)^-1."""
    m2 = M.shape[0]
    return np.linalg.inv(-1j * omega * np.eye(m2, dtype=complex) - M)


def _bath_diagonal(m2, n_bar):
    """Diagonal of `_bath_covariance` (the bath matrices are diagonal)."""
    half = m2 // 2
    return np.concatenate([np.full(half, float(n_bar) + 1.0),
                           np.full(half, float(n_bar))])


def _variance_from_rows(rows, baths):
    """sum_c sum_k N_c,kk |r_c,k|^2 for the rows r_c = u^dag T_c of the
    transfer matrices T_c and the diagonals N_c,kk of the bath matrices.

    This equals u^dag (sum_c T_c N_c T_c^dag) u (each N_c is diagonal
    and >= 0), but it is a sum of non-negative terms and keeps its
    accuracy near threshold. Forming the matrix first and then the
    quadratic form subtracts numbers of the size of the antisqueezed
    variance, which lost the squeezed one (fixed in 0.16; see the
    CHANGELOG). The callers also subtract the identity part of T_ex
    (the reflected input) after projecting on u, not inside the matrix,
    for the same reason."""
    total = 0.0
    for r, nd in zip(rows, baths):
        total += float(np.dot(r.real * r.real + r.imag * r.imag, nd))
    return total


def _quadrature_u(n_modes, mode_index, phi, mode_index_b=None):
    """The vector u with u^dag z = X_phi of one mode, or of the joint
    mode (a + b)/sqrt(2) (indices already checked)."""
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
    return u


def _variance_at(G, eta, n_modes, mode_index, phi, mode_index_b,
                 n_th_port, n_th_loss):
    """Output quadrature variance from the resolvent G (inputs already
    checked): z_out = T_ex z_ex + T_0 z_0 with T_ex = 2 eta G - I and
    T_0 = 2 sqrt(eta (1 - eta)) G, so the rows are
    u^dag T_ex = 2 eta (u^dag G) - u^dag and u^dag T_0."""
    m2 = G.shape[0]
    u = _quadrature_u(n_modes, mode_index, phi, mode_index_b)
    uc = u.conj()
    y = uc @ G
    rows = (2.0 * eta * y - uc, 2.0 * np.sqrt(eta * (1.0 - eta)) * y)
    return _variance_from_rows(
        rows, (_bath_diagonal(m2, n_th_port), _bath_diagonal(m2, n_th_loss)))


def output_quadrature_variance(M, eta: float, omega: float,
                               mode_index: int, n_modes: int,
                               phi: float = 0.0, mode_index_b=None,
                               n_th_port: float = 0.0,
                               n_th_loss: float = 0.0,
                               allow_marginal: bool = False) -> float:
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

    Fixed in 0.16: the variance is summed from the rows u^dag T of the
    transfer matrices, a sum of non-negative terms, instead of read off
    the assembled output matrix. Close to threshold the old reading
    subtracted numbers of the size of the antisqueezed variance and lost
    the squeezed one: `single_mode_parametric(0.9999)` at eta = 1,
    omega = 0, phi = pi/2 gave -2.1e-8 (a negative variance) instead of
    1.25e-9. Elsewhere the two agree to rounding.

    Fixed in 0.18: M must damp every mode at rate 1 (the Hermitian part
    of each diagonal block equal to -I; see `_check_damping`), the
    decay these formulas assume, or the call is refused. Up to 0.17.0 a
    passive mode with damping 0.5 (e.g. from `drift_from_qutip(H,
    [0.5])`) gave 4.5 at omega = 0 and eta = 1 instead of the vacuum
    level 0.5. Modes with other decay rates belong in
    `output_variance_ports`, which takes them as `gammas`. The same
    check applies to `output_covariance_xxpp`, `optimal_quadrature`,
    `optimal_band_quadrature` and the output-entanglement functions.
    """
    eta = _check_quadrature_inputs(M, eta, mode_index, n_modes,
                                   mode_index_b, n_th_port, n_th_loss,
                                   allow_marginal)
    G = _resolvent(np.asarray(M), omega)
    return _variance_at(G, eta, n_modes, mode_index, phi, mode_index_b,
                        n_th_port, n_th_loss)


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
    _check_damping(M)
    return eta


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
    _check_damping(M)
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


def quadrature_extremes(variance_at, rtol: float = 1e-8) -> dict:
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

    Accuracy (new in 0.16): the subtraction c0 - r loses the relative
    accuracy of V_min when V_max / V_min is huge (close to threshold the
    error is about 1e-16 V_max, which can exceed V_min and even make it
    negative). For a scalar variance_at, V_min and V_max are therefore
    the values of variance_at at phi_min and phi_max (two more
    evaluations), which are as accurate as variance_at itself; the
    angles come out accurate either way. An array-valued variance_at has
    a different best angle in each element, so it keeps c0 -/+ r;
    `optimal_quadrature` and `optimal_band_quadrature` evaluate each
    frequency at its own angle instead.

    Returns dict(v_min, v_max, phi_min, phi_max, mean=c0), each a float
    or an array shaped like the values returned by variance_at.
    """
    def ev(p):
        return np.asarray(variance_at(float(p)), dtype=float)

    v0, v45, v90, v22 = (ev(p) for p in (0.0, np.pi / 4, np.pi / 2,
                                          np.pi / 8))
    if not all(np.all(np.isfinite(v)) for v in (v0, v45, v90, v22)):
        raise ValueError("variance_at returned a non-finite value")
    c0, r, phi_min, phi_max = _harmonics(v0, v45, v90, v22, rtol)
    out = {"v_min": c0 - r, "v_max": c0 + r, "phi_min": phi_min,
           "phi_max": phi_max, "mean": c0}
    if np.ndim(v0) == 0:
        out = {k: float(v) for k, v in out.items()}
        v_lo, v_hi = ev(out["phi_min"]), ev(out["phi_max"])
        if not (np.isfinite(v_lo) and np.isfinite(v_hi)):
            raise ValueError("variance_at returned a non-finite value")
        out["v_min"], out["v_max"] = float(v_lo), float(v_hi)
    return out


def _harmonics(v0, v45, v90, v22, rtol):
    """c0, r and the extreme angles from V(0), V(pi/4), V(pi/2), with
    the form check at V(pi/8) (see `quadrature_extremes`)."""
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
    return c0, r, phi_min, phi_max


def optimal_quadrature(M, eta: float, omega, mode_index: int,
                       n_modes: int, mode_index_b=None,
                       n_th_port: float = 0.0, n_th_loss: float = 0.0,
                       allow_marginal: bool = False) -> dict:
    """The most squeezed and most antisqueezed output quadrature, exactly.

    Same arguments as `output_quadrature_variance`, except that omega
    may be an array (a spectrum) and there is no phi: the angle is
    optimized at each frequency by `quadrature_extremes`, with no scan
    and no grid error. With a detuning the best angle changes with
    frequency; `phi_min` gives it.

    Returns dict(v_min, v_max, phi_min, phi_max, mean, squeezing_db,
    antisqueezing_db); arrays over omega when omega is an array.

    Since 0.15 the inputs are checked and the output matrix is built
    once per frequency, instead of once per frequency and angle. Since
    0.16, v_min and v_max are the variances at phi_min and phi_max,
    each exactly what `output_quadrature_variance` returns there
    (tested for exact equality), instead of c0 -/+ r, which lost the
    squeezed variance near threshold (see the CHANGELOG). The angles
    and the mean are unchanged.
    """
    w = np.asarray(omega, dtype=float)
    ws = np.atleast_1d(w)
    var, var_k = _angle_function(M, eta, ws, mode_index, n_modes,
                                 mode_index_b, n_th_port, n_th_loss,
                                 allow_marginal)
    out = _extremes_per_frequency(var, var_k, ws.size)
    if w.ndim == 0:
        out = {k: float(v[0]) for k, v in out.items()}
    out["squeezing_db"] = squeezing_db(out["v_min"])
    out["antisqueezing_db"] = squeezing_db(out["v_max"])
    return out


def _extremes_per_frequency(var, var_k, n):
    """`quadrature_extremes` over frequencies, with v_min and v_max
    evaluated at each frequency's own angles."""
    out = quadrature_extremes(var)
    out["v_min"] = np.array([var_k(k, out["phi_min"][k]) for k in range(n)])
    out["v_max"] = np.array([var_k(k, out["phi_max"][k]) for k in range(n)])
    return out


def _angle_function(M, eta, ws, mode_index, n_modes, mode_index_b,
                    n_th_port, n_th_loss, allow_marginal):
    """(var, var_k): var(phi) is the array of output variances over the
    frequencies ws and var_k(k, phi) the one at ws[k], each exactly what
    `output_quadrature_variance` returns, with the checks run and the
    resolvent built once per frequency."""
    eta = _check_quadrature_inputs(M, eta, mode_index, n_modes,
                                   mode_index_b, n_th_port, n_th_loss,
                                   allow_marginal)
    Ms = np.asarray(M)
    Gs = [_resolvent(Ms, x) for x in ws]

    def var_k(k, p):
        return _variance_at(Gs[k], eta, n_modes, mode_index, float(p),
                            mode_index_b, n_th_port, n_th_loss)

    def var(p):
        return np.array([var_k(k, p) for k in range(len(Gs))])
    return var, var_k


def optimal_band_quadrature(M, eta, omegas, mode_index: int, n_modes: int,
                            weights=None, mode_index_b=None,
                            n_th_port: float = 0.0, n_th_loss: float = 0.0,
                            allow_marginal: bool = False,
                            goal: str = "mean") -> dict:
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
        frequency, not all zero. They enter the band average only.
    goal (new in 0.18) : what "best" means for the fixed angle.
        "mean" (default, as before): the lowest band average V_band.
        "worst": the best worst case, the angle at which the least
            squeezed frequency of the band is as squeezed as possible
            (it minimizes max_k V(phi, omega_k), and so the largest
            squeezing_db).
        "penalty": the angle at which the largest shortfall from each
            frequency's own best angle is as small as possible (it
            minimizes max_k penalty_db).
        Both worst-case goals are solved exactly, without a scan: as a
        function of t = 2 phi each V(phi, omega_k), divided by its
        reference (0.5 for "worst", the per-frequency optimum v_opt for
        "penalty"), is c0 + c1 cos t + c2 sin t, and the largest of
        such curves is lowest either at the minimum of one of them or
        where two of them cross. Those angles are found in closed form
        (a crossing solves c0 + c1 cos t + c2 sin t = 0 for the
        difference of two curves) and the best one is kept. The cost
        grows as the cube of the number of frequencies (0.7 s for 400
        on one test machine). Every frequency counts in the worst case,
        whatever its weight.

    Returns a dict:
        phi : the best fixed angle for the goal, in [0, pi);
        v_band, squeezing_db_band : the band average at phi (variance,
            vacuum 0.5) and in dB;
        v, squeezing_db : the variance and dB at phi at every frequency;
        v_opt, phi_opt : the per-frequency optimum of
            `optimal_quadrature`, for comparison;
        penalty_db : squeezing_db - 10 log10(v_opt / 0.5) at every
            frequency (>= 0): what holding the angle fixed costs there;
        worst_db, worst_penalty_db (new in 0.18) : the largest
            squeezing_db and the largest penalty_db over the band at
            phi; goal : the goal used.
    The angle is meaningless when the band average does not depend on
    it (phase-insensitive light); `quadrature_extremes` notes the same.
    The values at phi are evaluated there exactly (as in
    `optimal_quadrature`); for the worst-case goals the angle itself is
    chosen from the harmonics c0, c1, c2, which near threshold carry
    the rounding error described in `quadrature_extremes`.

    For the photonic molecule's ports, the same answer comes from
    ``quadrature_extremes(lambda p: np.average([output_variance_ports(
    M, g, eta, port, w, phi=p) for w in omegas], weights=weights))``.
    """
    if goal not in ("mean", "worst", "penalty"):
        raise ValueError(f"goal must be 'mean', 'worst' or 'penalty' "
                         f"(got {goal!r})")
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
    var, var_k = _angle_function(M, eta, ws, mode_index, n_modes,
                                 mode_index_b, n_th_port, n_th_loss,
                                 allow_marginal)
    cache = {}

    def cached(p):
        if p not in cache:
            cache[p] = var(p)
        return cache[p]

    per = _extremes_per_frequency(cached, var_k, ws.size)
    if goal == "mean":
        # scalar: quadrature_extremes evaluates the band average at its
        # best angle (since 0.16), so v_band is the average of v below
        band = quadrature_extremes(
            lambda p: float(np.dot(wt, cached(p)) / np.sum(wt)))
        phi = band["phi_min"]
        v_band = band["v_min"]
    else:
        # the same three evaluations quadrature_extremes used (cached)
        v0, v45, v90 = cached(0.0), cached(np.pi / 4), cached(np.pi / 2)
        c0 = 0.5 * (v0 + v90)
        ref = np.full(ws.size, 0.5) if goal == "worst" else per["v_min"]
        if not np.all(ref > 0.0):
            raise ValueError("a per-frequency optimum is not positive; "
                             "the penalty is not defined")
        phi = _minimax_angle(c0, 0.5 * (v0 - v90), v45 - c0, ref)
        v_band = float(np.dot(wt, cached(phi)) / np.sum(wt))
    v = cached(phi)
    sq = squeezing_db(v)
    pen = sq - squeezing_db(per["v_min"])
    return {"phi": phi, "v_band": v_band,
            "squeezing_db_band": float(squeezing_db(v_band)),
            "v": v, "squeezing_db": sq,
            "v_opt": per["v_min"], "phi_opt": per["phi_min"],
            "penalty_db": pen, "worst_db": float(np.max(sq)),
            "worst_penalty_db": float(np.max(pen)), "goal": goal}


def _minimax_angle(c0, c1, c2, ref):
    """The angle phi in [0, pi) that minimizes max_k g_k(2 phi) with
    g_k(t) = (c0_k + c1_k cos t + c2_k sin t) / ref_k, exactly (new in
    0.18; see `optimal_band_quadrature`).

    Where the largest curve is lowest, either one curve is the largest
    and sits at its own minimum (t = atan2(-c2, -c1)), or two curves
    are equal there: d0 + d1 cos t + d2 sin t = 0 for their difference,
    t = atan2(d2, d1) +/- arccos(-d0 / hypot(d1, d2)). Every such angle
    is a candidate; the envelope is evaluated at all of them (in chunks)
    and the lowest is returned. t = 0 is added so that the list is never
    empty (all curves flat)."""
    a0, a1, a2 = (np.asarray(c, dtype=float) / ref for c in (c0, c1, c2))
    k = a0.size
    cands = [np.zeros(1)]
    r = np.hypot(a1, a2)
    cands.append(np.arctan2(-a2, -a1)[r > 0.0])
    i, j = np.triu_indices(k, 1)
    d0, d1, d2 = a0[i] - a0[j], a1[i] - a1[j], a2[i] - a2[j]
    rr = np.hypot(d1, d2)
    ok = (rr > 0.0) & (np.abs(d0) <= rr)
    base = np.arctan2(d2[ok], d1[ok])
    off = np.arccos(np.clip(-d0[ok] / rr[ok], -1.0, 1.0))
    cands += [base + off, base - off]
    th = np.concatenate(cands)
    best_val, best_t = np.inf, 0.0
    chunk = max(1, 2_000_000 // k)
    for s0 in range(0, th.size, chunk):
        t = th[s0:s0 + chunk]
        env = np.max(a0[:, None] + a1[:, None] * np.cos(t)[None, :]
                     + a2[:, None] * np.sin(t)[None, :], axis=0)
        m = int(np.argmin(env))
        if env[m] < best_val:
            best_val, best_t = float(env[m]), float(t[m])
    return float(np.mod(0.5 * best_t, np.pi))
