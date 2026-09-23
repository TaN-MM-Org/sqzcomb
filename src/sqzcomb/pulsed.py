"""Pumps that change in time: Gaussian quantum noise, step by step in
time, and the squeezing of a pulse of output light (new in 0.13).

Every spectrum elsewhere in the package assumes a steady state: a
constant drift matrix M and a steady covariance. With a pulsed or
modulated pump the drift matrix changes in time, M = M(t), and there is
no steady state to speak of. What stays exact is the linear equation
itself,

    dz/dt = M(t) z + sqrt(2 gamma) (input noise),

so this module integrates the covariance forward in time instead of
solving for a steady state.

Covariance (`covariance_evolution`). With V = <z z^dag> (the same
convention as `intracavity_covariance`: <a a^dag> = n + 1 and
<a^dag a> = n for a mode with n photons),

    dV/dt = M(t) V + V M(t)^dag + D,
    D = diag(2 gamma (n_th + 1), 2 gamma n_th).

A constant stable M drives V to `intracavity_covariance(M, ...)`; the
tests check that limit.

A pulse of output light (`temporal_mode_variance`). A detector with a
pulsed local oscillator measures one temporal mode of the output,
A = int f(t)* a_out(t) dt with a mode function f normalized to
int |f|^2 dt = 1, and its quadrature X = (A e^{-i phi} + A^dag
e^{i phi}) / sqrt(2) (vacuum variance 1/2, as everywhere in the
package). The output field a_out = sqrt(2 eta gamma) a - a_in,ex holds
the reflected input noise as well as the cavity field, and the two are
correlated; the module keeps that correlation by carrying X along as
an extra variable of the same linear equation (dX/dt =
w(t)^dag z_out), so Var(X) comes out of the same covariance equation.
A detector that is not perfect is then handled with `detected_variance`
or `homodyne_readout` as usual.

What the tests hold this to: a passive cavity gives exactly the vacuum
value 1/2 for any mode function (which only works if the reflected
input noise and its correlation with the cavity field are right); with
a constant pump, a long pulse gives the steady-state answer
int |f~(w)|^2 S(w) dw / 2 pi with S from `output_quadrature_variance`;
and a seeded Monte-Carlo simulation of the equivalent classical
(Wigner) equations reproduces Var(X) for a pump pulse that briefly goes
above threshold.

Above threshold: the linear equation stays exact for a model that is
linear (like `single_mode_parametric`), and a brief excursion of the
gain above threshold simply amplifies. For a Kerr resonator the
linearization around the vacuum holds only while the generated photon
number stays small compared with 1/chi; the functions report the
largest intracavity photon number reached so this can be checked.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["covariance_evolution", "temporal_mode_variance",
           "parametric_pulse_drift"]


def _as_drift(drift):
    if callable(drift):
        return drift
    M = np.asarray(drift, dtype=complex)
    return lambda t: M


def _check_M(M, m2):
    M = np.asarray(M, dtype=complex)
    if M.shape != (m2, m2):
        raise ValueError("the drift matrix changed shape or is not the "
                         "doubled 2n x 2n form")
    if not np.all(np.isfinite(M)):
        raise ValueError("the drift matrix is not finite")
    return M


def _setup(drift, t_span, gammas, n_th):
    fM = _as_drift(drift)
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not (np.isfinite(t0) and np.isfinite(t1) and t1 > t0):
        raise ValueError("t_span must be (t0, t1) with t1 > t0")
    M0 = np.asarray(fM(t0), dtype=complex)
    m2 = M0.shape[0]
    if M0.ndim != 2 or M0.shape != (m2, m2) or m2 % 2:
        raise ValueError("the drift must be a square doubled 2n x 2n "
                         "matrix")
    n = m2 // 2
    g = np.ones(n) if gammas is None else np.asarray(gammas, dtype=float)
    if g.shape != (n,) or np.any(g <= 0) or not np.all(np.isfinite(g)):
        raise ValueError("gammas must hold one positive decay rate per "
                         "mode")
    nt = np.broadcast_to(np.asarray(n_th, dtype=float), (n,)).copy()
    if np.any(nt < 0) or not np.all(np.isfinite(nt)):
        raise ValueError("n_th must be finite and >= 0")
    return fM, t0, t1, m2, n, g, nt


def covariance_evolution(drift, t_span, t_eval=None, V0=None, gammas=None,
                         n_th=0.0, rtol=1e-10, atol=1e-12):
    """Integrate dV/dt = M(t) V + V M(t)^dag + D over t_span.

    drift : a callable t -> M(t) (doubled 2n x 2n drift matrix in the
        package's normalized units) or a constant matrix.
    t_span : (t0, t1) in normalized time (units of 2/kappa).
    t_eval : times at which to return V (default: t0 and t1).
    V0 : initial <z z^dag> (default: vacuum, diag(1, 0) per mode, or
        the thermal state of n_th).
    gammas : decay rate of each mode (default 1, the package's
        normalization); n_th : bath occupation (scalar or per mode).
    Returns (t, V) with V of shape (len(t), 2n, 2n). Integrated with
    scipy's DOP853 at the given tolerances.
    """
    fM, t0, t1, m2, n, g, nt = _setup(drift, t_span, gammas, n_th)
    D = np.diag(np.concatenate([2 * g * (nt + 1), 2 * g * nt])
                ).astype(complex)
    if V0 is None:
        V0 = np.diag(np.concatenate([nt + 1, nt])).astype(complex)
    V0 = np.asarray(V0, dtype=complex)
    if V0.shape != (m2, m2):
        raise ValueError("V0 must be 2n x 2n")
    if t_eval is None:
        t_eval = np.array([t0, t1])

    def rhs(t, y):
        M = _check_M(fM(t), m2)
        V = y.reshape(m2, m2)
        return (M @ V + V @ M.conj().T + D).ravel()

    sol = solve_ivp(rhs, (t0, t1), V0.ravel(), method="DOP853",
                    t_eval=np.asarray(t_eval, dtype=float), rtol=rtol,
                    atol=atol)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    V = sol.y.T.reshape(-1, m2, m2)
    V = 0.5 * (V + np.conj(np.transpose(V, (0, 2, 1))))
    return sol.t, V


def temporal_mode_variance(drift, t_span, mode_function, eta=1.0, phi=0.0,
                           mode_index=0, V0=None, gammas=None, n_th=0.0,
                           rtol=1e-10, atol=1e-12, max_step=None):
    """Quadrature variance of one temporal mode of the output light.

    drift, t_span, V0, gammas, n_th : as in `covariance_evolution`
        (the default V0 is the vacuum, i.e. the pump switches on at t0).
    mode_function : callable f(t) (complex allowed), the temporal mode
        the detector measures; it should vanish outside t_span. It is
        normalized here (the variance is that of the normalized mode,
        vacuum = 1/2), and its norm is returned so a badly chosen
        window is visible.
    eta : escape fraction of the monitored mode into the output port;
        phi : quadrature angle (X_phi = cos phi x + sin phi p, as in
        `output_quadrature_variance`); mode_index : the monitored mode.
    max_step : largest integration step (pass something below the
        shortest feature of f or M(t) if they are narrow).

    Returns a dict: variance (Var X_phi, vacuum 1/2), squeezing_db
    (10 log10(variance / 0.5)), norm (int |f|^2 dt before
    normalization) and max_photons (largest intracavity <a^dag a> of
    any mode during the pulse).
    """
    fM, t0, t1, m2, n, g, nt = _setup(drift, t_span, gammas, n_th)
    if not callable(mode_function):
        raise ValueError("mode_function must be a callable f(t)")
    eta = float(eta)
    if not (0.0 <= eta <= 1.0):
        raise ValueError("eta must lie in [0, 1]")
    m = int(mode_index)
    if not (0 <= m < n):
        raise ValueError("mode_index out of range")
    gm = g[m]
    # noise inputs: per mode j one input (weight sqrt(2 g_j), for the
    # monitored mode the loss part sqrt(2 (1 - eta) g_m)), plus the
    # extraction-port input of the monitored mode; doubled throughout.
    k = n + 1
    B = np.zeros((m2 + 1, 2 * k), dtype=complex)
    for j in range(n):
        w = np.sqrt(2 * g[j] * ((1 - eta) if j == m else 1.0))
        B[j, j] = w
        B[n + j, k + j] = w
    B[m, n] = np.sqrt(2 * eta * gm)
    B[n + m, k + n] = np.sqrt(2 * eta * gm)
    nin = np.concatenate([nt, [nt[m]]])
    N = np.diag(np.concatenate([nin + 1, nin])).astype(complex)
    ph = np.exp(1j * float(phi))
    s2 = np.sqrt(2.0)

    def noise(t, fval):
        # X row: dX = ... - w^dag dxi_ex, w^dag = (f* e^{-i phi},
        # f e^{i phi}) / sqrt(2)
        Bt = B.copy()
        Bt[m2, n] = -np.conj(fval) / ph / s2
        Bt[m2, k + n] = -fval * ph / s2
        return Bt @ N @ Bt.conj().T

    size = m2 + 1
    if V0 is None:
        V0 = np.diag(np.concatenate([nt + 1, nt])).astype(complex)
    V0 = np.asarray(V0, dtype=complex)
    if V0.shape != (m2, m2):
        raise ValueError("V0 must be 2n x 2n")
    S0 = np.zeros((size, size), dtype=complex)
    S0[:m2, :m2] = V0
    y0 = np.concatenate([S0.ravel(), [0.0]])

    def rhs(t, y):
        M = _check_M(fM(t), m2)
        fval = complex(mode_function(t))
        A = np.zeros((size, size), dtype=complex)
        A[:m2, :m2] = M
        c = np.sqrt(2 * eta * gm) / s2
        A[m2, m] = c * np.conj(fval) / ph
        A[m2, n + m] = c * fval * ph
        S = y[:-1].reshape(size, size)
        dS = A @ S + S @ A.conj().T + noise(t, fval)
        # the running norm int |f|^2 dt is carried alongside
        return np.concatenate([dS.ravel(), [abs(fval) ** 2]])

    kw = {} if max_step is None else {"max_step": float(max_step)}
    sol = solve_ivp(rhs, (t0, t1), y0.astype(complex), method="DOP853",
                    rtol=rtol, atol=atol, dense_output=False, **kw)
    if not sol.success:
        raise RuntimeError(f"integration failed: {sol.message}")
    norm = float(sol.y[-1, -1].real)
    if not (norm > 1e-12):
        raise ValueError("the mode function vanishes on t_span")
    S = sol.y[:-1, -1].reshape(size, size)
    var = float(S[m2, m2].real) / norm
    # photons: <a^dag a> = <a a^dag> - 1 on each mode at the integrator's
    # own steps (every accepted step is inspected)
    Sall = sol.y[:-1].T.reshape(-1, size, size)
    photons = np.real(Sall[:, np.arange(n), np.arange(n)]) - 1.0
    return {"variance": var,
            "squeezing_db": float(10 * np.log10(var / 0.5)),
            "norm": norm,
            "max_photons": float(np.max(photons))}


def parametric_pulse_drift(mu, delta=0.0):
    """Drift M(t) of `single_mode_parametric` with a time-dependent gain.

    mu : callable t -> complex gain mu(t) (|mu| = 1 is the steady-state
        threshold at delta = 0); delta : detuning (constant or callable).
    Returns a callable t -> 2x2 doubled drift matrix.
    """
    from .linearize import single_mode_parametric
    if not callable(mu):
        raise ValueError("mu must be a callable mu(t); for a constant "
                         "gain use single_mode_parametric directly")
    fd = delta if callable(delta) else (lambda t: float(delta))
    return lambda t: single_mode_parametric(complex(mu(t)), float(fd(t)))
