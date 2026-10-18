"""Tutorial: loss or phase jitter? Squeezing pairs at several pump levels.

A squeezing run rarely reaches the level the efficiency alone predicts,
and the usual suspects are loss and phase jitter of the local
oscillator. One squeezed / antisqueezed pair cannot tell them apart.
Pairs at several pump levels can: loss pulls every pair toward shot
noise by the same fraction, while jitter mixes in a part of the
antisqueezing, which grows with the pump. The steps:

1. raw analyzer traces in dBm at one analysis frequency (5 MHz) for
   six pump levels: 10 sweeps each of the squeezed and antisqueezed
   quadratures, plus the shot-noise and dark traces;
2. `shot_noise_normalize`: dB relative to shot noise, dark noise
   subtracted, with an error bar per level;
3. `fit_squeezing_pairs`: the total efficiency and the RMS jitter
   shared by all pairs, with error bars and a chi-square check;
4. for comparison, `infer_source` on each pair alone, which assumes
   no jitter: its efficiency falls at the higher pump levels, the sign of
   jitter that a single pair would have reported as loss;
5. what that means for the plan: the gain for -6 dB from
   `gain_for_squeezing` with the fitted numbers, and the deepest
   squeezing reachable with them.

The traces are synthetic: illustrative true values (escape efficiency
0.9, detection efficiency 0.9, so a total of 0.81; 25 mrad RMS jitter;
linewidth 50 MHz; gains mu = 0.3 to 0.9; dark noise 18 dB below shot
noise) with seeded 2 % sweep-to-sweep power noise, made with the
package's own forward model, so the script runs anywhere and prints the
same numbers every time.

Output (sqzcomb 0.17.0):

    mu     squeezed dB       antisqueezed dB   eta (pair alone)
    0.30   -3.56 +/- 0.04    +4.53 +/- 0.04    0.804
    0.45   -4.93 +/- 0.04    +7.12 +/- 0.04    0.811
    0.60   -5.94 +/- 0.05    +10.25 +/- 0.04   0.808
    0.70   -6.24 +/- 0.04    +12.68 +/- 0.04   0.797
    0.80   -6.31 +/- 0.04    +15.23 +/- 0.04   0.785
    0.90   -6.25 +/- 0.05    +17.72 +/- 0.04   0.773
    joint fit: eta   = 0.812 +/- 0.002
               theta = 26.2 +/- 1.3 mrad
               chi^2 / dof = 6.4 / 4
    -6 dB at 5 MHz: mu = 0.625 +/- 0.006
    deepest reachable: -6.37 +/- 0.02 dB at mu = 0.788
"""
import numpy as np

import sqzcomb as sc

rng = np.random.default_rng(1707)
ETA_ESC, ETA_DET, THETA, KAPPA = 0.9, 0.9, 0.025, 50e6
F = 5e6                                         # analysis frequency, Hz
MUS = np.array([0.30, 0.45, 0.60, 0.70, 0.80, 0.90])
omega = 4 * np.pi * F / (2 * np.pi * KAPPA)     # units of kappa/2
shot_mw = 10 ** (-75.0 / 10)                    # shot noise, mW
dark_mw = shot_mw * 10 ** (-18.0 / 10)          # dark noise, mW


def sweeps(level_mw, n=10, scatter=0.02):
    """n sweeps (rows) of a one-frequency trace with the given mean
    power, in dBm."""
    return 10 * np.log10(level_mw * (1 + scatter * rng.normal(size=(n, 1))))


def detected(mu):
    """Squeezed and antisqueezed variance at the detector (vacuum 1/2):
    cavity output, LO jitter, detection efficiency."""
    M = sc.single_mode_parametric(mu)
    vs = sc.output_quadrature_variance(M, ETA_ESC, omega, 0, 1,
                                       phi=np.pi / 2)
    va = sc.output_quadrature_variance(M, ETA_ESC, omega, 0, 1, phi=0.0)
    js = sc.phase_noise_variance(vs, va, THETA)
    ja = sc.phase_noise_variance(va, vs, THETA)
    return (sc.detected_variance(js, ETA_DET),
            sc.detected_variance(ja, ETA_DET))


# 1.-2. Raw traces at each pump level, normalized to shot noise.
sq, sq_err, an, an_err = [], [], [], []
for mu in MUS:
    vs, va = detected(mu)
    shot = sweeps(shot_mw + dark_mw)
    dark = sweeps(dark_mw)
    s = sc.shot_noise_normalize(sweeps(shot_mw * vs / 0.5 + dark_mw),
                                shot, dark)
    a = sc.shot_noise_normalize(sweeps(shot_mw * va / 0.5 + dark_mw),
                                shot, dark)
    sq.append(s["db"][0])
    sq_err.append(s["sigma_db"][0])
    an.append(a["db"][0])
    an_err.append(a["sigma_db"][0])

# 3.-4. The joint fit, and each pair alone (loss only).
fit = sc.fit_squeezing_pairs(sq, an, sigma_db=sq_err, sigma_anti_db=an_err)
print("mu     squeezed dB       antisqueezed dB   eta (pair alone)")
for mu, s, ds, a, da in zip(MUS, sq, sq_err, an, an_err):
    alone = sc.infer_source(s, a)["eta"]
    print(f"{mu:.2f}   {s:+.2f} +/- {ds:.2f}    "
          f"{f'{a:+.2f} +/- {da:.2f}':<18}{alone:.3f}")
print(f"joint fit: eta   = {fit['eta']:.3f} +/- {fit['eta_sigma']:.3f}")
print(f"           theta = {1e3 * fit['theta_rms']:.1f} +/- "
      f"{1e3 * fit['theta_rms_sigma']:.1f} mrad")
print(f"           chi^2 / dof = {fit['chi2']:.1f} / {fit['chi2_dof']}")

# 5. The plan with the fitted numbers. eta and the jitter are strongly
#    correlated (both pull the squeezed level up), so the error bars use
#    their covariance: first-order propagation through central
#    differences in eta and in the mixing fraction m = <sin^2 theta>.
eta, mix = fit["eta"], fit["mix"]
cov = fit["covariance"][:2, :2]                 # (eta, mix) block


def plan(e, m):
    theta = np.sqrt(-0.5 * np.log(1 - 2 * m))    # m -> RMS jitter
    return sc.gain_for_squeezing(-6.0, eta=e, omega=omega, theta_rms=theta)


p = plan(eta, mix)


def spread(key):
    he, hm = 0.1 * np.sqrt(cov[0, 0]), 0.1 * np.sqrt(cov[1, 1])
    grad = np.array([
        (plan(eta + he, mix)[key] - plan(eta - he, mix)[key]) / (2 * he),
        (plan(eta, mix + hm)[key] - plan(eta, mix - hm)[key]) / (2 * hm)])
    return float(np.sqrt(grad @ cov @ grad))


print(f"-6 dB at 5 MHz: mu = {p['mu']:.3f} +/- {spread('mu'):.3f}")
print(f"deepest reachable: {p['best_db']:.2f} +/- {spread('best_db'):.2f} "
      f"dB at mu = {p['mu_best']:.3f}")
