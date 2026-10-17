"""Tutorial: from raw spectrum-analyzer traces to the squeezer's numbers.

The measurement chain of a squeezing experiment, end to end:

1. raw traces as a spectrum analyzer records them, in dBm: 20 sweeps
   each of the squeezed quadrature, the antisqueezed quadrature, the
   shot-noise reference (signal blocked) and the dark trace (no light),
   at 30 analysis frequencies;
2. `shot_noise_normalize`: average the sweeps in linear power, subtract
   the dark noise, divide by the shot noise, with an error bar per point;
3. `fit_noise_spectra`: fit gain mu, total efficiency eta and linewidth
   kappa to both traces, with the 0.05 dB uncertainty of the shot-noise
   reference carried as a common error (`calib_sigma_db`);
4. `infer_source`: the closed-form inference from the measured pair at
   the lowest frequency, an independent route to the efficiency.

The traces are synthetic: illustrative true values (mu = 0.6,
eta = 0.75, kappa = 20 MHz, shot noise at -80 dBm, dark noise 15 dB
below it) with seeded 3 % sweep-to-sweep power noise, so the script
runs anywhere and prints the same numbers every time.

Output (sqzcomb 0.16.0):

    dark noise           = 0.0158 (vacuum units, mean over frequency)
    lowest frequency     : -5.23 +/- 0.05 dB and +10.65 +/- 0.05 dB
    fit: mu    = 0.598 +/- 0.002
         eta   = 0.749 +/- 0.002
         kappa = 20.07 +/- 0.07 MHz
         chi^2 / dof = 57.6 / 57
    pair at 1 MHz: eta = 0.750 +/- 0.006, source -11.81 +/- 0.10 dB
"""
import numpy as np

import sqzcomb as sc

rng = np.random.default_rng(16)
MU, ETA, KAPPA = 0.6, 0.75, 20e6
f = np.linspace(1e6, 40e6, 30)                 # analysis frequencies, Hz
shot_mw = 10 ** (-80.0 / 10)                   # shot-noise power, mW
dark_mw = shot_mw * 10 ** (-15.0 / 10)         # dark (electronic) noise


def sweeps(level_mw, n=20, scatter=0.03):
    """n sweeps of a trace with the given mean power, in dBm."""
    p = level_mw * (1.0 + scatter * rng.normal(size=(n, level_mw.size)))
    return 10 * np.log10(p)


# 1. Raw traces: optical noise relative to shot noise, plus dark noise.
rel_sq = 10 ** (sc.noise_spectrum_db(MU, ETA, KAPPA, f, "squeezed") / 10)
rel_an = 10 ** (sc.noise_spectrum_db(MU, ETA, KAPPA, f, "anti") / 10)
ones = np.ones(f.size)
sq_dbm = sweeps(shot_mw * rel_sq + dark_mw)
an_dbm = sweeps(shot_mw * rel_an + dark_mw)
shot_dbm = sweeps(shot_mw * ones + dark_mw)
dark_dbm = sweeps(dark_mw * ones)

# 2. Normalize to shot noise, dark noise subtracted.
sq = sc.shot_noise_normalize(sq_dbm, shot_dbm, dark_dbm)
an = sc.shot_noise_normalize(an_dbm, shot_dbm, dark_dbm)
print(f"dark noise           = {sq['dark_variance'].mean():.4f} "
      "(vacuum units, mean over frequency)")
print(f"lowest frequency     : {sq['db'][0]:.2f} +/- {sq['sigma_db'][0]:.2f}"
      f" dB and {an['db'][0]:+.2f} +/- {an['sigma_db'][0]:.2f} dB")

# 3. Fit both traces, with the shot-noise calibration error.
# One error per frequency: the larger of the two traces' errors (they
# share the shot and dark traces, so their errors are slightly
# correlated; this tutorial ignores that).
sig = np.maximum(sq["sigma_db"], an["sigma_db"])
fit = sc.fit_noise_spectra(f, sq_db=sq["db"], anti_db=an["db"],
                           sigma_db=sig, calib_sigma_db=0.05)
print(f"fit: mu    = {fit.mu:.3f} +/- {fit.sigma['mu']:.3f}")
print(f"     eta   = {fit.eta:.3f} +/- {fit.sigma['eta']:.3f}")
print(f"     kappa = {fit.kappa_hz / 1e6:.2f} +/- "
      f"{fit.sigma['kappa_hz'] / 1e6:.2f} MHz")
print(f"     chi^2 / dof = {fit.chi2:.1f} / {fit.chi2_dof}")

# 4. Independent route: the measured pair at the lowest frequency.
src = sc.infer_source(sq["db"][0], an["db"][0],
                      sigma_db=max(sq["sigma_db"][0], an["sigma_db"][0]),
                      calib_sigma_db=0.05)
print(f"pair at {f[0] / 1e6:.0f} MHz: eta = {src['eta']:.3f} +/- "
      f"{src['eta_sigma']:.3f}, source {src['sq_db_source']:.2f} +/- "
      f"{src['sq_db_source_sigma']:.2f} dB")
