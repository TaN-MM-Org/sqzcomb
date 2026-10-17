"""Tutorial: from two routine ring measurements to a squeezing plan.

A lab characterizes a new ring with two routine measurements, then asks
how hard to pump it for a target squeezing:

1. a slow, low-power laser sweep across one resonance, recording the
   power on the bus waveguide: the dip gives the loaded linewidth kappa
   and the escape efficiency eta_esc (`fit_resonance_transmission`);
2. the comb threshold power: it gives the single-photon Kerr shift g0
   (`ring_from_threshold`), and with it a complete `RingSpec`;
3. the plan: the parametric gain mu that gives -4 dB at the detector at
   a 10 MHz analysis frequency, with 90 % detection efficiency and
   20 mrad of local-oscillator phase jitter (`gain_for_squeezing`).

The "measured" data are synthetic: illustrative true values (kappa =
100 MHz, eta_esc = 0.8, threshold 0.15 mW) plus seeded Gaussian noise,
so the script runs anywhere and prints the same numbers every time.
The transmission alone cannot tell an over-coupled ring from an
under-coupled one (both give the same dip); this tutorial assumes the
ring is over-coupled, as squeezing rings are designed to be, which a
real lab confirms from the phase response or a gap sweep (see the
`sqzcomb.resonance` docstring).

Output (sqzcomb 0.16.0):

    linewidth kappa      = 100.11 +/- 0.44 MHz
    escape efficiency    = 0.7996 +/- 0.0007  (over-coupled branch)
      (under-coupled reading of the same dip: 0.2004)
    g0 / 2 pi            = 8.41 +/- 0.29 Hz
    gain for -4 dB at 10 MHz: mu = 0.4481 +/- 0.0010  (antisqueezing +6.76 dB)
    deepest reachable here:   -5.10 dB at mu = 0.8325
"""
import numpy as np

import sqzcomb as sc

rng = np.random.default_rng(2026)

# 1. Transmission scan: 201 points over +/- 400 MHz around the resonance,
#    power in microwatts, 0.5 % detector noise per point.
f = np.linspace(-400e6, 400e6, 201)
truth = sc.through_transmission(f, f0_hz=5e6, kappa_hz=100e6, eta_esc=0.8,
                                baseline=40.0)
sigma_t = 0.005 * 40.0
scan = truth + rng.normal(0.0, sigma_t, f.size)
res = sc.fit_resonance_transmission(f, scan, sigma=sigma_t, branch="over")
kappa, s_kappa = res["kappa_hz"], res["sigma"]["kappa_hz"]
eta_esc, s_eta = res["eta_esc"], res["eta_esc_sigma"]
print(f"linewidth kappa      = {kappa / 1e6:.2f} +/- {s_kappa / 1e6:.2f} MHz")
print(f"escape efficiency    = {eta_esc:.4f} +/- {s_eta:.4f}  "
      "(over-coupled branch)")
print(f"  (under-coupled reading of the same dip: {res['eta_under']:.4f})")

# 2. Comb threshold, measured at the detuning of lowest threshold
#    (alpha = 1): 0.150 mW +/- 0.005 mW (synthetic).
spec, s_g0 = sc.ring_from_threshold(
    kappa_hz=kappa, eta_esc=eta_esc, threshold_w=0.150e-3,
    lambda_pump_m=1.55e-6, sigma_kappa_hz=s_kappa,
    sigma_threshold_w=0.005e-3,
    reference="tutorial: synthetic transmission scan and threshold")
print(f"g0 / 2 pi            = {spec.g0_hz:.2f} +/- {s_g0:.2f} Hz")

# 3. Plan: total efficiency = escape x detection (90 %), 10 MHz offset.
#    The error bar of mu carries the error of eta_esc (central
#    difference over +/- 1 sigma).
omega = sc.normalized_frequency(spec, 10e6)


def plan_for(eta):
    return sc.gain_for_squeezing(-4.0, eta=eta * 0.90, omega=omega,
                                 theta_rms=0.020)


plan = plan_for(eta_esc)
s_mu = 0.5 * abs(plan_for(eta_esc + s_eta)["mu"]
                 - plan_for(eta_esc - s_eta)["mu"])
print(f"gain for -4 dB at 10 MHz: mu = {plan['mu']:.4f} +/- {s_mu:.4f}  "
      f"(antisqueezing {plan['anti_db']:+.2f} dB)")
print(f"deepest reachable here:   {plan['best_db']:.2f} dB at "
      f"mu = {plan['mu_best']:.4f}")
