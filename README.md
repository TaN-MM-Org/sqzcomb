# sqzcomb

[![PyPI](https://img.shields.io/pypi/v/sqzcomb)](https://pypi.org/project/sqzcomb/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015375-blue)](https://doi.org/10.5281/zenodo.22015375) [![tests](https://github.com/TaN-MM-Org/sqzcomb/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/sqzcomb/actions)

`sqzcomb` is a Python package for predicting **squeezed light from
Kerr microcombs**: tiny optical ring resonators that, when pumped with
a laser, can produce light whose noise in one quadrature is below the
normal quantum noise floor. The package follows the light end to end:
from the classical field circulating in the ring, through the quantum
noise around it, to **the noise a detector on the output fiber would
actually report**. That last step is the point. Squeezing inside the
ring is not what an experiment sees; only what leaves through the
output port is, and the ring coupling that is best for one is not best
for the other.

It answers questions such as:

- How much squeezing will my ring deliver to a detector, given its
  linewidth, output coupling and pump power?
- How much do optical loss, detector electronics and local-oscillator
  phase jitter take away, and what loss or jitter budget does a target
  need?
- Can a second, coupled ring get the light out better than one ring
  alone can?
- Are two comb lines entangled at the output, and would a detector be
  able to show it?
- From a measured pair of squeezed and antisqueezed levels, or from
  measured noise spectra, what did the source itself produce?
- Before measuring: will the planned frequencies pin down the
  parameters at all?

When a question falls outside what the model can answer, the package
stops with an error that says why, instead of returning a number that
looks fine but is not (see [When it refuses, and
why](#when-it-refuses-and-why)).

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install and requirements](#install-and-requirements)
- [Units and conventions](#units-and-conventions)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections](#corrections)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

- **Microring, Kerr microcomb** -- a small ring-shaped optical
  resonator. Its material has the **Kerr effect** (the refractive index
  changes slightly with light intensity), which lets a strong pump
  laser create light at new, evenly spaced frequencies (a **comb** of
  **lines**, or **modes**).
- **Linewidth (kappa)** -- how fast light leaks out of the ring; the
  width of its resonance. **Escape efficiency** `eta = kappa_ex /
  kappa` is the fraction of that leakage that goes into the useful
  output port; the rest is lost inside the ring. `eta = 1/2` is
  **critical coupling**.
- **Pump, detuning, threshold** -- the pump is the driving laser. Its
  **detuning** is how far it sits from the ring's resonance. Above the
  **threshold** pump power the ring starts to oscillate on new comb
  lines; this package's quantum-noise part works **below** that point
  (or around a stable steady state such as a soliton). In the model,
  `F` is the pump strength and `alpha` the detuning, both in
  normalized units; `mu` is the pump strength seen by the quantum
  noise (the "parametric gain"), with `mu = 1` the threshold of a
  single squeezed mode.
- **Dispersion** -- how the ring's resonance spacing changes from line
  to line (`D2` in Hz, `d2` in normalized units). "Anomalous"
  dispersion is what bright solitons need.
- **Lugiato-Lefever equation (LLE)** -- the standard equation for the
  field inside such a ring. `sqzcomb` solves it in its usual
  dimensionless ("normalized") form.
- **Soliton** -- a single stable pulse circulating in the ring; a
  **soliton crystal** is several equally spaced pulses.
- **Quadrature** -- light's noise can be split into two components,
  like the cosine and sine parts of a wave, selected by a phase `phi`.
  **Vacuum** (no light) has equal noise in every quadrature; this is
  the **shot-noise** level.
- **Squeezing / antisqueezing** -- noise below / above vacuum in a
  quadrature, quoted in **dB relative to vacuum** (negative is
  squeezed). The squeezed and antisqueezed quadratures are 90 degrees
  apart.
- **Homodyne detection** -- the standard way to measure one
  quadrature: the light is mixed with a strong reference beam, the
  **local oscillator (LO)**, whose phase selects the quadrature. The
  noise is recorded against **analysis frequency** (offset from the
  carrier, `omega` in normalized units, `f` in Hz).
- **Detection efficiency** -- the fraction of the light that reaches
  and is counted by the detector. Any loss mixes in vacuum noise and
  washes squeezing out.
- **Covariance matrix** -- a table of all the quadrature noises and
  their correlations; it fully describes the (Gaussian) states used
  here. **xxpp** is the ordering (all `x` quadratures, then all `p`).
- **Entanglement** -- quantum correlations between two lines stronger
  than any classical ones. **Logarithmic negativity** `E_N` measures
  it (0 means not entangled); it comes from the **PPT test** (positive
  partial transpose), a standard mathematical check for entanglement
  of such states. The **Duan test** is a simpler check based on the
  noise of sums and differences of the two lines' quadratures; it can
  miss entanglement that the PPT test finds.
- **Symplectic eigenvalues** -- numbers computed from a covariance
  matrix that say how noisy (mixed) a state is; all equal to the
  vacuum value means a pure state.
- **Photonic molecule** -- two coupled rings. Here the second ring is
  passive and can act as the output channel for the first. `J` is the
  rate at which light hops between the rings and `gamma` the decay
  rate of the second ring (both in units of `kappa/2`).
- **Drift matrix** -- the matrix `M` of the linear equations that the
  small quantum fluctuations obey. Almost every function here takes or
  returns one. It must be **stable** (all eigenvalues with negative
  real part) for a steady noise level to exist.

## Install and requirements

```
pip install sqzcomb              # numpy + scipy
pip install sqzcomb[interop]     # adds QuTiP, for drift_from_qutip only
```

For development: clone the repository and `pip install -e .[test]`.

It needs Python 3.9 or newer, NumPy 1.22 or newer and SciPy 1.10 or
newer. QuTiP (4.7 or newer) is needed only for `drift_from_qutip`.

## Units and conventions

- **Model units.** The physics runs in the normalized units of the
  LLE: rates in units of `kappa/2` (so a single ring's field decays at
  rate 1 and a single parametric mode reaches threshold at `mu = 1`),
  time in units of `2/kappa`, analysis frequency `omega` in units of
  `kappa/2`. The dispersion sign follows the LLE module: `d2 < 0` is
  anomalous.
- **Lab units.** `RingSpec` and its helpers take ordinary Hz (the
  linewidth `kappa_hz` is the full width at half maximum, `g0_hz` is
  `g0/2pi`), watts and metres, and convert once. `hbar` and `c` come
  from `scipy.constants`.
- **Noise level.** A quadrature variance of **0.5 is vacuum**
  (`squeezing_db(v) = 10 log10(v / 0.5)`). With a real pump parameter
  `mu > 0`, the quadrature at `phi = pi/2` is squeezed and `phi = 0`
  antisqueezed.
- **Covariance matrices** are in xxpp order with vacuum `(hbar/2) I`.
  `covariance_xxpp` and the entanglement functions use `hbar = 2` by
  default (vacuum = identity); `output_covariance_xxpp` returns
  `hbar = 1` (vacuum = 0.5 I, the same scale as the variances above).
- **Measured traces** (`fit_noise_spectra`, `infer_source`, the CSV
  files) are in dB relative to shot noise, the way a spectrum analyzer
  trace is calibrated.

## Examples

Each example below runs as written, and the output shown is what it
printed with sqzcomb 0.12.1. The device and pump numbers are
illustrative values chosen for the example, not measured data, except
the measured pair in example 6, which is quoted from the paper named
there.

### 1. From lab numbers to model units

```python
import sqzcomb as sc

# Illustrative ring values (not a real device).
ring = sc.RingSpec(kappa_hz=100e6,      # loaded linewidth (FWHM), Hz
                   eta_esc=0.8,         # escape efficiency kappa_ex / kappa
                   g0_hz=10.0,          # single-photon Kerr shift g0 / 2 pi, Hz
                   lambda_pump_m=1.55e-6,
                   reference="illustrative values for the README")

print(f"comb threshold at alpha = 1: {sc.threshold_power(ring) * 1e3:.3f} mW")
print(f"pump F at 2 mW:              {sc.normalized_pump(ring, 2e-3):.4f}")
print(f"detuning alpha for 50 MHz:   {sc.normalized_detuning(ring, 50e6):.4f}")
print(f"d2 for D2/2pi = 100 kHz:     {sc.normalized_dispersion(ring, 1e5):.4f}")
print(f"omega for a 10 MHz offset:   {sc.normalized_frequency(ring, 10e6):.4f}")
print(f"photons in the ring at rho=1: {sc.intracavity_photons(ring, 1.0):.3e}")
```

```
comb threshold at alpha = 1: 0.126 mW
pump F at 2 mW:              3.9870
detuning alpha for 50 MHz:   1.0000
d2 for D2/2pi = 100 kHz:     -0.0020
omega for a 10 MHz offset:   0.2000
photons in the ring at rho=1: 5.000e+06
```

The 2 mW pump is well above threshold for this ring (`F` = 3.99,
against `F = 1` at threshold for `alpha = 1`). `threshold_power` is the power at which the
flat-state cubic of the LLE has the root `|psi|^2 = 1`, so
`F_th^2 = 1 + (alpha - 1)^2`. Here `psi` is the normalized field in
the ring and `rho = |psi|^2` its normalized intensity. Every
`RingSpec` needs a `reference`
string saying where its numbers came from; an empty one is refused.

### 2. One ring against two: the 3 dB limit

```python
import numpy as np
from sqzcomb import (single_mode_parametric, output_quadrature_variance,
                     squeezing_db, photonic_molecule, molecule_threshold,
                     output_variance_ports)

# One ring, pumped at 99 % of threshold (mu = 0.99), noise read at the
# centre of the spectrum (omega = 0) in the squeezed quadrature (phi = pi/2).
M = single_mode_parametric(0.99)
for eta in (0.5, 0.8, 1.0):
    v = output_quadrature_variance(M, eta, 0.0, mode_index=0, n_modes=1,
                                   phi=np.pi / 2)
    print(f"single ring, eta = {eta}: {squeezing_db(v):7.2f} dB")

# Two rings: the Kerr ring has NO output port; light leaves through a
# second, passive ring (J^2 / gamma = 3), read at its own port.
J, gamma = np.sqrt(27.0), 9.0
mu_th = molecule_threshold(J, gamma)
M2, gammas = photonic_molecule(0.99 * mu_th, J, gamma=gamma)
v2 = output_variance_ports(M2, gammas, eta=1.0, port_mode=1, omega=0.0,
                           phi=0.0)
print(f"threshold of the pair: mu = {mu_th:.4f}")
print(f"two rings, read through ring 2: {squeezing_db(v2):7.2f} dB")
```

```
single ring, eta = 0.5:   -3.01 dB
single ring, eta = 0.8:   -6.99 dB
single ring, eta = 1.0:  -45.98 dB
threshold of the pair: mu = 4.0000
two rings, read through ring 2:   -6.02 dB
```

A single ring at critical coupling (`eta = 0.5`) cannot deliver more
than about 3 dB of squeezing to the detector, however hard it is
pumped: half of the squeezed light is lost inside the ring and replaced
by vacuum. Stronger output coupling does better. In the two-ring
**photonic molecule**, the pumped Kerr ring has no output port of its
own; its light leaves through the second ring, which acts as an output
channel with effective efficiency `(J^2/gamma) / (1 + J^2/gamma) =
3/4` here. The result passes the 3 dB value. Note the reading phase:
hopping to the second ring turns the squeezed quadrature by a quarter
turn, so it is read at `phi = 0` there.

### 3. A pumped comb: steady state, then detected squeezing

```python
import numpy as np
from sqzcomb import (lle_evolve, homogeneous_steady_states,
                     fluctuation_matrix, output_quadrature_variance,
                     squeezing_db)

F, alpha = 0.9, 0.3                          # illustrative, below threshold
psi = lle_evolve(np.full(16, 0.05 + 0j), F=F, alpha=alpha, t_end=100.0)
print(f"|psi|^2 from the solver: {abs(psi[0]) ** 2:.6f}")
print(f"root of the cubic:       {homogeneous_steady_states(F, alpha)[0]:.6f}")

M, modes = fluctuation_matrix(psi, alpha)    # quantum noise around it
i0 = int(np.where(modes == 0)[0][0])         # the pumped line
phis = np.linspace(0.0, np.pi, 181)
for eta in (0.2, 0.5, 0.8, 1.0):
    best = min(squeezing_db(output_quadrature_variance(
        M, eta, 0.0, i0, modes.size, phi=p)) for p in phis)
    print(f"eta = {eta}: best squeezing {best:6.3f} dB")
```

```
|psi|^2 from the solver: 0.698836
root of the cubic:       0.698836
eta = 0.2: best squeezing -0.730 dB
eta = 0.5: best squeezing -2.126 dB
eta = 0.8: best squeezing -4.194 dB
eta = 1.0: best squeezing -6.460 dB
```

`lle_evolve` finds the classical field by time-stepping the LLE; for a
flat state it lands on the root of the cubic `rho (1 + (alpha -
rho)^2) = F^2`. `fluctuation_matrix` then builds the quantum-noise
equations around it, and `output_quadrature_variance` gives the noise
at the output port. The script `examples/squeezing_vs_coupling.py`
does the same for six couplings from 0.1 to 1.0.

### 4. What the detector takes away

```python
from sqzcomb import (detected_squeezing_db, dark_from_clearance_db,
                     required_efficiency_db, phase_noise_squeezing_db,
                     max_phase_noise, thermal_occupation)

v_source = 0.05                        # 10 dB below vacuum (vacuum = 0.5)
print(f"70 % efficiency:            {detected_squeezing_db(v_source, 0.7):.3f} dB")
dark = dark_from_clearance_db(15.0)    # receiver noise 15 dB below shot noise
print(f"70 % and 15 dB clearance:   {detected_squeezing_db(v_source, 0.7, dark):.3f} dB")
print(f"efficiency needed for -6 dB: {required_efficiency_db(-6.0, -10.0):.4f}")

# Local-oscillator phase jitter mixes in the antisqueezed quadrature.
for jitter in (0.001, 0.03, 0.1):       # RMS, radians
    db = phase_noise_squeezing_db(-10.0, 11.0, jitter)
    print(f"-10 dB / +11 dB source, {jitter * 1e3:5.0f} mrad jitter: {db:.3f} dB")
print(f"largest jitter that still gives variance 0.1: {max_phase_noise(0.1, 0.05, 6.3):.4f} rad")

print(f"thermal photons at 193 THz and 300 K: {thermal_occupation(193e12, 300.0):.2e}")
```

```
70 % efficiency:            -4.318 dB
70 % and 15 dB clearance:   -3.962 dB
efficiency needed for -6 dB: 0.8320
-10 dB / +11 dB source,     1 mrad jitter: -9.999 dB
-10 dB / +11 dB source,    30 mrad jitter: -9.538 dB
-10 dB / +11 dB source,   100 mrad jitter: -6.504 dB
largest jitter that still gives variance 0.1: 0.0898 rad
thermal photons at 193 THz and 300 K: 3.90e-14
```

Loss mixes in vacuum: `V_detected = eta V + (1 - eta)/2 + V_dark`.
Local-oscillator phase jitter mixes in the antisqueezed quadrature;
for Gaussian jitter of RMS size `sigma` the average is the closed form
`<cos 2 theta> = exp(-2 sigma^2)`. The better the source, the larger its
antisqueezing and the more jitter hurts. At optical frequencies and
room temperature the thermal photon number is tiny, which is why the
baths default to vacuum.

### 5. Entanglement and supermodes

```python
import numpy as np
from sqzcomb import (photonic_molecule, intracavity_covariance,
                     covariance_xxpp, entanglement_report,
                     principal_quadratures, symplectic_eigenvalues,
                     output_entanglement_spectrum)

# The two coupled rings of example 2, driven at mu = 0.8 (illustrative).
M, gammas = photonic_molecule(mu=0.8, J=1.0)
sigma = covariance_xxpp(intracavity_covariance(M, gammas))   # vacuum = identity
rep = entanglement_report(sigma, 0, 1)
print(f"log. negativity: {rep['log_negativity']:.4f} "
      f"(entangled: {rep['entangled']})")
print(f"Duan sum {rep['duan_sum']:.4f}, bound {rep['duan_bound']:.1f} "
      f"(Duan test fires: {rep['duan_violated']})")
w, v = principal_quadratures(sigma)
print("principal variances:", np.round(w, 4))
print("symplectic eigenvalues:", np.round(symplectic_eigenvalues(sigma), 4))

# Output entanglement of a twin-beam pair, as a detector would see it.
twin = np.array([[-1, 0, 0, 0.5], [0, -1, 0.5, 0],
                 [0, 0.5, -1, 0], [0.5, 0, 0, -1]], dtype=complex)
omegas = [0.0, 1.0, 5.0]
print("E_N at the output:", np.round(output_entanglement_spectrum(
    twin, 0.8, omegas, 0, 1, 2), 4))
print("... with 70 % detection:", np.round(output_entanglement_spectrum(
    twin, 0.8, omegas, 0, 1, 2, detection_efficiency=0.7), 4))
```

```
log. negativity: 0.0911 (entangled: True)
Duan sum 2.6440, bound 2.0 (Duan test fires: False)
principal variances: [0.5795 0.9307 1.241  2.5368]
symplectic eigenvalues: [1.0484 1.0484]
E_N at the output: [1.2417 0.6779 0.0605]
... with 70 % detection: [0.6887 0.4225 0.042 ]
```

Inside the driven two-ring molecule the rings are entangled (`E_N >
0`), but the simple symmetric Duan test does not detect it; the
sharper PPT test used for `E_N` does. `principal_quadratures` finds the
most-squeezed combination of all quadratures (0.5795 here, against 1 for
vacuum in these units). `symplectic_eigenvalues` above 1 show the state
is mixed. The second part uses a hand-written drift matrix of a
two-line twin-beam source (`da1/dt = -a1 + 0.5 a2*` and the same with 1
and 2 swapped) and gives the entanglement a pair of detectors would
see at each analysis frequency, with and without 70 % detection
efficiency.

### 6. Inferring the source from a measured pair

```python
from sqzcomb import infer_source

# Measured pair of Ulanov et al., Nat. Commun. 16, 10791 (2025):
# 1.71 dB squeezing and 5.54 dB antisqueezing, after all losses.
out = infer_source(sq_db=-1.71, anti_db=5.54, sigma_db=0.1)
print(f"implied total efficiency:  {out['eta']:.4f} +/- {out['eta_sigma']:.4f}")
print(f"inferred source squeezing: {out['sq_db_source']:.2f} "
      f"+/- {out['sq_db_source_sigma']:.2f} dB")
print(f"inferred source antisqueezing: {out['anti_db_source']:.2f} dB")

try:
    infer_source(sq_db=-3.0, anti_db=1.0)
except ValueError as err:
    print("refused:", err)
```

```
implied total efficiency:  0.3724 +/- 0.0204
inferred source squeezing: -8.99 +/- 0.25 dB
inferred source antisqueezing: 8.99 dB
refused: the measured uncertainty product S*A = 0.157739 is below the vacuum product 0.25: impossible for any source followed by loss (loss only grows the product). Check the shot-noise calibration
```

Under the standard model -- a pure squeezed state followed by loss --
the measured squeezed and antisqueezed levels fix the total efficiency
and the source in closed form. This is a model statement, not a
measurement: if the source is less pure than assumed (excess phase
noise, thermal noise), the inferred numbers flatter it. The error bars
come from numerical derivatives of the closed form. The paper's own
inference is not reproduced here (the test suite checks only that this
pair gives a physical efficiency and a stronger source).

### 7. Fitting measured noise spectra, and planning the measurement

```python
import numpy as np
from sqzcomb import noise_spectrum_db, fit_noise_spectra, plan_noise_measurement

# Synthetic "measured" traces: illustrative true values plus 0.05 dB noise.
mu, eta, kappa_hz = 0.55, 0.62, 12e6
f = np.linspace(0.5e6, 30e6, 25)                  # analysis frequencies, Hz
rng = np.random.default_rng(1)
sq = noise_spectrum_db(mu, eta, kappa_hz, f, "squeezed") + rng.normal(0, 0.05, f.size)
an = noise_spectrum_db(mu, eta, kappa_hz, f, "anti") + rng.normal(0, 0.05, f.size)

fit = fit_noise_spectra(f, sq_db=sq, anti_db=an, sigma_db=0.05)
print(f"mu    = {fit.mu:.4f} +/- {fit.sigma['mu']:.4f}")
print(f"eta   = {fit.eta:.4f} +/- {fit.sigma['eta']:.4f}")
print(f"kappa = {fit.kappa_hz / 1e6:.3f} +/- {fit.sigma['kappa_hz'] / 1e6:.3f} MHz")
print(f"chi^2 / dof = {fit.chi2:.1f} / {fit.chi2_dof}")

# Before measuring: would one quadrature alone be enough?
one = plan_noise_measurement(mu, eta, kappa_hz, f, sigma_db=0.05,
                             quadratures=("squeezed",))
both = plan_noise_measurement(mu, eta, kappa_hz, f, sigma_db=0.05)
print("squeezed trace only identifiable:", one["identifiable"])
print("both traces identifiable:", both["identifiable"],
      f"(predicted sigma of mu {both['sigma']['mu']:.4f})")
try:
    fit_noise_spectra(f, sq_db=sq)
except ValueError as err:
    print("refused:", err)
```

```
mu    = 0.5474 +/- 0.0021
eta   = 0.6181 +/- 0.0035
kappa = 12.109 +/- 0.073 MHz
chi^2 / dof = 35.1 / 47
squeezed trace only identifiable: False
both traces identifiable: True (predicted sigma of mu 0.0021)
refused: one quadrature is a single Lorentzian -- two shape numbers for three unknowns (mu, eta, kappa). Supply both quadratures, or pass the independently measured kappa_hz
```

One quadrature alone is a single bell-shaped (Lorentzian) curve: two shape numbers
(depth and width) for three unknowns (`mu`, `eta`, `kappa`). The fit
therefore needs both traces, or a separately measured `kappa_hz`.
`plan_noise_measurement` gives the same error bars the fit would report
(the same `(J^T W J)^-1` matrix, the standard least-squares error
formula) before any data exist, and
`design_noise_frequencies` picks the most informative analysis
frequencies from a candidate list.

### 8. Calibrating g0 from the threshold power

```python
import os, tempfile
import numpy as np
from sqzcomb import (ring_from_threshold, threshold_power,
                     save_spectra_csv, load_spectra_csv)

# Illustrative measurement: threshold 0.15 mW +/- 0.005 mW,
# linewidth 100 MHz +/- 2 MHz.
ring, sigma_g0 = ring_from_threshold(
    kappa_hz=100e6, eta_esc=0.8, threshold_w=0.15e-3, lambda_pump_m=1.55e-6,
    reference="illustrative threshold sweep",
    sigma_kappa_hz=2e6, sigma_threshold_w=0.005e-3)
print(f"g0/2pi = {ring.g0_hz:.3f} +/- {sigma_g0:.3f} Hz")
print(f"threshold_power of the new ring: {threshold_power(ring) * 1e3:.6f} mW")
print(ring.reference)

# Measured spectra travel in a plain CSV file.
path = os.path.join(tempfile.mkdtemp(), "spectra.csv")
f = np.array([1e6, 2e6, 5e6])
save_spectra_csv(path, f, [-3.1, -2.8, -1.9], [6.2, 5.9, 4.0], sigma_db=0.1)
f2, sq2, an2, sig2 = load_spectra_csv(path)
print(open(path).read().splitlines()[0], "| round trip exact:",
      np.array_equal(f, f2) and np.array_equal(an2, [6.2, 5.9, 4.0]))
```

```
g0/2pi = 8.388 +/- 0.437 Hz
threshold_power of the new ring: 0.150000 mW
g0 calibrated by sqzcomb.lab.ring_from_threshold from the measured threshold power 0.00015 W at alpha=1; illustrative threshold sweep
f_hz,sq_db,anti_db,sigma_db | round trip exact: True
```

The single-photon Kerr shift `g0` is hard to measure directly; the comb
threshold power is routine. `ring_from_threshold` inverts
`threshold_power` for `g0` and propagates the measurement errors
(`g0` scales as `kappa^2 / P`).

### 9. A soliton, and the marginal mode

```python
import numpy as np
from sqzcomb import (soliton_seed, newton_state, fluctuation_matrix,
                     output_quadrature_variance, squeezing_db)

F, alpha, d2 = 1.9, 3.0, -0.25                 # illustrative, anomalous d2
sol, info = newton_state(soliton_seed(128, F, alpha, (d2,)), F, alpha, (d2,))
print("converged:", info["residual"] < 1e-12,
      f"| peak |psi|^2 = {np.abs(sol).max() ** 2:.3f}")

M, modes = fluctuation_matrix(sol, alpha, (d2,))
print("largest growth rate below 1e-8 in size:",
      abs(np.linalg.eigvals(M).real.max()) < 1e-8)
i0 = int(np.where(modes == 0)[0][0])
try:
    output_quadrature_variance(M, 0.9, 0.5, i0, modes.size)
except ValueError as err:
    print("refused:", str(err)[:60], "...")
v = output_quadrature_variance(M, 0.9, 0.5, i0, modes.size,
                               allow_marginal=True)
print(f"with allow_marginal=True, pump line at phi = 0: {squeezing_db(v):+.3f} dB")
```

```
converged: True | peak |psi|^2 = 7.290
largest growth rate below 1e-8 in size: True
refused: drift matrix is marginally stable (an eigenvalue's real part ...
with allow_marginal=True, pump line at phi = 0: +2.342 dB
```

`newton_state` solves the stationary LLE and never returns a state
whose residual is above the tolerance. Around a soliton the noise
equations have an eigenvalue at zero: sliding the pulse around the ring
costs nothing. The spectra functions refuse such a marginal matrix
unless you pass `allow_marginal=True`; a truly unstable one is refused
either way.

### 10. Bringing a model from QuTiP

```python
import numpy as np
import qutip
from sqzcomb import drift_from_qutip, single_mode_parametric

a = qutip.destroy(12)
H = 0.5j * 0.4 * (a.dag() ** 2 - a ** 2)       # quadratic: accepted
M, _ = drift_from_qutip(H, [1.0])
print("same drift matrix as single_mode_parametric(0.4):",
      np.allclose(M, single_mode_parametric(0.4)))
try:
    drift_from_qutip(a.dag() ** 2 * a ** 2, [1.0])   # a Kerr term
except ValueError as err:
    print("refused:", err)
```

```
same drift matrix as single_mode_parametric(0.4): True
refused: H is not quadratic in the mode operators; refusing to linearize it silently
```

`drift_from_qutip` reads the coefficients of a quadratic Hamiltonian
(the energy operator that defines a quantum model; "quadratic" means
it has at most two mode operators per term)
and rebuilds it; anything it cannot rebuild (such as a Kerr term) is
refused rather than silently linearized.

## What is in the package

**Classical field**

- `lle_evolve(psi0, F, alpha, dispersion, t_end, dt)` -- time-steps
  the LLE (split-step method; the Kerr half-steps and the linear step
  are each solved exactly).
- `homogeneous_steady_states(F, alpha)` -- the flat-state intensities,
  roots of `rho (1 + (alpha - rho)^2) = F^2`.
- `newton_state`, `soliton_seed`, `continuation` -- solitons and
  soliton crystals by Newton's method, a pulse-shaped (sech) starting guess, and a
  sweep over detuning; unconverged states are never returned.

**Quantum noise and output spectra**

- `fluctuation_matrix(psi_s, alpha, dispersion, modes)` -- the drift
  matrix around a steady state.
- `single_mode_parametric(mu, delta)` -- the drift matrix of one
  below-threshold parametric mode (the textbook squeezer).
- `output_quadrature_variance` -- the noise of one output quadrature
  (or the joint quadrature of two lines) at frequency `omega`, with
  optional thermal baths; `squeezing_db` converts to dB.
- `output_covariance_xxpp` -- the full output covariance matrix;
  `output_entanglement`, `output_entanglement_spectrum` -- `E_N`
  between two output lines, optionally after detection loss.

**Photonic molecules**

- `photonic_molecule(mu, J, delta_a, delta_b, gamma)` -- two coupled
  rings; `molecule_threshold(J, gamma)` -- its exact threshold on
  resonance; `output_variance_ports` -- the detected noise when rings
  have different decay rates and the output port sits on one (or
  several) of them.
- `molecule_fluctuation_matrix` -- the multimode version: every
  retained comb line coupled to a matching line of an auxiliary ring.

**Gaussian states and entanglement**

- `intracavity_covariance`, `covariance_xxpp`,
  `symplectic_eigenvalues`, `principal_quadratures` -- the state
  inside the ring, its covariance matrix, how mixed it is, and its
  most-squeezed collective quadratures.
- `drift_from_qutip` -- a drift matrix from a quadratic QuTiP
  Hamiltonian.
- `two_mode_reduction`, `ppt_symplectic_eigenvalue`,
  `logarithmic_negativity`, `duan_epr_sum`, `entanglement_report` --
  two-mode entanglement measures.
- `thermal_occupation(frequency_hz, temperature_k)` -- the
  thermal (Bose-Einstein) photon number, from the exact SI values of `h` and
  `k_B`.

**Imperfect detection**

- `detected_variance`, `detected_squeezing_db`,
  `dark_from_clearance_db`, `lossy_channel_xxpp` -- loss and
  electronic noise, on single variances or on covariance matrices.
- `required_efficiency`, `required_efficiency_db` -- the efficiency a
  target needs.
- `phase_noise_variance`, `phase_noise_squeezing_db`,
  `max_phase_noise` -- LO phase jitter, and the jitter a target
  allows.

**Lab units, fitting and planning**

- `RingSpec`, `normalized_pump`, `threshold_power`,
  `normalized_detuning`, `normalized_dispersion`,
  `normalized_frequency`, `physical_frequency`, `intracavity_photons`
  -- conversions between lab and model units.
- `fit_noise_spectra`, `NoiseFit`, `noise_spectrum_db` -- fit
  `(mu, eta, kappa)` and optionally a dark-noise floor to measured
  spectra, and the model curve itself.
- `infer_source` -- the source behind a measured squeezed /
  antisqueezed pair.
- `plan_noise_measurement`, `design_noise_frequencies`,
  `ring_from_threshold`, `save_spectra_csv`, `load_spectra_csv` --
  measurement planning, `g0` calibration, and a checked CSV format for
  spectra.

Each function's docstring (`help(sqzcomb.output_quadrature_variance)`,
for example) gives its inputs, units and conventions.

## When it refuses, and why

`sqzcomb` raises an error instead of guessing when:

- a drift matrix is unstable (above threshold), where a linearized
  noise level does not exist -- in the spectra, the port spectra, the
  output covariance and the intracavity covariance;
- a drift matrix is only marginally stable (an eigenvalue with zero
  real part, as around a soliton), unless `allow_marginal=True` says
  that direction is understood;
- a thermal occupation or dark noise is negative, a decay rate
  (`gamma`, `gamma_b`) is not positive, a detection efficiency is
  outside (0, 1] (`eta` outside [0, 1] for `output_variance_ports`), a
  variance or a dark clearance is negative, or `thermal_occupation`
  gets a non-positive frequency or a negative temperature;
- a detector is placed on a line that never enters its bus
  (`output_variance_ports`);
- a matrix is not a valid covariance matrix (odd size, not symmetric,
  not positive semidefinite, not real, invalid for the entanglement
  formulas);
- a QuTiP Hamiltonian is not quadratic, or a mode has fewer than 3
  Fock (photon-number) levels;
- Newton's method does not reach its tolerance (also inside
  `continuation`, naming the failing detuning), or a soliton seed is
  asked for with normal dispersion (`d2 >= 0`);
- a `RingSpec` has no `reference`, a non-positive linewidth, `g0` or
  wavelength, or an escape efficiency outside (0, 1]; a pump power or
  intensity is negative;
- a loss or jitter budget is asked for a target the source cannot
  reach, or one that loss or jitter could never produce;
- `fit_noise_spectra` gets only one trace and no known `kappa_hz`,
  an antisqueezed trace whose mean is more than 0.25 dB below shot
  noise, fewer than 5 frequencies, or non-positive `sigma_db`; it also
  stops if the fit fails or the data do not determine the parameters;
- `infer_source` gets a non-finite value, a non-positive `sigma_db`,
  an antisqueezed level at or below shot noise, a
  squeezed level at or above it, a squeezed-times-antisqueezed product
  below the vacuum value (impossible after loss), or a pair implying an
  efficiency outside (0, 1];
- `plan_noise_measurement` and `design_noise_frequencies` get `mu`
  outside (0, 1), `eta` outside (0, 1], a non-positive `kappa_hz`,
  frequency or `sigma_db`, unknown quadratures, a bad `n_pick`, or
  (design only) candidate frequencies that cannot determine the
  parameters at all;
- a spectra CSV file is empty, has the wrong header, no data rows,
  rows of the wrong length, a non-numeric value or a non-positive
  frequency, or the columns to be saved differ in length.

## How the results are checked

96 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.10 with the
oldest versions the package allows (NumPy 1.22.0, SciPy 1.10.0, QuTiP
4.7.0). The three QuTiP tests are skipped when QuTiP is not installed,
which is the case in the main matrix; they run in the oldest-versions
job. The numerical checks compare the package with a closed-form
result, an exact identity, or a second, independent calculation; none
compares against a number stored from an earlier run. The main ones,
with the tolerances the tests use:

**Output spectra**

- A passive cavity returns vacuum (0.5) at 10 random couplings,
  frequencies and phases, to 1e-12; with thermal baths at `n_bar`,
  exactly `(2 n_bar + 1)/2` on a grid of couplings, frequencies and
  phases, to 1e-12.
- The single parametric mode matches the closed form
  `0.5 (1 - 4 eta mu / ((1 + mu)^2 + omega^2))` to a relative 1e-10.
- At critical coupling and `mu = 0.9999` the detected squeezing lies
  between -3.011 and -2.99 dB; with full extraction at `mu = 0.99` it
  is below -20 dB.
- With thermal inputs the parametric spectrum is `(2 n_bar + 1)` times
  the vacuum one, to 1e-12; a hot-loss / cold-port case matches a
  hand-derived formula to 1e-12; the Bose function equals 1 exactly
  where `h f = k_B T ln 2` (1e-12).
- The LLE solver's free decay matches `exp(-t)` (relative 1e-3), and
  its flat state matches the cubic root (relative 1e-4).

**Photonic molecules**

- With `J = 0` the molecule reproduces the single-mode closed form to
  1e-12; undriven, it returns vacuum at 25 random settings of every
  parameter, to 1e-12.
- The undriven supermodes split by `2 J` (1e-12); the threshold
  formula agrees with the eigenvalues just below and just above it
  (0.1 % either side) for four `(J, gamma)` pairs.
- At `omega = 0` the output through the second ring equals a single
  mode with decay `1 + J^2/gamma` and efficiency
  `eta (J^2/gamma)/(1 + J^2/gamma)`, to 1e-12, for four parameter sets;
  the `J^2/gamma = 3` molecule gives between -6.03 and -6.01 dB.
- The multimode molecule equals the two-ring model at one line
  (1e-14), reduces to the plain comb machinery at `J = 0` (1e-12), and
  satisfies the same `omega = 0` equivalence for the twin-beam
  quadrature of a line pair (1e-12).

**Gaussian states and entanglement**

- Intracavity covariances of vacuum and of the parametric mode match
  closed forms (1e-12 and 1e-10); in the tested molecule and
  lossy-channel cases every symplectic eigenvalue stays at or above
  `hbar/2`.
- For the two-mode squeezed vacuum (the textbook entangled pair, with
  squeezing strength `r`), `E_N = 2r` and the Duan sum
  `= hbar e^{-2r}` to a relative 1e-12; the principal variances are
  `(hbar/2) e^{-/+2r}` to 1e-12.
- The fast PPT formula equals an explicit partial transposition on 10
  random states (relative 1e-9); `E_N` is unchanged by local
  operations (relative 1e-10).
- Every quadrature read off `output_covariance_xxpp` equals the
  separate `output_quadrature_variance` result to 1e-13; for a twin
  beam, `E_N(omega) = -ln(2 V_min)` to 1e-6 (limited by the phase grid
  used to find `V_min`); lowering the detection efficiency step by
  step (0.9, 0.5, 0.2) never increases `E_N`.
- The QuTiP adapter reproduces the parametric mode and the two-ring
  molecule, including the detuning sign, to 1e-9.

**Detection and phase noise**

- Vacuum is unchanged by loss (1e-15); two losses equal one of the
  product efficiency (1e-15 scalar, 1e-13 matrix); the scalar and
  matrix forms agree (1e-13).
- The jitter formula equals direct numerical integration over the
  Gaussian distribution to 1e-10; zero, static and very large jitter
  give the known limiting values (to 1e-12 or better); `max_phase_noise` inverts it to 1e-12; loss
  and jitter give the same result in either order (1e-12).

**Solitons**

- From a flat seed, Newton lands on the cubic root (1e-10); a soliton
  converges below 1e-12 and moves by less than 5e-3 when handed to the
  independent time-stepper for 20 time units.
- The translation mode (sliding the pulse around the ring) is a zero
  mode of the drift matrix, `M z = 0`, to a relative 1e-9, and the largest eigenvalue real part is below 1e-8 in
  size.
- The two-pulse crystal equals the single soliton at four times the
  dispersion, resampled, to 1e-8 in amplitude.

**Lab units, fitting, planning, inference**

- The frequency conversion round-trips (relative 1e-15), and `F`
  scales as the square root of power (relative 1e-12); at `threshold_power`,
  `F^2 = 1 + (alpha - 1)^2` (relative 1e-12) and the cubic has the
  root 1 (1e-10).
- The fit model equals the closed form to 1e-10 dB; noise-free spectra
  give back `mu`, `eta` and `kappa` to 1e-6; a dark floor is recovered
  to 1e-4. With 0.1 dB noise, the scatter of 40 seeded fits lies
  between 0.4 and 2 times the reported error bars.
- The planner's error bars equal the fit's within 2 %, also at
  `eta = 1`; one quadrature alone gives a scaled singular value below
  1e-8 of the largest. The greedy frequency choice was never beaten
  by any of 30 random subsets of the same size.
- `ring_from_threshold` round-trips through `threshold_power` to a
  relative 1e-12; its error bar matches finite differences within
  0.1 %. The CSV round trip is bit-exact.
- `infer_source` inverts `detected_variance` on a grid of 4 sources
  times 4 efficiencies (efficiency to 1e-9, squeezed variance to 1e-12);
  its error bars match the scatter of 600 seeded Monte Carlo trials
  (repeated simulated measurements with random noise) within 20 %.

## Corrections

**0.12.1 (this release)** fixes one bug:
`plan_noise_measurement` computed its sensitivities with a step that
went past `eta = 1`. At `eta = 1`, which the function accepts, the
model then returned NaN, the planner reported a measurable design as
"not identifiable", and `design_noise_frequencies` refused it. It now
steps backward at that edge. The fit itself was not affected.

This release also corrects the documentation. The old README's short
prediction example stopped with an "unstable" error (its 256-line
state is above threshold); it said the xxpp export uses `hbar = 2`,
which is true of `covariance_xxpp` but not of `output_covariance_xxpp`
(vacuum 0.5 I); it listed Python 3.9-3.14 in CI, but 3.10 was not
tested; and several "every" or "exact" statements were stronger than
the tests (see [How the results are checked](#how-the-results-are-checked)
for the real tolerances). The full history is in
[CHANGELOG.md](CHANGELOG.md).

## Limits

- Gaussian, linearized quantum noise only: no non-Gaussian states and
  no dynamics above threshold (unstable drift matrices are refused,
  never averaged).
- No technical noise of the resonator itself (thermorefractive or
  Raman noise); these are material physics with their own modelling
  choices.
- Continuous-wave pumping only.
- The multimode molecule assumes the auxiliary ring's line spacing
  matches the main ring's, so each line couples only to its partner.
- `infer_source` assumes a pure squeezed source followed by loss;
  `fit_noise_spectra` fits the single-mode parametric model.
- No material constants ship with the package: `g0`, linewidths and
  dispersion come from your own measurements, each `RingSpec` with a
  required `reference`.

## Where it comes from

The methods were developed for the associated paper:

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Overcoming the 3 dB
> squeezing extraction limit in silicon carbide microcombs with a
> photonic molecule," Optics Express 34(18), 34822-34834 (2026),
> https://doi.org/10.1364/OE.612248 (open access); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/sic-molecule-squeezer

This package is the general-purpose engine; the paper repository
reproduces the specific published study. Other literature the methods
rely on is cited in the docstrings and in [CHANGELOG.md](CHANGELOG.md).

## Citing, support and license

If `sqzcomb` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22015375](https://doi.org/10.5281/zenodo.22015375),
which always resolves to the latest release; every release is archived
on Zenodo. [CITATION.cff](CITATION.cff) has the details.

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of [CONTRIBUTING.md](CONTRIBUTING.md)
binds the maintainer exactly as it binds contributors: a change that
touches physics arrives with a test, and a constant arrives with its
source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/sqzcomb/issues). Usage
questions are welcome alongside bug reports; a docstring that left a
unit or a sign convention unclear is treated as a documentation bug,
not user error. While the version is below 1.0 the API may still move
between minor versions; such changes are called out in the release
notes. The normalized-unit conventions above are stable: any change
to them would be a breaking change named in the release notes, never
a quiet renormalization.

Licensed under Apache-2.0.
