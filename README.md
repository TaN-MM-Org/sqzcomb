# sqzcomb

[![PyPI](https://img.shields.io/pypi/v/sqzcomb)](https://pypi.org/project/sqzcomb/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015375-blue)](https://doi.org/10.5281/zenodo.22015375) [![tests](https://github.com/TaN-MM-Org/sqzcomb/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/sqzcomb/actions)

Quantum noise in Kerr microcombs, computed end to end: from the
classical field circulating in the ring, through the quantum
fluctuations around it, to **the noise a detector on the output fiber
would actually report**. The package exists because squeezing inside
the cavity is not the observable -- what leaves the extraction port
is -- and the coupling that maximizes one does not maximize the
other.

## Install

```
pip install sqzcomb              # numpy + scipy
pip install sqzcomb[interop]     # adds the QuTiP adapter
```

For development: clone the repository and `pip install -e .[test]`.

## Quick start, from laboratory numbers

The package computes in the standard normalized units of the
Lugiato-Lefever equation; `RingSpec` is the exact dictionary from the
numbers a lab actually has (v0.9):

```python
import sqzcomb as sc

ring = sc.RingSpec(kappa_hz=100e6, eta_esc=0.8, g0_hz=10.0,
                   lambda_pump_m=1.55e-6,
                   reference="our device, linewidth scan 2026-09")
F = sc.normalized_pump(ring, power_w=2e-3)       # pump amplitude
P_th = sc.threshold_power(ring, alpha=1.0)       # comb threshold, W
```

and `fit_noise_spectra` runs the model backward from measured data:
feed it the squeezed and antisqueezed homodyne traces (dB relative to
shot noise) and it returns the pump parameter, total detection
efficiency and cavity linewidth with uncertainties. Identifiability
is arithmetic, enforced rather than hoped for: one quadrature is a
single Lorentzian -- two shape numbers for three unknowns -- so the
fit requires both traces or an independently measured linewidth, and
refuses otherwise; an antisqueezed trace below shot noise (impossible
in this model) is refused too.

A minimal prediction in normalized units:

```python
import numpy as np
from sqzcomb import (lle_evolve, fluctuation_matrix,
                     output_quadrature_variance, squeezing_db)

psi = lle_evolve(np.full(256, 0.05 + 0j), F=1.2, alpha=0.8,
                 dispersion=(-0.02,), t_end=300.0)
M, modes = fluctuation_matrix(psi, alpha=0.8, dispersion=(-0.02,))
i0 = int(np.where(modes == 0)[0][0])
v = output_quadrature_variance(M, eta=0.5, omega=0.0,
                               mode_index=i0, n_modes=modes.size)
print(squeezing_db(v), "dB relative to vacuum")
```

Conventions, stated once and stable: time in photon lifetimes,
eta = kappa_ex / kappa, vacuum variance 1/2 (xxpp export uses
hbar = 2, vacuum exactly the identity).

## What is inside

- **The classical field**: a Lugiato-Lefever split-step solver whose
  two half-steps are each exact; homogeneous steady states checked
  against the exact cubic; and a Newton solver for localized states
  -- dissipative solitons and soliton crystals -- that refuses to
  return anything whose stationarity is not verified.
- **The quantum fluctuations**: the linearized fluctuation matrix
  around any steady state, with a stability guard that refuses
  unstable states (and marginal soliton states unless the translation
  mode is explicitly acknowledged); input-output quadrature spectra
  through an extraction port with intrinsic loss and thermal baths;
  single-mode, joint two-mode, and full covariance matrices.
- **Photonic molecules**: the two-ring coupled model and the full
  multimode comb molecule (every retained comb line coupled to an
  auxiliary ring), including the extraction mechanism the associated
  paper is about -- a ring with no extraction port of its own
  delivering deep detected squeezing through its neighbor.
- **Entanglement, as a detector would certify it**: two-mode Gaussian
  entanglement of comb lines (logarithmic negativity through the
  Simon invariants, the Duan-Simon sum with its bound) both inside
  the cavity and -- frequency-resolved, through loss and finite
  detection efficiency -- at the output.
- **Imperfect detection**: the standard beamsplitter loss model and
  additive electronic noise, as scalar maps and as the Gaussian
  channel on covariance matrices; `required_efficiency` inverts it
  into the number an experiment plans around.
- **Supermodes**: `principal_quadratures` finds the deepest squeezing
  any generalized quadrature of a multimode state attains, and the
  supermode carrying it -- exactly, by linear algebra, for pure and
  mixed states alike.
- **Interop**: xxpp covariance export, Williamson spectra, and a
  QuTiP adapter that turns any quadratic QuTiP Hamiltonian into a
  drift matrix this package accepts, refusing non-quadratic ones
  rather than silently linearizing.

## How it is checked

80 tests (Python 3.9-3.13, run in CI on every push), every physics
claim anchored to a closed form, an exact identity, or two
independent code paths -- never a stored number. Highlights: vacuum
passes any passive device unchanged at every coupling, port and
frequency; the degenerate parametric oscillator spectrum to 1e-10;
the textbook 3 dB detected-squeezing ceiling at critical coupling --
and its violation by full extraction -- emerging from the machinery
rather than being asserted; exact molecule thresholds, supermode
splittings and port equivalences; the converged soliton's exact
translation zero mode and the two-pulse crystal equal to a rescaled
single soliton grid point for grid point; thermal baths pinned to
(2 n_bar + 1)/2 closed forms with h and k_B from their exact SI
definitions; entanglement formulas cross-checked against explicit
partial transposition and the closed-form two-mode squeezed vacuum;
lab-unit conversions anchored by exact round trips and by
`threshold_power` hitting the root of the package's own cubic; and
the noise-spectrum fit recovering generating parameters with
Monte-Carlo scatter matching its reported uncertainties.

## Honest limits

Deliberate scope, designed out with reasons: no quantum noise beyond
the Gaussian linearization (no non-Gaussian states or above-threshold
dynamics -- unstable drift matrices are refused, never silently
averaged); no technical noise of the resonator itself
(thermorefractive and Raman noise are material physics with their own
modelling choices); and continuous-wave pumping only.

## Associated paper

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Overcoming the 3 dB
> squeezing extraction limit in silicon carbide microcombs with a
> photonic molecule," Optics Express 34(18), 34822-34834 (2026),
> https://doi.org/10.1364/OE.612248 (open access); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/sic-molecule-squeezer

This package is the general-purpose engine; the paper repository
reproduces the specific published study.

## Support and governance

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/sqzcomb/issues). Usage
questions are welcome alongside bug reports; a docstring that left a
unit or a sign convention unclear is treated as a documentation bug,
not user error. While the version is below 1.0 the API may still move
between minor versions; such changes are called out in the release
notes. The normalized-unit conventions above are stable: any change
to them would be a breaking change named in the release notes, never
a quiet renormalization.

## License

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22015375](https://doi.org/10.5281/zenodo.22015375),
which always resolves to the latest version.
