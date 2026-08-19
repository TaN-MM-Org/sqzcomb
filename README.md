# sqzcomb

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22015376.svg)](https://doi.org/10.5281/zenodo.22015376) [![tests](https://github.com/TaN-MM-Org/sqzcomb/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/sqzcomb/actions)

Squeezed light in Kerr microcombs, computed **end to end**: from the
classical Lugiato-Lefever steady state, through the linearized quantum
fluctuations around it, to the **output quadrature-noise spectrum a
homodyne detector would report**. The package exists because intracavity
squeezing is not the observable; what leaves the extraction port is, and
the coupling that maximizes one does not maximize the other.

## Status

v0.1.0 (alpha). Implemented and tested:

- Lugiato-Lefever solver (Strang splitting; the Kerr step and the
  linear-plus-pump step are each exact)
- homogeneous steady states against the exact cubic
- linearized fluctuation (Bogoliubov) matrix around an arbitrary steady
  state, with a stability guard that refuses above-threshold states
- input-output quadrature spectra with an extraction port and intrinsic
  loss; single-mode and joint two-mode quadratures

Verified against closed forms in the test-suite: vacuum passes a passive
cavity unchanged for every coupling and frequency; the degenerate
parametric oscillator output spectrum is reproduced to 1e-10; and the
textbook result that detectable squeezing saturates at 3 dB at critical
coupling, while full extraction breaks that limit, emerges from the
machinery rather than being asserted.

Not yet implemented (the v0.2+ roadmap, in order): the auxiliary-ring
photonic molecule and its Purcell extraction of below-threshold modes,
soliton-crystal steady-state continuation, supermode decomposition of the
multimode covariance, and thermal input noise.

## Install and use

```
pip install -e .
```

```python
import numpy as np
from sqzcomb import (lle_evolve, fluctuation_matrix,
                     output_quadrature_variance, squeezing_db)

# steady state at pump F and detuning alpha, anomalous dispersion d2
psi = lle_evolve(np.full(256, 0.05 + 0j), F=1.2, alpha=0.8,
                 dispersion=(-0.02,), t_end=300.0)

M, modes = fluctuation_matrix(psi, alpha=0.8, dispersion=(-0.02,))
i0 = int(np.where(modes == 0)[0][0])
v = output_quadrature_variance(M, eta=0.5, omega=0.0,
                               mode_index=i0, n_modes=modes.size)
print(squeezing_db(v), "dB relative to vacuum")
```

Units are the standard normalized LLE units: time in photon lifetimes,
eta = kappa_ex / kappa, vacuum variance 1/2.

## Methodological basis

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Overcoming the 3 dB
> squeezing extraction limit in silicon carbide microcombs with a
> photonic molecule" (under review); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/sic-molecule-squeezer

This package is the general-purpose engine; the paper repository
reproduces the specific published study, including the photonic-molecule
extraction that this package's v0.1 does not yet contain.

## License

Apache-2.0
