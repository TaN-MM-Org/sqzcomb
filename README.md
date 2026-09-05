# sqzcomb

[![PyPI](https://img.shields.io/pypi/v/sqzcomb)](https://pypi.org/project/sqzcomb/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22015375-blue)](https://doi.org/10.5281/zenodo.22015375) [![tests](https://github.com/TaN-MM-Org/sqzcomb/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/sqzcomb/actions)

Squeezed light in Kerr microcombs, computed **end to end**: from the
classical Lugiato-Lefever steady state, through the linearized quantum
fluctuations around it, to the **output quadrature-noise spectrum a
homodyne detector would report**. The package exists because intracavity
squeezing is not the observable; what leaves the extraction port is, and
the coupling that maximizes one does not maximize the other.

## Entanglement of the comb (new in v0.6)

The `entangle` module quantifies twin-beam entanglement directly from the
covariance matrices the package produces: the partial-transpose symplectic
eigenvalue through the Simon invariants (necessary and sufficient for
two-mode Gaussian states), the logarithmic negativity, and the symmetric
Duan-Simon EPR sum with its separability bound. The invariant formula is
cross-checked against an independent explicit-partial-transpose
computation, against the closed-form two-mode squeezed vacuum (E_N = 2r
exactly), and on the driven photonic molecule, whose two rings turn out to
be PPT-entangled below threshold while the symmetric Duan sum misses it,
a working demonstration of why the sharper criterion matters.

```python
from sqzcomb import (photonic_molecule, intracavity_covariance,
                     covariance_xxpp, entanglement_report)
M, gammas = photonic_molecule(mu=0.8, J=1.0)
sigma = covariance_xxpp(intracavity_covariance(M, gammas))
print(entanglement_report(sigma, 0, 1))
```

## Status

v0.7.0 (alpha). Implemented and tested (66 tests, Python 3.9-3.13):

- Lugiato-Lefever solver (Strang splitting; the Kerr step and the
  linear-plus-pump step are each exact)
- homogeneous steady states against the exact cubic
- linearized fluctuation (Bogoliubov) matrix around an arbitrary steady
  state, with a stability guard that refuses above-threshold states
- input-output quadrature spectra with an extraction port and intrinsic
  loss; single-mode and joint two-mode quadratures
- **photonic molecule (new in v0.2)**: the two-ring coupled-mode model
  (`photonic_molecule`), per-mode-coupling output spectra
  (`output_variance_ports`), and the exact instability threshold
  (`molecule_threshold`)
- **multimode comb molecule (new in v0.3)**: every retained comb line of
  the LLE fluctuation matrix coupled to a matching auxiliary-ring mode
  (`molecule_fluctuation_matrix`), with multi-line bus detection and
  joint twin-beam quadratures through the auxiliary ring
- **Gaussian-state interop (new in v0.4)**: steady-state covariance
  matrices of the intracavity Gaussian state (`intracavity_covariance`,
  numpy-only Lyapunov solve), export in the standard xxpp quadrature
  ordering with an explicit hbar convention (`covariance_xxpp`, vacuum
  exactly the identity at hbar = 2), Williamson symplectic spectra
  (`symplectic_eigenvalues`), and a QuTiP adapter (`drift_from_qutip`)
  that turns any quadratic QuTiP Hamiltonian into a drift matrix this
  package's spectra machinery accepts, refusing non-quadratic
  Hamiltonians rather than silently linearizing them. Install
  `sqzcomb[interop]` for the adapter; its test asserts that the released
  two-ring molecule is reproduced exactly from a QuTiP Hamiltonian.

Verified against closed forms in the test-suite: vacuum passes a passive
cavity, and a passive molecule, unchanged for every coupling, port and
frequency; the degenerate parametric oscillator output spectrum is
reproduced to 1e-10; and the textbook result that detectable squeezing
saturates at 3 dB at critical coupling, while full extraction breaks that
limit, emerges from the machinery rather than being asserted.

For the molecule, the test-suite additionally asserts: exact reduction to
the single ring at zero coupling; passive supermodes split by exactly 2J;
the resonant threshold mu = 1 + J^2/gamma (static branch, Hopf branch
1 + gamma beyond J = gamma); the quarter-turn quadrature rotation of the
-iJ hop; the exact zero-frequency equivalence of the auxiliary-ring port
to an effective single mode with escape efficiency
(J^2/gamma)/(1 + J^2/gamma); and a J^2/gamma = 3 molecule reaching 6 dB
detected squeezing through the auxiliary port although the Kerr ring
itself has no extraction port, which is the molecule extraction mechanism
in its simplest form.

For the multimode molecule, the asserts continue in the same spirit: at
one retained line the builder equals the released two-ring matrix to
machine precision; at zero coupling it reduces exactly to the plain
fluctuation matrix and the v0.1 spectra, single-line and twin-beam; a
passive multimode molecule returns exact vacuum through any bus; the
resonant auxiliary ring at zero frequency is exactly the single ring with
J^2/gamma_b extra loss per line and the quarter-turn rotation; and when
J^2/gamma_b exceeds one, twin-beam squeezing detected through the
auxiliary bus is strictly deeper than through the main bus of the same
device. The stability guard is also asserted to refuse a flat state that
is above a pair's modulational-instability threshold once the molecule's
added loss is removed.

Later releases are documented in their own sections below: imperfect
detection (v0.5), two-mode Gaussian entanglement of the comb (v0.6),
and solitons, supermodes and thermal baths (v0.7).

Every item of the original roadmap is now implemented. Deliberate
scope, stated plainly -- designed-out, not overlooked: quantum noise
beyond the Gaussian linearization (no non-Gaussian states, no
above-threshold dynamics -- unstable and marginal drift matrices are
refused or must be explicitly acknowledged, never silently averaged);
technical noise of the resonator itself (thermorefractive and Raman
noise are material physics with their own modelling choices); and
pulsed or synchronously pumped operation (the pump here is CW). The
maximally informative multimode decomposition for *pure* states
(Bloch-Messiah) is likewise out; `principal_quadratures` answers the
question experiments ask -- the deepest collective squeezing and its
supermode -- exactly, for pure and mixed states alike.

## Install and use

```
pip install sqzcomb
```

For development, clone the repository and `pip install -e .[test]`.

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

## Imperfect detection (new in v0.5)

Every spectrum above is the noise at the extraction port; the
photodiodes report less. `detection` applies the standard beamsplitter
model of optical loss and quantum efficiency plus additive electronic
noise, in both languages the package speaks: as a scalar map on
quadrature variances (`detected_variance`, `detected_squeezing_db`,
with `dark_from_clearance_db` converting a receiver's dark clearance
in dB into a variance) and as the lossy Gaussian channel on xxpp
covariance matrices (`lossy_channel_xxpp`, per-mode efficiencies
allowed). `required_efficiency` inverts the loss model into the number
an experiment plans around: the minimum efficiency that still delivers
a target squeezing from a given source. Loss stages compose by
multiplying efficiencies, and the test suite asserts that composition
exactly, along with the vacuum fixed point, physicality of the channel
(symplectic eigenvalues never fall below hbar/2), and agreement of the
scalar and matrix forms on a squeezed mode.

References: the beamsplitter model of detector inefficiency,
U. Leonhardt, Measuring the Quantum State of Light (Cambridge, 1997);
the Gaussian lossy channel, C. Weedbrook et al., Rev. Mod. Phys. 84,
621 (2012).

## Solitons, supermodes and thermal baths (new in v0.7)

The `soliton` module finds *localized* steady states -- dissipative
Kerr solitons and soliton crystals -- by Newton's method in Fourier
space (`newton_state`, seeded by `soliton_seed`, swept by
`continuation`). The Jacobian it inverts is the same linearization the
quantum-noise machinery builds, in its exact discrete form, so the
solver refuses to return anything whose stationary residual is not
verified below tolerance. The test suite then closes the loop through
independent code paths: the converged soliton barely moves under the
split-step time evolver, its background matches the exact cubic root,
its linearization carries the *exact* translation (Goldstone) zero
mode with eigenvector d psi / d theta, and a two-pulse crystal equals,
grid point for grid point, the single soliton of the equation with
dispersion scaled by four -- the exact rescaling of the periodic
domain. Because the Goldstone mode makes a soliton's drift matrix
marginally rather than asymptotically stable, the spectra now
distinguish the two: a marginal matrix is refused with an explanation
unless `allow_marginal=True` says the translation mode is understood,
while a genuinely unstable one stays refused regardless.

`principal_quadratures` adds the supermode decomposition of any
multimode covariance matrix: the eigendecomposition of sigma, whose
smallest eigenvalue is -- exactly, by linear algebra -- the deepest
squeezing any generalized quadrature of the state attains, and whose
eigenvector is the supermode carrying it. The two-mode squeezed vacuum
pins it in closed form ((hbar/2) e^{-/+ 2r} with EPR supermodes), and
it is deliberately distinct from the Williamson spectrum, which
measures mixedness, not squeezing.

Thermal input noise completes the bath model: `output_quadrature_variance`
takes Bose occupations for the extraction-port and loss baths, and
`intracavity_covariance` for each mode's bath, with `thermal_occupation`
supplying the physical number from the exact SI constants (h and k_B
are exact by definition since the 2019 redefinition). The anchors are
closed forms: a passive cavity with baths at n_bar emits exactly
(2 n_bar + 1)/2 at every frequency, coupling and phase; the parametric
oscillator's spectra scale by exactly (2 n_bar + 1); a passive mode
holds exactly n_bar photons; and the hot-loss/cold-port mixture matches
its hand-derived form.

## Methodological basis

> T. M. Mahim, M. M. Rahman and A. S. M. Mohsin, "Overcoming the 3 dB
> squeezing extraction limit in silicon carbide microcombs with a
> photonic molecule," Optics Express 34(18), 34822-34834 (2026),
> https://doi.org/10.1364/OE.612248 (open access); code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/sic-molecule-squeezer

This package is the general-purpose engine; the paper repository
reproduces the specific published study. v0.2 adds the two-mode photonic
molecule, the extraction mechanism in its simplest form; the paper's full
multimode comb molecule remains in the paper repository.

## Support and governance

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the issue tracker at
https://github.com/TaN-MM-Org/sqzcomb/issues. Usage questions are
welcome there alongside bug reports; a docstring that left a unit or a
sign convention unclear is treated as a documentation bug, not as user
error. The maintainer aims to respond within a week.

While the version is below 1.0 the API may still move between minor
versions; such changes are called out in the release notes. The
normalized-unit conventions stated above are stable: any change to them
would be a breaking change named in the release notes, never a quiet
renormalization.

## License

Apache-2.0
