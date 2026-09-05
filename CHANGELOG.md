# Changelog

Every physical claim added in any release is pinned by a test against
an exact result; the release notes on GitHub carry the full anchor
lists. Versions below 1.0 may move the API between minor versions;
such changes are called out here and in the release notes.

## v0.7.0 - 2026-09-05

- Soliton and soliton-crystal steady states by verified Newton
  continuation (`newton_state`, `soliton_seed`, `continuation`): the
  Jacobian is the exact discrete form of the package's own
  fluctuation matrix; anchors include the exact cubic root from a
  flat seed, confirmation by the independent split-step evolver, the
  exact translation (Goldstone) zero mode, and the exact rescaling
  identity between an N-pulse crystal and the single soliton at
  N^2-scaled dispersion.
- Marginal-vs-unstable distinction in the spectra: a soliton's
  Goldstone-marginal drift matrix is refused with an explanation
  unless `allow_marginal=True`; genuine instability stays refused.
- Supermode decomposition (`principal_quadratures`): the exact
  deepest-squeezing collective quadrature of any multimode covariance,
  pinned to the two-mode squeezed vacuum closed form.
- Thermal input noise: Bose occupations in the spectra and the
  intracavity covariance, `thermal_occupation` from SI-exact
  constants; anchors are the thermal fixed point of a passive cavity,
  the exact (2 n_bar + 1) scaling of the parametric oscillator, and
  the hand-derived hot-loss/cold-port mixture.
- CI now tests Python 3.9, 3.11, 3.12 and 3.13.

## v0.6.0

- Two-mode Gaussian entanglement of the comb (`entangle`): PPT
  symplectic eigenvalue via Simon invariants, logarithmic negativity,
  Duan-Simon EPR sum; cross-checked against explicit partial
  transposition and the two-mode squeezed vacuum (E_N = 2r exactly).

## v0.5.0

- Imperfect detection (`detection`): beamsplitter loss, dark noise,
  the lossy Gaussian channel on covariance matrices, and the inverse
  loss budget `required_efficiency`.

## v0.4.0

- Gaussian-state interop (`gaussian`): intracavity covariance
  (numpy-only Lyapunov solve), xxpp export with explicit hbar,
  Williamson symplectic spectra, and the QuTiP drift adapter that
  refuses non-quadratic Hamiltonians.

## v0.3.0

- Multimode comb molecule (`molecule_fluctuation_matrix`): every
  retained comb line coupled to a matching auxiliary-ring mode, with
  multi-line bus detection and twin-beam quadratures.

## v0.2.0

- Two-ring photonic molecule (`photonic_molecule`,
  `output_variance_ports`, `molecule_threshold`): the extraction
  mechanism in its simplest form, driven past the single-ring 3 dB
  detected-squeezing limit in the tests.

## v0.1.0

- Lugiato-Lefever solver, homogeneous steady states against the exact
  cubic, linearized fluctuation matrix with stability guard, and
  input-output quadrature spectra with extraction port and intrinsic
  loss.
