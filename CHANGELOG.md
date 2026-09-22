# Changelog

## 0.12.1 (2026-09-22)

A bug fix, a wider CI matrix, and a rewritten README.

### Fixed

- `lab.plan_noise_measurement` took its central-difference step in
  `eta` past 1. At `eta = 1`, which it accepts, the model returned NaN
  there, so the planner reported an identifiable design as not
  identifiable (`sigma` None) and `design_noise_frequencies` refused it
  ("even the full candidate list cannot determine the parameters").
  At that edge it now uses a backward difference. `fit_noise_spectra`
  was not affected.

### Tests

- `test_plan_and_design_work_at_full_efficiency` (tests/test_lab.py):
  at `eta = 1` the planner is identifiable, its information matrix is
  finite, its error bars match the fit's within 2 %, and
  `design_noise_frequencies` returns a design. It fails on 0.12.0.
  96 tests in total.

### Changed

- CI now also runs Python 3.10 (the matrix is 3.9 to 3.14), and a new
  `oldest-dependencies` job runs the suite on Python 3.10 with
  NumPy 1.22.0, SciPy 1.10.0 and QuTiP 4.7.0, the lowest versions
  `pyproject.toml` allows. No code change was needed for them.
- README rewritten for readers outside the field: a guide to the terms,
  units and conventions, ten examples each with the output it prints,
  the refusals, and the test checks with their real tolerances. The old
  README's short prediction example stopped with an "unstable" error;
  it said the xxpp export uses hbar = 2, which holds for
  `covariance_xxpp` but not for `output_covariance_xxpp` (vacuum
  0.5 I); and it listed Python 3.9-3.14 in CI, but 3.10 was not run.
- CONTRIBUTING.md: the dependency note said "NumPy only"; SciPy has
  been required since 0.9.0.

### Notes on earlier entries

- 0.12.0: "exact inverse ... machine precision" -- the test holds the
  efficiency to 1e-9 and the source variance to 1e-12, and the pure-state
  product V_s V_a = 1/4 to 1e-9. The error bars are central differences
  of the closed form, not exact derivatives; they match seeded Monte
  Carlo within 20 %.
- 0.11.0: "the greedy design never loses to a random subset" -- the
  test compares it with 30 random subsets of the same size.
- The old README: "Monte-Carlo scatter matching its reported
  uncertainties" (of the 0.9.0 noise-spectrum fit) -- the test accepts
  a scatter between 0.4 and 2 times the reported error bars (40 fits).

## 0.12.0 (2026-09-18)

Source inference, and a future-proofing pass.

- `inference.infer_source`: the closed-form inversion of a measured
  squeezing/anti-squeezing pair under the standard pure-state-plus-
  loss model -- the routine "inferred on-chip squeezing" of the
  2025-2026 literature (usage context: Ulanov et al., Nat. Commun.
  16, 10791 (2025); Karnik et al., arXiv:2605.27607) -- with its
  pure-state assumption stated, exact error propagation, and
  refusals for everything the algebra can catch (sub-vacuum
  uncertainty products, missing antisqueezing, implied efficiency
  outside (0, 1]).
- CI now also runs on Python 3.14.
- Anchors: the inversion is the exact inverse of the package's own
  `detected_variance` over a grid of sources and efficiencies (two
  code paths, machine precision); the inferred source restores
  V_s V_a = 1/4 exactly; error bars match seeded Monte Carlo; the
  published Ulanov pair runs to a physical efficiency with no paper
  number asserted; every refusal pinned.

## 0.11.0 (2026-09-17)

Lab adaptability: plan the measurement before taking it, and
calibrate the hardest ring number from a routine one.

- `lab.plan_noise_measurement`: predicted error bars for a planned
  noise-spectrum measurement -- the same (J^T W J)^-1 matrix the fit
  reports, evaluated before any data exists -- with a scale-invariant
  identifiability verdict. The single-quadrature degeneracy that
  `fit_noise_spectra` refuses after the fact is reported here before
  the beam time, by exact rank arithmetic.
- `lab.design_noise_frequencies`: greedy D-optimal choice of analysis
  frequencies (Pukelsheim, Optimal Design of Experiments, SIAM
  (2006)), refusing candidate lists that cannot identify the
  parameters.
- `lab.ring_from_threshold`: g0 calibrated from the measured comb
  threshold power by exact inversion of `threshold_power`'s
  F_th^2 = 1 + (alpha - 1)^2 identity, returning a ready-to-use
  `RingSpec` whose `reference` records the calibration, with the
  exactly propagated error bar.
- `lab.save_spectra_csv` / `load_spectra_csv`: a plain, checked CSV
  contract for measured spectrum pairs; round trips are exact.
- Anchors: planner and fit sigmas agree as two code paths of one
  matrix; the single-quadrature information matrix is exactly
  rank-deficient (unit-free singular value below 1e-8 of the largest)
  while the two-quadrature one is not; the greedy design never loses
  to a random subset; the g0 round trip through `threshold_power` is
  exact to machine precision and its error bar matches finite
  differences; CSV round trips are bit-exact.

## 0.10.0 (2026-09-13)

Physics upgrade from the 2026 literature: local-oscillator phase
noise, the practical limit the unified variance budget of integrated
squeezers identifies once loss is tamed (D. J. Dean et al.,
"Practical limits on integrated squeezers", npj Nanophotonics (2026),
doi:10.1038/s44310-026-00125-5; the mixing law itself is the standard
one of Dwyer et al., Opt. Express 21, 19047 (2013) and Oelker et al.,
Optica 3, 682 (2016)).

### Added

- `phase_noise_variance` / `phase_noise_squeezing_db`: the detected
  quadrature variance under Gaussian local-oscillator phase jitter
  (plus an optional static offset), via the exact closed-form
  Gaussian average <cos 2 theta> = exp(-2 sigma^2) of the rotation
  law V(theta) = V_sq cos^2 + V_anti sin^2. Works on scalars or
  whole spectra.
- `max_phase_noise`: the largest RMS jitter that still delivers a
  target variance -- the phase-noise counterpart of
  `required_efficiency`, closed-form inverted, refusing targets that
  are not phase-noise-limited (reachable only by fixing the loss
  budget) or that phase noise could never worsen a source into.

### Anchors (asserted in `tests/test_phase_noise.py`, not stated)

- The closed form equals direct numerical integration over the
  Gaussian jitter distribution (two independent code paths) to 1e-10
  at several jitters and offsets.
- Exact limits: zero jitter returns the source variance exactly; the
  static-offset case is the plain rotation law; infinite jitter gives
  the quadrature-blind (V_sq + V_anti)/2; vacuum is a fixed point at
  every jitter; degradation is monotone in jitter.
- The `max_phase_noise` inversion round-trips to 1e-12 and both
  refusals fire.
- Composition with the loss channel commutes exactly (both are affine
  maps sharing the vacuum fixed point), asserted rather than assumed.

## 0.9.0 (2026-09-12)

Adaptability release: the exact bridge between laboratory ring
parameters and the package's normalized units, and the fit that
extracts squeezer parameters from measured noise spectra.

### Added

- `RingSpec` and conversion helpers (`normalized_pump`,
  `normalized_detuning`, `normalized_dispersion`,
  `normalized_frequency`/`physical_frequency`, `threshold_power`,
  `intracavity_photons`). Anchors: exact round trips; the derived
  scaling structure of F; `threshold_power` cross-validated through
  the package's own flat-state cubic (rho = 1 root at the returned
  power); mandatory provenance (`reference`) on every RingSpec;
  hbar and c from scipy.constants, no hand-typed constants.
- `fit_noise_spectra` / `NoiseFit` / `noise_spectrum_db`: (mu, eta,
  kappa) with uncertainties from measured homodyne spectra in dB
  relative to shot noise, optional common dark-noise floor, model
  curves generated by the package's own input-output solver and
  cross-checked against the independent closed form in the tests.
  Identifiability handled by refusal: one quadrature alone (two
  Lorentzian shape numbers, three unknowns) is rejected unless the
  linewidth is supplied; a systematically sub-shot-noise
  antisqueezed trace is rejected as unphysical for this model.

### Changed

- The package now depends on scipy (>= 1.10) alongside numpy: the
  physical bridge takes hbar and c from `scipy.constants` rather than
  typing them, and the spectrum fit uses
  `scipy.optimize.least_squares`. (Every sibling package in the
  organization already carries this dependency.)
## v0.8.0 - 2026-09-10

Detected entanglement: the entanglement certified at the OUTPUT, not
inside the cavity.

- `output_covariance_xxpp`: full symmetrized xxpp covariance of the
  output field at analysis frequency omega (vacuum 0.5 I), composing
  the validated input-output spectra with the validated xxpp
  conversion; every single- and joint-quadrature entry equals the
  independent `output_quadrature_variance` path to 1e-13 (asserted).
- `output_entanglement` / `output_entanglement_spectrum`:
  frequency-resolved logarithmic negativity between output comb
  lines, with an optional detection-efficiency channel applied
  through the exact Gaussian lossy map before the PPT test. Anchors:
  passive thermal outputs are exactly ((2 nbar + 1)/2) I and
  separable at every frequency and coupling; symmetric twin-beam
  E_N(omega) = -ln(2 V_EPR_min(omega)) against the independent
  joint-quadrature spectra path (1e-6, phase-grid limited); E_N -> 0
  at large omega; detection loss never creates entanglement
  (monotone, asserted stepwise); above-threshold drift matrices are
  refused as everywhere else in the package.

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
