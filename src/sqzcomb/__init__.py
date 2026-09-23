"""sqzcomb: squeezed light in Kerr microcombs, from steady state to detected noise.

Solves the Lugiato-Lefever equation for the classical intracavity field,
linearizes the quantum fluctuations around that steady state, and computes
output quadrature-noise spectra through the standard input-output relations,
so that a resonator design can be judged by the number a homodyne detector
would actually report.

Methodological basis: T. M. Mahim, M. M. Rahman and A. S. M. Mohsin,
"Overcoming the 3 dB squeezing extraction limit in silicon carbide
microcombs with a photonic molecule" (under review).
"""
from .lle import lle_evolve, homogeneous_steady_states
from .linearize import fluctuation_matrix, single_mode_parametric
from .molecule import (molecule_fluctuation_matrix, molecule_threshold,
                       output_variance_ports, photonic_molecule,
                       ring_line_frequencies,
                       vernier_molecule_fluctuation_matrix)
from .spectra import (output_covariance_xxpp, output_entanglement,
                      output_entanglement_spectrum,
                      output_quadrature_variance, squeezing_db)
from .detection import (dark_from_clearance_db, detected_squeezing_db,
                        detected_variance, lossy_channel_xxpp,
                        max_phase_noise, phase_noise_squeezing_db,
                        phase_noise_variance, required_efficiency,
                        required_efficiency_db)
from .entangle import (duan_epr_sum, entanglement_report,
                       logarithmic_negativity, ppt_symplectic_eigenvalue,
                       two_mode_reduction)
from .gaussian import (covariance_xxpp, drift_from_qutip,
                       intracavity_covariance, principal_quadratures,
                       symplectic_eigenvalues)
from .soliton import continuation, newton_state, soliton_seed
from .thermal import thermal_occupation
from .physical import (RingSpec, intracavity_photons, kerr_shift_from_n2,
                       normalized_detuning,
                       normalized_dispersion, normalized_frequency,
                       normalized_pump, physical_frequency, threshold_power)
from .fitnoise import (ModelFit, NoiseFit, fit_noise_spectra,
                       fit_spectra_model, jitter_average,
                       molecule_spectra_model, noise_spectrum_db,
                       parametric_spectra_model)
from .inference import infer_source
from .lab import (design_noise_frequencies, load_spectra_csv,
                  plan_noise_measurement, ring_from_threshold,
                  save_spectra_csv)
from .kerrpo import kerr_parametric_drift, kerr_parametric_states
from .master import (coherent_state, fock_operators, fock_state,
                     kerr_parametric_master, liouvillian, master_evolve,
                     master_moments, master_output_spectrum, steady_state,
                     wigner, wigner_negativity)
from .technical import (classical_noise_variance,
                        classical_noise_variance_ports, gain_noise_drive,
                        lle_mode_amplitudes, normalized_psd,
                        pump_noise_drive, resonance_noise_drive)
from .pulsed import (covariance_evolution, parametric_pulse_drift,
                     temporal_mode_variance)

__version__ = "0.13.0"
__all__ = [
    "lle_evolve", "homogeneous_steady_states",
    "fluctuation_matrix", "single_mode_parametric",
    "photonic_molecule", "output_variance_ports", "molecule_threshold",
    "molecule_fluctuation_matrix",
    "newton_state", "soliton_seed", "continuation",
    "output_quadrature_variance", "squeezing_db",
    "output_covariance_xxpp", "output_entanglement",
    "output_entanglement_spectrum",
    "intracavity_covariance", "covariance_xxpp",
    "symplectic_eigenvalues", "principal_quadratures",
    "thermal_occupation",
    "RingSpec", "normalized_pump", "threshold_power",
    "normalized_detuning", "normalized_dispersion",
    "normalized_frequency", "physical_frequency", "intracavity_photons",
    "NoiseFit", "fit_noise_spectra", "noise_spectrum_db", "drift_from_qutip",
    "two_mode_reduction", "ppt_symplectic_eigenvalue",
    "logarithmic_negativity", "duan_epr_sum", "entanglement_report",
    "detected_variance", "detected_squeezing_db", "dark_from_clearance_db",
    "lossy_channel_xxpp", "required_efficiency", "required_efficiency_db",
    "phase_noise_variance", "phase_noise_squeezing_db", "max_phase_noise",
    "plan_noise_measurement", "design_noise_frequencies",
    "ring_from_threshold", "save_spectra_csv", "load_spectra_csv",
    "infer_source",
    # new in 0.13
    "ring_line_frequencies", "vernier_molecule_fluctuation_matrix",
    "ModelFit", "fit_spectra_model", "parametric_spectra_model",
    "molecule_spectra_model", "jitter_average",
    "kerr_parametric_drift", "kerr_parametric_states",
    "fock_operators", "liouvillian", "steady_state", "master_evolve",
    "fock_state", "coherent_state", "kerr_parametric_master",
    "master_moments", "master_output_spectrum", "wigner",
    "wigner_negativity",
    "classical_noise_variance", "classical_noise_variance_ports",
    "resonance_noise_drive", "pump_noise_drive", "gain_noise_drive",
    "lle_mode_amplitudes", "normalized_psd",
    "covariance_evolution", "temporal_mode_variance",
    "parametric_pulse_drift", "kerr_shift_from_n2",
]
