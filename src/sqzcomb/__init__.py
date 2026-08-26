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
                       output_variance_ports, photonic_molecule)
from .spectra import output_quadrature_variance, squeezing_db
from .gaussian import (covariance_xxpp, drift_from_qutip,
                       intracavity_covariance, symplectic_eigenvalues)

__version__ = "0.4.0"
__all__ = [
    "lle_evolve", "homogeneous_steady_states",
    "fluctuation_matrix", "single_mode_parametric",
    "photonic_molecule", "output_variance_ports", "molecule_threshold",
    "molecule_fluctuation_matrix",
    "output_quadrature_variance", "squeezing_db",
    "intracavity_covariance", "covariance_xxpp",
    "symplectic_eigenvalues", "drift_from_qutip",
]
