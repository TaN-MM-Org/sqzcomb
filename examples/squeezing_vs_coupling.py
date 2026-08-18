"""Detected squeezing against extraction ratio for a flat pumped state.

Reproduces the qualitative coupling trade-off: intracavity gain wants weak
extraction, detection wants strong extraction, and the observable saturates
in between.
"""
import numpy as np

from sqzcomb import (lle_evolve, fluctuation_matrix,
                     output_quadrature_variance, squeezing_db)

F, alpha = 0.9, 0.3
psi = lle_evolve(np.full(32, 0.05 + 0j), F=F, alpha=alpha,
                 t_end=300.0, dt=0.01)
M, modes = fluctuation_matrix(psi, alpha)
i0 = int(np.where(modes == 0)[0][0])

print(" eta   best squeezing over phase (dB)")
for eta in (0.1, 0.3, 0.5, 0.7, 0.9, 1.0):
    best = min(
        squeezing_db(output_quadrature_variance(M, eta, 0.0, i0,
                                                modes.size, phi=phi))
        for phi in np.linspace(0, np.pi, 60)
    )
    print(f" {eta:.1f}   {best:7.3f}")
