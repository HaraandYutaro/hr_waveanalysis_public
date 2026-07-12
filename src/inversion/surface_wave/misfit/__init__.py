"""Surface-wave misfit functions (canonical package).

All misfits exposed here are wave-agnostic by design. Two of them
(``DeterminantMisfit``, ``NearestNeighborMisfit``) follow a strategy pattern,
taking a wave-specific callable (``secular_func`` or ``predict_modes_func``)
at construction time.

References
----------
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380. (Determinant misfit + niching PSO)
- Wilken, D. & Rabbel, W. (2012). Geophys. J. Int., 190, 580-594.
  (Nearest-neighbour misfit)
"""

from src.inversion.surface_wave.misfit.base import DispersionMisfit
from src.inversion.surface_wave.misfit.determinant import DeterminantMisfit
from src.inversion.surface_wave.misfit.nearest_neighbor import NearestNeighborMisfit
from src.inversion.surface_wave.misfit.weighted_l2 import WeightedL2Misfit

__all__ = [
    "DispersionMisfit",
    "WeightedL2Misfit",
    "DeterminantMisfit",
    "NearestNeighborMisfit",
]
