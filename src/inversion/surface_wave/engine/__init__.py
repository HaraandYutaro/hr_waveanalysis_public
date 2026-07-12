"""Surface-wave inversion engines (canonical package).

All engines exposed here are wave-agnostic by design: they receive a
``RayleighForwardSolver`` / ``SurfaceWaveForwardSolver`` (same class) and a
``DispersionMisfit`` at construction time, and they do not reference any
wave-type-specific behaviour internally.

- ``DampedLeastSquaresEngine``  : Marquardt-damped Gauss-Newton (LSQ family)
- ``LCIEngine``                 : Laterally Constrained Inversion (LSQ family)
- ``NichingPSOEngine``          : Niching PSO + modified BSAS clustering
                                  (Zhang et al. 2023; DOI:10.1093/gji/ggac380)
"""

from src.inversion.surface_wave.engine.base import InversionEngine
from src.inversion.surface_wave.engine.damped_lsq import DampedLeastSquaresEngine
from src.inversion.surface_wave.engine.lci import LCIEngine
from src.inversion.surface_wave.engine.pso import NichingPSOEngine

__all__ = [
    "InversionEngine",
    "DampedLeastSquaresEngine",
    "LCIEngine",
    "NichingPSOEngine",
]
