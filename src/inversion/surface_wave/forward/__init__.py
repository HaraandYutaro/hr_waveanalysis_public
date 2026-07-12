"""Surface-wave forward solvers (canonical package).

Re-exports the wave-agnostic forward solver ABC. Wave-specific concrete
implementations remain in their wave packages:

  - Rayleigh:  src.inversion.rayleigh.forward.thomson_haskell.ThomsonHaskellSolver
  - Love:      src.inversion.love.forward.thomson_haskell_love.ThomsonHaskellLoveSolver
"""

from src.inversion.surface_wave.forward.base import (
    RayleighForwardSolver,
    SurfaceWaveForwardSolver,
)

__all__ = [
    "SurfaceWaveForwardSolver",     # canonical wave-agnostic ABC name
    "RayleighForwardSolver",        # legacy alias preserved
]
