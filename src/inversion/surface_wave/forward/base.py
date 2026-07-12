"""Surface-wave forward solver ABC (canonical name, currently aliased).

The ABC contract (``forward(model, freqs, mode) -> ndarray``;
``jacobian(model, freqs, mode) -> ndarray``) is wave-agnostic and applies
identically to Rayleigh and Love wave forward solvers. The class is
historically named ``RayleighForwardSolver`` (legacy from Step ?-1a) and
physically lives in ``src.inversion.rayleigh.forward.base``.

This module introduces the canonical wave-agnostic name
``SurfaceWaveForwardSolver`` as a literal alias of the existing class, so:

  - ``isinstance(love_solver, SurfaceWaveForwardSolver)`` works
  - ``issubclass(LoveSolver, SurfaceWaveForwardSolver)`` works
  - both names refer to the **same Python class object**

Future migration (Slice R3): the underlying class definition will be
physically moved to this module, and ``src.inversion.rayleigh.forward.base``
will become a deprecation shim that re-exports
``RayleighForwardSolver = SurfaceWaveForwardSolver``.
"""

from src.inversion.rayleigh.forward.base import (
    RayleighForwardSolver as SurfaceWaveForwardSolver,
)

# Legacy alias kept for users who import via the canonical path but expect
# the historical class name to be available here.
RayleighForwardSolver = SurfaceWaveForwardSolver

__all__ = [
    "SurfaceWaveForwardSolver",     # canonical name
    "RayleighForwardSolver",        # legacy alias
]
