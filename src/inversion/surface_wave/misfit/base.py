"""DispersionMisfit ABC (canonical re-export).

Wave-agnostic misfit ABC. Currently re-exports from
``src.inversion.rayleigh.misfit.base``. Slice R3 will physically move this
class to the surface_wave package.
"""

from src.inversion.rayleigh.misfit.base import DispersionMisfit

__all__ = ["DispersionMisfit"]
