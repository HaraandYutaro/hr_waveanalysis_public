"""InversionEngine ABC (canonical re-export).

Wave-agnostic engine ABC. Currently re-exports from
``src.inversion.rayleigh.engine.base``. Slice R3 will physically move
the implementation here.
"""

from src.inversion.rayleigh.engine.base import InversionEngine

__all__ = ["InversionEngine"]
