"""Data containers for surface-wave inversion (canonical re-export).

This module re-exports the wave-agnostic data containers that physically still
live under ``src.inversion.rayleigh.model``. Both import paths are guaranteed
to resolve to **the same class objects** (``is`` identity), so existing code
using the legacy path continues to work unchanged.

The Rayleigh-named ``RayleighInversionResult`` is additionally aliased as
``SurfaceWaveInversionResult`` to give wave-agnostic users a non-misleading
class name. Both names refer to the same dataclass.

Future migration: Slice R3 will physically move ``model.py`` to this package
and turn ``src.inversion.rayleigh.model`` into a deprecation shim.

References
----------
- See ``src/inversion/surface_wave/__init__.py`` for citations.
"""

from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    LCIProfileResult,
    PickedDispersionCurve,
    Pseudo2DVsSection,
    RayleighInversionResult,
)

# Wave-agnostic alias for the per-CMP 1D inversion result container.
# The underlying dataclass is identical (literal identity).
SurfaceWaveInversionResult = RayleighInversionResult

__all__ = [
    "LayeredEarthModel",
    "PickedDispersionCurve",
    "RayleighInversionResult",        # legacy name preserved
    "SurfaceWaveInversionResult",     # canonical wave-agnostic alias
    "Pseudo2DVsSection",
    "LCIProfileResult",
]
