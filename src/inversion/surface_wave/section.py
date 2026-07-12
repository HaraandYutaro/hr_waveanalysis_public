"""Pseudo-2D section builder for surface-wave inversion (canonical re-export).

Per-CMP 1D inversion results -> depth-grid mapping. Wave-agnostic.
Used by both Rayleigh and Love inversion pipelines.

Currently re-exports from ``src.inversion.rayleigh.section``. Slice R3 will
physically move the implementation to this package.
"""

from src.inversion.rayleigh.section import Pseudo2DSectionBuilder

__all__ = ["Pseudo2DSectionBuilder"]
