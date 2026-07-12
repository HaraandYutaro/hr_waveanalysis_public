"""Picking QC for surface-wave dispersion images (canonical re-export).

Wave-agnostic QC of an f-v dispersion energy image into a PickedDispersionCurve.
Applies to both Rayleigh and Love picking workflows.

Currently re-exports from ``src.inversion.rayleigh.picking_qc``. Slice R3 will
physically move the implementation to this package.

References
----------
- See ``src/inversion/surface_wave/__init__.py`` for citations.
- Future Hessian ridge picker (Hou et al. 2025; DOI:10.1038/s41598-025-04954-w)
  is planned as ``surface_wave/picking_qc_hessian.py`` in a follow-up slice.
"""

from src.inversion.rayleigh.picking_qc import (
    DispersionPickQCResult,
    quality_control_dispersion_pick,
)

__all__ = [
    "DispersionPickQCResult",
    "quality_control_dispersion_pick",
]
