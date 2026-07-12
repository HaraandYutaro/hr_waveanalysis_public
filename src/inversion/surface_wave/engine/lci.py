"""LCIEngine (canonical re-export).

Wave-agnostic Laterally Constrained Inversion engine.
Pairs with LSQ-compatible misfits (e.g. ``WeightedL2Misfit``).

Reference: Auken, E. & Christiansen, A. V. (2004). Geophysics, 69, 752-761.
DOI:10.1190/1.1759461. Boiero, D. & Socco, L. V. (2010). Geophysics, 75, B49-B59.

Currently re-exports from ``src.inversion.rayleigh.engine.lci``.
Slice R3 will physically move the implementation here.
"""

from src.inversion.rayleigh.engine.lci import LCIEngine

__all__ = ["LCIEngine"]
