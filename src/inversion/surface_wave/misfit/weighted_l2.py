"""WeightedL2Misfit (canonical re-export).

Wave-agnostic sqrt(W)-weighted L2 misfit. Used by LSQ-family engines
(DampedLeastSquaresEngine, LCIEngine).

Currently re-exports from ``src.inversion.rayleigh.misfit.weighted_l2``.
Slice R3 will physically move the implementation here.
"""

from src.inversion.rayleigh.misfit.weighted_l2 import WeightedL2Misfit

__all__ = ["WeightedL2Misfit"]
