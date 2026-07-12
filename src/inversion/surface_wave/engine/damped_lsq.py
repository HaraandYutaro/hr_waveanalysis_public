"""DampedLeastSquaresEngine (canonical re-export).

Wave-agnostic Marquardt-damped Gauss-Newton engine for Vs-only updates.
Pairs with LSQ-compatible misfits (e.g. ``WeightedL2Misfit``).

Currently re-exports from ``src.inversion.rayleigh.engine.damped_lsq``.
Slice R3 will physically move the implementation here.
"""

from src.inversion.rayleigh.engine.damped_lsq import DampedLeastSquaresEngine

__all__ = ["DampedLeastSquaresEngine"]
