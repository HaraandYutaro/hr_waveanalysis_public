"""NichingPSOEngine (canonical re-export).

Wave-agnostic distance-based locally informed PSO + modified BSAS clustering
+ nearest-neighbour ranking engine.

Reference: Zhang, K. et al. (2023). "A niching particle swarm optimization
strategy for the multimodal inversion of surface waves." Geophys. J. Int.,
232, 1140-1158. DOI:10.1093/gji/ggac380.

Pairs with PSO-compatible misfits (``DeterminantMisfit``,
``NearestNeighborMisfit``). Wave-specific behaviour enters only through the
``forward`` and ``misfit`` (and optional ``ranking_misfit``) arguments at
construction time; the engine itself is wave-type-agnostic.

Currently re-exports from ``src.inversion.rayleigh.engine.pso``.
Slice R3 will physically move the implementation here.
"""

from src.inversion.rayleigh.engine.pso import NichingPSOEngine

__all__ = ["NichingPSOEngine"]
