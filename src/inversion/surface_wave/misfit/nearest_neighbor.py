"""NearestNeighborMisfit (canonical re-export, strategy-pattern misfit).

PDF1 Eq. 8 of Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380.
Original: Wilken, D. & Rabbel, W. (2012). Geophys. J. Int., 190, 580-594.

Wave-agnostic via strategy injection: a wave-specific ``predict_modes_func``
callable supplies the per-mode forward predictions used to find the nearest
neighbour::

    # Love wave usage
    from src.inversion.surface_wave.misfit.nearest_neighbor import NearestNeighborMisfit
    from src.inversion.love.forward.thomson_haskell_love import ThomsonHaskellLoveSolver
    fwd = ThomsonHaskellLoveSolver()
    nn = NearestNeighborMisfit(predict_modes_func=fwd.predict_modes, max_mode=4)

Currently re-exports from ``src.inversion.rayleigh.misfit.nearest_neighbor``.
Slice R3 will physically move the implementation here.
"""

from src.inversion.rayleigh.misfit.nearest_neighbor import NearestNeighborMisfit

__all__ = ["NearestNeighborMisfit"]
