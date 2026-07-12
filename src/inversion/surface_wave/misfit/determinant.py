"""DeterminantMisfit (canonical re-export, strategy-pattern misfit).

PDF1 Eq. 10 of Zhang, K. et al. (2023). "A niching particle swarm optimization
strategy for the multimodal inversion of surface waves." Geophys. J. Int.,
232, 1140-1158. DOI:10.1093/gji/ggac380.

This misfit is wave-agnostic via strategy injection: the wave-specific
secular function is supplied at construction time::

    # Love wave usage
    from src.inversion.surface_wave.misfit.determinant import DeterminantMisfit
    from src.inversion.love.secular import love_secular_value
    misfit = DeterminantMisfit(love_secular_value, normalize="log")

    # Rayleigh wave usage (when a Rayleigh secular function becomes available)
    # misfit = DeterminantMisfit(rayleigh_secular_value, normalize="log")

Currently re-exports from ``src.inversion.rayleigh.misfit.determinant``.
Slice R3 will physically move the implementation here.
"""

from src.inversion.rayleigh.misfit.determinant import DeterminantMisfit

__all__ = ["DeterminantMisfit"]
