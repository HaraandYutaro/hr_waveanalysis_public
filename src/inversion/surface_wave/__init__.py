"""Wave-agnostic surface wave inversion building blocks (canonical package).

This package is the **canonical home** for surface-wave inversion abstractions
that do not depend on a specific wave type (Rayleigh vs Love). At the time of
introduction (Slice R1), the underlying classes still physically live under
``src.inversion.rayleigh.*``; this package re-exports them so that the
canonical import path is available immediately without breaking any existing
import. Future slices will physically migrate the files into this package and
leave ``src.inversion.rayleigh.*`` as deprecation shims.

Recommended import paths
------------------------
Wave-agnostic abstractions and algorithms (prefer this package going forward)::

    from src.inversion.surface_wave.model import (
        LayeredEarthModel,
        PickedDispersionCurve,
        SurfaceWaveInversionResult,
        Pseudo2DVsSection,
        LCIProfileResult,
    )
    from src.inversion.surface_wave.forward.base import SurfaceWaveForwardSolver
    from src.inversion.surface_wave.misfit.weighted_l2 import WeightedL2Misfit
    from src.inversion.surface_wave.misfit.determinant import DeterminantMisfit
    from src.inversion.surface_wave.misfit.nearest_neighbor import NearestNeighborMisfit
    from src.inversion.surface_wave.engine.damped_lsq import DampedLeastSquaresEngine
    from src.inversion.surface_wave.engine.lci import LCIEngine
    from src.inversion.surface_wave.engine.pso import NichingPSOEngine
    from src.inversion.surface_wave.picking_qc import (
        DispersionPickQCResult,
        quality_control_dispersion_pick,
    )
    from src.inversion.surface_wave.section import Pseudo2DSectionBuilder

Wave-specific implementations (continue to import from the wave-specific
sub-packages)::

    Rayleigh:
        from src.inversion.rayleigh.init_model import build_default_init_model
        from src.inversion.rayleigh.forward.thomson_haskell import ThomsonHaskellSolver

    Love:
        from src.inversion.love.init_model import build_default_love_init_model
        from src.inversion.love.forward.thomson_haskell_love import ThomsonHaskellLoveSolver
        from src.inversion.love.secular import love_secular_value

Strategy-pattern usage examples
-------------------------------
DeterminantMisfit and NearestNeighborMisfit are wave-agnostic by design and
take a ``secular_func`` / ``predict_modes_func`` callable that the wave-specific
package supplies::

    from src.inversion.surface_wave.misfit.determinant import DeterminantMisfit
    from src.inversion.love.secular import love_secular_value
    misfit = DeterminantMisfit(love_secular_value, normalize="log")

    from src.inversion.surface_wave.misfit.nearest_neighbor import NearestNeighborMisfit
    from src.inversion.love.forward.thomson_haskell_love import ThomsonHaskellLoveSolver
    fwd = ThomsonHaskellLoveSolver()
    nn = NearestNeighborMisfit(predict_modes_func=fwd.predict_modes, max_mode=4)

References
----------
- Zhang, K. et al. (2023). "A niching particle swarm optimization strategy for
  the multimodal inversion of surface waves." Geophys. J. Int., 232, 1140-1158.
  DOI:10.1093/gji/ggac380.
- Hou, X. et al. (2025). Sci. Rep., 15, 21595. DOI:10.1038/s41598-025-04954-w.
"""

# Top-level package marker. Submodules are imported explicitly by users
# (no eager submodule imports here to keep import cost minimal).
__all__: list = []
