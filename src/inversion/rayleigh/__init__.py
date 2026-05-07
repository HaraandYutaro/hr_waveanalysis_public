"""Rayleigh-wave Vs inversion (Step ?-1a skeleton).

Public surface of this subpackage:

  data classes (model.py)
    - LayeredEarthModel
    - PickedDispersionCurve
    - RayleighInversionResult
    - Pseudo2DVsSection

  forward (forward/)
    - RayleighForwardSolver        : ABC
    - ToyForwardSolver             : engine/misfit verification only

  misfit (misfit/)
    - DispersionMisfit             : ABC
    - WeightedL2Misfit             : LSQ-ready, sqrt(W)-weighted residual

  engine (engine/)
    - InversionEngine              : ABC
    - DampedLeastSquaresEngine     : Marquardt-damped Gauss-Newton

  section (section.py)
    - Pseudo2DSectionBuilder       : per-CMP results -> pseudo-2D section

  picking QC (picking_qc.py)
    - DispersionPickQCResult       : QC result dataclass
    - quality_control_dispersion_pick : dispersion image -> QC'd PickedDispersionCurve

  init model (init_model.py)
    - build_default_init_model     : dispersion curve -> default initial model
"""

from src.inversion.rayleigh.init_model import build_default_init_model
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    PickedDispersionCurve,
    Pseudo2DVsSection,
    RayleighInversionResult,
)
from src.inversion.rayleigh.picking_qc import (
    DispersionPickQCResult,
    quality_control_dispersion_pick,
)
from src.inversion.rayleigh.section import Pseudo2DSectionBuilder

__all__ = [
    "LayeredEarthModel",
    "PickedDispersionCurve",
    "RayleighInversionResult",
    "Pseudo2DVsSection",
    "Pseudo2DSectionBuilder",
    "DispersionPickQCResult",
    "quality_control_dispersion_pick",
    "build_default_init_model",
]