"""Love-wave Vs inversion subpackage.

Public surface
--------------
forward (forward/)
  - LoveForwardSolver         : ABC (RayleighForwardSolver を継承して契約共有)
  - ThomsonHaskellLoveSolver  : disba ``PhaseDispersion(..., wave="love")`` ラッパー

secular (secular.py)
  - love_secular_value        : 単一 (c, f, model) 点の Love 波 secular 関数値

init model (init_model.py)
  - build_default_love_init_model : picked dispersion -> 初期モデル

データクラス (LayeredEarthModel, PickedDispersionCurve, RayleighInversionResult,
Pseudo2DVsSection, LCIProfileResult) は ``src.inversion.rayleigh.model`` を
そのまま再 export して共用する。Love と Rayleigh は同一の層構造表現を取るため
モデルクラスを二重定義しない (DRY 原則)。

References
----------
- Zhang, K. et al. (2023). "A niching particle swarm optimization strategy for
  the multimodal inversion of surface waves." Geophys. J. Int., 232, 1140-1158.
  DOI:10.1093/gji/ggac380.
- disba (Luu, K.) — Rayleigh / Love phase & group dispersion. https://github.com/keurfonluu/disba
"""

from src.inversion.love.forward.thomson_haskell_love import ThomsonHaskellLoveSolver
from src.inversion.love.init_model import build_default_love_init_model
from src.inversion.love.secular import love_secular_value
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    LCIProfileResult,
    PickedDispersionCurve,
    Pseudo2DVsSection,
    RayleighInversionResult,
)

__all__ = [
    "ThomsonHaskellLoveSolver",
    "build_default_love_init_model",
    "love_secular_value",
    "LayeredEarthModel",
    "PickedDispersionCurve",
    "RayleighInversionResult",
    "Pseudo2DVsSection",
    "LCIProfileResult",
]
