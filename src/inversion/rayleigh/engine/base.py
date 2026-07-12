"""Inversion engine ABC

Step ?-1a 契約:
  - run_single(observed, init_model) は 1 CMP 分の逆解析を実行し
    RayleighInversionResult を返す (必須)。
  - run_profile(observations, init_models, cmp_x) は複数 CMP 分の逆解析を
    返す。既定実装は run_single の独立反復 (Luo et al. 2008 型 pseudo-2D)。
    [出典注意: Luo et al. 2008 は library/refs / Research.md に未収録のため
     出典未確認。pseudo-2D 構築の機能自体は動作する。Plan.md P2-G 参照。]
    LCI (Auken 系) は本メソッドを override して横方向拘束を導入する。
  - 更新対象は Vs のみ (Step ?-1a)。h は固定。
"""

from __future__ import annotations

import abc
from typing import List, Sequence

import numpy as np

from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    PickedDispersionCurve,
    RayleighInversionResult,
)


class InversionEngine(abc.ABC):
    """逆解析エンジン ABC。"""

    def __init__(
        self,
        forward: RayleighForwardSolver,
        misfit: DispersionMisfit,
        **opts,
    ) -> None:
        self.forward = forward
        self.misfit = misfit
        self.opts = dict(opts)

    @abc.abstractmethod
    def run_single(
        self,
        observed: PickedDispersionCurve,
        init_model: LayeredEarthModel,
    ) -> RayleighInversionResult:
        """1 CMP 分の逆解析を実行する。"""

    def run_profile(
        self,
        observations: Sequence[PickedDispersionCurve],
        init_models: Sequence[LayeredEarthModel],
        cmp_x: np.ndarray,
    ) -> List[RayleighInversionResult]:
        """複数 CMP の逆解析。既定は独立反復 (cmp_x は LCI 派生で参照される)。"""
        if len(observations) != len(init_models):
            raise ValueError("observations と init_models の長さが一致しません")
        results: List[RayleighInversionResult] = []
        for obs, m0 in zip(observations, init_models):
            results.append(self.run_single(obs, m0))
        return results
