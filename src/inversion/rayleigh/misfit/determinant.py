"""DeterminantMisfit — root-finding-free misfit for niching PSO.

PDF1 (Zhang, K. et al., 2023. "A niching particle swarm optimization strategy
for the multimodal inversion of surface waves." Geophys. J. Int., 232,
1140-1158. DOI:10.1093/gji/ggac380.) §2.2 Eq. (10) で導入された determinant
misfit を実装する::

    F_D(m) = sum_i  |D(f_i^obs, v_i^obs, m)| / sigma_i

ここで D は secular (characteristic) 関数。本 misfit は

  - 観測点 (f_i, v_i^obs) における secular 関数の絶対値の総和を取るのみで、
    内部で root-finding を行わない。これが PSO 内 loop での速度上の利点。
  - モード番号付け (modal pairing) が不要。観測曲線が基本モード単独でも、
    多モード混在でも同じ式で扱える。

本 misfit は LSQ engine とは整合せず、``residual()`` / ``sqrt_weights()``
は ``None`` を返す。Gauss-Newton 系 engine は明示的にエラーを返すこと
(``WeightedL2Misfit`` と対照的)。

Wave 種別への適用:
  - Love wave : ``src.inversion.love.secular.love_secular_value`` を inject。
  - Rayleigh : 自前 secular 関数の実装が必要 (本リポでは未実装。disba は
    secular 関数値を公開しないため、Rayleigh で本 misfit を使うには別実装が要る)。

References
----------
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380.
- Maraschini, M. et al. (2010). "A new misfit function for multimodal inversion
  of surface waves." Geophysics, 75, G31-G43. (Determinant misfit の原典)
"""

from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import LayeredEarthModel, PickedDispersionCurve


class DeterminantMisfit(DispersionMisfit):
    """Secular 関数値ベースの misfit (PDF1 Eq. 10)。

    Parameters
    ----------
    secular_func : callable
        ``secular_func(c: float, f: float, model: LayeredEarthModel) -> float``
        の形式で secular 関数値を返す callable を inject する。
        ``np.nan`` を返した点は misfit 合計から除外される (leaky regime 等)。
    normalize : str
        ``"raw"``  : sum_i |D_i| / sigma_i (PDF1 原典)
        ``"log"`` : sum_i log(1 + |D_i|) / sigma_i  (数値レンジ圧縮; PSO の
                     探索安定性のため。secular の絶対値は層厚 / 周波数で
                     非線形に scale するため log の方が PSO で扱いやすい)
        既定 ``"log"``。
    """

    def __init__(
        self,
        secular_func: Callable[[float, float, LayeredEarthModel], float],
        *,
        normalize: str = "log",
    ) -> None:
        if not callable(secular_func):
            raise TypeError("secular_func must be callable")
        if normalize not in ("raw", "log"):
            raise ValueError(f"normalize must be 'raw' or 'log'; got {normalize!r}")
        self._secular_func = secular_func
        self._normalize = normalize

    def value(
        self,
        observed: PickedDispersionCurve,
        predicted: Any,
    ) -> float:
        """PDF1 Eq. (10) の F_D(m) を返す。

        Parameters
        ----------
        observed : PickedDispersionCurve
            観測 (f, c, c_std) の集合。
        predicted : LayeredEarthModel
            現在のモデル。本 misfit では ``forward(model, freqs)`` の結果
            (位相速度配列) ではなく、**モデル本体**を受け取る点に注意。
            これは secular 関数が (c, f, model) を要求するためで、
            PSO engine 側は ``misfit.value(observed, model)`` の形で呼ぶ。
        """
        if not isinstance(predicted, LayeredEarthModel):
            raise TypeError(
                "DeterminantMisfit.value expects a LayeredEarthModel as `predicted`; "
                f"got {type(predicted).__name__}. "
                "PSO engines must pass the current model, not a velocity array."
            )
        fs = np.asarray(observed.f, dtype=float)
        cs = np.asarray(observed.c, dtype=float)
        if observed.c_std is None:
            sigma = np.ones_like(cs)
        else:
            sigma = np.asarray(observed.c_std, dtype=float)
            sigma = np.where(sigma > 0.0, sigma, 1.0)

        total = 0.0
        n_used = 0
        for i in range(cs.size):
            d = self._secular_func(float(cs[i]), float(fs[i]), predicted)
            if not np.isfinite(d):
                continue
            ad = abs(d)
            if self._normalize == "log":
                ad = float(np.log1p(ad))
            total += ad / float(sigma[i])
            n_used += 1

        if n_used == 0:
            # すべての観測点が leaky / failure。PSO に高いペナルティを返す。
            return float("inf")
        return float(total)

    # residual / sqrt_weights は LSQ engine 用。determinant は LSQ 非対応なので None。
    def residual(self, observed: Any, predicted: Any) -> Optional[np.ndarray]:
        return None

    def sqrt_weights(self, observed: Any) -> Optional[np.ndarray]:
        return None
