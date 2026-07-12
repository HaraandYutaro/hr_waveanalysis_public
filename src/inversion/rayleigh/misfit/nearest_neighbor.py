"""NearestNeighborMisfit — Wilken & Rabbel 2012 / PDF1 Eq. (8) misfit.

PDF1 (Zhang, K. et al. (2023). "A niching particle swarm optimization strategy
for the multimodal inversion of surface waves." Geophys. J. Int., 232,
1140-1158. DOI:10.1093/gji/ggac380.) §2.2 Eq. (8) で、Wilken & Rabbel (2012)
を引用して導入された 'nearest-neighbour' misfit::

    F_N(m) = (1/N) * sum_i  | v_i^obs - v_i^nearest(m) | / sigma_i

ここで ``v_i^nearest`` は、現モデルの全モード分散曲線上で観測点
(f_i, v_i^obs) に最も近い velocity 値。モード番号付けを必要としない点で
determinant misfit と同じ利点を持ち、計算には root-finding (forward 1 回 +
モード列挙) を要するが、その結果が比較可能な物理量 (m/s) として
解釈しやすいため、PSO 結果の cluster ranking で広く使われる (PDF1 §2.3 末尾)。

本 misfit は LSQ engine とは整合せず、``residual()`` / ``sqrt_weights()``
は ``None`` を返す。

References
----------
- Wilken, D. & Rabbel, W. (2012). "On the application of particle swarm
  optimization strategies on Scholte-wave inversion." Geophys. J. Int., 190,
  580-594. (Nearest-neighbour misfit の原典)
- O'Neill, A. (2003). PhD thesis, Univ. of Western Australia. (sigma_i の
  推定式; PDF1 Eq. 9)
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380.
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import LayeredEarthModel, PickedDispersionCurve


class NearestNeighborMisfit(DispersionMisfit):
    """Nearest-neighbour velocity-domain misfit (PDF1 Eq. 8)。

    Parameters
    ----------
    predict_modes_func : callable
        ``predict_modes_func(model, freqs, max_mode) -> list[(mode_index, ndarray)]``
        の形式。各モードについて存在する位相速度配列を返す。
        高次モードが存在しない (disba が DispersionError) 場合は当該モードを
        list から除外して返すことで、本 misfit は「存在するモードのみ」で
        nearest neighbor を計算する。
    max_mode : int
        探索する高次モード数の上限。既定 4。
    """

    def __init__(
        self,
        predict_modes_func: Callable[
            [LayeredEarthModel, np.ndarray, int],
            Sequence[Tuple[int, np.ndarray]],
        ],
        *,
        max_mode: int = 4,
    ) -> None:
        if not callable(predict_modes_func):
            raise TypeError("predict_modes_func must be callable")
        if int(max_mode) < 0:
            raise ValueError("max_mode must be >= 0")
        self._predict_modes_func = predict_modes_func
        self._max_mode = int(max_mode)

    def value(
        self,
        observed: PickedDispersionCurve,
        predicted: Any,
    ) -> float:
        """PDF1 Eq. (8) の F_N(m) を返す。

        Parameters
        ----------
        observed : PickedDispersionCurve
        predicted : LayeredEarthModel
            現在のモデル。``predict_modes_func(model, observed.f, max_mode)`` を
            呼んで利用可能なモード分散曲線を集める。
        """
        if not isinstance(predicted, LayeredEarthModel):
            raise TypeError(
                "NearestNeighborMisfit.value expects a LayeredEarthModel as `predicted`; "
                f"got {type(predicted).__name__}."
            )
        fs = np.asarray(observed.f, dtype=float)
        cs_obs = np.asarray(observed.c, dtype=float)
        if observed.c_std is None:
            sigma = np.ones_like(cs_obs)
        else:
            sigma = np.asarray(observed.c_std, dtype=float)
            sigma = np.where(sigma > 0.0, sigma, 1.0)

        mode_curves: List[Tuple[int, np.ndarray]] = list(
            self._predict_modes_func(predicted, fs, self._max_mode)
        )
        if len(mode_curves) == 0:
            return float("inf")

        # shape=(n_modes, m): 各 (mode, freq) の予測位相速度
        predicted_matrix = np.stack(
            [np.asarray(c, dtype=float) for _, c in mode_curves], axis=0
        )
        # 周波数ごとに観測との差の絶対値最小を取る
        diff = np.abs(predicted_matrix - cs_obs[None, :])  # (n_modes, m)
        nearest = np.min(diff, axis=0)  # (m,)

        weighted = nearest / sigma
        return float(np.mean(weighted))

    def residual(self, observed: Any, predicted: Any) -> Optional[np.ndarray]:
        return None

    def sqrt_weights(self, observed: Any) -> Optional[np.ndarray]:
        return None
