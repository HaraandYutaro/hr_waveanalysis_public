"""Misfit ABC for Rayleigh-wave dispersion inversion.

Step ?-1a 契約 (engine との整合):
  - residual(observed, predicted) は **sqrt(W)-weighted** な残差 r_w を返す。
    すなわち L2 系 misfit では r_w = sqrt(W) * (observed.c - predicted)。
    LSQ 系 engine は ||r_w||^2 / 2 を最小化対象として消費する。
  - sqrt_weights(observed) は同じ sqrt(W) を engine に提供する。
    engine はこれを raw Jacobian J に同様に掛けて weighted Jacobian
    J_w = sqrt(W)[:, None] * J を構成する。
  - residual と sqrt_weights は **同じ重み W** に従わなければならない。
    この整合が崩れると Gauss-Newton の正規方程式が偏った推定を返す。
  - Wasserstein など LSQ 非対応 misfit は residual / sqrt_weights のいずれも
    None を返してよい。その場合 LSQ 系 engine は明示的にエラーを出す。

observed と predicted は ABC レベルでは opaque に扱う:
  - L2 系   : observed = PickedDispersionCurve, predicted = ndarray (m,)
  - 将来    : observed = energy map ndarray, predicted = synthetic spectrum ndarray
  ペアリングは concrete forward + concrete misfit の組み合わせで保証する。
"""

from __future__ import annotations

import abc
from typing import Any, Optional

import numpy as np


class DispersionMisfit(abc.ABC):
    """Misfit ABC。observed / predicted は concrete pair が解釈する opaque 型。"""

    @abc.abstractmethod
    def value(self, observed: Any, predicted: Any) -> float:
        """スカラー misfit 値。LSQ 系では 0.5 * ||r_w||^2 を返す。"""

    def residual(self, observed: Any, predicted: Any) -> Optional[np.ndarray]:
        """Weighted residual r_w = sqrt(W) * (observed - predicted)。

        LSQ 系 engine 用。LSQ 非対応 misfit は None を返してよい。
        """
        return None

    def sqrt_weights(self, observed: Any) -> Optional[np.ndarray]:
        """Weighted Jacobian 構成のために engine に渡す sqrt(W)。

        LSQ 系 engine はこれを raw Jacobian に left-multiply する。
        residual() の重みと **必ず一致** させること。
        LSQ 非対応 misfit は None を返してよい。
        """
        return None
