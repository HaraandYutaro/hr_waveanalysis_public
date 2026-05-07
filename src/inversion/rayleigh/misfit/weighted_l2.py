"""WeightedL2Misfit — sqrt(W)-weighted L2 misfit for picked dispersion curves.

Engine 契約 (DampedLeastSquaresEngine と整合):
  W = diag(1 / c_std_j^2)        (c_std が None なら W = I)
  sqrt_weights(observed) = 1 / c_std        (None なら ones)
  residual(observed, predicted) = sqrt(W) * (observed.c - predicted)
                                = (observed.c - predicted) / c_std
  value(observed, predicted)    = 0.5 * sum( residual^2 )

engine 側はこの sqrt(W) を **そのまま** raw Jacobian に left-multiply して
weighted Jacobian J_w を構成し、normal equation
  (J_w^T J_w + lambda I) Δm = J_w^T r_w
を解く。これは
  (J^T W J + lambda I) Δm = J^T W (observed - predicted)
と等価。residual と sqrt_weights が同じ W に従うことが整合の必要条件。
"""

from __future__ import annotations

import numpy as np

from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import PickedDispersionCurve


class WeightedL2Misfit(DispersionMisfit):
    """observed.c_std を 1-σ とみなした sqrt(W)-weighted L2 misfit。"""

    def sqrt_weights(self, observed: PickedDispersionCurve) -> np.ndarray:
        if observed.c_std is None:
            return np.ones_like(observed.c)
        return 1.0 / observed.c_std

    def residual(
        self,
        observed: PickedDispersionCurve,
        predicted: np.ndarray,
    ) -> np.ndarray:
        predicted = np.asarray(predicted, dtype=float)
        if predicted.shape != observed.c.shape:
            raise ValueError(
                f"predicted shape {predicted.shape} != observed.c shape {observed.c.shape}"
            )
        sw = self.sqrt_weights(observed)
        return sw * (observed.c - predicted)

    def value(
        self,
        observed: PickedDispersionCurve,
        predicted: np.ndarray,
    ) -> float:
        r_w = self.residual(observed, predicted)
        return 0.5 * float(np.dot(r_w, r_w))
