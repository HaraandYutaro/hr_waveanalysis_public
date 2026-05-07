"""Forward solver ABC for Rayleigh-wave dispersion.

Step ?-1a 契約:
  - forward(model, freqs, mode) は位相速度 c(f) を返す。
  - jacobian(model, freqs, mode) は ∂c / ∂Vs_j を返す (Vs のみ)。
    base 実装は中心差分。Step ?-1b 以降の concrete で解析的 Jacobian に置換可。
  - 層厚 h, Vp, ρ は forward 入力としては受け取るが、Jacobian は Vs に対してのみ。
"""

from __future__ import annotations

import abc

import numpy as np

from src.inversion.rayleigh.model import LayeredEarthModel


class RayleighForwardSolver(abc.ABC):
    """Rayleigh 波分散曲線の前進解 ABC。"""

    @abc.abstractmethod
    def forward(
        self,
        model: LayeredEarthModel,
        freqs: np.ndarray,
        mode: int = 0,
    ) -> np.ndarray:
        """各 freqs における位相速度 c(f) を返す。

        Parameters
        ----------
        model : LayeredEarthModel
        freqs : ndarray, shape=(m,)
            周波数 [Hz]。
        mode : int
            モード番号 (0 = 基本モード)。Step ?-1a の Toy では 0 のみサポート。

        Returns
        -------
        ndarray, shape=(m,)
            予測位相速度 [m/s]。
        """

    def jacobian(
        self,
        model: LayeredEarthModel,
        freqs: np.ndarray,
        mode: int = 0,
        eps_rel: float = 1e-3,
        eps_floor: float = 1e-2,
    ) -> np.ndarray:
        """∂c / ∂Vs_j を中心差分で計算する汎用実装。

        Step ?-1a 契約:
          - 戻り値 shape=(m, n_layers)。
          - **重みなし** の純粋な感度行列。
            engine 側で sqrt(W) を後から掛けて weighted Jacobian を構成する。
          - 層厚 h, Vp, ρ に関する偏微分は提供しない (Vs only)。

        Parameters
        ----------
        eps_rel : float
            摂動量の Vs に対する相対比。実摂動は max(eps_rel * |vs_j|, eps_floor)。
        eps_floor : float
            vs_j がゼロに近いときの最小摂動量 [m/s]。
        """
        freqs = np.asarray(freqs, dtype=float)
        n = model.n_layers
        m = freqs.shape[0]
        jac = np.zeros((m, n), dtype=float)

        for j in range(n):
            eps = max(eps_rel * abs(float(model.vs[j])), eps_floor)
            vs_plus = model.vs.copy()
            vs_minus = model.vs.copy()
            vs_plus[j] += eps
            vs_minus[j] -= eps
            c_plus = self.forward(model.copy_with_vs(vs_plus), freqs, mode=mode)
            c_minus = self.forward(model.copy_with_vs(vs_minus), freqs, mode=mode)
            jac[:, j] = (c_plus - c_minus) / (2.0 * eps)

        return jac
