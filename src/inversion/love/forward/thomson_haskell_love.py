"""ThomsonHaskellLoveSolver — disba (Thomson-Haskell) wrapper for Love waves.

disba は Rayleigh / Love wave dispersion の確立実装で、内部は Thomson-Haskell
+ reduced delta-matrix によって数値的に安定。本ファイルは ``PhaseDispersion``
を ``wave="love"`` で薄くラップし、本リポの SI 単位 (m, m/s, kg/m^3, Hz) と
disba の単位 (km, km/s, g/cm^3, s) の橋渡しを行う。

Engine 契約との整合 (src.inversion.rayleigh.forward.base.RayleighForwardSolver):
  - ``forward(model, freqs, mode)`` は 重みなし c(f) を返す。
  - ``jacobian`` は base 実装の中心差分をそのまま継承
    (disba は解析的 Jacobian を公開していない)。
  - 戻り値は engine 側で重みを掛けることを前提に、本クラスでは **一切の重み付与なし**。

Love wave の物理特性:
  - Love wave は P 波速度に依存せず、Vs / ρ / h のみで決まる SH モード。
  - 入力 model の vp は disba の API 要件のため渡すが、Love 分散には寄与しない。
  - Halfspace の Vs を超える c では proper Love mode は存在せず、
    disba は対応する周期で部分結果 (NaN / 脱落) を返すことがある。

References
----------
- Haskell, N. A. (1953). "The dispersion of surface waves on multilayered media."
  Bull. Seismol. Soc. Am., 43, 17-34.
- disba (Luu, K.). https://github.com/keurfonluu/disba
- Zhang et al. (2023). DOI:10.1093/gji/ggac380. (Love secular function discussion)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.model import LayeredEarthModel

_M_TO_KM = 1.0 / 1000.0
_KGM3_TO_GCM3 = 1.0 / 1000.0


class ThomsonHaskellLoveSolver(RayleighForwardSolver):
    """disba PhaseDispersion による Love wave forward solver。

    ABC ``RayleighForwardSolver`` を継承するのは、契約 (``forward``,
    ``jacobian``) が wave 種別に依存しないため。ABC の名称は将来 slice で
    ``SurfaceWaveForwardSolver`` に汎用化される予定だが、本 slice では
    既存 public API を保つために継承構造をそのまま流用する。

    Parameters
    ----------
    halfspace_thickness_km : float
        ``LayeredEarthModel.h[-1]`` が +inf または非有限の場合に
        disba へ渡す物理的な厚さ [km]。浅部探査スケール (数 100 m) では
        既定の 1000 km で十分に半無限。
    dc_km_s : float or None
        disba の root-search ステップ [km/s] (``PhaseDispersion(dc=...)``)。
        None の場合、disba 既定値 (0.005 km/s) に従う。
    algorithm : str
        disba ``PhaseDispersion(algorithm=...)`` (既定 ``"dunkin"``)。
        Thomson-Haskell + reduced delta-matrix 系の root-finder バリアント。
    """

    def __init__(
        self,
        *,
        halfspace_thickness_km: float = 1000.0,
        dc_km_s: Optional[float] = None,
        algorithm: str = "dunkin",
    ) -> None:
        try:
            import disba  # noqa: F401  # 遅延 import 検証のため
        except ImportError as e:
            raise ImportError(
                "ThomsonHaskellLoveSolver requires `disba` to be installed. "
                "Install via `uv add disba` or `pip install disba`."
            ) from e
        self.halfspace_thickness_km = float(halfspace_thickness_km)
        self.dc_km_s = dc_km_s
        self.algorithm = str(algorithm)

    def forward(
        self,
        model: LayeredEarthModel,
        freqs: np.ndarray,
        mode: int = 0,
    ) -> np.ndarray:
        from disba import PhaseDispersion

        freqs = np.asarray(freqs, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError(f"freqs must be 1D and non-empty; got shape {freqs.shape}")
        if np.any(freqs <= 0):
            raise ValueError("freqs must be strictly positive")

        # 単位変換 (m -> km, m/s -> km/s, kg/m^3 -> g/cm^3)
        h_km = model.h * _M_TO_KM
        if not np.isfinite(h_km[-1]):
            h_km = h_km.copy()
            h_km[-1] = self.halfspace_thickness_km
        vp_km_s = model.vp * _M_TO_KM
        vs_km_s = model.vs * _M_TO_KM
        rho_gcm3 = model.rho * _KGM3_TO_GCM3

        # disba は ascending periods を期待。元の freqs 順に戻すための index を保持。
        periods = 1.0 / freqs
        sort_idx = np.argsort(periods)
        periods_sorted = periods[sort_idx]

        kwargs = dict(algorithm=self.algorithm)
        if self.dc_km_s is not None:
            kwargs["dc"] = float(self.dc_km_s)
        pd = PhaseDispersion(h_km, vp_km_s, vs_km_s, rho_gcm3, **kwargs)
        cp = pd(periods_sorted, mode=int(mode), wave="love")

        returned_periods = np.asarray(cp.period, dtype=float)
        returned_vel_km_s = np.asarray(cp.velocity, dtype=float)
        if returned_periods.size != periods_sorted.size:
            raise RuntimeError(
                "disba returned a partial Love dispersion curve "
                f"({returned_periods.size}/{periods_sorted.size} points). "
                "Adjust frequency range, layer parameters, or `dc_km_s`. "
                "NaN-padded results are not supported in this slice."
            )

        # km/s -> m/s に戻し、元 freqs 順に並び戻す。
        velocity_sorted_m_s = returned_vel_km_s * 1000.0
        unsort_idx = np.argsort(sort_idx)
        return velocity_sorted_m_s[unsort_idx]

    def predict_modes(
        self,
        model: LayeredEarthModel,
        freqs: np.ndarray,
        max_mode: int = 4,
    ) -> list:
        """各モードについて Love phase velocity 配列を返す (存在するものだけ)。

        Niching PSO の NearestNeighborMisfit や cluster ranking で
        「観測 mode が存在する場合のみ扱う」要件を満たすためのヘルパ。

        Returns
        -------
        list of tuple (mode_index, ndarray shape=(m,))
            disba が解を返したモードのみ含む。
            高次モードが存在しない周波数帯では空 list を返す。
        """
        results: list = []
        freqs = np.asarray(freqs, dtype=float)
        for mode in range(int(max_mode) + 1):
            try:
                c = self.forward(model, freqs, mode=mode)
            except RuntimeError:
                continue
            except Exception:
                # disba は内部で DispersionError を投げ得る。基本モードが取れない場合は中断。
                if mode == 0:
                    raise
                continue
            results.append((mode, c))
        return results
