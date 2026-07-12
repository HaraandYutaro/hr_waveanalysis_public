"""Default initial model builder for Love-wave Vs inversion.

Love 波は P 波速度に依存せず、Vs / 密度 / 層厚で完全に決まる。
そのため Rayleigh 用 ``build_default_init_model`` の Rayleigh-Vs 比 0.92
(Poisson 半空間に対する Rayleigh / S 比) は適用せず、本関数では:

  - 高周波端 (浅層) の代表 Vs は ``c(f_max)`` を直接採用
  - 低周波端 (深層) の代表 Vs は ``c(f_min)`` を直接採用

する。これは Love 半空間の極限速度が ``c_Love -> beta_halfspace``
(Aki & Richards 2002 §7.2) という性質に基づく安全側の初期推定であり、
ユーザ指示「観測 c の min/max を範囲に用いる方針」と整合する。

Vp は disba の API 要件で渡すが Love 分散には寄与しないため、
``vp_vs_ratio`` (既定 sqrt(3)) は disba root-finder の入力 sanity のために
保持する (Poisson 比が物理範囲に収まる値を選ぶ)。

References
----------
- Aki, K. & Richards, P. G. (2002). "Quantitative Seismology", 2nd ed. §7.2.
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.inversion.rayleigh.model import LayeredEarthModel, PickedDispersionCurve

_SQRT3 = float(np.sqrt(3.0))


def build_default_love_init_model(
    picked: PickedDispersionCurve,
    *,
    n_layers: int = 10,
    halfspace: bool = True,
    vp_vs_ratio: float = _SQRT3,
    rho: float = 2000.0,
    vs_bounds_factor: Tuple[float, float] = (0.5, 2.0),
    sensitivity_factor: float = 2.5,
) -> LayeredEarthModel:
    """picked Love dispersion curve から default 初期モデルを構成する。

    Parameters
    ----------
    picked : PickedDispersionCurve
        観測ピッキング済み Love 分散曲線 (基本モード推奨)。
    n_layers : int
        層数 (半空間を含む)。既定 10。
    halfspace : bool
        True (既定) の場合、最終層の h を +inf にする。
    vp_vs_ratio : float
        Vp / Vs 比。既定 sqrt(3) (ν=0.25)。Love 分散には寄与しないが、
        disba root-finder の物理整合のために用いる。
    rho : float
        全層一定の密度 [kg/m^3]。既定 2000。
    vs_bounds_factor : tuple of float
        Vs の下限・上限を vs_init に乗ずる係数。既定 (0.5, 2.0)。
        PSO の探索空間境界として ``vs_bounds`` に格納される。
    sensitivity_factor : float
        感度深度の係数: z_max ~ lambda_max / sensitivity_factor。既定 2.5。
        Love 波は同波長で Rayleigh より浅く感じる傾向があるが、
        実用上は同係数で十分 (Xia 2015 §3)。

    Returns
    -------
    LayeredEarthModel
        n_layers 層の初期モデル。
    """
    if picked.n_points < 2:
        raise ValueError(
            "picked must contain at least 2 frequency points to seed an init model"
        )
    if n_layers < 2:
        raise ValueError(f"n_layers must be >= 2, got {n_layers}")
    lo_f, hi_f = vs_bounds_factor
    if not (0.0 < lo_f <= hi_f):
        raise ValueError(f"invalid vs_bounds_factor: {vs_bounds_factor}")

    # --- 感度深度推定 ---
    f_min = float(np.min(picked.f))
    f_max = float(np.max(picked.f))
    c_max = float(np.max(picked.c))
    if f_min <= 0.0 or c_max <= 0.0:
        raise ValueError("picked must have positive frequencies and velocities")
    lambda_max = c_max / f_min
    z_max = lambda_max / sensitivity_factor

    # --- 層厚: 対数的に増加 ---
    edges = np.geomspace(z_max / 100.0, z_max, n_layers)
    thicknesses_finite = np.diff(np.concatenate([[0.0], edges]))
    h = thicknesses_finite.copy()
    if halfspace:
        h[-1] = np.inf

    # --- Vs 初期分布: 各層の中心深さに対し linear ---
    # 観測 c の min/max を直接 Vs 候補として採用 (ユーザ方針: Rayleigh-Vs 比 0.92 を使わない)。
    z_top = np.concatenate([[0.0], np.cumsum(thicknesses_finite[:-1])])
    z_bot = z_top + thicknesses_finite
    z_center = 0.5 * (z_top + z_bot)
    if n_layers >= 2:
        z_center[-1] = z_top[-1] + 0.5 * thicknesses_finite[-2]

    idx_fmax = int(np.argmax(picked.f))
    idx_fmin = int(np.argmin(picked.f))
    c_at_fmax = float(picked.c[idx_fmax])  # 浅層側 Vs 推定
    c_at_fmin = float(picked.c[idx_fmin])  # 深層側 Vs 推定
    vs_top = c_at_fmax
    vs_bot = c_at_fmin

    z_norm = (z_center - z_center[0]) / max(z_center[-1] - z_center[0], 1e-30)
    vs_init = vs_top + (vs_bot - vs_top) * z_norm

    vp_init = vp_vs_ratio * vs_init
    rho_init = np.full(n_layers, float(rho))
    vs_lo = lo_f * vs_init
    vs_hi = hi_f * vs_init

    return LayeredEarthModel(
        vs=vs_init,
        vp=vp_init,
        rho=rho_init,
        h=h,
        vs_bounds=(vs_lo, vs_hi),
    )
