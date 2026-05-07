"""Default initial model builder for Rayleigh-wave Vs inversion.

picked dispersion curve から物理的に整合した 1D 層構造の **初期** モデルを生成する。
Step ?-1a の "h は固定 / Vs のみ更新" 制約と、Step ?-1b の disba root-finder が要求する
Vp/Vs 比 (Poisson 比が物理範囲) を満たす値を返す。

設計方針:
  - 感度深度の上限を z_max ≈ lambda_max / 2.5 で見積もる
    (lambda_max = c_max / f_min)。
  - 層厚は対数的に増やす (浅層を薄く、深層を厚く)。
  - 最終層は半空間として h[-1] = +inf を入れる
    (ThomsonHaskellSolver 側で halfspace_thickness_km に置換される)。
  - Vs init は深さに対し linear: surface ≈ c(f_max)/0.92, 底 ≈ c(f_min)/0.92
    (0.92 は ν=0.25 半空間の Rayleigh-Vs 比)。
  - Vp = sqrt(3) * Vs (ν=0.25)、rho は一定 (既定 2000 kg/m^3)。
  - vs_bounds は per-layer に (lower_factor * vs_init, upper_factor * vs_init)。
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

from src.inversion.rayleigh.model import LayeredEarthModel, PickedDispersionCurve


_RAYLEIGH_VS_RATIO = 0.92
_SQRT3 = float(np.sqrt(3.0))


def build_default_init_model(
    picked: PickedDispersionCurve,
    *,
    n_layers: int = 10,
    halfspace: bool = True,
    vp_vs_ratio: float = _SQRT3,
    rho: float = 2000.0,
    vs_bounds_factor: Tuple[float, float] = (0.5, 2.0),
    sensitivity_factor: float = 2.5,
) -> LayeredEarthModel:
    """picked dispersion curve から default 初期モデルを構成する。

    Parameters
    ----------
    picked : PickedDispersionCurve
        観測ピッキング済み分散曲線。
    n_layers : int
        層数 (半空間を含む)。既定 10。
    halfspace : bool
        True (既定) の場合、最終層の h を +inf にする。
    vp_vs_ratio : float
        Vp / Vs 比。既定 sqrt(3) (ν=0.25)。disba root-finder の安定性のため
        Poisson 比が物理範囲 (0 < ν < 0.5) になる値を選ぶこと。
    rho : float
        全層一定の密度 [kg/m^3]。既定 2000。
    vs_bounds_factor : tuple of float
        Vs の下限・上限を vs_init に乗ずる係数。既定 (0.5, 2.0)。
    sensitivity_factor : float
        感度深度の係数: z_max ~ lambda_max / sensitivity_factor。既定 2.5。

    Returns
    -------
    LayeredEarthModel
        n_layers 層の初期モデル (Vs 初期分布、Vp = ratio*Vs、rho 一定、h 固定)。
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
    c_min = float(np.min(picked.c))
    c_max = float(np.max(picked.c))
    if f_min <= 0.0 or c_max <= 0.0:
        raise ValueError("picked must have positive frequencies and velocities")
    lambda_max = c_max / f_min
    z_max = lambda_max / sensitivity_factor

    # --- 層厚: 対数的に増加 ---
    # 半空間を含めて n_layers 層。浅 (n_layers - 1) 層で z_max まで到達するよう設計し、
    # 最終層を半空間 (h = inf) または z_max の最終層厚と同等の値で閉じる。
    edges = np.geomspace(z_max / 100.0, z_max, n_layers)  # shape=(n_layers,)
    thicknesses_finite = np.diff(np.concatenate([[0.0], edges]))  # shape=(n_layers,)
    h = thicknesses_finite.copy()
    if halfspace:
        h[-1] = np.inf

    # --- Vs 初期分布: 各層の中心深さに対し linear ---
    z_top = np.concatenate([[0.0], np.cumsum(thicknesses_finite[:-1])])
    z_bot = z_top + thicknesses_finite
    z_center = 0.5 * (z_top + z_bot)
    z_center[-1] = z_top[-1] + 0.5 * thicknesses_finite[-2] if n_layers >= 2 else z_center[-1]

    vs_top = c_max / _RAYLEIGH_VS_RATIO  # 高周波 -> 浅層 -> ?? 実際は HF が薄い層に対応
    vs_bot = c_min / _RAYLEIGH_VS_RATIO  # 低周波 -> 深層
    # 高周波での c は浅層 Vs に近い、低周波での c は深層 Vs に近い。
    # picked 上で f が大きいほど c が小さい (normal dispersion) ことを想定。
    # ただし、picked.c の最小値 = HF 側、最大値 = LF 側 とは限らないので、
    # f の最大/最小に対応する c を直接使う。
    idx_fmax = int(np.argmax(picked.f))
    idx_fmin = int(np.argmin(picked.f))
    c_at_fmax = float(picked.c[idx_fmax])
    c_at_fmin = float(picked.c[idx_fmin])
    vs_top = c_at_fmax / _RAYLEIGH_VS_RATIO  # surface 側
    vs_bot = c_at_fmin / _RAYLEIGH_VS_RATIO  # 底側

    # 単調になることは保証しない (inverse dispersion ケースを許容)。
    # 中心深さに対し linear interpolation。
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
