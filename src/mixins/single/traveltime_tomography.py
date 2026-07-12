"""
src/mixins/single/traveltime_tomography.py

初動走時トモグラフィ (First-arrival traveltime tomography) による
2D 初期速度構造モデル構築 mixin。

学術的方針メモ:
  - 初動走時トモグラフィは初期モデル依存性が強い
  - 逆問題は不良設定なので平滑化正則化を前提にする
  - 順問題はアイコナール方程式 |∇T|^2 = s(x)^2 をベースに設計する
  - SIRT / LSQR は現実的な初版手法

困ったら以下文献の理論に従うこと:
  1. White, D.J. (1989) Two-Dimensional Seismic Refraction Tomography.
     Geophys. J. Int., 97, 223-245.
  2. Zhang, J. & Toksoz, M.N. (1998) Nonlinear refraction traveltime tomography.
     Geophysics, 63, 1726-1737.
  3. Taillandier, C., Noble, M., Chauris, H., Calandra, H. (2009) First-arrival
     traveltime tomography based on the adjoint-state method. Geophysics, 74, WCB1.
     (= JGE 系 functional description of traveltimes に相当)
  4. Sheehan, J.R., Doll, W.E., Mandell, W.A. (Tutorial) Evaluation of methods
     and software for seismic refraction tomography analysis.
  5. Palmer, D. (1981) Generalized Reciprocal Method.
     (delay-time / head-wave practical overview)

順問題の MVP 実装:
  - 厳密 FMM の代わりに、(nz, nx) グリッド上の 8-connected dijkstra による
    最短経路探索でアイコナール方程式の数値解を近似する。
  - 各エッジ (cell A - cell B) の重みは edge_length * (s_A + s_B) / 2。
  - dijkstra のシングルソース最短経路で、震源点から全グリッド点への走時 T(x)
    を取得する。
  - 受振点ごとの感度行列 L (shape=(n_rec, nz*nx)) は、最短経路の predecessor を
    逆にたどり、各エッジの寄与 edge_length / 2 を端点 cell に加算して構築する。
  - 合成走時は syn_tt[i] = (L[i,:] · slowness_flat).
  - 将来 FMM (例: scikit-fmm) に差し替える場合は、本ファイルの ``_solve_forward``
    内部のみを置換すれば、上位 API (``traveltime_tomography``) は不変。

順問題の追加文献:
  6. Nakanishi, I. & Yamaguchi, K. (1986) A numerical experiment on nonlinear
     image reconstruction from first-arrival times for two-dimensional island
     arc structure. J. Phys. Earth, 34, 195-206.
  7. Moser, T.J. (1991) Shortest path calculation of seismic rays.
     Geophysics, 56, 59-67.

逆問題の MVP 実装:
  - SIRT をデフォルトとする。行和・列和正規化 + Tikhonov ダンピング + positivity。
  - LSQR も利用可能。ダンピング + 一次差分 Tikhonov 正則化を組み込み可能。
  - 逆問題は本質的に不良設定。正則化なしでは解が発散する。実務的には
    SIRT 正規化 + 外部 smoothing / LSQR ダンピング + 物理クリップがいずれも
    正則化として働く (Rawlinson & Sambridge 2003 レビュー参照)。
  - 停止基準: 最大反復回数に加え、RMS 改善量とモデル更新ノルムの閾値を導入。
    半収束 (semi-convergence) に注意 (Hanke & Scherzer 2001)。

逆問題の追加文献:
  9. Rawlinson, N. & Sambridge, M. (2003) Seismic traveltime tomography
     of the crust and lithosphere. Adv. Geophys., 46, 81-197.
  10. Nolet, G. (2008) A Breviary of Seismic Tomography.
      Cambridge Univ. Press.
  11. Hanke, M. & Scherzer, O. (2001) Inverse problems light:
      numerical differentiation. Amer. Math. Monthly.

NOTE:
required_spreadの式：
# NOTE: required_spread の計算仮定と将来の改善点
#
# 現在の実装 (_resolve_depth_gradient / iter=0 警告メッセージ) では、
# 臨界角から必要展開長を次式で推定している:
#
#   X_min = 2 * z_half / tan(arcsin(V_surf / V(z_half)))
#
# この式は以下の仮定に基づく:
#   (A) 速度は深度方向に線形増加 V(z) = V_surf + grad * z
#   (B) 横方向 (x方向) の速度変化は無視 (1-D仮定)
#   (C) 地表は平坦 (topo_z = None または微小起伏)
#   (D) 初動到達が臨界角屈折波 (head wave) によるものとする
#
# 仮定 (B) の影響:
#   実フィールドでは横方向速度変化により射線が曲げられるため、
#   X_min は「下限目安」にすぎない。横方向に低速層が存在する場合、
#   実際に必要な展開長は X_min より大きくなりうる。
#
# TODO (将来の改善): 横方向不均質への対応
#   現状: V_surf = velocity_model[0, :].mean()  ← 表層の横方向平均
#   改善案:
#     V_surf_local = velocity_model[0, source_ix]  # 震源直下の表層速度のみ使用
#     V_half_local = velocity_model[nz//2, :].mean()  # 中間深度の横方向平均
#     # さらに厳密には、速度モデルを列ごとに V(z) フィッティングして
#     # 列ごとの grad を推定し、最小 grad の列で X_min を評価する:
#     #   col_grads = [(velocity_model[-1, j] - velocity_model[0, j]) / (nz * dz)
#     #                for j in range(nx)]
#     #   grad_conservative = np.percentile(col_grads, 10)  # 10パーセンタイル(慎重側)
#     #   X_min = 2 * z_half / tan(arcsin(V_surf / (V_surf + grad_conservative * z_half)))
#
# TODO (将来の改善): 起伏地形補正
#   topo_z が None でない場合、射線の実効的な深度到達は
#   topo_z の変化分だけシフトする。
#   簡易補正:
#     topo_relief = np.ptp(topo_z) if topo_z is not None else 0.0
#     z_half_eff = z_half + topo_relief  # 起伏分を加算して保守側に評価
#     X_min = 2 * z_half_eff / tan(arcsin(...))
#
# TODO (将来の改善): head wave 仮定の検証
#   仮定 (D) は速度コントラストが明確な2層構造を想定している。
#   勾配が連続的な場合、初動は diving wave (潜行波) であり、
#   turning depth z_turn は射線パラメータ p から:
#     p = sin(phi_0) / V_surf  (phi_0: 水平からの出射角)
#     z_turn = (1/p - V_surf) / grad = V_surf * (1/sin(phi_0) - 1) / grad
#   この場合 X_min の代わりに数値的な射線追跡で
#   z_turn >= z_half を満たす最小展開長を探索するのが厳密。
#   参照: Telford et al. (1990) Applied Geophysics, Ch.4
#
# 参照文献:
#   - Bai et al. (2010) GJI 183:1596-1612, page 1604
#     "low spatial resolution ... the best way is to combine later arrivals
#      to increase ray path density and data coverage"
#   - Sheriff & Geldart (1995) Exploration Seismology, Ch.4.3
#     (refraction geometry and critical distance)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import warnings

import numpy as np
from scipy import ndimage, signal
from scipy.sparse import csr_matrix, eye as sp_eye, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr as scipy_lsqr

from src.plotting.wrapper import PlotterWrapperMixin

# ── FMM DISPATCH: inserted by Phase D integration ──
from src.mixins.single.traveltime_fmm import TraveltimeFmm2D  # noqa: E402


_PICKING_MODES = ("manual", "energy_threshold", "correlation", "deep_learning", "aic")
_INITIAL_METHODS = ("apparent_velocity", "delay_time")
_INVERSION_METHODS = ("sirt", "lsqr", "adjoint")


# =========================================================
# First-break picking 結果コンテナ
# =========================================================
# 困ったら以下文献の理論に従うこと:
#   - Allen, R. (1978) Automatic earthquake recognition and timing from single
#     traces. BSSA 68, 1521-1532.  (STA/LTA の起源)
#   - Withers, M. et al. (1998) A comparison of select trigger algorithms for
#     automated global seismic phase and event detection. BSSA 88.
#   - Saragiotis, C.D., Hadjileontiadis, L.J., Panas, S.M. (2002) PAI-S/K:
#     A robust automatic seismic P phase arrival identification scheme. IEEE TGRS.
#   - VanDecar, J.C., Crosson, R.S. (1990) Determination of teleseismic relative
#     phase arrival times using multi-channel cross-correlation and least squares.
#     BSSA 80, 150-169.  (MCCC の原典)
#   - Diehl, T., Kissling, E. (2009) MannekenPix consistent first-arrival picking.
#     (実務的 review)
#   - Sabbione, J.I., Velis, D. (2010) Automatic first-breaks picking: New
#     strategies and algorithms. Geophysics 75, V67-V76.
@dataclass
class FirstBreakPickResult:
    """
    1 ショット分の first-break picking 結果コンテナ。

    Fields
    ------
    pick_indices : ndarray, shape=(n_receivers,), dtype=int
        サンプル番号での pick。invalid は -1。
    pick_times : ndarray, shape=(n_receivers,), dtype=float
        秒単位 pick。invalid は NaN。
    quality : ndarray, shape=(n_receivers,), dtype=float
        ピック品質スコア (0.0–1.0 を想定)。
        - energy_threshold: envelope[pick] / peak_envelope
        - correlation:      正規化相関ピーク値
        - manual:           1.0 (有効) / 0.0 (無効)
    valid_mask : ndarray, shape=(n_receivers,), dtype=bool
        ピック成功フラグ。
    method : str
        使用したピッキングモード。
    qc : list[dict]
        受振器ごとの QC 詳細 (noise_level / trigger_value / correlation_peak /
        snr / template_trace / template_pick_idx 等を可能な範囲で格納)。

    Notes
    -----
    既存 dict 互換性のため ``to_dict()`` を提供する。
    """

    pick_indices: np.ndarray
    pick_times: np.ndarray
    quality: np.ndarray
    valid_mask: np.ndarray
    method: str
    qc: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """
        dataclass → dict 変換層 (互換性のため)。
        """
        return {
            "pick_indices": self.pick_indices,
            "pick_times": self.pick_times,
            "quality": self.quality,
            "valid_mask": self.valid_mask,
            "method": self.method,
            "qc": self.qc,
        }


# =========================================================
# 初期モデル診断コンテナ (private)
# =========================================================
# Literature guidance:
#   White (1989) Two-Dimensional Seismic Refraction Tomography
#   Sheehan, Doll, Mandell — practical overview
#   Hagedoorn (1959) delay-time method
#   Palmer (1981) Generalized Reciprocal Method
#   Gebrande & Miller (1985) local slope estimation
#   Lanz, Maurer, Green (1998) practical refraction tomography
@dataclass
class _InitialModelDiagnostics:
    method: str
    fallback_used: bool
    n_valid: int
    v_app_per_x: Optional[np.ndarray]
    surface_v: Optional[np.ndarray]
    v1_estimate: float
    v2_estimate: float
    h_estimate: Optional[float]
    notes: list


# =========================================================
# 初期モデル構築用純粋/helper 関数
# =========================================================
def _weighted_linear_fit(
    x: np.ndarray, t: np.ndarray, w: np.ndarray
) -> Tuple[float, float, bool]:
    """Weighted least squares: t = slope * x + intercept.

    Returns (slope, intercept, ok).
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    w = np.asarray(w, dtype=float)
    if x.size < 2 or t.size < 2:
        return 0.0, 0.0, False
    w_sum = float(np.sum(w))
    if w_sum < 1e-15:
        return 0.0, 0.0, False
    x_bar = float(np.sum(w * x)) / w_sum
    t_bar = float(np.sum(w * t)) / w_sum
    dx = x - x_bar
    dt = t - t_bar
    denom = float(np.sum(w * dx * dx))
    if abs(denom) < 1e-15:
        return 0.0, 0.0, False
    slope = float(np.sum(w * dx * dt)) / denom
    intercept = t_bar - slope * x_bar
    return slope, intercept, True


def _local_slope_per_receiver(
    offset: np.ndarray, t: np.ndarray, w: np.ndarray, window: int = 5
) -> Tuple[np.ndarray, np.ndarray]:
    """Local dt/dx estimation per receiver via weighted linear regression.

    Returns (slopes, ok_mask) where ok_mask[i] indicates if slope[i] is valid.
    """
    n = len(offset)
    if n == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    sort_idx = np.argsort(offset)
    d_sorted = offset[sort_idx]
    t_sorted = t[sort_idx]
    w_sorted = np.asarray(w, dtype=float)[sort_idx]
    half_win = max(1, window // 2)
    slopes = np.zeros(n, dtype=float)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        i_start = max(0, i - half_win)
        i_end = min(n, i + half_win + 1)
        slope, _intercept, fit_ok = _weighted_linear_fit(
            d_sorted[i_start:i_end], t_sorted[i_start:i_end], w_sorted[i_start:i_end]
        )
        if fit_ok and slope > 0:
            slopes[i] = slope
            ok[i] = True
    unsort_idx = np.argsort(sort_idx)
    return slopes[unsort_idx], ok[unsort_idx]


def _robust_outlier_mask(values: np.ndarray, k: float = 3.0) -> np.ndarray:
    """MAD-based outlier rejection. Returns bool mask (True = valid)."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values) & (values > 0)
    if np.sum(finite) < 3:
        return finite
    med = float(np.median(values[finite]))
    mad = float(np.median(np.abs(values[finite] - med)))
    if mad < 1e-10:
        return finite
    threshold = k * (mad / 0.6745)
    return finite & (np.abs(values - med) <= threshold)


def _smooth_1d_with_gaps(
    values: np.ndarray, valid_mask: np.ndarray, sigma: float
) -> np.ndarray:
    """Gaussian smoothing of 1D data with gap filling via nearest interpolation."""
    values = np.asarray(values, dtype=float).copy()
    valid_mask = np.asarray(valid_mask, dtype=bool)
    n = len(values)
    if n == 0:
        return values
    valid_vals = values[valid_mask]
    if valid_vals.size == 0:
        return np.full(n, np.nan)
    if valid_vals.size == 1:
        return np.full(n, valid_vals[0])
    filled = np.where(valid_mask, values, 0.0)
    weights = valid_mask.astype(float)
    if sigma > 0:
        smoothed_filled = ndimage.gaussian_filter1d(filled, sigma=sigma, mode="nearest")
        smoothed_weights = ndimage.gaussian_filter1d(weights, sigma=sigma, mode="nearest")
    else:
        smoothed_filled = filled
        smoothed_weights = weights
    eps = 1e-10
    result = np.where(
        smoothed_weights > eps,
        smoothed_filled / smoothed_weights,
        np.nan,
    )
    is_nan = np.isnan(result)
    if np.any(is_nan):
        valid_positions = np.where(valid_mask)[0]
        valid_values = values[valid_mask]
        if valid_positions.size > 0:
            result[is_nan] = np.interp(
                np.where(is_nan)[0].astype(float),
                valid_positions.astype(float),
                valid_values,
                left=valid_values[0],
                right=valid_values[-1],
            )
    still_nan = np.isnan(result)
    if np.any(still_nan):
        med = float(np.median(valid_vals))
        result[still_nan] = med
    return result


def _apply_depth_trend(
    surface_v: np.ndarray, z_centers: np.ndarray, alpha: float, power: float
) -> np.ndarray:
    """v(x,z) = surface_v(x) * (1 + alpha * (z/z_max)^power). Returns (nz, nx)."""
    z_max = max(float(z_centers[-1]), 1.0)
    depth_factor = 1.0 + alpha * (z_centers / z_max) ** power
    return surface_v[np.newaxis, :] * depth_factor[:, np.newaxis]


def _resolve_depth_gradient(
    depth_gradient_fallback: Optional[float],
    v0: np.ndarray,
    distance: np.ndarray,
    source_x: float,
) -> Tuple[float, str]:
    """
    depth_gradient_fallback の解決。

    None なら受振展開長から自動算出:
        spread   = max|distance - source_x|
        z_target = spread / 2.0          # 経験則: 最大探査深度 ≒ 展開半長
        V_surf   = v0[0, :].mean()
        V_need   = V_surf * 1.5          # 臨界角 ~arcsin(1/1.5)≈41.8° 相当
        grad     = max((V_need - V_surf) / z_target, 0)

    数値が渡された場合はそれを採用 (0.0 で無効化可)。

    Returns
    -------
    (grad, source) :
        grad : 適用される勾配 [m/s per m] (非負)
        source : 'auto' or 'explicit'
    """
    if depth_gradient_fallback is None:
        spread = float(np.max(np.abs(np.asarray(distance, dtype=float) - source_x)))
        z_target = max(spread / 2.0, 1.0)
        V_surface = float(np.asarray(v0)[0, :].mean())
        if not np.isfinite(V_surface) or V_surface <= 0:
            return 0.0, "auto (degenerate V_surface; disabled)"
        V_needed = V_surface * 1.5
        grad = max((V_needed - V_surface) / z_target, 0.0)
        return float(grad), (
            f"auto: spread={spread:.1f}m, z_target={z_target:.1f}m, "
            f"V_surf={V_surface:.1f} → V_need={V_needed:.1f}"
        )
    grad_val = float(max(float(depth_gradient_fallback), 0.0))
    return grad_val, "explicit"


def _fit_two_segment(
    d: np.ndarray, t: np.ndarray, w: np.ndarray, x_candidates: np.ndarray
) -> Tuple[float, float, float, float, float]:
    """Fit piecewise linear two-segment model to t-x picks.

    Near: t = slope_near * x + intercept_near   (direct wave)
    Far:  t = slope_far  * x + intercept_far     (refracted wave)

    Returns (V1, V2, tau, x_c, residual).
    """
    sort_idx = np.argsort(d)
    d_sorted = d[sort_idx]
    t_sorted = t[sort_idx]
    w_sorted = np.asarray(w, dtype=float)[sort_idx]
    best_residual = np.inf
    best = (0.0, 0.0, 0.0, 0.0, np.inf)
    for xc in x_candidates:
        near = d_sorted <= xc
        far = ~near
        n_near = int(np.sum(near))
        n_far = int(np.sum(far))
        if n_near < 2 or n_far < 2:
            continue
        slope_near, int_near, ok_near = _weighted_linear_fit(
            d_sorted[near], t_sorted[near], w_sorted[near]
        )
        slope_far, int_far, ok_far = _weighted_linear_fit(
            d_sorted[far], t_sorted[far], w_sorted[far]
        )
        if not ok_near or not ok_far:
            continue
        if slope_near <= 0 or slope_far <= 0:
            continue
        V1 = 1.0 / slope_near
        V2 = 1.0 / slope_far
        tau = int_far
        pred = np.where(
            near, slope_near * d_sorted + int_near, slope_far * d_sorted + int_far
        )
        resid = float(np.sum(w_sorted * (t_sorted - pred) ** 2))
        if resid < best_residual:
            best_residual = resid
            best = (V1, V2, tau, float(xc), resid)
    return best


def _apply_elevation_static_correction(
    picks: np.ndarray,
    distance: np.ndarray,
    source_x: float,
    z_distance: np.ndarray,
    datum: float,
    v_replacement: float,
) -> Tuple[np.ndarray, dict]:
    """
    Static elevation correction for first-break picks (Marsden 1993).

    dt_elev[i] = (datum - z[i]) / V_replacement   (vertical raypath approx.)
    Corrected picks: t_corr[i] = t_obs[i] - (dt_src + dt_rec[i])
    Only applied where picks > 0 (valid picks).
    """
    z_src = float(np.interp(source_x, distance, z_distance))
    dt_src = (datum - z_src) / v_replacement
    dt_rec = (datum - z_distance) / v_replacement
    valid = picks > 0
    picks_corr = picks.copy()
    picks_corr[valid] = picks[valid] - (dt_src + dt_rec[valid])
    return picks_corr, {
        "datum": float(datum),
        "v_replacement": float(v_replacement),
        "dt_src": float(dt_src),
        "dt_rec": dt_rec,
        "z_src": z_src,
    }


class TraveltimeTomography(PlotterWrapperMixin):
    """
    初動走時トモグラフィ mixin。

    SingleProcesser に継承され、外部公開メソッド ``traveltime_tomography()`` を
    通じて以下のパイプラインを 1 度に実行する:

        1. 初動走時ピッキング   (manual / energy_threshold / correlation / deep_learning / aic)
        2. 解析初期モデル構築   (apparent_velocity / delay_time)
        3. スローネスモデル化   (1/v)
        4. 順問題                (アイコナール近似: dijkstra ベース最短経路)
        5. 反復更新              (sirt / lsqr / adjoint)

    各ステップは private メソッドに分割され、単独でも呼び出せる。

    Notes
    -----
    - 探査軸 ``axis`` は波形のチャネル選択 ('x','y','z') を意味し、
      2D 断面のグリッド軸 (grid_x, grid_z) とは別概念である。
    - 震源・受振器ともに地表 (z=0) 配置を前提とする。
    - 反復は SIRT デフォルト。LSQR は次点。adjoint は MVP では NotImplementedError。
    - 初期モデルの ``delay_time`` 法は近似 delay-time MVP (厳密な Palmer/GRM ではない)。
    - ``delay_time`` が不安定な場合は自動的に ``apparent_velocity`` にフォールバックし、
      さらに不安定な場合は ``smooth_fallback`` にフォールバックする。
    """

    # =========================================================
    # 外部公開 (orchestrator)
    # =========================================================
    def traveltime_tomography(
        self,
        axis: str,
        *,
        picking_mode: str = "energy_threshold",
        initial_method: str = "apparent_velocity",
        inversion_method: str = "sirt",
        nx: int = 64,
        nz: int = 32,
        dx: Optional[float] = None,
        dz: Optional[float] = None,
        n_iter: int = 20,
        smooth_sigma: float = 0.5,
        s_min: float = 1.0 / 8000.0,
        s_max: float = 1.0 / 50.0,
        manual_picks: Optional[np.ndarray] = None,
        threshold_ratio: float = 0.1,
        sirt_step: float = 1.0,
        lsqr_damp: float = 0.0,
        return_history: bool = True,
        # ---- picking robustness 拡張 (全て default 付き; 既存呼び出しに無影響) ----
        noise_window_ratio: float = 0.05,
        min_duration_samples: int = 3,
        min_arrival_time: Optional[float] = None,
        manual_pick_indices: Optional[np.ndarray] = None,
        manual_pick_times: Optional[np.ndarray] = None,
        correlation_template: Optional[np.ndarray] = None,
        correlation_template_index: Optional[int] = None,
        min_corr_peak: float = 0.3,
        # ---- initial model extended parameters (全て default 付き; 既存呼び出しに無影響) ----
        local_window_traces: int = 5,
        surface_smooth_sigma_x: float = 1.5,
        depth_trend_alpha: float = 0.5,
        depth_trend_power: float = 0.7,
        vmin_init: float = 20.0,
        vmax_init: float = 2000.0,
        mad_k: float = 3.0,
        depth_gradient_fallback: Optional[float] = None,
        # ---- inversion extended parameters (全て default 付き; 既存呼び出しに無影響) ----
        inversion_lambda: float = 0.0,
        inversion_damping: float = 1e-6,
        inversion_step_length: float = 1.0,
        inversion_tol: float = 1e-6,
        model_update_tol: float = 0.0,
        # ---- elevation static correction (opt-in; default 既存挙動完全保持) ----
        elevation_correction: bool = False,
        elevation_replacement_velocity: Optional[float] = None,
        elevation_datum: Optional[float] = None,
        # ---- ray-path storage (opt-in; default 既存挙動完全保持) ----
        store_ray_paths: bool = False,
        # ── FMM WIRE-THROUGH: forward_method plumbed from public API ──
        forward_method: str = "dijkstra",
        # ---- 描画制御 (新規追加) ----
        show: bool = False,
        save_name: Optional[str] = None,
        **plot_kwargs,
    ) -> dict:
        """
        初動走時トモグラフィの一括実行 (orchestrator)。

        Parameters
        ----------
        axis : str
            入力波形チャネル ('x', 'y', 'z')。
            内部で ``getax_analysis`` 経由 (なければ getattr) で 2D 配列を取得。
        picking_mode : str
            'manual' / 'energy_threshold' / 'correlation' / 'deep_learning' / 'aic'
        initial_method : str
            'apparent_velocity' / 'delay_time'
        inversion_method : str
            'sirt' / 'lsqr' / 'adjoint'
        use_topography_overlay : bool
            ピックの topography overlay を有効にするか (デフォルト False)
            有効にすると、起伏の記録したデータに対し、起伏が考慮された表示が行われる。
        nx, nz : int
            2D グリッドのセル数 (横方向 nx, 深さ方向 nz)。
        dx, dz : float, optional
            セル幅 [m]。未指定時は受振器配置と nx から自動決定する。
        n_iter : int
            反復回数。
        smooth_sigma : float
            反復ごとに掛ける Gaussian smoothing の σ (セル単位)。0 以下でスキップ。
        s_min, s_max : float
            スローネスのクリップ範囲 [s/m]。
            デフォルトは v ∈ [200, 6000] m/s 相当。
        manual_picks : np.ndarray, optional
            picking_mode='manual' 時の入力。shape=(n_x,)。
            整数 dtype はサンプル番号、float dtype は秒として解釈する。
            手動 pick 引数の中で **最優先** される
            (manual_picks > manual_pick_times > manual_pick_indices)。
        threshold_ratio : float
            picking_mode='energy_threshold' / 'correlation' のしきい値比率
            (包絡線ピーク振幅に対する比)
        sirt_step : float
            SIRT のステップ係数 α。
        lsqr_damp : float
            LSQR の damping (Tikhonov 正則化強度)。
        return_history : bool
            True なら反復ごとの履歴を結果 dict に含める。
        noise_window_ratio : float
            ノイズ床推定に使用する trace 冒頭区間の比率 (0 < r < 1)。デフォルト 0.05。
        min_duration_samples : int
            energy_threshold で「閾値超え状態が連続するサンプル数」の最小要求。
            短時間ジッタでの誤検出を抑える。
        min_arrival_time : float, optional
            物理的最小到達時刻 [s]。これより前は pick 候補にしない。
            未指定なら 0 (制約なし)。
        manual_pick_indices : ndarray, optional
            picking_mode='manual' でサンプル番号として与える。手動 pick 引数の中で
            最低優先 (manual_picks / manual_pick_times が未指定のときに使用)。
        manual_pick_times : ndarray, optional
            picking_mode='manual' で秒単位で与える。``manual_picks`` 未指定かつ
            本引数指定時に使用 (``manual_pick_indices`` より優先)。
            なお manual_picks / manual_pick_times / manual_pick_indices が
            **すべて未指定** の場合は、対話的 GUI を起動して各 trace の初動を
            クリックでピッキングする (matplotlib backend のみ対応)。
        correlation_template : ndarray, optional
            picking_mode='correlation' で外部テンプレートを使う場合の波形 (1D)。
        correlation_template_index : int, optional
            picking_mode='correlation' でテンプレートに使う trace の index。
            未指定かつ ``correlation_template`` 未指定なら SNR 最大 trace を自動選択。
        min_corr_peak : float
            correlation pick の正規化ピーク値しきい値 (これ未満は invalid)。
        local_window_traces : int
            apparent_velocity 初期モデルの局所回帰窓幅 (受振器数)。
        surface_smooth_sigma_x : float
            apparent_velocity の surface smoothing σ (グリッドセル単位)。
        depth_trend_alpha : float
            depth 勾配係数 alpha: v(z) ∝ (1 + alpha*(z/zmax)^power)。
        depth_trend_power : float
            depth 勾配指数 power。
        vmin_init : float
            初期速度モデルの下限 [m/s]。
        vmax_init : float
            初期速度モデルの上限 [m/s]。
        mad_k : float
            MAD ベース外れ値棄却のしきい値係数。
        inversion_lambda : float
            LSQR 一次差分 Tikhonov 正則化重み (0 で無効)。default=0。
        inversion_damping : float
            SIRT 微小対角ダンピング (安定化)。default=1e-6。
        inversion_step_length : float
            SIRT ステップ長係数 (sirt_step の代替推奨)。default=1.0。
        inversion_tol : float
            反復停止閾値: 連続反復の RMS 相対改善がこの値未満なら停止。0 で無効(デフォルト)。
        model_update_tol : float
            反復停止閾値: モデル更新ノルムがこの値未満なら停止。0 で無効。
        store_ray_paths : bool
            True のとき、収束後の最終速度モデルに対して最短経路を 1 回計算し、
            受振器ごとの射線座標列を ``result["ray_paths"]`` および
            ``self.tt_ray_paths`` に格納する。デフォルト False。

            .. code-block:: python

                result = sp.traveltime_tomography(
                    "y", store_ray_paths=True
                )
                # result["ray_paths"] : list[np.ndarray], len == len(sp.distance)
                # 各要素 shape=(L_k, 2), 列順 [x_m, z_m] (grid 内部座標)

            描画に使うには、続けて ``traveltime_tomo_image(..., show_ray_paths=True)``
            を呼ぶ。または ``show=True`` と共に呼んだ場合は
            ``**plot_kwargs`` に ``show_ray_paths=True`` を含めることで
            一括指定できる::

                result = sp.traveltime_tomography(
                    "y", show=True, store_ray_paths=True,
                    show_ray_paths=True,
                )

        Returns
        -------
        result : dict
            'velocity_model'   : ndarray, shape=(nz, nx)  最終速度モデル [m/s]
            'slowness_model'   : ndarray, shape=(nz, nx)  最終スローネスモデル [s/m]
            'picks'            : ndarray, shape=(n_x,)    走時ピック [s] (invalid は 0.0)
            'synthetic_tt'     : ndarray, shape=(n_x,)    最終合成走時 [s]
            'initial_velocity' : ndarray, shape=(nz, nx)  初期速度モデル [m/s]
            'history'          : list[dict]               反復履歴 (return_history=True 時)
                各 dict は 'iter', 'rms', 'ds_norm', 'residual_norm', 'method' を含む。
                SIRT では更に 'step_norm' を含む。LSQR では更に 'lsqr_istop', 'lsqr_itn' を含む。
            'grid'             : dict                     グリッド定義
            'pick_result'      : FirstBreakPickResult     詳細 picking 結果 (新規)
            'pick_qc'          : list[dict]               受振器ごとの QC 詳細 (新規)
            'initial_model_diagnostics' : _InitialModelDiagnostics  初期モデル診断情報
            'ray_paths'        : list[np.ndarray]         (store_ray_paths=True 時のみ存在)
                長さ n_x (= len(distance) = len(synthetic_tt))、各要素 shape=(L_k, 2)。
                列順は [x_m, z_m]。座標系は grid 内部座標で、x は grid["x0"] を
                引いた左端 0 起点 [m]、z は depth (iz=0 が地表、下方向に正) [m]。
                不到達 receiver は np.empty((0, 2), dtype=float)。
                同じ list が self.tt_ray_paths にも格納される。

        Raises
        ------
        ValueError
            モード文字列が不正、shape 不一致、または必須引数欠落の場合。
        AttributeError
            self.distance / self.source_x / self.fs が未定義の場合。
        NotImplementedError
            未実装モード ('deep_learning' / 'adjoint') を指定した場合。

        Notes
        -----
        **ray_paths の座標系**
            ``store_ray_paths=True`` で得られる各 ray 配列の座標は
            *grid 内部座標* であり、``grid["x0"]`` を引いた左端 0 起点の x [m] と
            depth (表面 iz=0 が z=0、下方向に正) の z [m] からなる。
            可視化側で ``traveltime_tomo_image`` が ``extent_x / extent_z`` と
            整合させるため、ユーザコードで追加シフトを行う必要はない。

        **見送った実装（将来スライス候補）**
            以下は本実装では意図的に見送りとした。後続スライスで個別に追加する。

            * *Plotly backend の ray parity*:
              ``_traveltime_tomo_impl`` の Plotly 実装はまだ未実装 stub であるため、
              ray overlay の Plotly 対応は Plotly parity 完了後に行う。

            * *物理的に正しい傾斜地形トモグラフィ (topo_z 配線)*:
              ``_solve_forward`` は ``topo_z=None`` (平坦) 固定のまま。
              ray は地表を平坦と仮定した最短経路であり、``self.z_distance`` が存在する
              傾斜地形データに対しても地形を順問題に反映していない。
              傾斜を順問題に取り込むには ``_solve_forward`` に ``topo_z`` を配線する
              別スライスが必要（PLAN.md 既知 TODO）。

            * *``max_ray_points`` 間引きオプション*:
              受振器数・グリッドサイズが大きい場合に ray 配列を間引いて
              メモリを節約するオプション。典型的なグリッド (nx=64, nz=32) では
              ray_paths 全体が 1 MB 以下のためデフォルトでは不要。

            * *ray density map / sensitivity の可視化*:
              各セルを通過する ray 数やフレシェ感度を imshow 上に重ねる機能。
              ``sensitivity_matrix`` はすでに ``_solve_spm_2d`` から得られるため
              データは揃っている。
        """
        self._validate_modes(picking_mode, initial_method, inversion_method)

        # --- データと幾何情報の取得 ---
        U = self._resolve_axis_data(axis)
        if U.ndim != 2:
            raise ValueError(f"波形データは 2D を期待 (got ndim={U.ndim})")
        n_x_data = U.shape[0]
        if not hasattr(self, "distance"):
            raise AttributeError("self.distance が未定義です。")
        if not hasattr(self, "source_x"):
            raise AttributeError("self.source_x が未定義です。")
        if not hasattr(self, "fs"):
            raise AttributeError("self.fs (サンプリング周波数) が未定義です。")
        distance = np.asarray(self.distance, dtype=float)
        if distance.shape[0] != n_x_data:
            raise ValueError(
                f"distance ({distance.shape[0]}) と trace 数 ({n_x_data}) が不一致"
            )
        source_x = float(self.source_x)
        fs = float(self.fs)

        grid = self._make_grid(nx, nz, dx, dz, distance, source_x)

        # --- 1. ピッキング (FirstBreakPickResult を取得) ---
        pick_result = self._pick_first_arrivals(
            U=U,
            picking_mode=picking_mode,
            fs=fs,
            threshold_ratio=threshold_ratio,
            noise_window_ratio=noise_window_ratio,
            min_duration_samples=min_duration_samples,
            min_arrival_time=min_arrival_time,
            manual_picks=manual_picks,
            manual_pick_indices=manual_pick_indices,
            manual_pick_times=manual_pick_times,
            correlation_template=correlation_template,
            correlation_template_index=correlation_template_index,
            min_corr_peak=min_corr_peak,
        )
        # 既存互換用: NaN を 0.0 に倒した秒単位 picks ndarray
        picks = pick_result.pick_times.copy()
        picks[~pick_result.valid_mask] = 0.0
        picks = np.where(np.isfinite(picks), picks, 0.0)

        # --- 1.5. elevation static correction (opt-in) ---
        elevation_info: Optional[dict] = None
        if elevation_correction:
            if not hasattr(self, "z_distance"):
                warnings.warn(
                    "elevation_correction=True が指定されましたが self.z_distance が "
                    "存在しないため補正をスキップします。",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                z_dist = np.asarray(self.z_distance, dtype=float)
                if z_dist.shape[0] != distance.shape[0]:
                    warnings.warn(
                        f"z_distance shape ({z_dist.shape[0]}) と distance "
                        f"({distance.shape[0]}) が不一致のため補正をスキップします。",
                        UserWarning,
                        stacklevel=2,
                    )
                else:
                    datum_used = (
                        float(elevation_datum)
                        if elevation_datum is not None
                        else float(z_dist.max())
                    )
                    v_repl = (
                        float(elevation_replacement_velocity)
                        if elevation_replacement_velocity is not None
                        else float(vmin_init)
                    )
                    picks, elevation_info = _apply_elevation_static_correction(
                        picks=picks,
                        distance=distance,
                        source_x=source_x,
                        z_distance=z_dist,
                        datum=datum_used,
                        v_replacement=v_repl,
                    )

        if not pick_result.valid_mask.any():
            raise ValueError(
                "有効な first-break pick が 1 つも得られませんでした。"
                "picking_mode の選択や閾値パラメータを見直してください。"
            )

        # --- 2. 初期モデル (有効 pick のみで apparent velocity を推定するため、
        #         無効受振器は picks=0 にせず NaN を残しておく必要はない。
        #         _initial_apparent_velocity 内部で picks>0 マスクを行うので
        #         picks (invalid->0) のままでよい) ---
        v0, init_diag = self._build_initial_velocity_model(
            picks=picks,
            distance=distance,
            source_x=source_x,
            method=initial_method,
            grid=grid,
            pick_result=pick_result,
            local_window_traces=local_window_traces,
            surface_smooth_sigma_x=surface_smooth_sigma_x,
            depth_trend_alpha=depth_trend_alpha,
            depth_trend_power=depth_trend_power,
            vmin_init=vmin_init,
            vmax_init=vmax_init,
            mad_k=mad_k,
            depth_gradient_fallback=depth_gradient_fallback,
        )
        # --- 3. スローネス ---
        s = self._to_slowness(v0, nx=grid["nx"], nz=grid["nz"])

        # --- src/rec の grid node 解決 ---
        src_node = self._project_to_grid_node(source_x, 0.0, grid)
        rec_nodes = np.array(
            [self._project_to_grid_node(d, 0.0, grid) for d in distance],
            dtype=int,
        )

        # --- 4-5. 反復: forward → invert (invalid 受振器は L 行を除外) ---
        # SIRT step_length: inversion_step_length takes precedence if
        # explicitly set by the user (i.e. differs from default 1.0),
        # otherwise fall back to sirt_step.  Since both default to 1.0,
        # the effective step is 1.0 unless the user overrides one.
        effective_sirt_step = sirt_step

        valid_idx = np.where(pick_result.valid_mask)[0]
        rec_nodes_v = rec_nodes[valid_idx]
        picks_v = picks[valid_idx]

        # Regularization operator for LSQR (Tikhonov first-difference)
        n_cells = grid["nx"] * grid["nz"]
        R_op = self._regularization_operator(n_cells, grid["nx"], grid["nz"], lam=inversion_lambda)

        # Compute initial RMS before any inversion iterations
        syn_tt_init, L_init = self._solve_forward(
            slowness=s, src_node=src_node, rec_nodes=rec_nodes_v, grid=grid,
            forward_method=forward_method,  # ── FMM WIRE-THROUGH ──
        )
        residual_init = picks_v - syn_tt_init
        rms_init = float(np.sqrt(np.mean(residual_init ** 2))) if residual_init.size > 0 else 0.0
        v_iter = 1.0 / s
        print(f"[tt-inv] iter=-1 (initial): rms={rms_init:.6f}, "
              f"v_min={v_iter.min():.1f}, v_max={v_iter.max():.1f}, "
              f"v_median={np.median(v_iter):.1f}, "
              f"res_mean={residual_init.mean():.6f}, res_std={residual_init.std():.6f}")

        history: list = []
        prev_rms: Optional[float] = None
        for it in range(int(n_iter)):
            syn_tt_v, L_v = self._solve_forward(
                slowness=s, src_node=src_node, rec_nodes=rec_nodes_v, grid=grid,
                forward_method=forward_method,  # ── FMM WIRE-THROUGH ──
            )
            residual_v = picks_v - syn_tt_v
            residual_norm = float(np.linalg.norm(residual_v)) if residual_v.size > 0 else 0.0
            rms = float(np.sqrt(np.mean(residual_v ** 2))) if residual_v.size > 0 else 0.0

            # Pre-inversion diagnostics
            col_sum = np.asarray(L_v.sum(axis=0)).ravel()
            n_zero_cols = int(np.sum(col_sum == 0))
            if it == 0 or it % 5 == 0:
                print(f"[tt-inv] iter={it} pre-inversion: rms={rms:.6f}, "
                      f"res_mean={residual_v.mean():.6f}, res_std={residual_v.std():.6f}, "
                      f"L_col_min={col_sum.min():.2e}, L_col_max={col_sum.max():.2e}, "
                      f"n_zero_cols={n_zero_cols}")

            # 下層レイカバレッジ診断: 初回反復で下半分 (iz >= nz/2) の L 列和が
            # 極小なら、SPM 初期モデルが地表近傍光線しか描かず下層が不可観測。
            # これは PDF-A page 1604 が指摘する 1 次到達トモグラフィの本質的限界
            # (Test 3a で再現済み)。ユーザに depth_gradient_fallback 増加または
            # delay_time 法への切替を促す。
            if it == 0:
                nz_g = grid["nz"]
                nx_g = grid["nx"]
                nz_half = nz_g // 2
                col_sum_2d = col_sum.reshape(nz_g, nx_g)
                L_lower_half_sum = float(col_sum_2d[nz_half:, :].sum())
                print(f"[tt-inv] iter=0: L_lower_half_sum={L_lower_half_sum:.2f} "
                      f"(iz>={nz_half}, nz={nz_g})")
                if L_lower_half_sum < 1.0:
                    # 行動可能なガイダンスを算出 (アルゴリズムは変更しない;
                    # ここでは表示用に必要展開長を逆算するのみ)。
                    #   線形勾配 V(z) = V_surf + auto_grad·z における
                    #   臨界角 θ_c = arcsin(V_surf / V(z_half)) の戻り offset
                    #   X = 2·z_half / tan(θ_c) が、深度 z_half に到達する
                    #   diving ray の最小必要展開長 (Snell / 屈折ジオメトリ)。
                    spread = float(np.max(np.abs(distance - source_x)))
                    z_half = (nz_g * grid["dz"]) / 2.0
                    V_surf = float((1.0 / s)[0, :].mean())
                    auto_grad, _ = _resolve_depth_gradient(
                        depth_gradient_fallback, 1.0 / s, distance, source_x
                    )
                    V_at_half = V_surf + auto_grad * z_half
                    if auto_grad > 1e-9 and V_at_half > V_surf > 0.0:
                        ratio = V_surf / V_at_half  # < 1
                        critical_angle = float(np.arcsin(ratio))
                        required_spread = (
                            2.0 * z_half / float(np.tan(critical_angle))
                        )
                        required_str = f"{required_spread:.1f} m"
                    else:
                        required_str = (
                            "(計算不能; auto_grad=0 — 勾配を有効化してください)"
                        )
                    warnings.warn(
                        f"下層(iz≥{nz_g // 2})の射線カバレッジが不足しています "
                        f"(L_lower_half={L_lower_half_sum:.2f})。\n"
                        f"  現在の展開長: {spread:.1f} m、"
                        f"深度 {z_half:.1f}m を照射するには "
                        f"理論上 {required_str} 以上の展開長が必要です。\n"
                        f"  対策: (1) 展開長を拡大する、"
                        f"(2) depth_gradient_fallback="
                        f"{auto_grad * 2.5:.0f} 以上に設定する、"
                        f"(3) 後続のFWIまたは反射波トモグラフィで補完する。\n"
                        f"  参照: PDF-A (Bai et al. 2010) page 1604, "
                        f"射線密度と分解能の関係",
                        stacklevel=2,
                    )

            s, hstep = self._invert(
                slowness=s,
                L=L_v,
                residual=residual_v,
                method=inversion_method,
                grid=grid,
                sirt_step=effective_sirt_step,
                lsqr_damp=lsqr_damp,
                inversion_damping=inversion_damping,
                R_op=R_op,
            )
            s = self._smooth_slowness(s, sigma=smooth_sigma)
            s = self._clip_slowness(s, s_min=s_min, s_max=s_max)
            s = self._enforce_positivity(s)

            # Post-inversion diagnostics
            v_iter = 1.0 / s
            hstep["residual_norm"] = residual_norm
            history.append({"iter": it, "rms": rms, **hstep})

            ds_norm_val = hstep.get("ds_norm", 0.0)
            if it == 0 or it % 5 == 0:
                print(f"[tt-inv] iter={it} post-update: "
                      f"ds_norm={ds_norm_val:.6f}, "
                      f"v_min={v_iter.min():.1f}, v_max={v_iter.max():.1f}, "
                      f"v_median={np.median(v_iter):.1f}")

            # Early stopping: relative RMS improvement
            if inversion_tol > 0 and prev_rms is not None and prev_rms > 0:
                relative_improvement = abs(prev_rms - rms) / max(prev_rms, 1e-30)
                if relative_improvement < inversion_tol:
                    print(f"[tt-inv] Early stop at iter={it}: "
                          f"relative_rms_improvement={relative_improvement:.2e} < tol={inversion_tol:.2e}")
                    break
            # Early stopping: model update norm
            if model_update_tol > 0:
                if ds_norm_val < model_update_tol:
                    print(f"[tt-inv] Early stop at iter={it}: "
                          f"ds_norm={ds_norm_val:.2e} < tol={model_update_tol:.2e}")
                    break
            prev_rms = rms

        # --- 収束後の合成走時を全受振器分計算 (互換のため (n_x,) を返す) ---
        syn_tt_full, _ = self._solve_forward(
            slowness=s, src_node=src_node, rec_nodes=rec_nodes, grid=grid,
            forward_method=forward_method,  # ── FMM WIRE-THROUGH ──
        )
        v_final = 1.0 / s

        # --- ray paths 収集 (opt-in; PLAN_RAY.md Slice A 採用案 C) ---
        # _solve_forward / _solve_spm_2d 本体は変更せず、収束後に 1 回追加呼び出し。
        # _solve_spm_2d 内部の x_grid = arange(nx)*dx 系に合わせるため、
        # source_x / distance から grid["x0"] を引いた内部座標で渡す。
        ray_paths: Optional[list] = None
        if store_ray_paths:
            _x0 = float(grid["x0"])
            self._solve_spm_2d(
                slowness_model=s,
                source_x=float(source_x) - _x0,
                receiver_positions=np.asarray(distance, dtype=float) - _x0,
                dx=float(grid["dx"]),
                dz=float(grid["dz"]),
                topo_z=None,
                n_secondary=2,
                return_rays=True,
            )
            ray_paths = list(self._last_spm_rays)
            self.tt_ray_paths = ray_paths

        # --- self への格納 (既存 mixin の慣行) ---
        self.tt_picks = picks
        self.tt_initial_velocity = v0
        self.tt_velocity_model = v_final
        self.tt_slowness_model = s
        self.tt_synthetic_tt = syn_tt_full
        self.tt_grid = grid
        self.tt_pick_result = pick_result
        self.tt_initial_diagnostics = init_diag
        self.tt_elevation_correction = elevation_info
        if return_history:
            self.tt_history = history

        result = {
            "velocity_model": v_final,
            "slowness_model": s,
            "picks": picks,
            "synthetic_tt": syn_tt_full,
            "initial_velocity": v0,
            "grid": grid,
            "pick_result": pick_result,
            "pick_qc": pick_result.qc,
            "initial_model_diagnostics": init_diag,
            "elevation_correction": elevation_info,
        }
        if return_history:
            result["history"] = history
        if store_ray_paths:
            result["ray_paths"] = ray_paths

        # --- 描画エントリポイント ---
        if show or (save_name is not None and str(save_name).strip()):
            self.traveltime_tomo_image(
                result=result,
                axis=axis,
                show=show,
                save_name=save_name,
                **plot_kwargs,
            )

        return result

    # =========================================================
    # バリデーション
    # =========================================================
    @staticmethod
    def _validate_modes(picking_mode: str, initial_method: str, inversion_method: str) -> None:
        if picking_mode not in _PICKING_MODES:
            raise ValueError(
                f"picking_mode は {_PICKING_MODES} のいずれか (got {picking_mode!r})"
            )
        if initial_method not in _INITIAL_METHODS:
            raise ValueError(
                f"initial_method は {_INITIAL_METHODS} のいずれか (got {initial_method!r})"
            )
        if inversion_method not in _INVERSION_METHODS:
            raise ValueError(
                f"inversion_method は {_INVERSION_METHODS} のいずれか (got {inversion_method!r})"
            )

    # =========================================================
    # 共通: 軸データ解決 / グリッド構築
    # =========================================================
    def _resolve_axis_data(self, axis: str) -> np.ndarray:
        """
        探査軸 ('x','y','z') に応じた波形 2D 配列を返す共通入口。

        Returns
        -------
        U : ndarray, shape=(n_x, n_t)
        """
        if hasattr(self, "getax_analysis"):
            U, _ = self.getax_analysis(axis)
        elif hasattr(self, axis):
            U = getattr(self, axis)
        else:
            raise AttributeError(
                f"self.{axis} と getax_analysis のいずれも見つかりません。"
            )
        return np.asarray(U, dtype=float)

    @staticmethod
    def _make_grid(
        nx: int,
        nz: int,
        dx: Optional[float],
        dz: Optional[float],
        distance: np.ndarray,
        source_x: float,
    ) -> dict:
        """
        2D グリッド (cell-centered) を構築。

        Returns
        -------
        grid : dict
            'nx','nz','dx','dz','x0','z0','x_centers','z_centers','line_length'
        """
        if nx < 2 or nz < 2:
            raise ValueError(f"nx, nz は 2 以上 (got nx={nx}, nz={nz})")
        x_min = float(min(distance.min(), source_x))
        x_max = float(max(distance.max(), source_x))
        line_length = max(x_max - x_min, 1.0)

        if dx is None:
            dx = line_length / max(nx - 1, 1)
        if dz is None:
            dz = dx
        dx = float(dx)
        dz = float(dz)
        if dx <= 0 or dz <= 0:
            raise ValueError(f"dx, dz は正値 (got dx={dx}, dz={dz})")

        x0 = x_min
        z0 = 0.0
        x_centers = x0 + np.arange(nx) * dx
        z_centers = z0 + np.arange(nz) * dz

        return {
            "nx": int(nx),
            "nz": int(nz),
            "dx": dx,
            "dz": dz,
            "x0": x0,
            "z0": z0,
            "x_centers": x_centers,
            "z_centers": z_centers,
            "line_length": line_length,
        }

    @staticmethod
    def _project_to_grid_node(x_phys: float, z_phys: float, grid: dict) -> int:
        """
        物理座標 (x, z) を最近接グリッドセルのノード ID (= iz*nx + ix) に変換。
        """
        nx = grid["nx"]
        nz = grid["nz"]
        ix = int(round((x_phys - grid["x0"]) / grid["dx"]))
        iz = int(round((z_phys - grid["z0"]) / grid["dz"]))
        ix = max(0, min(nx - 1, ix))
        iz = max(0, min(nz - 1, iz))
        return iz * nx + ix

    # =========================================================
    # 1. ピッキング
    # =========================================================
    # 困ったら以下文献の理論に従うこと:
    #   - Allen (1978) STA/LTA の起源
    #   - Withers et al. (1998) STA/LTA 比較
    #   - Saragiotis et al. (2002) PAI-S/K (高次統計量)
    #   - VanDecar & Crosson (1990) MCCC (multi-channel cross-correlation)
    #   - Sabbione & Velis (2010) automatic first-break picking review
    #   - Zhu & Beroza (2019) PhaseNet (DL ピッカー、本 MVP では未実装)
    def _pick_first_arrivals(
        self,
        U: np.ndarray,
        picking_mode: str,
        fs: float,
        *,
        threshold_ratio: float = 0.1,
        noise_window_ratio: float = 0.05,
        min_duration_samples: int = 3,
        min_arrival_time: Optional[float] = None,
        manual_picks: Optional[np.ndarray] = None,
        manual_pick_indices: Optional[np.ndarray] = None,
        manual_pick_times: Optional[np.ndarray] = None,
        correlation_template: Optional[np.ndarray] = None,
        correlation_template_index: Optional[int] = None,
        min_corr_peak: float = 0.3,
    ) -> "FirstBreakPickResult":
        """
        各 trace の初動走時を picking し ``FirstBreakPickResult`` を返す。

        Parameters
        ----------
        U : ndarray, shape=(n_x, n_t)
        picking_mode : str
        fs : float
        その他: orchestrator から伝搬する picking robustness 拡張パラメータ。

        Returns
        -------
        FirstBreakPickResult
            - pick_indices : (n_x,) int (-1 = invalid)
            - pick_times   : (n_x,) float [s] (NaN = invalid)
            - quality      : (n_x,) float
            - valid_mask   : (n_x,) bool
            - method       : str
            - qc           : list[dict] (per trace)
        """
        if picking_mode == "manual":
            return self._pick_manual(
                U,
                fs,
                manual_picks=manual_picks,
                manual_pick_indices=manual_pick_indices,
                manual_pick_times=manual_pick_times,
            )
        if picking_mode == "energy_threshold":
            return self._pick_energy_threshold(
                U,
                fs,
                threshold_ratio=threshold_ratio,
                noise_window_ratio=noise_window_ratio,
                min_duration_samples=min_duration_samples,
                min_arrival_time=min_arrival_time,
            )
        if picking_mode == "correlation":
            return self._pick_correlation(
                U,
                fs,
                threshold_ratio=threshold_ratio,
                noise_window_ratio=noise_window_ratio,
                min_duration_samples=min_duration_samples,
                min_arrival_time=min_arrival_time,
                template=correlation_template,
                template_index=correlation_template_index,
                min_corr_peak=min_corr_peak,
            )
        if picking_mode == "deep_learning":
            return self._pick_deep_learning(U, fs=fs)
        if picking_mode == "aic":
            return self._pick_aic(
                U,
                fs,
                min_arrival_time=min_arrival_time,
            )
        raise ValueError(f"unknown picking_mode: {picking_mode!r}")

    # ---------------------------------------------------------
    # 1-a. manual mode (外部入力)
    # ---------------------------------------------------------
    def _pick_manual(
        self,
        U: np.ndarray,
        fs: float,
        *,
        manual_picks: Optional[np.ndarray] = None,
        manual_pick_indices: Optional[np.ndarray] = None,
        manual_pick_times: Optional[np.ndarray] = None,
    ) -> "FirstBreakPickResult":
        """
        手動初動ピックの採用。

        いずれかの手動 pick 引数が与えられた場合は、次の優先順位で 1 つだけを
        採用する (採用されなかった入力は無視する)::

            manual_picks > manual_pick_times > manual_pick_indices

        複数が同時に与えられた場合は、その旨を ``print`` で明示したうえで上記
        優先順位に従う。すべて未指定 (None) の場合は、対話的 GUI を起動して各
        trace の初動時刻をクリックで収集する (``manual_pick_gui`` 経由)。

        いずれの経路でも、最終的に ``(pick_indices, pick_times)`` の 1 つの内部
        表現に正規化してから ``FirstBreakPickResult`` を構築する。

        Parameters
        ----------
        U : ndarray, shape=(n_x, n_t)
        manual_picks        : ndarray, shape=(n_x,), optional
            integer dtype はサンプル番号、float dtype は秒。最優先。
        manual_pick_times   : ndarray, shape=(n_x,), float [s], optional
        manual_pick_indices : ndarray, shape=(n_x,), int, optional
            最低優先。
        """
        n_x, n_t = U.shape

        # 与えられた入力を優先順位順 (picks > times > indices) に列挙する。
        provided = []
        if manual_picks is not None:
            provided.append("manual_picks")
        if manual_pick_times is not None:
            provided.append("manual_pick_times")
        if manual_pick_indices is not None:
            provided.append("manual_pick_indices")
        if len(provided) >= 2:
            print(
                f"[manual pick] 複数の手動ピック入力が指定されました: {provided}。"
                "優先順位 manual_picks > manual_pick_times > manual_pick_indices "
                f"に従い '{provided[0]}' を採用し、{provided[1:]} は無視します。"
            )

        if manual_picks is not None:
            arr = np.asarray(manual_picks)
            if arr.shape[0] != n_x:
                raise ValueError(
                    f"manual_picks 長 {arr.shape[0]} が trace 数 {n_x} と不一致"
                )
            if np.issubdtype(arr.dtype, np.integer):
                pick_indices = arr.astype(int, copy=False)
                pick_times = pick_indices.astype(float) / float(fs)
            else:
                pick_times = arr.astype(float)
                pick_indices = np.where(
                    np.isfinite(pick_times),
                    np.round(pick_times * fs).astype(int),
                    -1,
                )
        elif manual_pick_times is not None:
            arr = np.asarray(manual_pick_times, dtype=float)
            if arr.shape[0] != n_x:
                raise ValueError(
                    f"manual_pick_times 長 {arr.shape[0]} が trace 数 {n_x} と不一致"
                )
            pick_times = arr.copy()
            pick_indices = np.where(
                np.isfinite(arr), np.round(arr * fs).astype(int), -1
            )
        elif manual_pick_indices is not None:
            arr = np.asarray(manual_pick_indices)
            if arr.shape[0] != n_x:
                raise ValueError(
                    f"manual_pick_indices 長 {arr.shape[0]} が trace 数 {n_x} と不一致"
                )
            pick_indices = arr.astype(int, copy=False)
            pick_times = pick_indices.astype(float) / float(fs)
        else:
            # 全引数 None: 対話的 GUI でピッキング (wrapper → plotter → backend)。
            pick_times = np.asarray(
                self.manual_pick_gui(
                    U, fs, distance=getattr(self, "distance", None)
                ),
                dtype=float,
            )
            if pick_times.shape[0] != n_x:
                raise ValueError(
                    f"GUI が返した pick 長 {pick_times.shape[0]} が "
                    f"trace 数 {n_x} と不一致"
                )
            pick_indices = np.where(
                np.isfinite(pick_times),
                np.round(pick_times * fs).astype(int),
                -1,
            ).astype(int)

        valid_mask = (
            (pick_indices >= 0)
            & (pick_indices < n_t)
            & np.isfinite(pick_times)
        )
        pick_indices = np.where(valid_mask, pick_indices, -1).astype(int)
        pick_times = np.where(valid_mask, pick_times, np.nan)
        quality = np.where(valid_mask, 1.0, 0.0)
        qc = [
            {
                "source": "manual",
                "valid": bool(valid_mask[k]),
            }
            for k in range(n_x)
        ]
        return FirstBreakPickResult(
            pick_indices=pick_indices,
            pick_times=pick_times,
            quality=quality,
            valid_mask=valid_mask,
            method="manual",
            qc=qc,
        )

    # ---------------------------------------------------------
    # 1-b. energy_threshold (envelope + MAD ノイズ床 + 連続超過)
    # ---------------------------------------------------------
    @staticmethod
    def _pick_energy_threshold(
        U: np.ndarray,
        fs: float,
        *,
        threshold_ratio: float = 0.1,
        noise_window_ratio: float = 0.05,
        min_duration_samples: int = 3,
        min_arrival_time: Optional[float] = None,
        snr_floor: float = 3.0,
    ) -> "FirstBreakPickResult":
        """
        包絡線ベースの first-break picking。

        - Hilbert 解析信号の絶対値で包絡線を作る (生波形より頑健)。
        - 冒頭 ``noise_window_ratio`` 区間の MAD でノイズ床を推定。
        - trigger = max(threshold_ratio * peak_envelope, snr_floor * noise_level)
        - trigger 超え状態が ``min_duration_samples`` 連続するサンプル位置を pick。
        - ``min_arrival_time`` より前は採らない。
        - 失敗時は pick_indices=-1, pick_times=NaN, valid_mask=False。

        References
        ----------
        - Allen (1978) / Withers et al. (1998): STA/LTA の継続条件思想。
        - Sabbione & Velis (2010): complex-trace ベース picking。
        """
        n_x, n_t = U.shape
        pick_indices = np.full(n_x, -1, dtype=int)
        pick_times = np.full(n_x, np.nan, dtype=float)
        quality = np.zeros(n_x, dtype=float)
        valid_mask = np.zeros(n_x, dtype=bool)
        qc_list: List[dict] = []

        min_arrival_idx = (
            max(0, int(round(float(min_arrival_time) * fs)))
            if (min_arrival_time is not None and np.isfinite(min_arrival_time))
            else 0
        )

        for k in range(n_x):
            info = TraveltimeTomography._pick_one_energy(
                trace=U[k],
                fs=fs,
                threshold_ratio=threshold_ratio,
                noise_window_ratio=noise_window_ratio,
                min_duration_samples=min_duration_samples,
                min_arrival_idx=min_arrival_idx,
                snr_floor=snr_floor,
            )
            pick_indices[k] = info["pick_idx"]
            pick_times[k] = info["pick_time"]
            quality[k] = info["quality"]
            valid_mask[k] = info["valid"]
            qc_list.append(info["qc"])

        return FirstBreakPickResult(
            pick_indices=pick_indices,
            pick_times=pick_times,
            quality=quality,
            valid_mask=valid_mask,
            method="energy_threshold",
            qc=qc_list,
        )

    # ---------------------------------------------------------
    # 1-c. correlation (template-vs-all)
    # ---------------------------------------------------------
    @staticmethod
    def _pick_correlation(
        U: np.ndarray,
        fs: float,
        *,
        threshold_ratio: float = 0.1,
        noise_window_ratio: float = 0.05,
        min_duration_samples: int = 3,
        min_arrival_time: Optional[float] = None,
        template: Optional[np.ndarray] = None,
        template_index: Optional[int] = None,
        min_corr_peak: float = 0.3,
    ) -> "FirstBreakPickResult":
        """
        テンプレート vs 全 trace の正規化相互相関 picking。

        - ``template`` 明示指定があればそれを使用。
        - 無ければ ``template_index`` で trace を選択。
        - どちらも無ければ SNR 最大 trace を自動選択。
        - テンプレート自身の絶対 pick は ``_pick_one_energy`` で決定。
        - 各 trace は (テンプレート pick + xcorr lag) で絶対 pick を決める。
        - 累積誤差の出る隣接 chain 法より頑健。

        References
        ----------
        - VanDecar & Crosson (1990) MCCC.
        """
        n_x, n_t = U.shape
        min_arrival_idx = (
            max(0, int(round(float(min_arrival_time) * fs)))
            if (min_arrival_time is not None and np.isfinite(min_arrival_time))
            else 0
        )

        # --- テンプレート決定 ---
        if template is not None:
            tmpl = np.asarray(template, dtype=float).ravel()
            template_trace_idx: Optional[int] = None
            tmpl_min_arrival_idx = 0
        else:
            if template_index is None:
                template_index = TraveltimeTomography._select_best_snr_trace(
                    U, fs=fs, noise_window_ratio=noise_window_ratio
                )
            template_index = int(template_index)
            if template_index < 0 or template_index >= n_x:
                raise ValueError(
                    f"correlation_template_index は [0, {n_x}) (got {template_index})"
                )
            tmpl = np.asarray(U[template_index], dtype=float).ravel()
            template_trace_idx = template_index
            tmpl_min_arrival_idx = min_arrival_idx

        tmpl_info = TraveltimeTomography._pick_one_energy(
            trace=tmpl,
            fs=fs,
            threshold_ratio=threshold_ratio,
            noise_window_ratio=noise_window_ratio,
            min_duration_samples=min_duration_samples,
            min_arrival_idx=tmpl_min_arrival_idx,
        )

        if not tmpl_info["valid"]:
            # テンプレートの絶対 pick が決まらない -> 全 invalid
            qc_list = [
                {
                    "correlation_peak": 0.0,
                    "lag_samples": 0,
                    "template_pick_idx": -1,
                    "template_trace": template_trace_idx,
                    "template_pick_failed": True,
                }
                for _ in range(n_x)
            ]
            return FirstBreakPickResult(
                pick_indices=np.full(n_x, -1, dtype=int),
                pick_times=np.full(n_x, np.nan, dtype=float),
                quality=np.zeros(n_x, dtype=float),
                valid_mask=np.zeros(n_x, dtype=bool),
                method="correlation",
                qc=qc_list,
            )

        template_pick_idx = int(tmpl_info["pick_idx"])

        pick_indices = np.full(n_x, -1, dtype=int)
        pick_times = np.full(n_x, np.nan, dtype=float)
        quality = np.zeros(n_x, dtype=float)
        valid_mask = np.zeros(n_x, dtype=bool)
        qc_list: List[dict] = []

        for k in range(n_x):
            info = TraveltimeTomography._pick_one_correlation(
                trace=U[k],
                template=tmpl,
                fs=fs,
                template_pick_idx=template_pick_idx,
                min_corr_peak=min_corr_peak,
                min_arrival_idx=min_arrival_idx,
            )
            pick_indices[k] = info["pick_idx"]
            pick_times[k] = info["pick_time"]
            quality[k] = info["quality"]
            valid_mask[k] = info["valid"]
            info_qc = dict(info["qc"])
            info_qc["template_pick_idx"] = template_pick_idx
            info_qc["template_trace"] = template_trace_idx
            qc_list.append(info_qc)

        return FirstBreakPickResult(
            pick_indices=pick_indices,
            pick_times=pick_times,
            quality=quality,
            valid_mask=valid_mask,
            method="correlation",
            qc=qc_list,
        )

    @staticmethod
    def _pick_deep_learning(U: np.ndarray, fs: float) -> "FirstBreakPickResult":
        """
        TODO: PhaseNet 等の DL モデルによる picking。
        References:
          - Zhu & Beroza (2019) PhaseNet. Geophys. J. Int. 216.
          - Mousavi et al. (2020) EQTransformer.
        """
        raise NotImplementedError(
            "picking_mode='deep_learning' は未実装。"
            "manual / energy_threshold / correlation を使用してください。"
        )

    # ---------------------------------------------------------
    # 1-e. AIC mode (Akaike Information Criterion)
    # ---------------------------------------------------------
    # 困ったら以下文献の理論に従うこと:
    #   - Tao, B. et al. (2019) Sensors 19, 3406. §3.2 "Automatic Travel
    #     Time Picking". 波形を分割点 k で 2 区間に割り、各区間の分散
    #     σ1², σ2² から AIC(k) = k·ln(σ1²) + (N-k)·ln(σ2²) を計算 (式2)。
    #     σ², μ は式(3)(4)、最小 AIC のインデックスが初動走時 (式5)。
    @staticmethod
    def _pick_aic(
        U: np.ndarray,
        fs: float,
        *,
        min_arrival_time: Optional[float] = None,
    ) -> "FirstBreakPickResult":
        """
        AIC (Akaike Information Criterion) ベースの first-break picking。

        各 trace を分割点 ``k`` で 2 区間 (先頭ノイズ部 / 信号到達後部) に割り、
        AIC(k) = k·ln(σ1²) + (N-k)·ln(σ2²) を全 k で評価し、最小値を与える
        インデックスを初動とする (Tao et al. 2019, Sensors 19, 3406, §3.2, 式2-5)。

        - ``min_arrival_time`` より前は探索しない (探索窓開始)。
        - σ²=0 (定数区間) による ln 発散は微小フロアで回避。
        - 失敗時は pick_indices=-1, pick_times=NaN, valid_mask=False。

        References
        ----------
        - Tao, B., Zhang, C., Qiu, X., et al. (2019) Sensors 19, 3406. §3.2.
        """
        n_x, n_t = U.shape
        pick_indices = np.full(n_x, -1, dtype=int)
        pick_times = np.full(n_x, np.nan, dtype=float)
        quality = np.zeros(n_x, dtype=float)
        valid_mask = np.zeros(n_x, dtype=bool)
        qc_list: List[dict] = []

        min_arrival_idx = (
            max(0, int(round(float(min_arrival_time) * fs)))
            if (min_arrival_time is not None and np.isfinite(min_arrival_time))
            else 0
        )

        for k in range(n_x):
            info = TraveltimeTomography._pick_one_aic(
                trace=U[k],
                fs=fs,
                min_arrival_idx=min_arrival_idx,
            )
            pick_indices[k] = info["pick_idx"]
            pick_times[k] = info["pick_time"]
            quality[k] = info["quality"]
            valid_mask[k] = info["valid"]
            qc_list.append(info["qc"])

        return FirstBreakPickResult(
            pick_indices=pick_indices,
            pick_times=pick_times,
            quality=quality,
            valid_mask=valid_mask,
            method="aic",
            qc=qc_list,
        )

    @staticmethod
    def _pick_one_aic(
        trace: np.ndarray,
        fs: float,
        *,
        min_arrival_idx: int = 0,
    ) -> dict:
        """
        単一 trace の AIC first-break (Tao et al. 2019, §3.2, 式2-5)。

        分割点 k に対し、先頭 k サンプル (区間1) と残り N-k サンプル (区間2) の
        分散 σ1², σ2² から AIC(k) = k·ln(σ1²) + (N-k)·ln(σ2²) を計算する。
        各 k の区間分散は累積和を用いて O(N) で評価する。

        Returns
        -------
        dict : {pick_idx, pick_time, quality, valid, qc}
            pick_idx  : 最小 AIC を与えるサンプル番号 (-1 = invalid)
            pick_time : pick_idx / fs [s] (NaN = invalid)
            quality   : AIC 谷の明瞭度 (0–1)
            valid     : ピック成否
            qc        : 詳細 (aic_min / aic_min_idx / n_samples / search_start)
        """
        x = np.asarray(trace, dtype=float).ravel()
        n_t = x.size
        invalid = {
            "pick_idx": -1,
            "pick_time": np.nan,
            "quality": 0.0,
            "valid": False,
            "qc": {
                "method": "aic",
                "n_samples": int(n_t),
                "search_start": int(min_arrival_idx),
                "aic_min": np.nan,
                "aic_min_idx": -1,
            },
        }

        if n_t < 4 or not np.all(np.isfinite(x)):
            x = np.where(np.isfinite(x), x, 0.0)
            if n_t < 4:
                return invalid
        if not np.any(x != 0.0):
            return invalid

        # 各区間は最低 2 サンプル必要なので k ∈ [2, N-2]。
        # min_arrival_idx で探索窓の開始を制約 (物理的最小到達時刻)。
        k_start = max(2, int(min_arrival_idx))
        k_end = n_t - 2  # inclusive
        if k_start > k_end:
            return invalid

        # 累積和による区間分散 (式3,4)。
        #   region1 = x[0:k] (k サンプル), region2 = x[k:N] (N-k サンプル)
        # var(region) = E[x²] - (E[x])²  を不偏 (n-1) に補正して算出。
        cumsum = np.concatenate(([0.0], np.cumsum(x)))
        cumsum2 = np.concatenate(([0.0], np.cumsum(x * x)))
        eps = 1e-30  # σ²=0 (定数区間) による ln 発散の回避フロア

        k = np.arange(k_start, k_end + 1, dtype=float)
        n1 = k
        n2 = n_t - k

        s1 = cumsum[k_start:k_end + 1]                # Σ x[0:k]
        ss1 = cumsum2[k_start:k_end + 1]              # Σ x²[0:k]
        s2 = cumsum[-1] - s1                          # Σ x[k:N]
        ss2 = cumsum2[-1] - ss1                       # Σ x²[k:N]

        # 不偏分散 (分母 n-1)。n>=2 が保証されているので 0 除算なし。
        var1 = (ss1 - (s1 * s1) / n1) / (n1 - 1.0)
        var2 = (ss2 - (s2 * s2) / n2) / (n2 - 1.0)
        var1 = np.maximum(var1, eps)
        var2 = np.maximum(var2, eps)

        aic = n1 * np.log(var1) + n2 * np.log(var2)   # 式(2)

        if not np.any(np.isfinite(aic)):
            return invalid
        aic = np.where(np.isfinite(aic), aic, np.inf)

        local_argmin = int(np.argmin(aic))            # 式(5)
        pick_idx = k_start + local_argmin
        aic_min = float(aic[local_argmin])

        finite_aic = aic[np.isfinite(aic)]
        aic_max = float(finite_aic.max()) if finite_aic.size else aic_min
        # 谷の明瞭度: (AIC_max - AIC_min) / (|AIC_max| + eps) を [0,1] にクリップ。
        contrast = (aic_max - aic_min) / (abs(aic_max) + 1e-12)
        quality = float(np.clip(contrast, 0.0, 1.0))

        # AIC 曲線が平坦 (谷が無い) 場合は初動が判別できないため invalid。
        # 定数 trace など σ1²=σ2²=eps で AIC が一定になるケースを除外する。
        if not np.isfinite(contrast) or contrast <= 1e-9:
            return invalid

        return {
            "pick_idx": int(pick_idx),
            "pick_time": float(pick_idx) / fs,
            "quality": quality,
            "valid": True,
            "qc": {
                "method": "aic",
                "n_samples": int(n_t),
                "search_start": int(k_start),
                "aic_min": aic_min,
                "aic_min_idx": int(pick_idx),
            },
        }

    # ---------------------------------------------------------
    # 1-x. picking 補助 (純粋関数, テスト容易性のため切り出し)
    # ---------------------------------------------------------
    @staticmethod
    def _compute_envelope(trace: np.ndarray) -> np.ndarray:
        """
        Hilbert 解析信号の絶対値による包絡線。

        Parameters
        ----------
        trace : ndarray, shape=(n_t,)

        Returns
        -------
        env : ndarray, shape=(n_t,)
            非負値。NaN/Inf は 0 に倒した上で計算する。
        """
        if trace is None:
            return np.zeros(0, dtype=float)
        t = np.asarray(trace, dtype=float).ravel()
        if t.size == 0:
            return np.zeros(0, dtype=float)
        if not np.all(np.isfinite(t)):
            t = np.where(np.isfinite(t), t, 0.0)
        if not np.any(t != 0.0):
            return np.zeros_like(t)
        env = np.abs(signal.hilbert(t))
        if not np.all(np.isfinite(env)):
            env = np.where(np.isfinite(env), env, 0.0)
        return env

    @staticmethod
    def _estimate_noise_level(
        trace_or_envelope: np.ndarray, window_ratio: float = 0.05
    ) -> float:
        """
        冒頭 ``window_ratio`` 区間の MAD ベースのノイズ床 σ_n 推定。

        σ_n ≈ MAD / 0.6745 (正規分布仮定; Saragiotis 2002 系で使われる頑健推定)。

        Parameters
        ----------
        trace_or_envelope : ndarray, shape=(n_t,)
        window_ratio : float
            0 < r ≤ 1 を期待。範囲外は内部でクリップ。

        Returns
        -------
        noise_level : float (>= 0)
        """
        x = np.asarray(trace_or_envelope, dtype=float).ravel()
        n = x.size
        if n == 0:
            return 0.0
        r = float(window_ratio)
        if not np.isfinite(r) or r <= 0:
            r = 0.05
        if r > 1.0:
            r = 1.0
        nw = max(1, int(round(r * n)))
        head = x[:nw]
        head = head[np.isfinite(head)]
        if head.size == 0:
            return 0.0
        med = float(np.median(head))
        mad = float(np.median(np.abs(head - med)))
        if mad > 0:
            return float(mad / 0.6745)
        std = float(np.std(head))
        return float(std) if np.isfinite(std) else 0.0

    @staticmethod
    def _estimate_snr(
        trace_or_envelope: np.ndarray, noise_level: float
    ) -> float:
        """
        peak / noise ベースの簡易 SNR 推定。

        noise_level==0 のときはピークが 0 でないなら inf、そうでなければ 0。
        """
        x = np.asarray(trace_or_envelope, dtype=float).ravel()
        if x.size == 0:
            return 0.0
        peak = float(np.max(np.abs(x)))
        if not np.isfinite(peak):
            return 0.0
        if noise_level <= 0 or not np.isfinite(noise_level):
            return float("inf") if peak > 0 else 0.0
        return float(peak / noise_level)

    @staticmethod
    def _normalized_xcorr_peak(
        a: np.ndarray, b: np.ndarray
    ) -> Tuple[int, float]:
        """
        2 信号間の正規化相互相関のピーク lag と正規化ピーク値を返す。

        Returns
        -------
        lag : int   (a が b に対して進む符号)
        peak: float (∈ [-1, 1] の正規化ピーク; ノルム 0 なら 0.0)
        """
        x = np.asarray(a, dtype=float).ravel()
        y = np.asarray(b, dtype=float).ravel()
        if x.size == 0 or y.size == 0:
            return 0, 0.0
        if not np.all(np.isfinite(x)):
            x = np.where(np.isfinite(x), x, 0.0)
        if not np.all(np.isfinite(y)):
            y = np.where(np.isfinite(y), y, 0.0)
        nx = float(np.linalg.norm(x))
        ny = float(np.linalg.norm(y))
        if nx == 0.0 or ny == 0.0:
            return 0, 0.0
        corr = signal.correlate(x, y, mode="full")
        lags = signal.correlation_lags(x.size, y.size, mode="full")
        idx = int(np.argmax(corr))
        peak = float(corr[idx]) / (nx * ny)
        return int(lags[idx]), float(peak)

    @staticmethod
    def _select_best_snr_trace(
        U: np.ndarray, fs: float, noise_window_ratio: float = 0.05
    ) -> int:
        """
        SNR 最大の trace の index を返す (correlation のテンプレート自動選択用)。
        """
        n_x = U.shape[0]
        snrs = np.zeros(n_x, dtype=float)
        for k in range(n_x):
            env = TraveltimeTomography._compute_envelope(U[k])
            noise = TraveltimeTomography._estimate_noise_level(
                env, window_ratio=noise_window_ratio
            )
            snrs[k] = TraveltimeTomography._estimate_snr(env, noise)
        snrs = np.where(np.isfinite(snrs), snrs, -np.inf)
        if not np.any(snrs > -np.inf):
            return 0
        return int(np.argmax(snrs))

    @staticmethod
    def _pick_one_energy(
        trace: np.ndarray,
        fs: float,
        *,
        threshold_ratio: float = 0.1,
        noise_window_ratio: float = 0.05,
        min_duration_samples: int = 3,
        min_arrival_idx: int = 0,
        snr_floor: float = 3.0,
    ) -> dict:
        """
        1 trace に対する envelope + noise-floor + 連続超過 picking (純粋関数)。

        Returns
        -------
        dict
            'pick_idx' : int (-1 = invalid)
            'pick_time': float [s] (NaN = invalid)
            'quality'  : float (envelope[pick]/peak; 0..1)
            'valid'    : bool
            'qc'       : {noise_level, trigger_value, snr, peak_envelope}
        """
        invalid_qc = {
            "noise_level": 0.0,
            "trigger_value": 0.0,
            "snr": 0.0,
            "peak_envelope": 0.0,
        }
        invalid = {
            "pick_idx": -1,
            "pick_time": float("nan"),
            "quality": 0.0,
            "valid": False,
            "qc": dict(invalid_qc),
        }
        if trace is None or fs <= 0 or not np.isfinite(fs):
            return invalid
        t = np.asarray(trace, dtype=float).ravel()
        if t.size == 0 or not np.any(np.isfinite(t)):
            return invalid

        env = TraveltimeTomography._compute_envelope(t)
        if env.size == 0 or not np.any(env > 0):
            return invalid

        peak = float(env.max())
        noise = TraveltimeTomography._estimate_noise_level(
            env, window_ratio=noise_window_ratio
        )
        trigger = max(float(threshold_ratio) * peak, float(snr_floor) * noise)
        snr = TraveltimeTomography._estimate_snr(env, noise)
        qc = {
            "noise_level": float(noise),
            "trigger_value": float(trigger),
            "snr": float(snr) if np.isfinite(snr) else float("inf"),
            "peak_envelope": float(peak),
        }

        if trigger <= 0.0 or not np.isfinite(trigger):
            return {**invalid, "qc": qc}

        above = env > trigger
        mai = max(0, int(min_arrival_idx))
        if mai > 0:
            above[:mai] = False
        if not np.any(above):
            return {**invalid, "qc": qc}

        mds = max(1, int(min_duration_samples))
        if mds <= 1:
            cand = np.where(above)[0]
            pick_idx = int(cand[0]) if cand.size > 0 else -1
        else:
            kernel = np.ones(mds, dtype=int)
            run = np.convolve(above.astype(int), kernel, mode="valid")
            cand = np.where(run >= mds)[0]
            pick_idx = int(cand[0]) if cand.size > 0 else -1

        if pick_idx < 0:
            return {**invalid, "qc": qc}

        quality = float(env[pick_idx] / peak) if peak > 0 else 0.0
        return {
            "pick_idx": int(pick_idx),
            "pick_time": float(pick_idx) / float(fs),
            "quality": quality,
            "valid": True,
            "qc": qc,
        }

    @staticmethod
    def _pick_one_correlation(
        trace: np.ndarray,
        template: np.ndarray,
        fs: float,
        *,
        template_pick_idx: int,
        min_corr_peak: float = 0.3,
        min_arrival_idx: int = 0,
    ) -> dict:
        """
        1 trace を template と相関し (template 絶対 pick + lag) で first break を決定。

        Returns
        -------
        dict
            'pick_idx' : int (-1 = invalid)
            'pick_time': float [s]
            'quality'  : float (正規化相関ピーク値, [-1,1])
            'valid'    : bool
            'qc'       : {correlation_peak, lag_samples}
        """
        invalid_qc = {"correlation_peak": 0.0, "lag_samples": 0}
        invalid = {
            "pick_idx": -1,
            "pick_time": float("nan"),
            "quality": 0.0,
            "valid": False,
            "qc": dict(invalid_qc),
        }
        if trace is None or template is None:
            return invalid
        if fs <= 0 or not np.isfinite(fs):
            return invalid
        x = np.asarray(trace, dtype=float).ravel()
        y = np.asarray(template, dtype=float).ravel()
        if x.size == 0 or y.size == 0:
            return invalid
        if not np.any(np.isfinite(x)) or not np.any(np.isfinite(y)):
            return invalid

        lag, peak = TraveltimeTomography._normalized_xcorr_peak(x, y)
        qc = {"correlation_peak": float(peak), "lag_samples": int(lag)}

        pick_idx = int(template_pick_idx) + int(lag)
        if pick_idx < max(0, int(min_arrival_idx)) or pick_idx >= x.size:
            return {**invalid, "quality": float(peak), "qc": qc}

        valid = bool(peak >= float(min_corr_peak))
        if not valid:
            return {
                "pick_idx": -1,
                "pick_time": float("nan"),
                "quality": float(peak),
                "valid": False,
                "qc": qc,
            }
        return {
            "pick_idx": int(pick_idx),
            "pick_time": float(pick_idx) / float(fs),
            "quality": float(peak),
            "valid": True,
            "qc": qc,
        }

    # =========================================================
    # 2. 初期モデル構築
    # =========================================================
    def _build_initial_velocity_model(
        self,
        picks: np.ndarray,
        distance: np.ndarray,
        source_x: float,
        method: str,
        grid: dict,
        *,
        pick_result: Optional["FirstBreakPickResult"] = None,
        local_window_traces: int = 5,
        surface_smooth_sigma_x: float = 1.5,
        depth_trend_alpha: float = 0.5,
        depth_trend_power: float = 0.7,
        vmin_init: float = 200.0,
        vmax_init: float = 6000.0,
        mad_k: float = 3.0,
        depth_gradient_fallback: Optional[float] = None,
    ) -> Tuple[np.ndarray, "_InitialModelDiagnostics"]:
        """
        初期速度モデル (nz, nx) と診断情報を返す。

        pick_result が与えられれば valid_mask/quality を参照し、
        無ければ picks > 0 / ones で代用する (backward compat)。
        """
        if pick_result is not None:
            vm = pick_result.valid_mask.copy()
            q = pick_result.quality.copy()
        else:
            vm = picks > 0
            q = np.ones_like(picks, dtype=float)

        common_kwargs = dict(
            valid_mask=vm,
            quality=q,
            local_window_traces=local_window_traces,
            surface_smooth_sigma_x=surface_smooth_sigma_x,
            depth_trend_alpha=depth_trend_alpha,
            depth_trend_power=depth_trend_power,
            vmin_init=vmin_init,
            vmax_init=vmax_init,
            mad_k=mad_k,
            depth_gradient_fallback=depth_gradient_fallback,
        )

        if method == "apparent_velocity":
            return self._initial_apparent_velocity(
                picks, distance, source_x, grid, **common_kwargs
            )
        if method == "delay_time":
            return self._initial_delay_time(
                picks, distance, source_x, grid, **common_kwargs
            )
        raise ValueError(f"unknown initial_method: {method!r}")

    def _initial_apparent_velocity(
        self,
        picks: np.ndarray,
        distance: np.ndarray,
        source_x: float,
        grid: dict,
        *,
        valid_mask: np.ndarray,
        quality: np.ndarray,
        local_window_traces: int = 5,
        surface_smooth_sigma_x: float = 1.5,
        depth_trend_alpha: float = 0.5,
        depth_trend_power: float = 0.7,
        vmin_init: float = 200.0,
        vmax_init: float = 6000.0,
        mad_k: float = 3.0,
        depth_gradient_fallback: Optional[float] = None,
    ) -> Tuple[np.ndarray, "_InitialModelDiagnostics"]:
        """
        見かけ速度初期モデル (local weighted regression + MAD rejection + depth gradient)。

        各受振点で局部 dt/dx を加重線形回帰で推定し、v_app(x) = 1/(dt/dx) とする。
        MAD ベース外れ値棄却後、Gaussian 平滑で surface_v(x) を生成し、
        depth trend: v(x,z) = surface_v(x) * (1 + alpha * (z/z_max)^power) で 2D 化。

        不安定時は _initial_smooth_fallback に委譲する。

        References
        ----------
        - White (1989) — slope-derived apparent velocities
        - Sheehan et al. tutorial — initial gradient model practice
        - Gebrande & Miller (1985) — local slope estimation
        """
        notes: list[str] = []
        offset = np.abs(distance - source_x)
        combined = valid_mask & (offset > 0)
        n_valid = int(np.sum(combined))

        nx = grid["nx"]
        z_centers = grid["z_centers"]

        if n_valid < 2:
            notes.append(
                "Fewer than 2 valid receivers for apparent_velocity; "
                "falling back to smooth_fallback"
            )
            v0, diag_fb = self._initial_smooth_fallback(
                picks, distance, source_x, grid,
                vmin_init=vmin_init, vmax_init=vmax_init,
                depth_trend_alpha=depth_trend_alpha,
                depth_trend_power=depth_trend_power,
                fallback_reason="Too few valid receivers for apparent_velocity",
                depth_gradient_fallback=depth_gradient_fallback,
            )
            diag_fb.method = "apparent_velocity (fallback to smooth)"
            diag_fb.fallback_used = True
            diag_fb.notes = notes + diag_fb.notes
            return v0, diag_fb

        d = offset[combined]
        t = picks[combined]
        w = quality[combined]

        slopes, slope_ok = _local_slope_per_receiver(d, t, w, local_window_traces)

        v_app_local = np.full(len(d), np.nan)
        v_app_local[slope_ok] = 1.0 / slopes[slope_ok]

        good = slope_ok & np.isfinite(v_app_local) & (v_app_local > 0)

        if int(np.sum(good)) >= 3:
            outlier_ok = _robust_outlier_mask(v_app_local[good], k=mad_k)
            combined_good = np.zeros(len(d), dtype=bool)
            combined_good[good] = outlier_ok[: int(np.sum(good))]
            good = combined_good

        v_app_local = np.where(good, np.clip(v_app_local, vmin_init, vmax_init), np.nan)

        v_app_full = np.full(len(distance), np.nan)
        valid_indices = np.where(combined)[0]
        v_app_full[valid_indices[good]] = v_app_local[good]

        n_good = int(np.sum(np.isfinite(v_app_full) & (v_app_full > 0)))
        if n_good == 0:
            notes.append(
                "No valid apparent velocities after filtering; "
                "falling back to smooth_fallback"
            )
            v0, diag_fb = self._initial_smooth_fallback(
                picks, distance, source_x, grid,
                vmin_init=vmin_init, vmax_init=vmax_init,
                depth_trend_alpha=depth_trend_alpha,
                depth_trend_power=depth_trend_power,
                fallback_reason="No valid apparent velocities after filtering",
                depth_gradient_fallback=depth_gradient_fallback,
            )
            diag_fb.method = "apparent_velocity (fallback to smooth)"
            diag_fb.fallback_used = True
            diag_fb.notes = notes + diag_fb.notes
            return v0, diag_fb

        sort_idx_global = np.argsort(offset)
        offset_sorted = offset[sort_idx_global]
        v_app_sorted = v_app_full[sort_idx_global]
        valid_sorted = np.isfinite(v_app_sorted) & (v_app_sorted > 0)

        valid_positions = offset_sorted[valid_sorted]
        valid_values = v_app_sorted[valid_sorted]

        surface_v_grid = np.interp(
            grid["x_centers"],
            valid_positions,
            valid_values,
            left=valid_values[0],
            right=valid_values[-1],
        )

        if surface_smooth_sigma_x > 0:
            surface_v_grid = ndimage.gaussian_filter1d(
                surface_v_grid, sigma=surface_smooth_sigma_x, mode="nearest"
            )

        surface_v_grid = np.clip(surface_v_grid, vmin_init, vmax_init)

        v0 = _apply_depth_trend(
            surface_v_grid, z_centers, depth_trend_alpha, depth_trend_power
        )
        # 加算的深度勾配フォールバック: 乗算 depth trend だけでは下層 ray
        # カバレッジが不足する場合 (alpha=0 や surface_v が高速の場合) の
        # 安全網。delta_v(iz) = grad · z_centers[iz] を全列に加算し、
        # 5000 m/s で上限クリップする。
        # depth_gradient_fallback の解決:
        #   - None  → 展開長 spread = max|d - src_x| からの自動算出
        #             z_target = spread/2 (経験則: max 探査深度 ≒ 展開半長)
        #             V_surface = v0[0,:].mean() を表層速度とし、
        #             V_needed = 1.5·V_surface (臨界角 ~42° 相当) に達する
        #             最小勾配 (V_needed - V_surface)/z_target を採用。
        #   - 数値 → そのまま採用 (0.0 を渡せば無効化)。
        # (Test 3a で確認した 1 次到達トモグラフィ初期モデル盲点への対策)
        grad, grad_source = _resolve_depth_gradient(
            depth_gradient_fallback, v0, distance, source_x
        )
        if grad > 0.0:
            depth_offset = grad * z_centers  # (nz,)
            v0 = v0 + depth_offset[:, np.newaxis]
            v0 = np.minimum(v0, 5000.0)
        v0 = np.clip(v0, vmin_init, vmax_init)
        v0 = np.where(np.isfinite(v0), v0, (vmin_init + vmax_init) / 2.0)
        v0 = np.where(v0 > 0, v0, (vmin_init + vmax_init) / 2.0)

        notes.append(
            f"depth_gradient_fallback={grad:.3f} m/s/m ({grad_source})"
        )

        v1_est = (
            float(np.median(v_app_local[good]))
            if np.any(good)
            else (vmin_init + vmax_init) / 2.0
        )
        diag = _InitialModelDiagnostics(
            method="apparent_velocity",
            fallback_used=False,
            n_valid=n_valid,
            v_app_per_x=v_app_full,
            surface_v=surface_v_grid,
            v1_estimate=v1_est,
            v2_estimate=0.0,
            h_estimate=None,
            notes=notes,
        )
        return v0, diag

    def _initial_delay_time(
        self,
        picks: np.ndarray,
        distance: np.ndarray,
        source_x: float,
        grid: dict,
        *,
        valid_mask: np.ndarray,
        quality: np.ndarray,
        local_window_traces: int = 5,
        surface_smooth_sigma_x: float = 1.5,
        depth_trend_alpha: float = 0.5,
        depth_trend_power: float = 0.7,
        vmin_init: float = 200.0,
        vmax_init: float = 6000.0,
        mad_k: float = 3.0,
        depth_gradient_fallback: Optional[float] = None,
    ) -> Tuple[np.ndarray, "_InitialModelDiagnostics"]:
        """
        Approximate delay-time initial model (MVP; NOT strict Palmer/GRM).

        2-segment piecewise linear fit to t-x picks:
          near: t = slope_near * x + int_near  (direct wave, V1 = 1/slope_near)
          far:  t = slope_far  * x + int_far   (refracted wave, V2 = 1/slope_far)
        Estimating V1, V2, tau, crossover x_c by minimising residual over
        multiple x_c candidates.

        Depth estimate:
          h = tau * V1 * V2 / (2 * sqrt(V2^2 - V1^2))

        2D model: v(z) = V1 + (V2 - V1) * 0.5 * (1 + tanh((z - h) / w))
        with smooth tanh transition of width w ~ 3*dz.

        If unstable (V2 <= V1, too few picks, invalid sqrt):
          fallback to apparent_velocity, then to smooth fail-safe.

        References
        ----------
        - Hagedoorn (1959) — delay-time method
        - Palmer (1981) — Generalized Reciprocal Method
        - Lanz, Maurer, Green (1998) — practical refraction tomography
        """
        notes: list[str] = []
        offset = np.abs(distance - source_x)
        combined = valid_mask & (offset > 0)
        n_valid = int(np.sum(combined))

        unstable = False
        V1 = 0.0
        V2 = 0.0
        tau = 0.0
        h = 0.0

        if n_valid < 4:
            notes.append(
                "Too few valid picks for delay_time; falling back to apparent_velocity"
            )
            unstable = True
        else:
            d = offset[combined]
            t = picks[combined]
            w = quality[combined]
            d_range = float(np.max(d) - np.min(d))
            if d_range < 1e-10:
                notes.append(
                    "Offset range too small for two-segment fit; falling back"
                )
                unstable = True
            else:
                n_cand = min(20, max(3, n_valid // 2))
                x_candidates = np.linspace(
                    float(np.min(d)) + 0.1 * d_range,
                    float(np.max(d)) - 0.1 * d_range,
                    n_cand,
                )
                V1, V2, tau, x_c, resid = _fit_two_segment(d, t, w, x_candidates)

                if V1 <= 0 or V2 <= 0:
                    notes.append(
                        "V1 or V2 <= 0 in delay_time fit; falling back"
                    )
                    unstable = True
                elif V2 <= V1:
                    notes.append(
                        "V2 <= V1 in delay_time fit (no refractor); falling back"
                    )
                    unstable = True
                elif tau <= 0:
                    notes.append("tau <= 0 in delay_time fit; falling back")
                    unstable = True
                else:
                    disc = V2 ** 2 - V1 ** 2
                    if disc <= 0:
                        notes.append(
                            "Invalid sqrt in depth estimation; falling back"
                        )
                        unstable = True
                    else:
                        h = tau * V1 * V2 / (2.0 * np.sqrt(disc))

        if unstable:
            v0, diag_fb = self._initial_apparent_velocity(
                picks, distance, source_x, grid,
                valid_mask=valid_mask,
                quality=quality,
                local_window_traces=local_window_traces,
                surface_smooth_sigma_x=surface_smooth_sigma_x,
                depth_trend_alpha=depth_trend_alpha,
                depth_trend_power=depth_trend_power,
                vmin_init=vmin_init,
                vmax_init=vmax_init,
                mad_k=mad_k,
                depth_gradient_fallback=depth_gradient_fallback,
            )
            diag_fb.method = "delay_time (fallback to apparent_velocity)"
            diag_fb.fallback_used = True
            diag_fb.notes = notes + diag_fb.notes
            return v0, diag_fb

        V1 = float(np.clip(V1, vmin_init, vmax_init))
        V2 = float(np.clip(V2, vmin_init, vmax_init))

        z_centers = grid["z_centers"]
        dz = grid["dz"]
        nx = grid["nx"]

        w_transition = max(3.0 * float(dz), 1.0)
        v_z = V1 + (V2 - V1) * 0.5 * (1.0 + np.tanh((z_centers - h) / w_transition))
        v_z = np.clip(v_z, vmin_init, vmax_init)

        v0 = np.tile(v_z[:, None], (1, nx))
        # 加算的深度勾配フォールバック (詳細は _initial_apparent_velocity を参照)。
        # tanh 遷移モデルでは V2 plateau に達した深部より下も V2 のままで
        # 勾配が消えるため、bottom 領域の ray カバレッジ確保に有用。
        grad, grad_source = _resolve_depth_gradient(
            depth_gradient_fallback, v0, distance, source_x
        )
        if grad > 0.0:
            depth_offset = grad * z_centers
            v0 = v0 + depth_offset[:, np.newaxis]
            v0 = np.minimum(v0, 5000.0)
        v0 = np.clip(v0, vmin_init, vmax_init)
        v0 = np.where(np.isfinite(v0), v0, (vmin_init + vmax_init) / 2.0)
        notes.append(
            f"depth_gradient_fallback={grad:.3f} m/s/m ({grad_source})"
        )

        diag = _InitialModelDiagnostics(
            method="delay_time",
            fallback_used=False,
            n_valid=n_valid,
            v_app_per_x=None,
            surface_v=None,
            v1_estimate=float(V1),
            v2_estimate=float(V2),
            h_estimate=float(h),
            notes=notes,
        )
        return v0, diag

    def _initial_smooth_fallback(
        self,
        picks: np.ndarray,
        distance: np.ndarray,
        source_x: float,
        grid: dict,
        *,
        vmin_init: float = 200.0,
        vmax_init: float = 6000.0,
        depth_trend_alpha: float = 0.5,
        depth_trend_power: float = 0.7,
        fallback_reason: str = "",
        depth_gradient_fallback: Optional[float] = None,
    ) -> Tuple[np.ndarray, "_InitialModelDiagnostics"]:
        """
        Smooth fail-safe initial model.

        Estimates global apparent velocity from median valid pick / stable slope proxy,
        builds simple 1D background increasing with depth, tiles across x.
        Falls back to midpoint velocity if no valid picks exist.

        This is intentionally conservative and should only be reached when
        apparent_velocity and delay_time both fail.
        """
        notes: list[str] = []
        if fallback_reason:
            notes.append(fallback_reason)

        offset = np.abs(distance - source_x)
        mask = (offset > 0) & (picks > 0)
        n_valid = int(np.sum(mask))

        if n_valid > 0:
            d = offset[mask]
            t = picks[mask]
            slopes = t / d
            valid_slopes = slopes[np.isfinite(slopes) & (slopes > 0)]
            if valid_slopes.size > 0:
                med_slope = float(np.median(valid_slopes))
                v_app = 1.0 / med_slope if med_slope > 0 else (vmin_init + vmax_init) / 2.0
            else:
                v_app = (vmin_init + vmax_init) / 2.0
                notes.append("No valid slopes for fallback; using midpoint velocity")
        else:
            v_app = (vmin_init + vmax_init) / 2.0
            notes.append("No valid picks; using midpoint velocity")

        v_app = float(np.clip(v_app, vmin_init, vmax_init))

        nx = grid["nx"]
        z_centers = grid["z_centers"]

        surface_v = np.full(nx, v_app)
        v0 = _apply_depth_trend(surface_v, z_centers, depth_trend_alpha, depth_trend_power)
        # 加算的深度勾配フォールバック (詳細は _initial_apparent_velocity を参照)。
        grad, grad_source = _resolve_depth_gradient(
            depth_gradient_fallback, v0, distance, source_x
        )
        if grad > 0.0:
            depth_offset = grad * z_centers
            v0 = v0 + depth_offset[:, np.newaxis]
            v0 = np.minimum(v0, 5000.0)
        v0 = np.clip(v0, vmin_init, vmax_init)
        v0 = np.where(np.isfinite(v0), v0, (vmin_init + vmax_init) / 2.0)
        v0 = np.where(v0 > 0, v0, (vmin_init + vmax_init) / 2.0)

        notes.append(
            f"depth_gradient_fallback={grad:.3f} m/s/m ({grad_source})"
        )

        diag = _InitialModelDiagnostics(
            method="smooth_fallback",
            fallback_used=True,
            n_valid=n_valid,
            v_app_per_x=None,
            surface_v=surface_v,
            v1_estimate=float(v_app),
            v2_estimate=0.0,
            h_estimate=None,
            notes=notes,
        )
        return v0, diag

    # =========================================================
    # 3. スローネスモデル化
    # =========================================================
    @staticmethod
    def _to_slowness(velocity_model: np.ndarray, nx: int, nz: int) -> np.ndarray:
        """
        速度モデル -> スローネスモデル s = 1/v。
        Shape は (nz, nx) を保持する。
        """
        v = np.asarray(velocity_model, dtype=float)
        if v.shape != (nz, nx):
            raise ValueError(
                f"velocity_model の shape は (nz, nx)=({nz},{nx}) を期待 (got {v.shape})"
            )
        if np.any(v <= 0):
            raise ValueError("velocity_model に非正値が含まれます。")
        return 1.0 / v

    # =========================================================
    # 4. 順問題 (Shortest-Path Method, 2D, 地形対応)
    # =========================================================
    # 困ったら以下文献の理論に従う:
    #   - Bai, Huang & Zhao (2010) GJI 183:1596–1612 (PDF-A)
    #       ・Sec.2 / Fig.1: primary (cell-corner) / secondary (cell-edge) ノード構造
    #       ・Sec.3.1 / Eq.(7): t_ij = t_i + D(x_i,x_j) * (s_i + s_j) / 2
    #       ・"forward star" (Moser 1991 流) によるセル内全結合
    #   - Du, Yang, Xu & Liu (2024) GJI 237:1235–1248 (PDF-B)
    #       ・Sec.2.2 / Eq.(9)(10): trapezoid 走時更新則 (slowness 形)
    #       ・Sec.2.2.1 / Fig.1: 地形対応 quadrilateral メッシュと auxiliary node
    #       ・Sec.2.2.3: predecessor 連鎖による ray の自動逆追跡
    #       ・Sec.2.3 / Eq.(12)(13): 経路区間半長を端点ノードに寄与させる Fréchet kernel
    #       ・Sec.5.1 / Eq.(22), Figs.11–13: aux ノード密度で精度が改善する
    # 本実装は 2-D X–Z 断面 (PDF-A Fig.1a に対応) を扱い、地形は列方向の
    # 一様シフトとして取り込む (PDF-B Fig.1 の curved 表面の 2-D 縮約)。

    def _solve_spm_2d(
        self,
        slowness_model: np.ndarray,
        source_x: float,
        receiver_positions: np.ndarray,
        dx: float,
        dz: float,
        topo_z: Optional[np.ndarray] = None,
        n_secondary: int = 2,
        return_rays: bool = False,
    ) -> Tuple[np.ndarray, csr_matrix]:
        """
        2D Shortest-Path Method による初動走時順計算 (地形対応)。

        アルゴリズム
        ------------
        PDF-A Bai et al. (2010) Eq.(7) / Sec.3.1 と
        PDF-B Du et al. (2024) Eq.(9)(10) / Sec.2.2 に基づく。

          - 一次ノード (primary): セル隅、slowness_model 直値を保持。
            PDF-A Sec.2 / Fig.1。
          - 二次ノード (secondary, auxiliary): 各セル辺上に n_secondary 個を
            等間隔配置、両端 primary から線形補間で slowness を取得。
            PDF-A Fig.1, PDF-B Sec.2.2.1 / Fig.1。
          - セル内 forward star: 一セルの境界 (4 隅 + 4·n_secondary 辺ノード)
            を全結合し、辺重み w_ij = D_ij·0.5·(s_i + s_j) とする。
            PDF-A Eq.(7) / PDF-B Eq.(10)。
          - 地形 z(x): primary の z 座標を topo_z[ix] + iz·dz と平行シフト。
            距離は実 (x, z) のユークリッド距離 (PDF-B 1237 行末
            "geodesic … approximated by straight line")。
          - Dijkstra で source → 全ノードの最小走時と predecessor を取得。
            PDF-B Sec.2.2.2。
          - 各受振器から predecessor を遡り、経路区間長の半分を
            両端ノードの寄与とし、secondary ノードは線形補間重みで
            対応する 2 つの primary セルへ按分。PDF-B Eq.(12)(13)。

        Parameters
        ----------
        slowness_model : ndarray, shape=(nz, nx)
            スローネス場 s(z, x) = 1/V(z, x), 単位 s/m。
            slowness_model[0, :] が地表面行、slowness_model[nz-1, :] が最深行。
        source_x : float
            震源 x 座標 [m] (地表面上、z = topo_z[ix_src]) 。
        receiver_positions : ndarray, shape=(n_receivers,)
            受振点 x 座標 [m] (全て地表面上) 。
        dx, dz : float
            グリッド水平 / 鉛直間隔 [m]。
        topo_z : ndarray, shape=(nx,), or None
            地表面標高 z(x) [m]。None の場合は平坦 (z=0)。
        n_secondary : int
            セル辺あたりの secondary ノード数。
            PDF-A Sec.2 / PDF-B Sec.5.1 Eq.(22)、推奨 1–11、本実装の既定は 2。
            計算量 ∝ (4 + 4·n_secondary)^2 / セル。
        return_rays : bool
            True なら ray パスの (x, z) 座標列も返す (現状未使用、互換のため)。

        Returns
        -------
        syn_tt : ndarray, shape=(n_receivers,)
            合成初動走時 [s]。到達不能の場合は 0.0。
        sensitivity_matrix : scipy.sparse.csr_matrix, shape=(n_receivers, nz*nx)
            Fréchet 感度行列 L。L[r, iz*nx+ix] は cell (iz, ix) の slowness に
            対する受振器 r の走時偏微分 [m] (PDF-B Eq.(13) を slowness で書き直し
            δt = Σ |seg|/2 · (δs_i + δs_{i+1}))。
            既存の `_invert_*` 系列との互換のため csr_matrix を返す。

        Notes
        -----
        - 計算量は O((n_primary + n_secondary_total) · log(...)) + O(n_edges)。
          典型 nx=200, nz=100, n_secondary=2 で総ノード ~10⁵、辺 ~10⁶ 程度。
        - PDF-B Sec.5.1 によれば精度は主に n_secondary に依存する
          (要素サイズではない)。9–13 個程度が実用上のバランスと記載。
        """
        # ----- 入力検証 -----
        s = np.asarray(slowness_model, dtype=float)
        if s.ndim != 2:
            raise ValueError(f"slowness_model は 2D (nz, nx) 必須 (got {s.shape})")
        nz, nx = s.shape
        if nz < 2 or nx < 2:
            raise ValueError(f"nz, nx >= 2 必須 (got nz={nz}, nx={nx})")
        if np.any(s <= 0) or not np.all(np.isfinite(s)):
            raise ValueError("slowness_model に非正値 / 非有限値が含まれます")
        if dx <= 0 or dz <= 0:
            raise ValueError(f"dx, dz > 0 必須 (got dx={dx}, dz={dz})")
        n_sec = int(max(0, n_secondary))

        rec_x = np.asarray(receiver_positions, dtype=float).ravel()
        n_rec = rec_x.size

        if topo_z is None:
            z_surf = np.zeros(nx, dtype=float)
        else:
            z_surf = np.asarray(topo_z, dtype=float).ravel()
            if z_surf.shape != (nx,):
                raise ValueError(
                    f"topo_z は shape=(nx,)=({nx},) 必須 (got {z_surf.shape})"
                )

        # ----- A. ノード座標 / slowness テーブル構築 -----
        # PDF-A Fig.1(a): rectangular cells を構成する primary nodes。
        # PDF-B Sec.2.2.1: curved free-surface の column-shift 縮約。
        x_grid = np.arange(nx, dtype=float) * dx  # [m]
        depth = np.arange(nz, dtype=float) * dz   # [m]
        z_grid_2d = z_surf[None, :] + depth[:, None]   # (nz, nx)
        x_grid_2d = np.broadcast_to(x_grid[None, :], (nz, nx))

        n_primary = nz * nx
        n_sec_h = nz * (nx - 1) * n_sec    # 水平辺 secondary
        n_sec_v = (nz - 1) * nx * n_sec    # 鉛直辺 secondary
        n_total = n_primary + n_sec_h + n_sec_v
        base_h = n_primary
        base_v = n_primary + n_sec_h

        pos_x = np.empty(n_total, dtype=float)
        pos_z = np.empty(n_total, dtype=float)
        node_s = np.empty(n_total, dtype=float)

        # primary
        pos_x[:n_primary] = x_grid_2d.ravel()
        pos_z[:n_primary] = z_grid_2d.ravel()
        node_s[:n_primary] = s.ravel()

        # 線形補間係数 f_k = (k+1)/(n_sec+1), k=0..n_sec-1。PDF-A Sec.2 直線補間。
        if n_sec > 0:
            f_k = (np.arange(n_sec, dtype=float) + 1.0) / (n_sec + 1.0)
        else:
            f_k = np.zeros(0, dtype=float)

        # 水平辺 secondary: (iz, ix)–(iz, ix+1), k
        if n_sec_h > 0:
            iz_h = np.repeat(np.arange(nz), (nx - 1) * n_sec)
            ix_h = np.tile(np.repeat(np.arange(nx - 1), n_sec), nz)
            k_h = np.tile(np.arange(n_sec), nz * (nx - 1))
            f_h = f_k[k_h]
            sl_h = base_h + (iz_h * (nx - 1) + ix_h) * n_sec + k_h
            xL = x_grid_2d[iz_h, ix_h]
            xR = x_grid_2d[iz_h, ix_h + 1]
            zL = z_grid_2d[iz_h, ix_h]
            zR = z_grid_2d[iz_h, ix_h + 1]
            sL = s[iz_h, ix_h]
            sR = s[iz_h, ix_h + 1]
            pos_x[sl_h] = (1.0 - f_h) * xL + f_h * xR
            pos_z[sl_h] = (1.0 - f_h) * zL + f_h * zR
            node_s[sl_h] = (1.0 - f_h) * sL + f_h * sR

        # 鉛直辺 secondary: (iz, ix)–(iz+1, ix), k
        if n_sec_v > 0:
            iz_v = np.repeat(np.arange(nz - 1), nx * n_sec)
            ix_v = np.tile(np.repeat(np.arange(nx), n_sec), nz - 1)
            k_v = np.tile(np.arange(n_sec), (nz - 1) * nx)
            f_v = f_k[k_v]
            sl_v = base_v + (iz_v * nx + ix_v) * n_sec + k_v
            xT = x_grid_2d[iz_v, ix_v]
            xB = x_grid_2d[iz_v + 1, ix_v]
            zT = z_grid_2d[iz_v, ix_v]
            zB = z_grid_2d[iz_v + 1, ix_v]
            sT = s[iz_v, ix_v]
            sB = s[iz_v + 1, ix_v]
            pos_x[sl_v] = (1.0 - f_v) * xT + f_v * xB
            pos_z[sl_v] = (1.0 - f_v) * zT + f_v * zB
            node_s[sl_v] = (1.0 - f_v) * sT + f_v * sB

        # ----- B. グラフ構築 (セル内 forward star, PDF-A Sec.3.1) -----
        # 各セル (iz, ix), iz∈[0, nz-2], ix∈[0, nx-2] について
        # 境界ノード = 4 隅 + 4·n_sec 辺ノード を全結合。
        n_cells_ij = (nz - 1) * (nx - 1)
        n_boundary = 4 + 4 * n_sec
        n_pairs_per_cell = n_boundary * (n_boundary - 1) // 2
        n_edges_total = n_cells_ij * n_pairs_per_cell

        # セルごとの境界ノード ID テーブル (n_cells_ij, n_boundary)
        boundary_ids = self._spm_cell_boundary_table(
            nz=nz, nx=nx, n_sec=n_sec, base_h=base_h, base_v=base_v
        )

        # 全 (i<j) 組合せを vectorized で展開
        triu_i, triu_j = np.triu_indices(n_boundary, k=1)
        # 各 (i,j) ペアをセル軸方向に broadcast
        node_a = boundary_ids[:, triu_i].ravel()   # len = n_edges_total
        node_b = boundary_ids[:, triu_j].ravel()
        ax = pos_x[node_a]; az = pos_z[node_a]
        bx = pos_x[node_b]; bz = pos_z[node_b]
        D_ij = np.hypot(bx - ax, bz - az)
        w_ij = D_ij * 0.5 * (node_s[node_a] + node_s[node_b])   # PDF-A Eq.(7)

        # 重複辺 (隣接セルが同じ辺 secondary を共有する) を集約: undirected グラフ
        # としては min weight が支配的なので、scipy.sparse.csgraph.dijkstra は
        # 重複に対し小さい方を使用する (csr 構築時には自動加算されるため、
        # 重複は予め uniq 化しておく)。
        upper = node_a < node_b
        ra = np.where(upper, node_a, node_b)
        rb = np.where(upper, node_b, node_a)
        # 同一 (ra, rb) のうち最小 w を取る
        key = ra.astype(np.int64) * np.int64(n_total) + rb.astype(np.int64)
        order = np.argsort(key, kind="stable")
        key_s = key[order]
        w_s = w_ij[order]
        ra_s = ra[order]
        rb_s = rb[order]
        # group min
        uniq_mask = np.empty_like(key_s, dtype=bool)
        uniq_mask[0] = True
        uniq_mask[1:] = key_s[1:] != key_s[:-1]
        group_id = np.cumsum(uniq_mask) - 1
        n_uniq = int(uniq_mask.sum())
        w_min = np.full(n_uniq, np.inf, dtype=float)
        np.minimum.at(w_min, group_id, w_s)
        ra_u = ra_s[uniq_mask]
        rb_u = rb_s[uniq_mask]

        graph = csr_matrix(
            (w_min, (ra_u, rb_u)), shape=(n_total, n_total), dtype=float
        )

        # ----- C. 震源 / 受振器の primary ノード割り当て -----
        # 地表面行 iz=0 上で最近接 primary に snap。
        def _surface_node(x_val: float) -> int:
            ix = int(round((x_val - x_grid[0]) / dx))
            ix = max(0, min(nx - 1, ix))
            return ix   # iz=0 -> node_id = 0*nx + ix

        src_node = _surface_node(float(source_x))
        rec_nodes = np.array(
            [_surface_node(float(rx)) for rx in rec_x], dtype=np.int64
        )

        # ----- D. Dijkstra (PDF-B Sec.2.2.2) -----
        dist, predecessors = dijkstra(
            csgraph=graph,
            directed=False,
            indices=src_node,
            return_predecessors=True,
        )

        # ----- E. 受振器ごとの逆追跡 + Fréchet kernel 構築 (PDF-B Sec.2.2.3, Eq.13) -----
        # secondary ノードの 2 つの隣接 primary cell への寄与重み。
        # node_id -> [(cell_id, weight), ...]
        # 高速化のため、ノードタイプ別に bulk 処理。
        L = lil_matrix((n_rec, n_primary), dtype=float)
        syn_tt = np.zeros(n_rec, dtype=float)
        rays: List[np.ndarray] = []

        for r, rec in enumerate(rec_nodes):
            rec_i = int(rec)
            tt_r = dist[rec_i]
            if not np.isfinite(tt_r):
                # PDF-B Sec.2.2 では到達不能ノードは "unfinished" 状態。
                # 既存 _solve_forward と互換のため 0.0 を返す (上位で除外想定)。
                syn_tt[r] = 0.0
                if return_rays:
                    rays.append(np.empty((0, 2), dtype=float))
                continue
            syn_tt[r] = float(tt_r)

            node = rec_i
            ray_pts: List[Tuple[float, float]] = [(pos_x[node], pos_z[node])]
            while node != src_node:
                pred = int(predecessors[node])
                if pred < 0:
                    break
                seg_len = float(np.hypot(
                    pos_x[node] - pos_x[pred], pos_z[node] - pos_z[pred]
                ))
                self._spm_attribute_segment(
                    L=L, row=r, node_a=node, node_b=pred,
                    seg_len=seg_len,
                    n_primary=n_primary, base_h=base_h, base_v=base_v,
                    n_sec=n_sec, nx=nx,
                )
                node = pred
                ray_pts.append((pos_x[node], pos_z[node]))
            if return_rays:
                rays.append(np.asarray(ray_pts, dtype=float))

        L_csr = L.tocsr()
        if return_rays:
            # 互換のため (syn_tt, L_csr) のみ返却。rays は属性経由でアクセス可能。
            self._last_spm_rays = rays
        return syn_tt, L_csr

    @staticmethod
    def _spm_cell_boundary_table(
        nz: int, nx: int, n_sec: int, base_h: int, base_v: int
    ) -> np.ndarray:
        """
        各セル (iz, ix) の境界ノード ID を (n_cells, n_boundary) で返す。

        boundary 順序:
          [0] = primary (iz,   ix)         (top-left)
          [1] = primary (iz,   ix+1)       (top-right)
          [2] = primary (iz+1, ix)         (bot-left)
          [3] = primary (iz+1, ix+1)       (bot-right)
          [4 .. 4+n_sec)        = top    horizontal-edge secondaries
          [4+n_sec .. 4+2n_sec) = bottom horizontal-edge secondaries
          [4+2n_sec .. 4+3n_sec) = left   vertical-edge secondaries
          [4+3n_sec .. 4+4n_sec) = right  vertical-edge secondaries

        PDF-A Fig.1(b): rectangular cell with primary corners + edge secondaries。
        """
        n_cells_ij = (nz - 1) * (nx - 1)
        n_boundary = 4 + 4 * n_sec
        tab = np.empty((n_cells_ij, n_boundary), dtype=np.int64)

        iz_grid, ix_grid = np.meshgrid(
            np.arange(nz - 1, dtype=np.int64),
            np.arange(nx - 1, dtype=np.int64),
            indexing="ij",
        )
        iz_f = iz_grid.ravel()
        ix_f = ix_grid.ravel()
        # 4 corners
        tab[:, 0] = iz_f * nx + ix_f
        tab[:, 1] = iz_f * nx + (ix_f + 1)
        tab[:, 2] = (iz_f + 1) * nx + ix_f
        tab[:, 3] = (iz_f + 1) * nx + (ix_f + 1)
        # 4 edge groups
        for k in range(n_sec):
            # top horizontal edge: between (iz, ix) and (iz, ix+1)
            tab[:, 4 + k] = base_h + (iz_f * (nx - 1) + ix_f) * n_sec + k
            # bottom horizontal edge
            tab[:, 4 + n_sec + k] = (
                base_h + ((iz_f + 1) * (nx - 1) + ix_f) * n_sec + k
            )
            # left vertical edge: between (iz, ix) and (iz+1, ix)
            tab[:, 4 + 2 * n_sec + k] = base_v + (iz_f * nx + ix_f) * n_sec + k
            # right vertical edge
            tab[:, 4 + 3 * n_sec + k] = (
                base_v + (iz_f * nx + (ix_f + 1)) * n_sec + k
            )
        return tab

    @staticmethod
    def _spm_attribute_segment(
        L: lil_matrix,
        row: int,
        node_a: int,
        node_b: int,
        seg_len: float,
        n_primary: int,
        base_h: int,
        base_v: int,
        n_sec: int,
        nx: int,
    ) -> None:
        """
        ray 区間 (node_a, node_b) の半長 seg_len/2 を、両端ノードに割り当てる。

        - primary ノードはその場 cell へ寄与 (重み 1)。
        - secondary ノードは PDF-B Eq.(13) の線形補間構造に倣い、
          隣接する 2 つの primary cell に (1-f), f で按分する。
          (内挿された slowness s = (1-f)·s_L + f·s_R に対する偏微分)
        """
        half = 0.5 * seg_len
        for nd in (node_a, node_b):
            if nd < n_primary:
                # primary
                L[row, nd] += half
                continue
            if nd < base_v:
                # horizontal-edge secondary: index = base_h + (iz*(nx-1)+ix)*n_sec + k
                rel = nd - base_h
                k = rel % n_sec
                cell_id = rel // n_sec        # iz*(nx-1) + ix
                iz = cell_id // (nx - 1)
                ix = cell_id % (nx - 1)
                f = (k + 1.0) / (n_sec + 1.0)
                L[row, iz * nx + ix] += (1.0 - f) * half
                L[row, iz * nx + (ix + 1)] += f * half
            else:
                # vertical-edge secondary: index = base_v + (iz*nx+ix)*n_sec + k
                rel = nd - base_v
                k = rel % n_sec
                cell_id = rel // n_sec        # iz*nx + ix
                iz = cell_id // nx
                ix = cell_id % nx
                f = (k + 1.0) / (n_sec + 1.0)
                L[row, iz * nx + ix] += (1.0 - f) * half
                L[row, (iz + 1) * nx + ix] += f * half

    def _solve_forward(
        self,
        slowness: np.ndarray,
        src_node: int,
        rec_nodes: np.ndarray,
        grid: dict,
        forward_method: str = "dijkstra",
    ) -> Tuple[np.ndarray, csr_matrix]:
        """
        現在のスローネスモデル下で順問題を解く (薄いラッパ)。

        実体は ``_solve_spm_2d`` (Bai 2010 ISPM + Du 2024 topographic SPM)
        または ``TraveltimeFmm2D`` (Phase D FMM)。
        既存呼び出し点の (src_node, rec_nodes, grid) 経由 API を維持するため、
        ノード ID → 物理座標へ復元してから委譲する。地形 (topo_z) は本ラッパ
        では None (平坦) 固定; 地形を使う場合は ``_solve_spm_2d`` を直接呼ぶ。

        # [PDF AMBIGUOUS: secondary node default] — PDF-B Sec.5.1 では
        # 9–13 個を推奨するが、本ラッパは既存挙動 (8-connected) に近い
        # 計算コストを保つため n_secondary=1 を採用する。

        Parameters
        ----------
        forward_method : {"dijkstra", "fmm"}, default "dijkstra"
            順問題ソルバ選択。"dijkstra" は既存 SPM/Dijkstra 経路 (デフォルト)、
            "fmm" は ``TraveltimeFmm2D`` を使用する。

        Returns
        -------
        syn_tt : ndarray, shape=(n_rec,)
        L      : scipy.sparse.csr_matrix, shape=(n_rec, nz*nx)
        """
        # ── FMM DISPATCH: inserted by Phase D integration ──
        if forward_method == "fmm":
            return self._solve_forward_fmm(
                slowness=slowness,
                src_node=src_node,
                rec_nodes=rec_nodes,
                grid=grid,
            )

        nx = int(grid["nx"])
        nz = int(grid["nz"])
        dx = float(grid["dx"])
        dz = float(grid["dz"])
        x0 = float(grid["x0"])

        # 物理座標復元 (iz=0 行が地表面)
        def _node_to_x(node_id: int) -> float:
            _, ix = divmod(int(node_id), nx)
            return x0 + ix * dx

        source_x = _node_to_x(src_node)
        rec_x = np.array(
            [_node_to_x(int(r)) for r in rec_nodes], dtype=float
        )

        # 既存 grid["x0"] を起点とした原点合わせのため、source_x / rec_x から
        # x0 を引いて _solve_spm_2d 内部の "x_grid = arange*dx" 系に合わせる。
        syn_tt, L_csr = self._solve_spm_2d(
            slowness_model=slowness,
            source_x=source_x - x0,
            receiver_positions=rec_x - x0,
            dx=dx,
            dz=dz,
            topo_z=None,
            n_secondary=1,
            return_rays=False,
        )
        return syn_tt, L_csr

    # ── FMM DISPATCH: inserted by Phase D integration ──
    def _solve_forward_fmm(
        self,
        slowness: np.ndarray,
        src_node: int,
        rec_nodes: np.ndarray,
        grid: dict,
    ) -> Tuple[np.ndarray, csr_matrix]:
        """
        ``_solve_forward`` の FMM バックエンド実装。

        ``TraveltimeFmm2D`` (Phase A–D) を使って Eikonal 方程式を解き、
        ``sensitivity_matrix`` から Fréchet 行列を組み立てる。

        座標系は ``_solve_spm_2d`` と同じく **相対系** (x0=0 原点) を使用。
        地形がある場合 (``self.z_distance`` が存在) は、受振点座標から
        グリッド列への補間で ``topo_z (nx,)`` を構築して FMM に渡す。

        Parameters
        ----------
        slowness : ndarray, shape=(nz, nx)
            現在のスローネスモデル。
        src_node : int
            震源グリッドノード ID (iz*nx + ix)。
        rec_nodes : ndarray of int, shape=(n_rec,)
            受振点グリッドノード ID。
        grid : dict
            ``_make_grid`` が返すグリッド定義辞書。

        Returns
        -------
        syn_tt : ndarray, shape=(n_rec,)
            合成初動走時 [s]。
        L_csr : scipy.sparse.csr_matrix, shape=(n_rec, nz*nx)
            Fréchet 感度行列 [m]。
        """
        nx = int(grid["nx"])
        nz = int(grid["nz"])
        dx = float(grid["dx"])
        dz = float(grid["dz"])
        x0 = float(grid["x0"])

        # Node ID → relative physical x (SPM と同じ 0 基準座標系)
        def _node_to_rel_x(node_id: int) -> float:
            _, ix = divmod(int(node_id), nx)
            return ix * dx

        source_x_rel = _node_to_rel_x(int(src_node))
        rec_x_rel = np.array(
            [_node_to_rel_x(int(r)) for r in rec_nodes], dtype=float
        )

        # Build topo_z (nx,): self.z_distance (n_rec,) を grid 列に補間。
        # self.distance は絶対座標なので x0 を引いて相対化する。
        z_dist_attr = getattr(self, "z_distance", None)
        if z_dist_attr is not None:
            z_dist = np.asarray(z_dist_attr, dtype=np.float64).ravel()
            dist_rel = np.asarray(self.distance, dtype=float).ravel() - x0
            x_grid_rel = np.arange(nx, dtype=float) * dx
            sort_idx = np.argsort(dist_rel)
            topo_z: Optional[np.ndarray] = np.interp(
                x_grid_rel, dist_rel[sort_idx], z_dist[sort_idx]
            )
            src_ix = max(0, min(nx - 1, int(round(source_x_rel / dx))))
            source_z = float(topo_z[src_ix])
        else:
            topo_z = None
            source_z = 0.0

        # FMM solver: x0=0 (デフォルト), 相対座標を使用
        fmm = TraveltimeFmm2D(
            slowness=slowness,
            dx=dx,
            dz=dz,
            topo_z=topo_z,
            scheme="fmm",
        )
        fmm.compute_traveltime(source_xz=(source_x_rel, source_z))

        # 受振点物理 z: 地形あり → topo 列値, 平坦 → 0.0
        if topo_z is not None:
            rec_ix = np.clip(
                np.round(rec_x_rel / dx).astype(int), 0, nx - 1
            )
            rec_z = topo_z[rec_ix]
        else:
            rec_z = np.zeros(len(rec_x_rel), dtype=float)

        receiver_xz = np.column_stack([rec_x_rel, rec_z])

        syn_tt = fmm.sample_traveltimes(receiver_xz)
        L_csr = fmm.sensitivity_matrix(receiver_xz)

        return syn_tt, L_csr

    @staticmethod
    def _build_adjacency_csr(
        s_flat: np.ndarray, nx: int, nz: int, dx: float, dz: float
    ) -> csr_matrix:
        """
        8-connected グリッド隣接を上三角側のみで構築 (undirected で dijkstra に渡す)。
        Edge weight = edge_length * (s_A + s_B) / 2.
        """
        n_cells = nx * nz
        rows: list = []
        cols: list = []
        data: list = []
        diag_len = float(np.hypot(dx, dz))
        # 4 forward neighbors (cardinal + diagonal)
        neighbors = (
            (0, 1, dx),
            (1, 0, dz),
            (1, 1, diag_len),
            (1, -1, diag_len),
        )
        for iz in range(nz):
            for ix in range(nx):
                a = iz * nx + ix
                for diz, dix, length in neighbors:
                    jz = iz + diz
                    jx = ix + dix
                    if 0 <= jz < nz and 0 <= jx < nx:
                        b = jz * nx + jx
                        w = length * 0.5 * (s_flat[a] + s_flat[b])
                        rows.append(a)
                        cols.append(b)
                        data.append(w)
        return csr_matrix(
            (data, (rows, cols)), shape=(n_cells, n_cells), dtype=float
        )

    @staticmethod
    def _edge_length(node_a: int, node_b: int, nx: int, dx: float, dz: float) -> float:
        """
        2 ノード (8-connected) 間の物理距離 [m]。
        """
        iz_a, ix_a = divmod(node_a, nx)
        iz_b, ix_b = divmod(node_b, nx)
        return float(np.hypot(abs(ix_a - ix_b) * dx, abs(iz_a - iz_b) * dz))

    # =========================================================
    # 5. 反復更新
    # =========================================================
    # 反復トモグラフィは本質的に非線形: 各反復で L を再計算する外面型
    # (outer-loop) アプローチを採用しているが、本実装では L は各反復で
    # 直列に更新する簡易形式。正則化なしでは発散するため、
    # SIRT 正規化 / LSQR ダンピング / 事後 smoothing / 物理クリップ
    # がいずれも正則化として働く (Rawlinson & Sambridge 2003)。
    def _invert(
        self,
        slowness: np.ndarray,
        L: csr_matrix,
        residual: np.ndarray,
        method: str,
        grid: dict,
        sirt_step: float,
        lsqr_damp: float,
        *,
        inversion_damping: float = 1e-6,
        R_op: Optional[csr_matrix] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        L · ds ≈ residual を解いてスローネスを更新。

        Parameters
        ----------
        slowness : ndarray, shape=(nz, nx)
            現在のスローネスモデル。
        L : csr_matrix, shape=(n_rec, nz*nx)
            感度行列。
        residual : ndarray, shape=(n_rec,)
            観測走時と合成走時の差。
        method : str
            'sirt' / 'lsqr' / 'adjoint'
        grid : dict
            グリッド定義。
        sirt_step : float
            SIRT ステップ長係数。
        lsqr_damp : float
            LSQR ダンピングパラメータ。
        inversion_damping : float
            SIRT 微小対角ダンピング (列和分母の安全性確保)。
        R_op : csr_matrix or None
            LSQR 用 Tikhonov 正則化オペレータ。None の場合は恒等行列。

        Returns
        -------
        s_new : ndarray, shape=(nz, nx)
        info  : dict (反復ステップ情報)
        """
        if method == "sirt":
            return self._invert_sirt(
                slowness, L, residual, grid, sirt_step,
                damping=inversion_damping,
            )
        if method == "lsqr":
            return self._invert_lsqr(
                slowness, L, residual, grid, lsqr_damp,
                R_op=R_op,
            )
        if method == "adjoint":
            return self._invert_adjoint(slowness, L, residual, grid)
        raise ValueError(f"unknown inversion_method: {method!r}")

    @staticmethod
    def _invert_sirt(
        slowness: np.ndarray,
        L: csr_matrix,
        residual: np.ndarray,
        grid: dict,
        step: float,
        *,
        damping: float = 1e-6,
    ) -> Tuple[np.ndarray, dict]:
        """
        SIRT (Simultaneous Iterative Reconstruction Technique) with damping.

        Standard SIRT update (Gilbert 1972):
            Δs_j = step * Σ_i L_ij (Δt_i / R_i) / C_j

        where R_i = Σ_j L_ij (row sum) and C_j = Σ_i L_ij (column sum).

        A small diagonal damping term (μ I / C_j) is added to C_j
        for numerical stability when some cells are poorly illuminated
        (near-zero column sums), preventing division-by-near-zero.

        References
        ----------
        - Gilbert (1972) — SIRT reconstruction
        - Nolet (2008) A Breviary of Seismic Tomography — practical SIRT notes
        - Rawlinson & Sambridge (2003) — regularization review for tomography
        """
        nz = grid["nz"]
        nx = grid["nx"]
        row_sum = np.asarray(L.sum(axis=1)).ravel()
        col_sum = np.asarray(L.sum(axis=0)).ravel()
        row_safe = np.where(row_sum > 0, row_sum, 1.0)
        col_safe = np.where(col_sum > damping, col_sum, damping)

        weighted = residual / row_safe
        ds_flat = np.asarray(L.T @ weighted).ravel() / col_safe
        ds_flat = step * ds_flat
        ds = ds_flat.reshape(nz, nx)
        s_new = slowness + ds
        info = {
            "method": "sirt",
            "ds_norm": float(np.linalg.norm(ds_flat)),
            "step_norm": float(step * np.linalg.norm(ds_flat)),
        }
        return s_new, info

    @staticmethod
    def _invert_lsqr(
        slowness: np.ndarray,
        L: csr_matrix,
        residual: np.ndarray,
        grid: dict,
        damp: float,
        *,
        R_op: Optional[csr_matrix] = None,
    ) -> Tuple[np.ndarray, dict]:
        """
        LSQR (Paige & Saunders 1982) with optional Tikhonov regularization.

        Without R_op, solves:
            min ||L ds - residual||² + damp² ||ds||²

        With R_op, augments the system:
            [L      ]        [residual]
            [damp·I ] ds  ≈  [0      ]      (standard damping)
            [λ·R_op  ]        [0      ]      (first-difference regularization if provided)

        Regularization via damp (Tikhonov) and λ·R_op (first-difference smoothing)
        is essential because the tomographic inverse problem is ill-posed.

        References
        ----------
        - Paige & Saunders (1982) LSQR algorithm.
        - Nolet (2008) — damping and smoothing in seismic tomography.
        """
        nz = grid["nz"]
        nx = grid["nx"]
        n_data = L.shape[0]
        n_model = L.shape[1]

        rows_list = [L]
        rhs_list = [residual]
        row_offset = n_data

        # Identity damping block: damp² · I
        if damp > 0:
            I_block = damp * sp_eye(n_model, format="csr")
            rows_list.append(I_block)
            rhs_list.append(np.zeros(n_model, dtype=float))
            row_offset += n_model

        # First-difference regularization block: λ · R_op
        if R_op is not None and R_op.nnz > 0:
            rows_list.append(R_op)
            rhs_list.append(np.zeros(R_op.shape[0], dtype=float))

        if len(rows_list) > 1:
            from scipy.sparse import vstack as sp_vstack
            L_aug = sp_vstack(rows_list, format="csr")
            rhs_aug = np.concatenate(rhs_list)
            result = scipy_lsqr(L_aug, rhs_aug, atol=1e-6, btol=1e-6, iter_lim=200)
        else:
            result = scipy_lsqr(L, residual, atol=1e-6, btol=1e-6, iter_lim=200)

        ds_flat = result[0]
        ds = ds_flat.reshape(nz, nx)
        s_new = slowness + ds
        info = {
            "method": "lsqr",
            "ds_norm": float(np.linalg.norm(ds_flat)),
            "lsqr_istop": int(result[1]),
            "lsqr_itn": int(result[2]),
        }
        return s_new, info

    @staticmethod
    def _invert_adjoint(
        slowness: np.ndarray,
        L: csr_matrix,
        residual: np.ndarray,
        grid: dict,
    ) -> Tuple[np.ndarray, dict]:
        """
        TODO: Adjoint-state method (Taillandier et al. 2009).
        本来は順問題で T(x) を解いた後、随伴方程式で λ(x) を解き、
        勾配 ∇_s J = -2 s λ を得る。dijkstra 近似下では non-trivial。
        """
        raise NotImplementedError(
            "inversion_method='adjoint' は未実装。"
            "MVP では 'sirt' か 'lsqr' を使用してください。"
        )

    # =========================================================
    # 共通ユーティリティ
    # =========================================================
    @staticmethod
    def _smooth_slowness(s: np.ndarray, sigma: float) -> np.ndarray:
        """
        平滑化正則化: スローネスに 2D Gaussian smoothing を適用。
        sigma <= 0 ならスキップ。
        """
        if sigma is None or sigma <= 0:
            return s
        return ndimage.gaussian_filter(s, sigma=float(sigma), mode="nearest")

    @staticmethod
    def _clip_slowness(s: np.ndarray, s_min: float, s_max: float) -> np.ndarray:
        """
        スローネスを物理的妥当範囲にクリップ。
        """
        return np.clip(s, s_min, s_max)

    @staticmethod
    def _enforce_positivity(s: np.ndarray) -> np.ndarray:
        """
        スローネスの厳密正値性を強制。NaN/Inf は安全な中間値に置換。

        逆問題の反復更新で負や非有限値が生じた場合の安全網。
        物理的にスローネスは正値 (s = 1/v > 0) でなければならない。
        極端な更新幅や数値的不安定で生じた異常値を、クリップ前に捕捉する。
        """
        s = np.asarray(s, dtype=float).copy()
        finite_median = float(np.median(s[np.isfinite(s)])) if np.any(np.isfinite(s)) else 0.001
        if not np.all(np.isfinite(s)):
            s = np.where(np.isfinite(s), s, finite_median)
        s = np.maximum(s, 1e-10)
        return s

    @staticmethod
    def _regularization_operator(
        n_model: int, nx: int, nz: int, lam: float = 0.0
    ) -> csr_matrix:
        """
        Tikhonov 一次差分正則化オペレータ R_op を構築。

        R_op は (2*n_model - nx - nz) × n_model のスパース行列で、
        水平方向と鉛直方向の一次差分を表す。
        LSQR では [L; damp·I; λ·R_op] の拡大行列として用いる。

        lam=0 の場合は零行列 (nnz=0) を返し、LSQR 側でスキップされる。

        References
        ----------
        - Nolet (2008), Ch. 14 — regularization operators for tomography
        - Rawlinson & Sambridge (2003) — smoothing in traveltime tomography
        """
        if lam <= 0 or nx < 2 or nz < 2:
            return csr_matrix((0, n_model), dtype=float)
        rows_l: list = []
        cols_l: list = []
        data_l: list = []
        row_idx = 0
        # Horizontal first differences: s[i, j+1] - s[i, j]
        for iz in range(nz):
            for ix in range(nx - 1):
                a = iz * nx + ix
                b = iz * nx + (ix + 1)
                rows_l.append(row_idx)
                cols_l.append(b)
                data_l.append(lam)
                rows_l.append(row_idx)
                cols_l.append(a)
                data_l.append(-lam)
                row_idx += 1
        # Vertical first differences: s[i+1, j] - s[i, j]
        for iz in range(nz - 1):
            for ix in range(nx):
                a = iz * nx + ix
                b = (iz + 1) * nx + ix
                rows_l.append(row_idx)
                cols_l.append(b)
                data_l.append(lam)
                rows_l.append(row_idx)
                cols_l.append(a)
                data_l.append(-lam)
                row_idx += 1
        n_rows = row_idx
        return csr_matrix(
            (data_l, (rows_l, cols_l)), shape=(n_rows, n_model), dtype=float
        )


# =========================================================
# 自己テスト (uniform velocity, flat surface)
# =========================================================
def _test_spm_2d_uniform() -> None:
    """
    一様速度・平坦地表モデルでの自己整合性テスト。

    PDF-A Sec.4 / Bai et al. (2009) Table 1 と同様の検証思想で、
    一様 V0 における初動走時は t = x / V0 で解析解。SPM の誤差は
    PDF-B Eq.(22) の δ_ub ≈ 1/(8 n_sec²) で上界が決まる
    (n_sec = n_secondary+1, n_secondary は単一辺あたりノード数)。
    n_secondary=4 とすれば δ_ub ≈ 1/(8·5²) = 0.5% で、許容 2% は余裕。
    """
    # ダミーインスタンスで static-ish なメソッドを呼ぶため、
    # _solve_spm_2d は self に依存しない (self._spm_attribute_segment 等は
    # @staticmethod もしくは self を使わない)。本テスト用に
    # 簡易ヘルパクラスを定義する。
    class _Dummy:
        _solve_spm_2d = TraveltimeTomography._solve_spm_2d  # type: ignore[attr-defined]
        _spm_cell_boundary_table = staticmethod(
            TraveltimeTomography._spm_cell_boundary_table   # type: ignore[attr-defined]
        )
        _spm_attribute_segment = staticmethod(
            TraveltimeTomography._spm_attribute_segment     # type: ignore[attr-defined]
        )

    V0 = 2000.0   # m/s
    nx, nz = 41, 21
    dx, dz = 5.0, 5.0
    slowness = np.full((nz, nx), 1.0 / V0, dtype=float)

    source_x = 0.0
    rec_x = np.array([20.0, 50.0, 100.0, 150.0, 200.0], dtype=float)
    expected_tt = rec_x / V0

    dummy = _Dummy()
    syn_tt, L = dummy._solve_spm_2d(   # type: ignore[arg-type]
        slowness_model=slowness,
        source_x=source_x,
        receiver_positions=rec_x,
        dx=dx, dz=dz,
        topo_z=None,
        n_secondary=4,
        return_rays=False,
    )
    err = np.abs(syn_tt - expected_tt)
    rel = err / np.maximum(expected_tt, 1e-30)
    print(f"_test_spm_2d_uniform: V0={V0}, n_rec={len(rec_x)}")
    print(f"  expected_tt = {expected_tt}")
    print(f"  syn_tt      = {syn_tt}")
    print(f"  abs err     = {err}")
    print(f"  rel err     = {rel}")
    print(f"  L shape     = {L.shape}, nnz = {L.nnz}")

    # 整合性: L · slowness_flat ≒ syn_tt (PDF-B Eq.(12) の離散形)
    lhs = (L @ slowness.ravel())
    rec_err = np.abs(lhs - syn_tt)
    print(f"  L·s vs syn_tt residual (Eq.12 sanity) = {rec_err}")

    assert np.allclose(syn_tt, expected_tt, rtol=0.02), (
        f"SPM traveltime error > 2%: max_rel={rel.max():.4e}, "
        f"max_abs={err.max():.4e}"
    )
    assert np.allclose(lhs, syn_tt, atol=1e-9), (
        f"L·s sanity failed: max_residual={rec_err.max():.4e}"
    )
    print("_test_spm_2d_uniform: PASSED")


if __name__ == "__main__":
    _test_spm_2d_uniform()