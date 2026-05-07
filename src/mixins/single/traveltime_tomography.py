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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from scipy import ndimage, signal
from scipy.sparse import csr_matrix, eye as sp_eye, lil_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.sparse.linalg import lsqr as scipy_lsqr

from src.plotting.wrapper import PlotterWrapperMixin


_PICKING_MODES = ("manual", "energy_threshold", "correlation", "deep_learning")
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


class TraveltimeTomography(PlotterWrapperMixin):
    """
    初動走時トモグラフィ mixin。

    SingleProcesser に継承され、外部公開メソッド ``traveltime_tomography()`` を
    通じて以下のパイプラインを 1 度に実行する:

        1. 初動走時ピッキング   (manual / energy_threshold / correlation / deep_learning)
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
        # ---- inversion extended parameters (全て default 付き; 既存呼び出しに無影響) ----
        inversion_lambda: float = 0.0,
        inversion_damping: float = 1e-6,
        inversion_step_length: float = 1.0,
        inversion_tol: float = 1e-6,
        model_update_tol: float = 0.0,
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
            'manual' / 'energy_threshold' / 'correlation' / 'deep_learning'
        initial_method : str
            'apparent_velocity' / 'delay_time'
        inversion_method : str
            'sirt' / 'lsqr' / 'adjoint'
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
            (legacy) picking_mode='manual' 時の入力。shape=(n_x,)。
            整数 dtype はサンプル番号、float dtype は秒として解釈する。
            ``manual_pick_indices`` / ``manual_pick_times`` の方が優先される。
        threshold_ratio : float
            picking_mode='energy_threshold' / 'correlation' のしきい値比率
            (包絡線ピーク振幅に対する比)。
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
            picking_mode='manual' でサンプル番号として与える。最優先。
        manual_pick_times : ndarray, optional
            picking_mode='manual' で秒単位で与える。``manual_pick_indices`` 未指定時に使用。
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
            SIRT 微小対角ダンピング (반정化安定化)。default=1e-6。
        inversion_step_length : float
            SIRT ステップ長係数 (sirt_step の代替推奨)。default=1.0。
        inversion_tol : float
            反復停止閾値: 連続反復の RMS 相対改善がこの値未満なら停止。0 で無効(デフォルト)。
        model_update_tol : float
            反復停止閾値: モデル更新ノルムがこの値未満なら停止。0 で無効。

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

        Raises
        ------
        ValueError
            モード文字列が不正、shape 不一致、または必須引数欠落の場合。
        AttributeError
            self.distance / self.source_x / self.fs が未定義の場合。
        NotImplementedError
            未実装モード ('deep_learning' / 'adjoint') を指定した場合。
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

        if not pick_result.valid_mask.any():
            raise ValueError(
                "有効な first-break pick が 1 件も得られませんでした。"
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
            slowness=s, src_node=src_node, rec_nodes=rec_nodes_v, grid=grid
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
                slowness=s, src_node=src_node, rec_nodes=rec_nodes_v, grid=grid
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
            slowness=s, src_node=src_node, rec_nodes=rec_nodes, grid=grid
        )
        v_final = 1.0 / s

        # --- self への格納 (既存 mixin の慣行) ---
        self.tt_picks = picks
        self.tt_initial_velocity = v0
        self.tt_velocity_model = v_final
        self.tt_slowness_model = s
        self.tt_synthetic_tt = syn_tt_full
        self.tt_grid = grid
        self.tt_pick_result = pick_result
        self.tt_initial_diagnostics = init_diag
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
        }
        if return_history:
            result["history"] = history

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
        raise ValueError(f"unknown picking_mode: {picking_mode!r}")

    # ---------------------------------------------------------
    # 1-a. manual mode (外部入力)
    # ---------------------------------------------------------
    @staticmethod
    def _pick_manual(
        U: np.ndarray,
        fs: float,
        *,
        manual_picks: Optional[np.ndarray] = None,
        manual_pick_indices: Optional[np.ndarray] = None,
        manual_pick_times: Optional[np.ndarray] = None,
    ) -> "FirstBreakPickResult":
        """
        外部入力ピックの採用 (GUI 不要)。
        優先順位: manual_pick_indices > manual_pick_times > (legacy) manual_picks。

        Parameters
        ----------
        U : ndarray, shape=(n_x, n_t)
        manual_pick_indices : ndarray, shape=(n_x,), int, optional
        manual_pick_times   : ndarray, shape=(n_x,), float [s], optional
        manual_picks        : ndarray, shape=(n_x,), optional
            integer dtype はサンプル番号、float dtype は秒。
        """
        n_x, n_t = U.shape

        if manual_pick_indices is not None:
            arr = np.asarray(manual_pick_indices)
            if arr.shape[0] != n_x:
                raise ValueError(
                    f"manual_pick_indices 長 {arr.shape[0]} が trace 数 {n_x} と不一致"
                )
            pick_indices = arr.astype(int, copy=False)
            pick_times = pick_indices.astype(float) / float(fs)
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
        elif manual_picks is not None:
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
        else:
            raise NotImplementedError(
                "picking_mode='manual' には manual_pick_indices / manual_pick_times "
                "/ manual_picks のいずれかが必須です (GUI モードは未実装)。"
            )

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
        v0 = np.clip(v0, vmin_init, vmax_init)
        v0 = np.where(np.isfinite(v0), v0, (vmin_init + vmax_init) / 2.0)
        v0 = np.where(v0 > 0, v0, (vmin_init + vmax_init) / 2.0)

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
        v0 = np.where(np.isfinite(v0), v0, (vmin_init + vmax_init) / 2.0)

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
        v0 = np.clip(v0, vmin_init, vmax_init)
        v0 = np.where(np.isfinite(v0), v0, (vmin_init + vmax_init) / 2.0)
        v0 = np.where(v0 > 0, v0, (vmin_init + vmax_init) / 2.0)

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
    # 4. 順問題 (アイコナール近似 = dijkstra)
    # =========================================================
    def _solve_forward(
        self,
        slowness: np.ndarray,
        src_node: int,
        rec_nodes: np.ndarray,
        grid: dict,
    ) -> Tuple[np.ndarray, csr_matrix]:
        """
        現在のスローネスモデル下でアイコナール方程式を最短経路で近似的に解く。

        Returns
        -------
        syn_tt : ndarray, shape=(n_rec,)
            合成走時 [s]
        L : scipy.sparse.csr_matrix, shape=(n_rec, nz*nx)
            感度行列 (各受振器に対する経路通過長 [m])

        Notes
        -----
        将来 FMM (skfmm 等) に差し替える場合は本メソッドを置換するだけで、
        上位 API は変更不要 (合成走時と感度行列の契約のみ守ればよい)。
        """
        nx = grid["nx"]
        nz = grid["nz"]
        dx = grid["dx"]
        dz = grid["dz"]
        s_flat = slowness.ravel()

        graph = self._build_adjacency_csr(s_flat, nx, nz, dx, dz)

        dist, predecessors = dijkstra(
            csgraph=graph,
            directed=False,
            indices=src_node,
            return_predecessors=True,
        )

        n_rec = len(rec_nodes)
        n_cells = nx * nz
        L = lil_matrix((n_rec, n_cells), dtype=float)
        syn_tt = np.zeros(n_rec, dtype=float)

        for i, rec in enumerate(rec_nodes):
            tt_i = dist[int(rec)]
            if not np.isfinite(tt_i):
                syn_tt[i] = 0.0
                continue
            syn_tt[i] = float(tt_i)
            node = int(rec)
            while node != src_node:
                pred = int(predecessors[node])
                if pred < 0:
                    break  # unreachable / source
                seg_len = self._edge_length(node, pred, nx, dx, dz)
                L[i, node] += 0.5 * seg_len
                L[i, pred] += 0.5 * seg_len
                node = pred
        return syn_tt, L.tocsr()

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