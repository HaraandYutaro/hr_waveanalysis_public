"""Quality-control layer for converting dispersion images into reliable
PickedDispersionCurve objects.

Step ?-6 の責務:
  - 分散曲線画像 (res, f_mesh, c_mesh) から QC 済みの PickedDispersionCurve を生成
  - 周波数ごとの品質メタデータ (quality_score, peak_width, flags 等) を提供
  - WeightedL2Misfit で使用する c_std の推定

設計方針:
  - 単一モード (mode=0) の QC に限定する。多モード分離は将来スライス。
  - ヒューリスティックによる近似品質評価であることを明示する。
  - numpy ベースで実装し、新規依存を追加しない。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np

from src.inversion.rayleigh.model import PickedDispersionCurve


@dataclass
class DispersionPickQCResult:
    """QC 済み分散曲線ピッキング結果。

    Fields
    ------
    picked : PickedDispersionCurve or None
        QC 後のピッキング結果。有効点が min_valid_points 未満の場合は None。
    mask_valid : ndarray, shape=(m,)
        各周波数点の有効/無効マスク。True = 有効。
    quality_score : ndarray, shape=(m,)
        各周波数点の品質スコア [0, 1]。
        ヒューリスティックな信頼度指標であり、確率ではない。
    peak_width : ndarray, shape=(m,)
        各周波数点のピーク半値幅 [m/s]。
    secondary_peak_ratio : ndarray, shape=(m,)
        各周波数点の第2ピーク比 (0–1)。1.0 に近いほど曖昧。
    c_raw : ndarray, shape=(m,)
        QC 前の argmax ピック速度 [m/s]。
    c_filtered : ndarray, shape=(m,)
        QC + 平滑化後の速度 [m/s]。無効点は NaN。
    flags : dict[str, ndarray]
        各周波数点のフラグ。キーごとに shape=(m,) の bool 配列。
        現在のキー:
          "low_energy"         — エネルギー閾値以下
          "ambiguous_peak"     — 第2ピーク競合
          "continuity_outlier" — 連続性逸脱
          "below_kmax_limit"   — c_raw < c_min(f) = 2*f*dx（空間エイリアシング）
          "above_kmin_limit"   — c_raw > c_max(f) = f*(N-1)*dx（分解能上限）
    c_min_valid : ndarray, shape=(m,) or None
        周波数ごとの有効速度下限 [m/s]。dx/n_receivers が None のとき None。
    c_max_valid : ndarray, shape=(m,) or None
        周波数ごとの有効速度上限 [m/s]。dx/n_receivers が None のとき None。
    metadata : dict
        QC パラメータ等の補助情報。
    """

    picked: Optional[PickedDispersionCurve]
    mask_valid: np.ndarray
    quality_score: np.ndarray
    peak_width: np.ndarray
    secondary_peak_ratio: np.ndarray
    c_raw: np.ndarray
    c_filtered: np.ndarray
    flags: Dict[str, np.ndarray]
    c_min_valid: Optional[np.ndarray] = None
    c_max_valid: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


def quality_control_dispersion_pick(
    res: np.ndarray,
    f_mesh: np.ndarray,
    c_mesh: np.ndarray,
    *,
    dx: Optional[float] = None,
    n_receivers: Optional[int] = None,
    min_energy_ratio: float = 0.05,
    continuity_rel_jump: float = 0.15,
    max_secondary_peak_ratio: float = 0.85,
    min_valid_points: int = 8,
    smooth_window: int = 3,
    estimate_uncertainty: bool = True,
) -> DispersionPickQCResult:
    """分散曲線画像から QC 済みの PickedDispersionCurve を生成する。

    Algorithmic steps:
      A.  形状検証と軸推定
      B.  各周波数で argmax による raw pick
      B2. 空間波数有効域チェック（dx, n_receivers が与えられた場合のみ）
      C.  エネルギー閾値 QC (low_energy flag)
      D.  ピーク競合 QC (ambiguous_peak flag, secondary_peak_ratio)
      E.  ピークシャープネス / 半値幅 (peak_width)
      F.  連続性 QC (continuity_outlier flag)
      G.  メディアン平滑化 (smooth_window が 3 以上の場合)
      H.  不確実性推定 (c_std)
      I.  品質スコア統合
      J.  PickedDispersionCurve 構築

    Parameters
    ----------
    res : ndarray
        分散曲線画像。周波数軸と速度軸の一方が行でもう一方が列。
        2D であり、f_mesh と c_mesh はそれぞれ異なる軸に対応する。
    f_mesh : ndarray, shape=(n_f,)
        周波数グリッド [Hz]。1D 昇順を想定。
    c_mesh : ndarray, shape=(n_c,)
        速度グリッド [m/s]。1D 昇順を想定。
    dx : float or None, optional
        受振間隔 [m]。n_receivers とともに与えると空間波数制約を適用する。
        有効速度下限: c_min(f) = 2 * f * dx（空間エイリアシング境界）。
        None の場合、波数制約をスキップ（後方互換）。
    n_receivers : int or None, optional
        受振器数 N。dx とともに与えると空間波数制約を適用する。
        有効速度上限: c_max(f) = f * (N - 1) * dx（空間分解能境界）。
        None の場合、波数制約をスキップ（後方互換）。
    min_energy_ratio : float
        各周波数スライスのピーク振幅が global_max * min_energy_ratio 未満の場合に
        low_energy flag を立てる。既定 0.05。
    continuity_rel_jump : float
        隣接ピック間の相対ジャンプがこの閾値を超えた場合に
        continuity_outlier flag を立てる。既定 0.15。
    max_secondary_peak_ratio : float
        ambiguous_peak flag の閾値。
        secondary_peak_ratio > max_secondary_peak_ratio の場合にフラグを立てる。
        既定 0.85。
    min_valid_points : int
        picked が None になる最低有効点数。既定 8。
    smooth_window : int
        メディアン平滑化のウィンドウサイズ。3 以上で有効。既定 3。
    estimate_uncertainty : bool
        True の場合 c_std を推定して PickedDispersionCurve に含める。既定 True。

    Returns
    -------
    DispersionPickQCResult

    Raises
    ------
    ValueError
        res が 2D でない、f_mesh と c_mesh の次元が不正、
        または軸推定が不可能な場合。

    Notes
    -----
    dx または n_receivers が None の場合、波数制約はスキップされ、
    既存の呼び出し (quality_control_dispersion_pick(res, f_mesh, c_mesh)) と
    完全に同一の出力を返す（後方互換）。
    """

    res = np.asarray(res, dtype=float)
    f_mesh = np.asarray(f_mesh, dtype=float)
    c_mesh = np.asarray(c_mesh, dtype=float)

    if res.ndim != 2:
        raise ValueError(f"res must be 2D, got shape {res.shape}")
    if f_mesh.ndim != 1 or c_mesh.ndim != 1:
        raise ValueError(
            f"f_mesh and c_mesh must be 1D, got {f_mesh.shape} and {c_mesh.shape}"
        )

    # --- Step A: 軸推定 ---
    # c_mesh が単調増加なら速度軸。f_mesh の要素数と res の軸の対応を判定。
    n_f = f_mesh.size
    n_c = c_mesh.size

    # res は (n_f, n_c) または (n_c, n_f) のいずれか
    if res.shape == (n_f, n_c):
        freq_axis = 0
        vel_axis = 1
        c_array = c_mesh
        f_array = f_mesh
    elif res.shape == (n_c, n_f):
        freq_axis = 1
        vel_axis = 0
        c_array = c_mesh
        f_array = f_mesh
    else:
        raise ValueError(
            f"res shape {res.shape} does not match (n_f={n_f}, n_c={n_c}) "
            f"or (n_c={n_c}, n_f={n_f})"
        )

    m = n_f  # 周波数点数

    # --- Step B: Raw pick (argmax) ---
    peak_indices = np.argmax(res, axis=vel_axis)
    if vel_axis == 1:
        c_raw = c_array[peak_indices]
    else:
        c_raw = c_array[peak_indices]

    # 各周波数スライスの最大エネルギー
    max_energy_per_freq = np.max(res, axis=vel_axis)
    global_max_energy = float(np.max(max_energy_per_freq))

    # --- Step B2: 空間波数有効域チェック ---
    below_kmax_limit = np.zeros(m, dtype=bool)
    above_kmin_limit = np.zeros(m, dtype=bool)
    c_min_valid_arr: Optional[np.ndarray] = None
    c_max_valid_arr: Optional[np.ndarray] = None

    # mask_valid の初期化（後続ステップで更新）
    mask_valid = np.ones(m, dtype=bool)

    if dx is not None and n_receivers is not None:
        f_1d = f_array  # shape=(n_f,), 周波数軸 1D 配列
        c_min_valid_arr = 2.0 * f_1d * float(dx)
        c_max_valid_arr = f_1d * float(n_receivers - 1) * float(dx)

        below_kmax_limit = c_raw < c_min_valid_arr
        above_kmin_limit = c_raw > c_max_valid_arr

        mask_valid[below_kmax_limit] = False
        mask_valid[above_kmin_limit] = False

    # --- Step C: Energy threshold QC ---
    low_energy_mask = max_energy_per_freq < (min_energy_ratio * global_max_energy)

    # --- Step D: Peak competition QC ---
    secondary_peak_ratio = np.zeros(m, dtype=float)
    ambiguous_peak_mask = np.zeros(m, dtype=bool)

    for i in range(m):
        if freq_axis == 0:
            slice_i = res[i, :]
        else:
            slice_i = res[:, i]

        max_val = float(slice_i[peak_indices[i] if freq_axis == 0 else peak_indices[i]])
        max_val = max_energy_per_freq[i]

        # 第2ピークの探索: 主ピーク周辺をマスクして最大値を探す
        pi = int(peak_indices[i]) if freq_axis == 0 else int(peak_indices[i])

        # 主ピークの近傍 (±3 グリッド点) をマスク
        lo = max(0, pi - 3)
        hi = min(len(c_array), pi + 4)
        masked_slice = slice_i.copy()
        masked_slice[lo:hi] = 0.0

        second_max = float(np.max(masked_slice))
        if max_val > 0:
            secondary_peak_ratio[i] = second_max / max_val
        else:
            secondary_peak_ratio[i] = 0.0

    ambiguous_peak_mask = secondary_peak_ratio > max_secondary_peak_ratio

    # --- Step E: Peak sharpness / width ---
    peak_width = np.full(m, float(c_array[-1] - c_array[0]), dtype=float)

    for i in range(m):
        if freq_axis == 0:
            slice_i = res[i, :]
        else:
            slice_i = res[:, i]

        pi = int(peak_indices[i]) if freq_axis == 0 else int(peak_indices[i])
        peak_val = float(slice_i[pi])

        if peak_val <= 0:
            continue

        half_max = peak_val / 2.0

        # 右側半値幅
        right_idx = pi
        while right_idx < len(c_array) - 1 and slice_i[right_idx + 1] > half_max:
            right_idx += 1
        # 左側半値幅
        left_idx = pi
        while left_idx > 0 and slice_i[left_idx - 1] > half_max:
            left_idx -= 1

        peak_width[i] = float(c_array[min(right_idx, len(c_array) - 1)] - c_array[max(left_idx, 0)])

    # --- Step F: Continuity QC ---
    continuity_outlier_mask = np.zeros(m, dtype=bool)

    # 3点メディアン整合性チェック
    # 有効点間での相対ジャンプを評価
    for i in range(1, m):
        if low_energy_mask[i] or low_energy_mask[i - 1]:
            continue
        rel_jump = abs(c_raw[i] - c_raw[i - 1]) / max(abs(c_raw[i - 1]), 1.0)
        if rel_jump > continuity_rel_jump:
            # 3点チェック: 前後の点で整合確認
            if i >= 2 and not low_energy_mask[i - 2]:
                median_prev = np.median([c_raw[i - 2], c_raw[i - 1], c_raw[i]])
                if abs(c_raw[i] - median_prev) / max(abs(median_prev), 1.0) > continuity_rel_jump:
                    continuity_outlier_mask[i] = True
            else:
                continuity_outlier_mask[i] = True

    # --- 統合マスク ---
    mask_valid = mask_valid & ~(low_energy_mask | continuity_outlier_mask)

    # ambiguous_peak は品質スコアに影響するが、ピック自体は無効にしない
    # (曖昧でもピーク位置自体は意味を持つ場合がある)

    # --- Step G: Smoothing ---
    c_filtered = c_raw.copy().astype(float)

    valid_indices = np.where(mask_valid)[0]
    if len(valid_indices) >= 3 and smooth_window >= 3:
        c_valid = c_raw[valid_indices]
        half_w = smooth_window // 2
        smoothed = np.empty_like(c_valid)
        for j in range(len(c_valid)):
            lo_j = max(0, j - half_w)
            hi_j = min(len(c_valid), j + half_w + 1)
            smoothed[j] = float(np.median(c_valid[lo_j:hi_j]))
        c_filtered[valid_indices] = smoothed

    # 無効点を NaN に
    c_filtered[~mask_valid] = np.nan

    # ギャップをまたぐ平滑化は行わない (有効点内のみ)

    # --- Step H: Uncertainty estimation ---
    c_std_valid: Optional[np.ndarray] = None
    if estimate_uncertainty and np.any(mask_valid):
        valid_peak_widths = peak_width[mask_valid]
        valid_secondary_ratios = secondary_peak_ratio[mask_valid]
        valid_c_filtered = c_filtered[mask_valid]

        # c_std_i = max(0.5 * peak_width_i, (secondary_ratio_i / max_ratio) * peak_width_i)
        # clip to max(1.0, 0.01 * c_filtered_i)
        c_std_valid_arr = np.maximum(
            0.5 * valid_peak_widths,
            (valid_secondary_ratios / max_secondary_peak_ratio) * valid_peak_widths,
        )
        lower_bounds = np.maximum(1.0, 0.01 * np.abs(valid_c_filtered))
        c_std_valid_arr = np.maximum(c_std_valid_arr, lower_bounds)
        c_std_valid = c_std_valid_arr

    # --- Step I: Quality score ---
    # sharpness_score = 1 / (1 + peak_width / median_valid_peak_width)
    # ambiguity_score = 1 - secondary_peak_ratio
    # continuity_score = 0 if flagged else 1
    # quality_score = clip(0.4*s + 0.4*a + 0.2*c, 0, 1)

    valid_pw = peak_width[mask_valid]
    median_pw = float(np.median(valid_pw)) if len(valid_pw) > 0 else 1.0
    if median_pw <= 0:
        median_pw = 1.0

    quality_score_full = np.zeros(m, dtype=float)

    sharpness_score = np.zeros(m, dtype=float)
    ambiguity_score = np.zeros(m, dtype=float)
    continuity_score = np.where(continuity_outlier_mask, 0.0, 1.0)
    sharpness_score = 1.0 / (1.0 + peak_width / median_pw)
    ambiguity_score = 1.0 - secondary_peak_ratio

    quality_score_full = np.clip(
        0.4 * sharpness_score + 0.4 * ambiguity_score + 0.2 * continuity_score,
        0.0,
        1.0,
    )

    # 波数制約で除外された点の quality_score を 0.0 に強制上書き
    wavenumber_excluded = below_kmax_limit | above_kmin_limit
    quality_score_full[wavenumber_excluded] = 0.0

    # --- Step J: Build PickedDispersionCurve ---
    valid_f = f_array[mask_valid]
    valid_c = c_filtered[mask_valid]

    # NaN を除去 (ギャップの境界等)
    finite_mask = np.isfinite(valid_c)
    valid_f = valid_f[finite_mask]
    valid_c = valid_c[finite_mask]

    picked: Optional[PickedDispersionCurve] = None
    if valid_f.size >= min_valid_points:
        # 昇順ソート
        sort_idx = np.argsort(valid_f)
        valid_f = valid_f[sort_idx]
        valid_c = valid_c[sort_idx]

        # 重複周波数除去
        unique_mask = np.diff(valid_f, prepend=-np.inf) > 0
        valid_f = valid_f[unique_mask]
        valid_c = valid_c[unique_mask]

        if valid_f.size >= min_valid_points:
            c_std_final = None
            if c_std_valid is not None:
                c_std_final_arr = c_std_valid[finite_mask]
                c_std_final_arr = c_std_final_arr[sort_idx]
                c_std_final_arr = c_std_final_arr[unique_mask]
                c_std_final = c_std_final_arr

            picked = PickedDispersionCurve(
                f=valid_f,
                c=valid_c,
                c_std=c_std_final,
                mode=0,
            )

    return DispersionPickQCResult(
        picked=picked,
        mask_valid=mask_valid,
        quality_score=quality_score_full,
        peak_width=peak_width,
        secondary_peak_ratio=secondary_peak_ratio,
        c_raw=c_raw,
        c_filtered=c_filtered,
        flags={
            "low_energy": low_energy_mask,
            "ambiguous_peak": ambiguous_peak_mask,
            "continuity_outlier": continuity_outlier_mask,
            "below_kmax_limit": below_kmax_limit,
            "above_kmin_limit": above_kmin_limit,
        },
        c_min_valid=c_min_valid_arr,
        c_max_valid=c_max_valid_arr,
        metadata=dict(
            min_energy_ratio=min_energy_ratio,
            continuity_rel_jump=continuity_rel_jump,
            max_secondary_peak_ratio=max_secondary_peak_ratio,
            min_valid_points=min_valid_points,
            smooth_window=smooth_window,
            estimate_uncertainty=estimate_uncertainty,
            n_f=n_f,
            n_c=n_c,
            dx=dx,
            n_receivers=n_receivers,
        ),
    )
