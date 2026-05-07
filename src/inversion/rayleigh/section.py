"""Pseudo-2D section builder for Rayleigh-wave Vs inversion.

Step ?-3 の責務:
  - Pseudo2DSectionBuilder: per-CMP 1D inversion 結果を深度グリッドに写像して
    pseudo-2D section を組み立てる。
  - MVP では nearest 写像のみ実装。linear 写像は将来スライスで拡張。

設計方針:
  - 全 CMP が同一層構造 (n_layers, h[:-1]) を共有することを前提とする。
    異なる層構造の CMP が混在する場合はエラーとする (将来のスライスで拡張)。
  - confidencegrid はグローバル正規化済みの Jacobian 列ノルム比。
    全 CMP の最大列ノルムで正規化するため、CMP 間横断比較が可能。
    Jacobian が1つでも利用不可の場合は全体を None とする。
  - sensitivitycutoff は vsgrid の低信頼セルを NaN マスクする閾値。
    グローバル正規化済みの confidencegrid に対して適用される。
    confidencegrid 自体はマスク対象の値を保持する
    (ユーザーが異なる閾値で再マスクできるよう)。
  - depth_fraction マスク: 各 CMP の有効探査深度 Z_max = lambda_max * depth_fraction
    を超えるセルを NaN にマスクする（Rix & Leipski 1991 標準: 0.5, Tokimatsu 1995: 0.33）。
"""

from __future__ import annotations

import warnings
from typing import List, Optional, Tuple

import numpy as np

from src.inversion.rayleigh.model import (
    PickedDispersionCurve,
    Pseudo2DVsSection,
    RayleighInversionResult,
)


def _compute_zmax(
    picked: Optional[PickedDispersionCurve],
    depth_fraction: float = 0.5,
) -> Tuple[Optional[float], Optional[float]]:
    """有効探査深度 Z_max を計算する。

    Z_max = lambda_max * depth_fraction
    lambda_max = max(c_obs[i] / f[i])

    Parameters
    ----------
    picked : PickedDispersionCurve or None
        QC 後の分散曲線。None または有効点が 0 の場合は (None, None) を返す。
    depth_fraction : float
        lambda_max に乗じる係数。0.5 → lambda/2 (Rix & Leipski 1991 標準)。

    Returns
    -------
    (lambda_max, z_max) : (float, float) or (None, None)
    """
    if picked is None:
        return None, None
    c = np.asarray(picked.c, dtype=float)
    f = np.asarray(picked.f, dtype=float)
    if c.size == 0 or f.size == 0:
        return None, None
    valid = np.isfinite(c) & np.isfinite(f) & (f > 0.0)
    if not np.any(valid):
        return None, None
    lambda_max = float(np.max(c[valid] / f[valid]))
    z_max = lambda_max * depth_fraction
    return lambda_max, z_max


class Pseudo2DSectionBuilder:
    """Per-CMP 1D inversion 結果を深度グリッドに写像して pseudo-2D section を構築する。"""

    @staticmethod
    def build(
        cmpx: np.ndarray,
        results: List[RayleighInversionResult],
        sensitivitycutoff: Optional[float] = None,
        maxdepthfactor: float = 2.5,
        interpolation: str = "nearest",
        picked_curves: Optional[List[Optional[PickedDispersionCurve]]] = None,
        depth_fraction: float = 0.5,
    ) -> Pseudo2DVsSection:
        """per-CMP 1D inversion 結果から Pseudo2DVsSection を構築する。

        Parameters
        ----------
        cmpx : ndarray, shape=(K,)
            CMP x 位置 [m]。昇順を想定。
        results : list[RayleighInversionResult], length K
            各 CMP の 1D inversion 結果。
            全結果の model.n_layers および model.h[:-1] が一致している必要がある。
        sensitivitycutoff : float or None
            confidencegrid の閾値 (0 < cutoff <= 1)。
            グローバル正規化済み confidencegrid に対して適用される。
            指定時、confidencegrid < cutoff の vsgrid セルを NaN にマスクする。
            None の場合はマスクしない。
            Jacobian が1つでも利用不可の場合は警告を出力しマスクをスキップする。
        maxdepthfactor : float
            半空間の深さ推定に用いる係数。既定 2.5。
            半空間層厚を maxdepthfactor * mean(h_finite) で推定する。
        interpolation : str
            深度写像モード。'nearest' のみ実装済み。
            'linear' は将来のスライスで拡張予定。
        picked_curves : list[PickedDispersionCurve or None] or None
            各 CMP の QC 後分散曲線。長さ K。None の場合は深度マスクをスキップする。
            有効探査深度マスクを適用するために必要。
        depth_fraction : float
            有効探査深度係数。Z_max = lambda_max * depth_fraction。
            0.5 → lambda/2（Rix & Leipski 1991 標準）。
            0.33 → lambda/3（Tokimatsu 1995 保守的）。既定 0.5。

        Returns
        -------
        Pseudo2DVsSection
            cmpx, depthz, vsgrid, confidencegrid, percmpresults, metadata を含む。
            metadata に z_max_per_cmp, lambda_max_per_cmp, depth_fraction を格納。

        Raises
        ------
        ValueError
            cmpx と results の長さが不一致。
            全 CMP の n_layers が一致しない。
            全 CMP の層厚 h[:-1] が一致しない。
            sensitivitycutoff が (0, 1] の範囲外。
        NotImplementedError
            interpolation が 'linear' の場合。
        """
        if interpolation not in ("nearest", "linear"):
            raise ValueError(
                f"interpolation must be 'nearest' or 'linear', "
                f"got {interpolation!r}"
            )
        if interpolation == "linear":
            raise NotImplementedError(
                "linear interpolation is not yet implemented; "
                "use interpolation='nearest' for now"
            )

        cmpx = np.asarray(cmpx, dtype=float)

        if len(results) == 0:
            raise ValueError(
                "results must contain at least one RayleighInversionResult"
            )
        if cmpx.shape[0] != len(results):
            raise ValueError(
                f"cmpx length ({cmpx.shape[0]}) != results length ({len(results)})"
            )

        if sensitivitycutoff is not None:
            if not (0.0 < sensitivitycutoff <= 1.0):
                raise ValueError(
                    f"sensitivitycutoff must be in (0, 1], "
                    f"got {sensitivitycutoff}"
                )

        first_model = results[0].model
        n_layers = first_model.n_layers

        for k, r in enumerate(results):
            if r.model.n_layers != n_layers:
                raise ValueError(
                    f"result at index {k} has {r.model.n_layers} layers, "
                    f"expected {n_layers}. "
                    f"Different layer structures are not yet supported."
                )

        for k, r in enumerate(results):
            if k == 0:
                continue
            if not np.allclose(r.model.h[:-1], first_model.h[:-1]):
                raise ValueError(
                    f"result at index {k} has different finite layer thicknesses "
                    f"than result 0. All CMPs must share the same depth structure "
                    f"for Pseudo2DSectionBuilder. "
                    f"Consider providing a common initial model."
                )
            first_is_halfspace = bool(np.isinf(first_model.h[-1]))
            k_is_halfspace = bool(np.isinf(r.model.h[-1]))
            if first_is_halfspace != k_is_halfspace:
                raise ValueError(
                    f"result at index {k} has inconsistent half-space indicator "
                    f"(h[-1]={r.model.h[-1]}) compared to result 0 "
                    f"(h[-1]={first_model.h[-1]}). "
                    f"All CMPs must have consistent half-space convention."
                )

        # --- depth grid ---
        h = first_model.h.copy()
        is_halfspace = bool(np.isinf(h[-1]))
        if is_halfspace:
            mean_finite_h = float(np.mean(h[:-1]))
            # [BUG-1 FIX] maxdepthfactor を参照する (ハードコード 2.0 は使用しない)
            h[-1] = maxdepthfactor * mean_finite_h

        z_top = np.concatenate([[0.0], np.cumsum(h[:-1])])
        depthz = z_top + h / 2.0

        # --- vsgrid ---
        K = cmpx.shape[0]
        vsgrid = np.empty((n_layers, K), dtype=float)
        for k, r in enumerate(results):
            vsgrid[:, k] = r.model.vs

        # --- confidencegrid from Jacobian column norms ---
        # [DESIGN-1] グローバル正規化: 全 CMP の最大列ノルムで正規化する。
        # これにより confidencegrid は CMP 間横断比較が可能な相対感度指標となる。
        all_have_jac = all(r.jacobian is not None for r in results)
        if all_have_jac:
            all_col_norms = np.stack(
                [
                    np.linalg.norm(np.asarray(r.jacobian, dtype=float), axis=0)
                    for r in results
                ],
                axis=1,
            )  # shape: (n_layers, K)
            global_max = float(np.max(all_col_norms))
            if global_max > 0.0:
                confidencegrid = all_col_norms / global_max
            else:
                confidencegrid = np.zeros_like(all_col_norms)
        else:
            confidencegrid = None

        # [DESIGN-2] Jacobian が無い場合に sensitivitycutoff が無視されることを警告
        if sensitivitycutoff is not None and confidencegrid is None:
            warnings.warn(
                "sensitivitycutoff is specified but confidencegrid could not be built "
                "(some results have jacobian=None). "
                "The cutoff mask will not be applied.",
                UserWarning,
                stacklevel=2,
            )

        # --- sensitivity cutoff mask ---
        if sensitivitycutoff is not None and confidencegrid is not None:
            mask = confidencegrid < sensitivitycutoff
            vsgrid[mask] = np.nan

        # --- 有効探査深度マスク (Rix & Leipski 1991 / Tokimatsu 1995) ---
        # Z_max = lambda_max * depth_fraction, lambda_max = max(c/f)
        z_max_per_cmp = np.full(K, np.nan, dtype=float)
        lambda_max_per_cmp = np.full(K, np.nan, dtype=float)

        if picked_curves is not None:
            if len(picked_curves) != K:
                warnings.warn(
                    f"picked_curves length ({len(picked_curves)}) != K ({K}). "
                    "Depth masking will be skipped.",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                for k, picked in enumerate(picked_curves):
                    lm, zm = _compute_zmax(picked, depth_fraction)
                    if lm is None or zm is None:
                        # picked が None → 全層 NaN
                        vsgrid[:, k] = np.nan
                        continue
                    lambda_max_per_cmp[k] = lm
                    z_max_per_cmp[k] = zm
                    # depth > Z_max のセルを NaN にマスク
                    for j, depth_mid in enumerate(depthz):
                        if depth_mid > zm:
                            vsgrid[j, k] = np.nan

                # サマリー出力
                valid_zm = z_max_per_cmp[np.isfinite(z_max_per_cmp)]
                if valid_zm.size > 0:
                    print(
                        f"  有効探査深度 (depth_fraction={depth_fraction:.2f}): "
                        f"min={valid_zm.min():.1f}m, "
                        f"max={valid_zm.max():.1f}m, "
                        f"mean={np.nanmean(valid_zm):.1f}m"
                    )

        return Pseudo2DVsSection(
            cmpx=cmpx,
            depthz=depthz,
            vsgrid=vsgrid,
            confidencegrid=confidencegrid,
            percmpresults=list(results),
            metadata=dict(
                maxdepthfactor=maxdepthfactor,
                sensitivitycutoff=sensitivitycutoff,
                is_halfspace=is_halfspace,
                n_layers=n_layers,
                interpolation=interpolation,
                builder="Pseudo2DSectionBuilder",
                # [DESIGN-1] グローバル正規化を採用していることを記録
                confidence_normalization="global",
                # 有効探査深度マスク関連
                depth_fraction=depth_fraction,
                z_max_per_cmp=z_max_per_cmp,
                lambda_max_per_cmp=lambda_max_per_cmp,
            ),
        )
