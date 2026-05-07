"""Data classes for Rayleigh-wave Vs inversion.

Step ?-1a の責務範囲:
  - 層パラメータコンテナ (LayeredEarthModel)
  - 観測分散曲線コンテナ (PickedDispersionCurve)
  - 1 CMP 分の逆解析結果コンテナ (RayleighInversionResult)

Step ?-3 の追加:
  - Pseudo-2D Vs section コンテナ (Pseudo2DVsSection)

Step ?-8 の追加:
  - Laterally Constrained Inversion (LCI) 結果コンテナ (LCIProfileResult)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class LayeredEarthModel:
    """1D 層構造モデル。

    Step ?-1a では vs のみが逆解析対象。h / vp / rho は与えられた値で固定。

    Fields
    ------
    vs : ndarray, shape=(n_layers,)
        各層の S 波速度 [m/s]。逆解析の更新対象。
    vp : ndarray, shape=(n_layers,)
        各層の P 波速度 [m/s]。固定 (Step ?-1a)。
    rho : ndarray, shape=(n_layers,)
        各層の密度 [kg/m^3]。固定 (Step ?-1a)。
    h : ndarray, shape=(n_layers,)
        各層の厚さ [m]。最下層は半空間 (慣習として np.inf を入れる)。
        Step ?-1a では固定 (engine による更新対象外)。h の更新は将来スライス。
    vs_bounds : tuple of two ndarrays, optional
        (lower, upper) の 1 対。各 shape=(n_layers,)。
        engine が各反復後に np.clip で射影する。None なら無拘束。
    """

    vs: np.ndarray
    vp: np.ndarray
    rho: np.ndarray
    h: np.ndarray
    vs_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None

    def __post_init__(self) -> None:
        self.vs = np.asarray(self.vs, dtype=float)
        self.vp = np.asarray(self.vp, dtype=float)
        self.rho = np.asarray(self.rho, dtype=float)
        self.h = np.asarray(self.h, dtype=float)
        n = self.vs.shape[0]
        if not (self.vp.shape == self.rho.shape == self.h.shape == (n,)):
            raise ValueError(
                "vs / vp / rho / h must share shape (n_layers,); got "
                f"{self.vs.shape}, {self.vp.shape}, {self.rho.shape}, {self.h.shape}"
            )
        if self.vs_bounds is not None:
            lo, hi = self.vs_bounds
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            if lo.shape != (n,) or hi.shape != (n,):
                raise ValueError(
                    f"vs_bounds entries must have shape ({n},); got {lo.shape}, {hi.shape}"
                )
            if np.any(lo > hi):
                raise ValueError("vs_bounds lower exceeds upper at some layers")
            self.vs_bounds = (lo, hi)

    @property
    def n_layers(self) -> int:
        return int(self.vs.shape[0])

    def copy_with_vs(self, new_vs: np.ndarray) -> "LayeredEarthModel":
        """vs のみ差し替えた新しいインスタンスを返す (h / vp / rho は不変)。"""
        return LayeredEarthModel(
            vs=np.asarray(new_vs, dtype=float).copy(),
            vp=self.vp.copy(),
            rho=self.rho.copy(),
            h=self.h.copy(),
            vs_bounds=(
                (self.vs_bounds[0].copy(), self.vs_bounds[1].copy())
                if self.vs_bounds is not None
                else None
            ),
        )


@dataclass
class PickedDispersionCurve:
    """観測ピッキング済み分散曲線 (1 CMP, 1 mode)。

    Fields
    ------
    f : ndarray, shape=(m,)
        ピックされた周波数 [Hz]。昇順を想定。
    c : ndarray, shape=(m,)
        各 f での位相速度 [m/s]。
    c_std : ndarray or None, shape=(m,)
        各 f での 1-σ 観測不確実性 [m/s]。
        None の場合、misfit 側で一様重み (= 1) として扱う。
    mode : int
        モード番号。基本モード = 0。
    """

    f: np.ndarray
    c: np.ndarray
    c_std: Optional[np.ndarray] = None
    mode: int = 0

    def __post_init__(self) -> None:
        self.f = np.asarray(self.f, dtype=float)
        self.c = np.asarray(self.c, dtype=float)
        if self.f.shape != self.c.shape or self.f.ndim != 1:
            raise ValueError(
                f"f / c must be 1D and share shape; got {self.f.shape}, {self.c.shape}"
            )
        if self.c_std is not None:
            self.c_std = np.asarray(self.c_std, dtype=float)
            if self.c_std.shape != self.f.shape:
                raise ValueError(
                    f"c_std must share shape with f; got {self.c_std.shape}"
                )
            if np.any(self.c_std <= 0):
                raise ValueError("c_std must be strictly positive")

    @property
    def n_points(self) -> int:
        return int(self.f.shape[0])


@dataclass
class RayleighInversionResult:
    """1 CMP 分の逆解析結果コンテナ。

    Fields
    ------
    model : LayeredEarthModel
        最終的な (best-rms) Vs モデル。h / vp / rho は入力どおり。
    predicted_c : ndarray, shape=(m,)
        最終モデルでの予測位相速度。
    residuals : ndarray, shape=(m,)
        最終モデルでの **重みなし** 残差 (observed.c - predicted_c)。
        重み付き残差は engine 内部の中間量で、結果には保持しない。
    rms : float
        重みなし残差の RMS [m/s]。
    n_iter : int
        実反復回数 (best-rms に到達するまでに要した反復数)。
    converged : bool
        rms_tol / step_tol を満たして収束したか。max_iter で打ち切られた場合は False。
    jacobian : ndarray or None, shape=(m, n_layers)
        最終モデルでの ∂c/∂Vs_j (重みなし)。posterior 共分散などの解析用。
    posterior_cov : ndarray or None, shape=(n_layers, n_layers)
        簡易的な近似事後共分散。Step ?-1a では None でも可。
    metadata : dict
        engine name, lambda 履歴, RMS 履歴などの補助情報。
    """

    model: LayeredEarthModel
    predicted_c: np.ndarray
    residuals: np.ndarray
    rms: float
    n_iter: int
    converged: bool
    jacobian: Optional[np.ndarray] = None
    posterior_cov: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pseudo2DVsSection:
    """Per-CMP 1D inversion 結果を深度グリッドに写像した pseudo-2D Vs section。

    Step ?-3 で定義。Pseudo2DSectionBuilder で構築する。

    Fields
    ------
    cmpx : ndarray, shape=(K,)
        CMP x 位置 [m]。
    depthz : ndarray, shape=(Z,)
        深度グリッド [m]。層中心深さ。
    vsgrid : ndarray, shape=(Z, K)
        各 (深度, CMP) グリッド点の S 波速度 [m/s]。
        sensitivitycutoff でマスクされたセルは NaN。
    confidencegrid : ndarray or None, shape=(Z, K)
        各グリッド点の信頼度。
        グローバル正規化済みの Jacobian 列ノルム比 (0-1)。
        全 CMP の最大列ノルムで正規化されるため、CMP 間横断比較が可能。
        None の場合は信頼度不明。
    percmpresults : list[RayleighInversionResult]
        各 CMP の 1D inversion 結果。長さ K。
    metadata : dict
        builder パラメータ等の補助情報。
    """

    cmpx: np.ndarray
    depthz: np.ndarray
    vsgrid: np.ndarray
    confidencegrid: Optional[np.ndarray] = None
    percmpresults: List[RayleighInversionResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.cmpx = np.asarray(self.cmpx, dtype=float)
        self.depthz = np.asarray(self.depthz, dtype=float)
        self.vsgrid = np.asarray(self.vsgrid, dtype=float)
        K = self.cmpx.shape[0]
        Z = self.depthz.shape[0]
        if K == 0:
            raise ValueError("cmpx must contain at least one element")
        if Z == 0:
            raise ValueError("depthz must contain at least one element")
        if self.vsgrid.shape != (Z, K):
            raise ValueError(
                f"vsgrid shape {self.vsgrid.shape} does not match "
                f"expected (Z={Z}, K={K})"
            )
        if self.confidencegrid is not None:
            self.confidencegrid = np.asarray(self.confidencegrid, dtype=float)
            if self.confidencegrid.shape != (Z, K):
                raise ValueError(
                    f"confidencegrid shape {self.confidencegrid.shape} "
                    f"does not match expected (Z={Z}, K={K})"
                )
        if len(self.percmpresults) != K:
            raise ValueError(
                f"percmpresults length ({len(self.percmpresults)}) "
                f"does not match cmpx length ({K})"
            )


@dataclass
class LCIProfileResult:
    """Laterally Constrained Inversion (LCI) のプロファイル結果コンテナ。

    Step ?-8 で定義。LCIEngine.run_profile が返す。

    LCI は K 個の CMP を **同時に** 逆解析し、隣接 CMP 間の
    Vs 列方向への滑らかさ拘束 (lateral smoothness) を導入する。
    その結果、per-CMP 1D inversion を独立に走らせる pseudo-2D
    (rayleigh_vs_inversion_profile) では実現できない、
    地質学的整合のとれた 2D 構造を得られる。

    Reference
    ---------
    Auken & Christiansen (2004); Boiero & Socco (2010).

    Fields
    ------
    per_cmp_results : list[RayleighInversionResult]
        各 CMP の最終結果。長さ K。
        n_iter / converged は LCI 全体の値が複製される (per-CMP 個別の
        収束判定は LCI では成立しない: 全 CMP が同時に最適化されるため)。
    joint_rms_history : list[float]
        各 outer iteration での joint unweighted RMS [m/s]。
        joint = 全 CMP 残差を結合してとった RMS。
    lambda_v : float
        垂直ラフネス正則化重み。
    lambda_h : float
        横方向 (CMP 間) スムージング正則化重み。
    converged : bool
        joint RMS が rms_tol に到達 (または step が改善しない場合に
        早期終了したか) の総合判定。
    n_iter : int
        実際の outer iteration 数。
    metadata : dict
        engine name, K (CMP 数), n_layers, use_sparse 等。
    """

    per_cmp_results: List[RayleighInversionResult]
    joint_rms_history: List[float]
    lambda_v: float
    lambda_h: float
    converged: bool
    n_iter: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.per_cmp_results) == 0:
            raise ValueError("per_cmp_results must contain at least one result")
        n0 = self.per_cmp_results[0].model.n_layers
        for k, r in enumerate(self.per_cmp_results):
            if r.model.n_layers != n0:
                raise ValueError(
                    f"per_cmp_results[{k}] has {r.model.n_layers} layers, "
                    f"expected {n0} (all CMPs must share n_layers in LCI)"
                )