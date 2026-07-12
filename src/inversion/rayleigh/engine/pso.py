"""NichingPSOEngine — Niching Particle Swarm Optimization for surface-wave inversion.

PDF1 (Zhang, K. et al., 2023. "A niching particle swarm optimization strategy
for the multimodal inversion of surface waves." Geophys. J. Int., 232,
1140-1158. DOI:10.1093/gji/ggac380.) §2.3 の distance-based locally informed
particle swarm (Qu et al. 2013) と modified BSAS (Orujpour et al. 2019) を
組み合わせた multimodal optimizer を実装する。

Why niching, not standard PSO?
------------------------------
determinant misfit や nearest-neighbor misfit は **多数の局所極小** を持つため
(PDF1 Fig. 22)、標準 PSO では global best particle 1 個が全体を引きずって
local optimum に閉じ込められやすい。niching PSO は粒子ごとに局所 best を
近傍から選ぶことで複数の極小を同時に探索する。PSO 結果から cluster analysis
で代表モデルを抽出し、nearest-neighbor misfit でランキングする。

主要パラメータ (PDF1 Section 2.3)
---------------------------------
- 集団サイズ N = 10 * D (D = layer 数 = 探索次元)
- inertia ω = 0.729843788 (Eberhart & Shi 2000)
- φ_j ∈ [0, 4.1 / nsize] (粒子ごと、近傍ごとに乱数)
- neighborhood size nsize: 2 -> 4 に反復で動的増加
- 停滞判定: F_t-m - F_t < η, m = 10*D, η = 1e-4
- 50 独立 run -> 上位 1/4 を modified BSAS (閾値 0.3*sqrt(D)) でクラスタリング
- 各クラスタ平均を候補モデルとし nearest-neighbor misfit で最終ランキング

本実装の Rayleigh / Love 共用設計
---------------------------------
本 engine は ``forward`` / ``misfit`` を inject 形式で受け取り、wave 種別に
依存しない。``DeterminantMisfit`` (Love 用 secular 関数を inject 済み) と
組み合わせれば Love wave、Rayleigh 用 secular 関数の実装が用意できれば
Rayleigh wave にも適用可能。

Engine 契約との整合 (src.inversion.rayleigh.engine.base.InversionEngine):
  - ``run_single(observed, init_model) -> RayleighInversionResult`` を実装。
  - ``misfit.value(observed, model)`` の呼び出しで、第二引数に LayeredEarthModel
    を渡す (DeterminantMisfit / NearestNeighborMisfit は model を期待する)。
    WeightedL2Misfit など LSQ 系は予測 velocity 配列を期待するため、両者は
    本 engine で混用できない (LSQ 系 misfit は本 engine では未対応)。
  - 戻り値の ``predicted_c`` / ``residuals`` / ``jacobian`` フィールドは、
    cluster の代表モデルを再 forward した結果で埋める。
    Jacobian は本 engine では計算しないため None。

References
----------
- Qu, B., Suganthan, P. N. & Das, S. (2013). "A distance-based locally informed
  particle swarm model for multimodal optimization." IEEE Trans. Evol. Comput.,
  17, 387-402.
- Orujpour, M. et al. (2019). "Multi-modal forest optimization algorithm."
  Neural Comput. Appl., 32, 6159-6173. (modified BSAS の参照)
- Eberhart, R. C. & Shi, Y. (2000). "Comparing inertia weights and constriction
  factors in particle swarm optimization." Proc. IEEE Congress on Evolutionary
  Computation, 84-88.
- Mendes, R., Kennedy, J. & Neves, J. (2004). "The fully informed particle
  swarm: simpler, maybe better." IEEE Trans. Evol. Comput., 8, 204-210.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from src.inversion.rayleigh.engine.base import InversionEngine
from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    PickedDispersionCurve,
    RayleighInversionResult,
)

_INERTIA = 0.729843788  # PDF1 Eq. (13), Eberhart & Shi 2000


@dataclass
class _PSORunResult:
    best_position: np.ndarray  # shape=(D,)
    best_value: float
    n_iter: int


def _normalized_distance(
    a: np.ndarray,
    b: np.ndarray,
    bounds_lo: np.ndarray,
    bounds_hi: np.ndarray,
) -> float:
    """探索範囲で正規化した Euclidean 距離 (PDF1 §2.3)。"""
    span = np.maximum(bounds_hi - bounds_lo, 1e-30)
    return float(np.linalg.norm((a - b) / span))


class NichingPSOEngine(InversionEngine):
    """Niching PSO + modified BSAS clustering + nearest-neighbor ranking。

    Parameters
    ----------
    forward : RayleighForwardSolver
        前進解 (cluster 代表モデルの再 forward と nearest-neighbor ranking 用)。
    misfit : DispersionMisfit
        PSO 内 loop の目的関数。``DeterminantMisfit`` を想定。
        ``misfit.value(observed, model)`` の形で呼び出される。
    ranking_misfit : DispersionMisfit or None
        cluster 代表モデルの最終ランキング用 (典型的に ``NearestNeighborMisfit``)。
        None の場合は PSO misfit と同じものを使う。
    n_runs : int
        独立 PSO run 数。既定 50 (PDF1 §2.3)。
    population_size : int or None
        粒子数。None の場合 ``10 * D`` (D = init_model.n_layers; PDF1 §2.3)。
    max_iter : int
        1 run あたり最大反復数。既定 500。
    eta : float
        停滞判定の閾値 (PDF1 Eq. 15)。既定 1e-4。
    nsize_min, nsize_max : int
        近傍サイズの動的範囲。既定 (2, 4) (PDF1 §2.3)。
    cluster_threshold_factor : float
        modified BSAS の閾値: ``factor * sqrt(D)``。既定 0.3 (PDF1 §2.3)。
    top_fraction : float
        独立 run 結果のうちクラスタリング対象とする上位比率。既定 0.25 (上位 1/4)。
    max_clusters : int
        BSAS が生成するクラスタ数の上限。安全のため設けるが、PDF1 では明示なし。
        既定 50。
    halfspace_monotonic_guard : bool
        True (既定) の場合、cluster 代表モデルで ``vs[-1] < max(vs[:-1])`` の
        ものは「halfspace Vs が overlying 層より小さい unphysical model」として
        ranking から除外する (PDF1 §5 で言及される banded low-value region への
        対策)。
    rng_seed : int or None
        乱数シード。None の場合は決定的でない。
    """

    def __init__(
        self,
        forward: RayleighForwardSolver,
        misfit: DispersionMisfit,
        *,
        ranking_misfit: Optional[DispersionMisfit] = None,
        n_runs: int = 50,
        population_size: Optional[int] = None,
        max_iter: int = 500,
        eta: float = 1e-4,
        nsize_min: int = 2,
        nsize_max: int = 4,
        cluster_threshold_factor: float = 0.3,
        top_fraction: float = 0.25,
        max_clusters: int = 50,
        halfspace_monotonic_guard: bool = True,
        rng_seed: Optional[int] = None,
        **opts,
    ) -> None:
        super().__init__(forward=forward, misfit=misfit, **opts)
        if int(n_runs) < 1:
            raise ValueError("n_runs must be >= 1")
        if int(max_iter) < 10:
            raise ValueError("max_iter must be >= 10")
        if not (0.0 < float(top_fraction) <= 1.0):
            raise ValueError("top_fraction must be in (0, 1]")
        if int(nsize_min) < 1 or int(nsize_max) < int(nsize_min):
            raise ValueError("require 1 <= nsize_min <= nsize_max")
        self._ranking_misfit = ranking_misfit if ranking_misfit is not None else misfit
        self._n_runs = int(n_runs)
        self._population_size = (
            int(population_size) if population_size is not None else None
        )
        self._max_iter = int(max_iter)
        self._eta = float(eta)
        self._nsize_min = int(nsize_min)
        self._nsize_max = int(nsize_max)
        self._cluster_threshold_factor = float(cluster_threshold_factor)
        self._top_fraction = float(top_fraction)
        self._max_clusters = int(max_clusters)
        self._halfspace_monotonic_guard = bool(halfspace_monotonic_guard)
        self._rng = np.random.default_rng(rng_seed)

    # =================================================================
    #  Public engine contract
    # =================================================================

    def run_single(
        self,
        observed: PickedDispersionCurve,
        init_model: LayeredEarthModel,
    ) -> RayleighInversionResult:
        if init_model.vs_bounds is None:
            raise ValueError(
                "NichingPSOEngine requires init_model.vs_bounds to be set "
                "(use build_default_love_init_model / build_default_init_model)."
            )
        bounds_lo, bounds_hi = init_model.vs_bounds
        bounds_lo = np.asarray(bounds_lo, dtype=float)
        bounds_hi = np.asarray(bounds_hi, dtype=float)
        D = int(init_model.n_layers)

        # 50 independent runs
        run_results: List[_PSORunResult] = []
        for run_i in range(self._n_runs):
            res = self._run_one_pso(
                observed=observed,
                init_model=init_model,
                bounds_lo=bounds_lo,
                bounds_hi=bounds_hi,
            )
            run_results.append(res)

        # 上位 top_fraction を抽出
        run_results.sort(key=lambda r: r.best_value)
        n_top = max(1, int(np.ceil(len(run_results) * self._top_fraction)))
        top_results = run_results[:n_top]

        # modified BSAS clustering
        clusters = self._modified_bsas(
            positions=[r.best_position for r in top_results],
            values=[r.best_value for r in top_results],
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            D=D,
        )

        # 各クラスタ代表モデルを作り、ranking_misfit で再ランキング
        candidate_models: List[LayeredEarthModel] = []
        candidate_misfits: List[float] = []
        for cluster_positions in clusters:
            if len(cluster_positions) == 0:
                continue
            mean_vs = np.mean(np.stack(cluster_positions, axis=0), axis=0)
            # bounds clamp (BSAS 平均が境界外に行くことは稀だが安全のため)
            mean_vs = np.clip(mean_vs, bounds_lo, bounds_hi)
            cand_model = init_model.copy_with_vs(mean_vs)
            if self._halfspace_monotonic_guard and (
                cand_model.vs[-1] < float(np.max(cand_model.vs[:-1]))
            ):
                # halfspace が overlying 層より小さい unphysical model は除外
                continue
            try:
                misfit_val = float(self._ranking_misfit.value(observed, cand_model))
            except Exception:
                misfit_val = float("inf")
            candidate_models.append(cand_model)
            candidate_misfits.append(misfit_val)

        if len(candidate_models) == 0:
            # halfspace guard で全候補が落ちた場合のフォールバック:
            # 最良 raw PSO 解を採用する
            best_raw = top_results[0]
            cand_model = init_model.copy_with_vs(
                np.clip(best_raw.best_position, bounds_lo, bounds_hi)
            )
            candidate_models.append(cand_model)
            candidate_misfits.append(best_raw.best_value)
            warnings.warn(
                "NichingPSOEngine: all clusters failed halfspace monotonic guard; "
                "falling back to raw best PSO solution.",
                stacklevel=2,
            )

        best_idx = int(np.argmin(candidate_misfits))
        best_model = candidate_models[best_idx]
        best_ranking_misfit = candidate_misfits[best_idx]

        # 最終モデルでの予測 c(f) と残差 (unweighted)
        try:
            predicted_c = self.forward.forward(best_model, observed.f, mode=int(observed.mode))
            residuals = observed.c - predicted_c
            rms = float(np.sqrt(np.mean(residuals * residuals)))
        except Exception:
            predicted_c = np.full_like(observed.c, np.nan)
            residuals = np.full_like(observed.c, np.nan)
            rms = float("nan")

        # n_iter / converged は PSO の全 run の合計 / 最良 run の収束として記録
        total_iters = int(sum(r.n_iter for r in run_results))
        best_run_converged = bool(run_results[0].n_iter < self._max_iter)

        metadata = {
            "engine": "niching_pso",
            "n_runs": self._n_runs,
            "population_size": (
                self._population_size if self._population_size is not None else 10 * D
            ),
            "max_iter": self._max_iter,
            "n_clusters": len(candidate_models),
            "best_pso_misfit": float(run_results[0].best_value),
            "best_ranking_misfit": float(best_ranking_misfit),
            "candidate_ranking_misfits": [float(v) for v in candidate_misfits],
            "halfspace_monotonic_guard": self._halfspace_monotonic_guard,
        }

        return RayleighInversionResult(
            model=best_model,
            predicted_c=np.asarray(predicted_c, dtype=float),
            residuals=np.asarray(residuals, dtype=float),
            rms=rms,
            n_iter=total_iters,
            converged=best_run_converged,
            jacobian=None,
            posterior_cov=None,
            metadata=metadata,
        )

    # =================================================================
    #  PSO single-run core (PDF1 §2.3 Eqs. 11-15)
    # =================================================================

    def _run_one_pso(
        self,
        *,
        observed: PickedDispersionCurve,
        init_model: LayeredEarthModel,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
    ) -> _PSORunResult:
        D = int(init_model.n_layers)
        N = self._population_size if self._population_size is not None else 10 * D
        N = max(N, 4)

        # 一様初期化 (探索空間内ランダム)
        positions = (
            bounds_lo[None, :]
            + self._rng.random((N, D)) * (bounds_hi - bounds_lo)[None, :]
        )
        velocities = np.zeros_like(positions)
        # 各粒子の pbest
        pbest_positions = positions.copy()
        pbest_values = np.full(N, np.inf)

        # 初期評価
        for i in range(N):
            v = self._eval_misfit(observed, init_model, positions[i], bounds_lo, bounds_hi)
            pbest_values[i] = v
            pbest_positions[i] = positions[i]

        global_best_value = float(np.min(pbest_values))
        history: List[float] = [global_best_value]

        stagnation_window = max(self._max_iter // 5, 10 * D)

        for t in range(self._max_iter):
            # 動的 neighborhood size: 2 -> max (PDF1 §2.3)
            frac = min(1.0, t / max(self._max_iter - 1, 1))
            nsize = int(round(
                self._nsize_min + (self._nsize_max - self._nsize_min) * frac
            ))
            nsize = max(self._nsize_min, min(nsize, self._nsize_max))

            # 各粒子に対し近傍 nsize 個を pbest 空間から取る
            # (距離は正規化 Euclidean; pbest_positions 上で計算)
            for i in range(N):
                dists = np.array(
                    [
                        _normalized_distance(
                            pbest_positions[i], pbest_positions[j], bounds_lo, bounds_hi
                        )
                        for j in range(N)
                    ]
                )
                dists[i] = np.inf  # 自分を除外
                nn_idx = np.argpartition(dists, kth=nsize)[:nsize]

                # φ_j ∈ [0, 4.1/nsize], P_i = Σ (nbest_j * φ_j / φ_sum) / nsize (PDF1 Eq. 14)
                phi_max = 4.1 / nsize
                phis = self._rng.random(nsize) * phi_max
                phi_sum = float(np.sum(phis)) if np.sum(phis) > 0.0 else 1e-30
                weighted = np.sum(
                    pbest_positions[nn_idx] * (phis / phi_sum)[:, None], axis=0
                ) / nsize
                P_i = weighted

                # V = ω * (V + φ * (P - X))  ; PDF1 Eq. 13
                # ここで φ は scalar として phi_sum を採用 (PDF1 表記の "φ")
                velocities[i] = _INERTIA * (
                    velocities[i] + phi_sum * (P_i - positions[i])
                )
                positions[i] = positions[i] + velocities[i]
                # bounds clamp (PDF1 では明示されないが、探索範囲外は無意味なため射影)
                positions[i] = np.clip(positions[i], bounds_lo, bounds_hi)

                # 評価 + pbest 更新
                v_new = self._eval_misfit(
                    observed, init_model, positions[i], bounds_lo, bounds_hi
                )
                if v_new < pbest_values[i]:
                    pbest_values[i] = v_new
                    pbest_positions[i] = positions[i].copy()

            cur_best = float(np.min(pbest_values))
            history.append(cur_best)

            # 停滞判定 (PDF1 Eq. 15)
            if t > stagnation_window:
                if history[-stagnation_window] - history[-1] < self._eta:
                    return _PSORunResult(
                        best_position=pbest_positions[int(np.argmin(pbest_values))],
                        best_value=cur_best,
                        n_iter=t + 1,
                    )

        return _PSORunResult(
            best_position=pbest_positions[int(np.argmin(pbest_values))],
            best_value=float(np.min(pbest_values)),
            n_iter=self._max_iter,
        )

    def _eval_misfit(
        self,
        observed: PickedDispersionCurve,
        init_model: LayeredEarthModel,
        vs_vector: np.ndarray,
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
    ) -> float:
        """1 粒子分の misfit 評価。bounds 外で +inf を返す。"""
        if np.any(vs_vector < bounds_lo) or np.any(vs_vector > bounds_hi):
            return float("inf")
        cand_model = init_model.copy_with_vs(np.asarray(vs_vector, dtype=float))
        try:
            return float(self.misfit.value(observed, cand_model))
        except Exception:
            return float("inf")

    # =================================================================
    #  Modified BSAS clustering (PDF1 §2.3 Eq. 16)
    # =================================================================

    def _modified_bsas(
        self,
        positions: Sequence[np.ndarray],
        values: Sequence[float],
        bounds_lo: np.ndarray,
        bounds_hi: np.ndarray,
        D: int,
    ) -> List[List[np.ndarray]]:
        """Orujpour 2019 流の modified BSAS。閾値 = factor * sqrt(D)。

        positions / values は misfit 昇順でソート済を期待 (PDF1: 上位 1/4 を
        昇順に並べて先頭を最初の cluster head とする)。
        """
        if len(positions) == 0:
            return []
        threshold = self._cluster_threshold_factor * float(np.sqrt(D))

        clusters: List[List[np.ndarray]] = [[np.asarray(positions[0], dtype=float)]]
        heads: List[np.ndarray] = [np.asarray(positions[0], dtype=float)]

        for k in range(1, len(positions)):
            p = np.asarray(positions[k], dtype=float)
            min_dist = float("inf")
            min_idx = 0
            for hi, head in enumerate(heads):
                d = _normalized_distance(p, head, bounds_lo, bounds_hi)
                if d < min_dist:
                    min_dist = d
                    min_idx = hi
            if min_dist > threshold and len(heads) < self._max_clusters:
                clusters.append([p])
                heads.append(p)
            else:
                clusters[min_idx].append(p)

        return clusters

    # =================================================================
    #  run_profile: independent per-CMP PSO. LCI は別 engine。
    # =================================================================

    def run_profile(
        self,
        observations: Sequence[PickedDispersionCurve],
        init_models: Sequence[LayeredEarthModel],
        cmp_x: np.ndarray,
    ) -> List[RayleighInversionResult]:
        if len(observations) != len(init_models):
            raise ValueError("observations と init_models の長さが一致しません")
        results: List[RayleighInversionResult] = []
        for obs, m0 in zip(observations, init_models):
            results.append(self.run_single(obs, m0))
        return results
