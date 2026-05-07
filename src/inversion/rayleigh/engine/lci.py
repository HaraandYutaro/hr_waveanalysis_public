"""LCIEngine — Laterally Constrained Inversion (Auken & Christiansen 2004).

Step ?-8 契約 (DampedLeastSquaresEngine との比較):
  - DampedLSQ: 各 CMP を独立に 1D 逆解析する (pseudo-2D)。
  - LCI:       全 K CMP を **同時に** 逆解析する。
               各 CMP のモデル列を結合した m = [m_1; m_2; ...; m_K]
               (長さ K * n_layers) を構築し、隣接 CMP 間の同一層 Vs
               の差分にペナルティ (lateral smoothness) を課す。
               これにより地質学的整合性のとれた 2D 構造が得られる。

Objective:
    Phi(m) = || W_d (d - g(m)) ||^2
           + lambda_v || D_v m ||^2          # vertical roughness
           + lambda_h || D_h m ||^2          # lateral smoothness

  - W_d : misfit.sqrt_weights から構成される観測重み (block-diag per CMP)
  - D_v : 各 CMP 内の隣接層差分 (block-diag of (n_layers-1) x n_layers)
  - D_h : 隣接 CMP 間の同一層差分 ((K-1)*n_layers x K*n_layers)

Gauss-Newton 線形化:
  各反復で予測差分を線形化し、augmented least-squares を解く:

      | W_d J_full       |        | W_d (d - g(m_curr))  |
      | sqrt(lam_v) D_v  | dm  =  | -sqrt(lam_v) D_v m   |
      | sqrt(lam_h) D_h  |        | -sqrt(lam_h) D_h m   |

  J_full は block-diagonal (各 CMP の forward は独立)。
  解は use_sparse=True で scipy.sparse.linalg.lsqr、
       use_sparse=False で scipy.linalg.lstsq を用いる。

更新対象は Vs のみ。h / Vp / rho は init_models のものを保持して固定。
bounds (vs_bounds) は各反復後に np.clip で射影する。

停止条件:
  - n_iter >= max_iter
  - joint unweighted RMS <= rms_tol
  - step が改善せず joint RMS が増加 → 早期終了 (converged=False のままでよい)
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.linalg import lstsq

from src.inversion.rayleigh.engine.base import InversionEngine
from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    LCIProfileResult,
    PickedDispersionCurve,
    RayleighInversionResult,
)


_DEFAULT_OPTS = dict(
    lambda_v=1.0,
    lambda_h=1.0,
    max_iter=20,
    rms_tol=0.5,
    use_sparse=True,
)


class LCIEngine(InversionEngine):
    """Laterally Constrained Inversion (Vs-only update, h fixed)。"""

    def __init__(
        self,
        forward: RayleighForwardSolver,
        misfit: DispersionMisfit,
        *,
        lambda_v: float = 1.0,
        lambda_h: float = 1.0,
        max_iter: int = 20,
        rms_tol: float = 0.5,
        use_sparse: bool = True,
        **opts,
    ) -> None:
        merged = dict(_DEFAULT_OPTS)
        merged.update(opts)
        merged.update(
            dict(
                lambda_v=float(lambda_v),
                lambda_h=float(lambda_h),
                max_iter=int(max_iter),
                rms_tol=float(rms_tol),
                use_sparse=bool(use_sparse),
            )
        )
        super().__init__(forward=forward, misfit=misfit, **merged)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run_single(
        self,
        observed: PickedDispersionCurve,
        init_model: LayeredEarthModel,
    ) -> RayleighInversionResult:
        """単一 CMP に対しても LCI 形式で実行 (lateral 項は自動的に消える)。"""
        prof = self.run_profile(
            cmp_x=np.array([0.0]),
            picks=[observed],
            init_models=[init_model],
        )
        return prof.per_cmp_results[0]

    def run_profile(
        self,
        cmp_x: np.ndarray,
        picks: Sequence[PickedDispersionCurve],
        init_models: Sequence[LayeredEarthModel],
    ) -> LCIProfileResult:
        """全 K CMP を同時逆解析する LCI のメインルーチン。"""
        K = len(picks)
        if len(init_models) != K:
            raise ValueError(
                f"init_models length ({len(init_models)}) != picks length ({K})"
            )
        cmp_x = np.asarray(cmp_x, dtype=float)
        if cmp_x.shape[0] != K:
            raise ValueError(
                f"cmp_x length ({cmp_x.shape[0]}) != picks length ({K})"
            )
        if K == 0:
            raise ValueError("picks must contain at least one CMP")

        N = init_models[0].n_layers
        for k, m in enumerate(init_models):
            if m.n_layers != N:
                raise ValueError(
                    f"init_models[{k}] has {m.n_layers} layers, expected {N} "
                    f"(LCI requires all CMPs share n_layers)"
                )

        opts = self.opts
        lam_v = float(opts["lambda_v"])
        lam_h = float(opts["lambda_h"])
        max_iter = int(opts["max_iter"])
        rms_tol = float(opts["rms_tol"])
        use_sparse = bool(opts["use_sparse"])
        if lam_v < 0.0 or lam_h < 0.0:
            raise ValueError("lambda_v / lambda_h must be non-negative")

        bounds = [self._extract_bounds(m, N) for m in init_models]

        # 初期モデル (clip 済み) を作成
        models: List[LayeredEarthModel] = []
        for k in range(K):
            lo, hi = bounds[k]
            models.append(
                init_models[k].copy_with_vs(np.clip(init_models[k].vs, lo, hi))
            )

        # 差分作用素を一度だけ構成
        D_v = self._build_vertical_diff(N, K)   # (K*(N-1)) x (K*N)
        D_h = self._build_lateral_diff(N, K)    # ((K-1)*N) x (K*N)

        # misfit 整合性チェック (sqrt_weights 必須)
        sw_per_cmp: List[np.ndarray] = []
        for k in range(K):
            sw = self.misfit.sqrt_weights(picks[k])
            if sw is None:
                raise TypeError(
                    f"{type(self.misfit).__name__} does not provide sqrt_weights; "
                    "LCIEngine requires LSQ-compatible misfit "
                    "(must implement residual() and sqrt_weights())."
                )
            sw = np.asarray(sw, dtype=float)
            if sw.shape != picks[k].c.shape:
                raise ValueError(
                    f"sqrt_weights[{k}] shape {sw.shape} != observed.c shape "
                    f"{picks[k].c.shape}"
                )
            sw_per_cmp.append(sw)

        # 初期 forward 評価
        per_cmp_predicted: List[np.ndarray] = []
        per_cmp_rms: List[float] = []
        for k in range(K):
            pred = self.forward.forward(models[k], picks[k].f, mode=picks[k].mode)
            per_cmp_predicted.append(pred)
            per_cmp_rms.append(self._unweighted_rms(picks[k].c, pred))

        joint_rms = self._joint_rms(picks, per_cmp_predicted)
        joint_rms_history: List[float] = [joint_rms]

        converged = joint_rms <= rms_tol
        n_iter_total = 0

        for it in range(1, max_iter + 1):
            if converged:
                break

            # 各 CMP の重み付き Jacobian と weighted residual を組む
            jac_blocks = []
            r_w_blocks = []
            for k in range(K):
                jac_raw = self.forward.jacobian(
                    models[k], picks[k].f, mode=picks[k].mode
                )
                sw = sw_per_cmp[k]
                jac_w = sw[:, None] * jac_raw
                r_w = self.misfit.residual(picks[k], per_cmp_predicted[k])
                jac_blocks.append(sp.csr_matrix(jac_w))
                r_w_blocks.append(r_w)

            J_full = sp.block_diag(jac_blocks, format="csr")
            r_w_full = np.concatenate(r_w_blocks)
            m_curr = np.concatenate([models[k].vs for k in range(K)])

            # augmented system: [J_full ; sqrt(lam_v) D_v ; sqrt(lam_h) D_h]
            blocks = [J_full]
            rhs_blocks = [r_w_full]
            sqrt_v = float(np.sqrt(lam_v))
            sqrt_h = float(np.sqrt(lam_h))
            if D_v.shape[0] > 0 and sqrt_v > 0.0:
                blocks.append(sqrt_v * D_v)
                rhs_blocks.append(-sqrt_v * (D_v @ m_curr))
            if D_h.shape[0] > 0 and sqrt_h > 0.0:
                blocks.append(sqrt_h * D_h)
                rhs_blocks.append(-sqrt_h * (D_h @ m_curr))

            A = sp.vstack(blocks, format="csr")
            b = np.concatenate(rhs_blocks)

            step = self._solve_least_squares(A, b, use_sparse=use_sparse)

            # bound projection
            new_models: List[LayeredEarthModel] = []
            for k in range(K):
                dm_k = step[k * N:(k + 1) * N]
                lo, hi = bounds[k]
                trial_vs = np.clip(models[k].vs + dm_k, lo, hi)
                new_models.append(models[k].copy_with_vs(trial_vs))

            new_predicted: List[np.ndarray] = []
            new_rms: List[float] = []
            for k in range(K):
                pred = self.forward.forward(
                    new_models[k], picks[k].f, mode=picks[k].mode
                )
                new_predicted.append(pred)
                new_rms.append(self._unweighted_rms(picks[k].c, pred))
            new_joint = self._joint_rms(picks, new_predicted)

            # joint RMS が改善したら受理。改善しなければ早期終了。
            if new_joint < joint_rms:
                models = new_models
                per_cmp_predicted = new_predicted
                per_cmp_rms = new_rms
                joint_rms = new_joint
                n_iter_total = it
                joint_rms_history.append(joint_rms)
                if joint_rms <= rms_tol:
                    converged = True
                    break
            else:
                joint_rms_history.append(joint_rms)
                break

        # per-CMP results を構築 (最終モデルでの Jacobian を再評価)
        per_cmp_results: List[RayleighInversionResult] = []
        for k in range(K):
            jac_final = self.forward.jacobian(
                models[k], picks[k].f, mode=picks[k].mode
            )
            residuals = picks[k].c - per_cmp_predicted[k]
            per_cmp_results.append(
                RayleighInversionResult(
                    model=models[k],
                    predicted_c=per_cmp_predicted[k],
                    residuals=residuals,
                    rms=per_cmp_rms[k],
                    n_iter=n_iter_total,
                    converged=converged,
                    jacobian=jac_final,
                    posterior_cov=None,
                    metadata=dict(
                        engine="lci",
                        cmp_index=k,
                        cmp_x=float(cmp_x[k]),
                        lambda_v=lam_v,
                        lambda_h=lam_h,
                        joint_rms_history=list(joint_rms_history),
                    ),
                )
            )

        return LCIProfileResult(
            per_cmp_results=per_cmp_results,
            joint_rms_history=joint_rms_history,
            lambda_v=lam_v,
            lambda_h=lam_h,
            converged=converged,
            n_iter=n_iter_total,
            metadata=dict(
                engine="lci",
                K=K,
                n_layers=N,
                use_sparse=use_sparse,
                opts=dict(self.opts),
            ),
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_bounds(
        model: LayeredEarthModel, n: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        if model.vs_bounds is None:
            return (np.full(n, -np.inf), np.full(n, np.inf))
        return (model.vs_bounds[0].copy(), model.vs_bounds[1].copy())

    @staticmethod
    def _build_vertical_diff(N: int, K: int) -> sp.csr_matrix:
        """各 CMP 内の隣接層差分 D_v_single = [(N-1) x N] を K 個 block-diag。"""
        if N < 2 or K < 1:
            return sp.csr_matrix((0, K * N), dtype=float)
        rows = np.repeat(np.arange(N - 1), 2)
        cols = np.empty(2 * (N - 1), dtype=int)
        data = np.empty(2 * (N - 1), dtype=float)
        for i in range(N - 1):
            cols[2 * i] = i
            cols[2 * i + 1] = i + 1
            data[2 * i] = -1.0
            data[2 * i + 1] = 1.0
        D_v_single = sp.csr_matrix((data, (rows, cols)), shape=(N - 1, N))
        return sp.block_diag([D_v_single] * K, format="csr")

    @staticmethod
    def _build_lateral_diff(N: int, K: int) -> sp.csr_matrix:
        """隣接 CMP 間の同一層差分。shape: ((K-1)*N) x (K*N)。"""
        if K < 2 or N < 1:
            return sp.csr_matrix((0, K * N), dtype=float)
        n_rows = (K - 1) * N
        rows = np.repeat(np.arange(n_rows), 2)
        cols = np.empty(2 * n_rows, dtype=int)
        data = np.empty(2 * n_rows, dtype=float)
        idx = 0
        for k in range(K - 1):
            for i in range(N):
                cols[idx] = k * N + i
                cols[idx + 1] = (k + 1) * N + i
                data[idx] = -1.0
                data[idx + 1] = 1.0
                idx += 2
        return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, K * N))

    @staticmethod
    def _solve_least_squares(
        A: sp.csr_matrix,
        b: np.ndarray,
        *,
        use_sparse: bool,
    ) -> np.ndarray:
        """augmented least-squares dm = argmin || A dm - b ||^2 を解く。"""
        if use_sparse:
            sol = spla.lsqr(A, b, atol=1e-10, btol=1e-10)
            return np.asarray(sol[0], dtype=float)
        sol, *_ = lstsq(A.toarray(), b)
        return np.asarray(sol, dtype=float)

    @staticmethod
    def _unweighted_rms(c_obs: np.ndarray, c_pred: np.ndarray) -> float:
        diff = c_obs - c_pred
        return float(np.sqrt(np.mean(diff * diff)))

    @staticmethod
    def _joint_rms(
        picks: Sequence[PickedDispersionCurve],
        predicted_list: Sequence[np.ndarray],
    ) -> float:
        """全 CMP 残差を結合した重みなし joint RMS [m/s]。"""
        all_resid = np.concatenate(
            [picks[k].c - predicted_list[k] for k in range(len(picks))]
        )
        return float(np.sqrt(np.mean(all_resid * all_resid)))
