"""DampedLeastSquaresEngine — Marquardt-damped Gauss-Newton for Vs-only update.

Step ?-1a 契約 (misfit との整合):
  - misfit.residual(observed, predicted)        -> r_w = sqrt(W) * (obs - pred)
  - misfit.sqrt_weights(observed)               -> sqrt(W)  (residual と同じ W)
  - forward.jacobian(model, freqs)              -> J_raw  (∂c/∂Vs_j, 重みなし)
  - engine が J_w = sqrt(W)[:, None] * J_raw として weighted Jacobian を構成
  - normal equation:
        ( J_w^T J_w + lambda I ) Δm = J_w^T r_w
        ⇔  ( J^T W J + lambda I ) Δm = J^T W (obs - pred)
  - **residual と Jacobian に同じ sqrt(W) が掛かることが整合の必要条件**。
    misfit と engine の双方でこれを破らないこと。

Reference (Xia, Miller & Park 1999) との相違 [Plan.md P2-B]:
  本実装は正規方程式 (J^T W J + lambda I) を直接解く Marquardt-damped Gauss-Newton。
  Xia99 は等価な重み付き減衰最小二乗を SVD (式5) で解き、減衰因子を変えながら
  分解能 / 分散のトレードオフを評価する。本実装は SVD と分解能解析を持たない
  (posterior_cov は None)。式の骨格は一致し、解そのものに誤りはない。

更新対象は Vs のみ。h / Vp / ρ は init_model のものを保持して固定する。
bounds (vs_bounds) は各反復後に np.clip で射影する単純 bound projection。

停止条件 (いずれかで停止):
  - n_iter >= max_iter
  - rms <= rms_tol                              (重みなし RMS [m/s])
  - ||Δm|| / ||m|| <= step_tol
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np

from src.inversion.rayleigh.engine.base import InversionEngine
from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    PickedDispersionCurve,
    RayleighInversionResult,
)

try:
    from disba import DispersionError as _DisbaDispersionError
    _FORWARD_ERRORS = (_DisbaDispersionError, ValueError, RuntimeError)
except ImportError:
    _FORWARD_ERRORS = (ValueError, RuntimeError)


_DEFAULT_OPTS = dict(
    lambda0=1e-2,
    lambda_inc=2.0,
    lambda_dec=0.5,
    lambda_min=1e-8,
    lambda_max=1e+8,
    max_iter=20,
    max_inner=10,        # Marquardt 増加方向のリトライ上限 / iter
    rms_tol=1e-3,        # m/s
    step_tol=1e-4,       # 相対 ||Δm||/||m||
)


class DampedLeastSquaresEngine(InversionEngine):
    """Marquardt-damped Gauss-Newton (Vs-only update, h fixed)。"""

    def __init__(
        self,
        forward: RayleighForwardSolver,
        misfit: DispersionMisfit,
        **opts,
    ) -> None:
        merged = dict(_DEFAULT_OPTS)
        merged.update(opts)
        super().__init__(forward=forward, misfit=misfit, **merged)

    # ------------------------------------------------------------------
    # public
    # ------------------------------------------------------------------

    def run_single(
        self,
        observed: PickedDispersionCurve,
        init_model: LayeredEarthModel,
    ) -> RayleighInversionResult:
        sw = self.misfit.sqrt_weights(observed)
        if sw is None:
            raise TypeError(
                f"{type(self.misfit).__name__} does not provide sqrt_weights; "
                "DampedLeastSquaresEngine requires LSQ-compatible misfit "
                "(must implement residual() and sqrt_weights())."
            )
        sw = np.asarray(sw, dtype=float)
        if sw.shape != observed.c.shape:
            raise ValueError(
                f"sqrt_weights shape {sw.shape} != observed.c shape {observed.c.shape}"
            )

        opts = self.opts
        lo, hi = self._extract_bounds(init_model)

        # 初期評価
        model = init_model.copy_with_vs(np.clip(init_model.vs, lo, hi))

        # PSO の方針に合わせた安全処理: 初期 forward 失敗時は converged=False で即返却
        try:
            predicted = self.forward.forward(model, observed.f, mode=observed.mode)
        except _FORWARD_ERRORS as e:
            warnings.warn(
                f"DampedLSQ: initial forward failed ({type(e).__name__}: {e}). "
                "Returning unconverged result.",
                stacklevel=2,
            )
            return RayleighInversionResult(
                model=init_model,
                predicted_c=np.full_like(observed.c, np.nan),
                residuals=np.full_like(observed.c, np.nan),
                rms=float("inf"),
                n_iter=0,
                converged=False,
                jacobian=None,
                posterior_cov=None,
                metadata=dict(
                    engine="damped_lsq",
                    forward_failed=True,
                    forward_error=str(e),
                ),
            )

        rms = self._unweighted_rms(observed.c, predicted)

        best_model = model
        best_predicted = predicted
        best_rms = rms
        best_jac = None
        best_iter = 0

        lam = float(opts["lambda0"])
        rms_history = [rms]
        lam_history = [lam]
        converged = False

        for it in range(1, int(opts["max_iter"]) + 1):
            # Jacobian 計算 — forward と同様にエラーを捕捉して安全に停止
            # PSO の方針に合わせた安全処理: Jacobian 失敗時は反復を打ち切る
            try:
                jac_raw = self.forward.jacobian(model, observed.f, mode=observed.mode)
            except _FORWARD_ERRORS as e:
                warnings.warn(
                    f"DampedLSQ iter {it}: jacobian failed ({type(e).__name__}: {e}). "
                    "Stopping iteration.",
                    stacklevel=2,
                )
                break

            r_w = self.misfit.residual(observed, predicted)
            jac_w = sw[:, None] * jac_raw  # weighted Jacobian (engine 責務)

            # Marquardt inner loop: lambda を増加させながら受理ステップを探す
            accepted = False
            for _ in range(int(opts["max_inner"])):
                step = self._solve_normal_equation(jac_w, r_w, lam)
                trial_vs = np.clip(model.vs + step, lo, hi)
                trial_model = model.copy_with_vs(trial_vs)

                # PSO の方針に合わせた安全処理: trial forward 失敗は np.inf でペナルティ
                try:
                    trial_predicted = self.forward.forward(
                        trial_model, observed.f, mode=observed.mode
                    )
                    trial_rms = self._unweighted_rms(observed.c, trial_predicted)
                except _FORWARD_ERRORS:
                    trial_rms = float("inf")
                    trial_predicted = None

                if trial_rms < rms:
                    # 受理
                    rel_step = float(np.linalg.norm(step) / max(np.linalg.norm(model.vs), 1e-30))
                    model = trial_model
                    predicted = trial_predicted
                    rms = trial_rms
                    lam = max(lam * float(opts["lambda_dec"]), float(opts["lambda_min"]))
                    accepted = True

                    if rms < best_rms:
                        best_model = model
                        best_predicted = predicted
                        best_rms = rms
                        best_jac = jac_raw
                        best_iter = it

                    rms_history.append(rms)
                    lam_history.append(lam)

                    if rms <= float(opts["rms_tol"]) or rel_step <= float(opts["step_tol"]):
                        converged = True
                    break
                else:
                    # 棄却 → lambda を増やしてやり直し
                    lam = min(lam * float(opts["lambda_inc"]), float(opts["lambda_max"]))
                    rms_history.append(rms)
                    lam_history.append(lam)

            if not accepted:
                # 内側で改善できなかったら大域停止 (半収束)
                break
            if converged:
                break

        if best_jac is None:
            # 一度も改善せず終了したケース。最終的な Jacobian を best_model で評価。
            # PSO の方針に合わせた安全処理: Jacobian 失敗時は None のまま返す
            try:
                best_jac = self.forward.jacobian(best_model, observed.f, mode=observed.mode)
            except _FORWARD_ERRORS:
                best_jac = None
            best_iter = 0

        residuals = observed.c - best_predicted
        return RayleighInversionResult(
            model=best_model,
            predicted_c=best_predicted,
            residuals=residuals,
            rms=best_rms,
            n_iter=best_iter,
            converged=converged,
            jacobian=best_jac,
            posterior_cov=None,
            metadata=dict(
                engine="damped_lsq",
                rms_history=rms_history,
                lambda_history=lam_history,
                opts=dict(self.opts),
            ),
        )

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _solve_normal_equation(
        jac_w: np.ndarray,
        r_w: np.ndarray,
        lam: float,
    ) -> np.ndarray:
        """ ( J_w^T J_w + lambda I ) Δm = J_w^T r_w を解く。"""
        n = jac_w.shape[1]
        gram = jac_w.T @ jac_w + lam * np.eye(n)
        rhs = jac_w.T @ r_w
        return np.linalg.solve(gram, rhs)

    @staticmethod
    def _unweighted_rms(c_obs: np.ndarray, c_pred: np.ndarray) -> float:
        """重みなし残差の RMS [m/s]。停止条件・report 用。"""
        diff = c_obs - c_pred
        return float(np.sqrt(np.mean(diff * diff)))

    @staticmethod
    def _extract_bounds(model: LayeredEarthModel) -> Tuple[np.ndarray, np.ndarray]:
        if model.vs_bounds is None:
            n = model.n_layers
            return (np.full(n, -np.inf), np.full(n, np.inf))
        return model.vs_bounds
