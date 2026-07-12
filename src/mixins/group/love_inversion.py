"""LoveInversion mixin for GroupProcesser.

Rayleigh wave 用 ``src.mixins.group.rayleigh_inversion.RayleighInversion`` と
対称な構造で、SH-Love wave の per-CMP 1D inversion + pseudo-2D / LCI 写像を
GroupProcesser に提供する薄い orchestration layer。

責務
----
- forward / misfit / engine の文字列 -> クラス解決 (Love 専用 registry)
- 初期モデル生成は ``build_default_love_init_model`` に委譲
- 一元化 API ``love_inversion(...)`` で
  CMP 重合 -> 分散曲線 -> QC -> 初期モデル生成 -> PSO -> Pseudo-2D 写像
  を内部オーケストレーション
- 個別 API ``love_vs_inversion_1d`` / ``love_vs_inversion_profile`` も提供

公開挙動の境界
--------------
- 既存 RayleighInversion mixin の公開メソッドは一切呼ばないし変えない。
- 分散曲線抽出は GroupDispersion.dispersion_curve(axis="y", ...) を再利用。
  axis="y" (SH 成分) を既定とすることで「SH-Love wave 」として運用する
  (examples/quickstart_group.py と同じ運用慣行)。
- QC は ``quality_control_dispersion_pick`` を共用。Hessian 系 picker への
  差し替えは後続 slice (Hou et al. 2025; DOI:10.1038/s41598-025-04954-w)。

References
----------
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380. (Niching PSO + cluster
  analysis + Love wave determinant misfit)
- Wilken, D. & Rabbel, W. (2012). Geophys. J. Int., 190, 580-594.
  (Nearest-neighbor misfit; PSO 後のクラスタランキングで利用)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from src.inversion.love.forward.thomson_haskell_love import ThomsonHaskellLoveSolver
from src.inversion.love.init_model import build_default_love_init_model
from src.inversion.love.secular import love_secular_value
from src.inversion.rayleigh.engine.base import InversionEngine
from src.inversion.rayleigh.engine.pso import NichingPSOEngine
from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.misfit.determinant import DeterminantMisfit
from src.inversion.rayleigh.misfit.nearest_neighbor import NearestNeighborMisfit
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    PickedDispersionCurve,
    Pseudo2DVsSection,
    RayleighInversionResult,
)
from src.inversion.rayleigh.picking_qc import (
    DispersionPickQCResult,
    quality_control_dispersion_pick,
)
from src.inversion.rayleigh.section import Pseudo2DSectionBuilder
from src.plotting.wrapper import PlotterWrapperMixin


# Love 専用 misfit registry。
# - "determinant"     : PDF1 Eq. 10 を Love secular 関数で実装したもの (PSO 内 loop 用)
# - "nearest_neighbor": PDF1 Eq. 8。PSO 内 loop でも使えるが計算重め (root-finding 経由)。
def _make_determinant_misfit() -> DeterminantMisfit:
    return DeterminantMisfit(love_secular_value, normalize="log")


def _make_nearest_neighbor_misfit(
    forward: ThomsonHaskellLoveSolver,
    max_mode: int = 4,
) -> NearestNeighborMisfit:
    return NearestNeighborMisfit(
        predict_modes_func=forward.predict_modes,
        max_mode=max_mode,
    )


_FORWARD_REGISTRY: Dict[str, Any] = {
    "thomson_haskell_love": ThomsonHaskellLoveSolver,
}

_MISFIT_REGISTRY: Dict[str, Any] = {
    "determinant": _make_determinant_misfit,
    # "nearest_neighbor" は forward が必要なので factory ではなく
    # _resolve_misfit 内で個別生成する。
}

_ENGINE_REGISTRY: Dict[str, Any] = {
    "niching_pso": NichingPSOEngine,
}


def _resolve_forward(
    spec: Union[str, RayleighForwardSolver],
) -> RayleighForwardSolver:
    if isinstance(spec, RayleighForwardSolver):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"forward must be str or RayleighForwardSolver, got {type(spec).__name__}"
        )
    cls = _FORWARD_REGISTRY.get(spec)
    if cls is None:
        raise ValueError(
            f"unknown Love forward name {spec!r}; available: {sorted(_FORWARD_REGISTRY)}"
        )
    return cls()


def _resolve_misfit(
    spec: Union[str, DispersionMisfit],
    forward_obj: ThomsonHaskellLoveSolver,
    max_mode: int,
) -> DispersionMisfit:
    if isinstance(spec, DispersionMisfit):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"misfit must be str or DispersionMisfit, got {type(spec).__name__}"
        )
    if spec == "determinant":
        return _make_determinant_misfit()
    if spec == "nearest_neighbor":
        return _make_nearest_neighbor_misfit(forward_obj, max_mode=max_mode)
    raise ValueError(
        f"unknown Love misfit name {spec!r}; "
        "available: ['determinant', 'nearest_neighbor']"
    )


def _resolve_engine_cls(
    spec: Union[str, InversionEngine, type],
) -> Tuple[str, Any]:
    if isinstance(spec, InversionEngine):
        return ("instance", spec)
    if isinstance(spec, type) and issubclass(spec, InversionEngine):
        return ("class", spec)
    if isinstance(spec, str):
        cls = _ENGINE_REGISTRY.get(spec)
        if cls is None:
            raise ValueError(
                f"unknown Love engine name {spec!r}; "
                f"available: {sorted(_ENGINE_REGISTRY)}"
            )
        return ("class", cls)
    raise TypeError(
        f"engine must be str, InversionEngine subclass, or instance; "
        f"got {type(spec).__name__}"
    )


class LoveInversion(PlotterWrapperMixin):
    """GroupProcesser に SH-Love wave Vs 1D 逆解析メソッドを提供する mixin。"""

    # =================================================================
    #  Atomic 1-CMP API
    # =================================================================

    def love_vs_inversion_1d(
        self,
        target,
        picked: PickedDispersionCurve,
        *,
        init_model: Optional[LayeredEarthModel] = None,
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell_love",
        misfit: Union[str, DispersionMisfit] = "determinant",
        engine: Union[str, InversionEngine, type] = "niching_pso",
        n_layers: int = 10,
        higher_mode_max: int = 4,
        **engine_opts,
    ) -> RayleighInversionResult:
        """1 CMP 分の SH-Love wave Vs 逆解析を実行する。

        Parameters
        ----------
        target : Any
            CMP 識別子 (metadata タグとしてのみ保持)。
        picked : PickedDispersionCurve
            観測ピッキング済み Love 分散曲線。
        init_model : LayeredEarthModel, optional
            初期モデル。None の場合 ``build_default_love_init_model`` で生成。
        forward, misfit, engine : str or instance
            registry 経由で実体化、または instance を直接渡す。
            既定は ``"thomson_haskell_love"`` / ``"determinant"`` / ``"niching_pso"``。
        n_layers : int
            init_model 自動生成時の層数。
        higher_mode_max : int
            nearest_neighbor misfit / cluster ranking 用の高次モード探索上限。
            既定 4。**高次モードが disba root-finder で取れない周波数帯では
            自動的にスキップされる** ため、本値は「上限」を意味する。
        **engine_opts
            engine class に転送する追加 option (PSO の n_runs, max_iter, rng_seed 等)。

        Returns
        -------
        RayleighInversionResult
            ``metadata`` に ``cmp_target`` / ``forward_name`` / ``misfit_name`` /
            ``engine_name`` / ``wave="love"`` を付与。
        """
        fwd_obj = _resolve_forward(forward)
        if not isinstance(fwd_obj, ThomsonHaskellLoveSolver):
            # nearest_neighbor misfit が forward.predict_modes に依存するため、
            # 現状 ThomsonHaskellLoveSolver 以外の forward は未サポート。
            warnings.warn(
                "Non-ThomsonHaskellLoveSolver forward used with LoveInversion; "
                "nearest_neighbor misfit may fail.",
                stacklevel=2,
            )
        mis_obj = _resolve_misfit(misfit, fwd_obj, max_mode=higher_mode_max)
        engine_kind, engine_arg = _resolve_engine_cls(engine)

        if engine_kind == "instance":
            eng = engine_arg
        else:
            # NichingPSOEngine は forward + misfit を必須に取り、
            # ranking_misfit はオプション (nearest_neighbor を別途生成して inject)。
            ranking_misfit = engine_opts.pop("ranking_misfit", None)
            if ranking_misfit is None and isinstance(fwd_obj, ThomsonHaskellLoveSolver):
                ranking_misfit = _make_nearest_neighbor_misfit(
                    fwd_obj, max_mode=higher_mode_max
                )
            eng = engine_arg(
                forward=fwd_obj,
                misfit=mis_obj,
                ranking_misfit=ranking_misfit,
                **engine_opts,
            )

        if init_model is None:
            init_model = build_default_love_init_model(picked, n_layers=n_layers)

        result = eng.run_single(picked, init_model)

        result.metadata["cmp_target"] = target
        result.metadata["wave"] = "love"
        result.metadata["forward_name"] = (
            forward if isinstance(forward, str) else type(fwd_obj).__name__
        )
        result.metadata["misfit_name"] = (
            misfit if isinstance(misfit, str) else type(mis_obj).__name__
        )
        result.metadata["engine_name"] = (
            engine if isinstance(engine, str) else type(eng).__name__
        )
        return result

    # =================================================================
    #  Profile API (independent per-CMP PSO + pseudo-2D)
    # =================================================================

    def love_vs_inversion_profile(
        self,
        cmp_x: np.ndarray,
        picks: List[PickedDispersionCurve],
        initmodels: Optional[List[LayeredEarthModel]] = None,
        *,
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell_love",
        misfit: Union[str, DispersionMisfit] = "determinant",
        engine: Union[str, InversionEngine, type] = "niching_pso",
        n_layers: int = 10,
        higher_mode_max: int = 4,
        sensitivitycutoff: Optional[float] = None,
        maxdepthfactor: float = 2.5,
        **engine_opts,
    ) -> Pseudo2DVsSection:
        """複数 CMP の Love-wave Vs 1D 逆解析を実行し、pseudo-2D section を返す。

        各 CMP に対して ``love_vs_inversion_1d`` を呼び出し、
        ``Pseudo2DSectionBuilder`` で pseudo-2D section を組み立てる。
        ``initmodels`` を省略した場合、最初の CMP の picks から共通初期モデルを
        生成し、各 CMP に独立コピーを渡す (層構造 h の整合保証 + in-place 汚染回避)。
        """
        K = len(picks)
        cmp_x = np.asarray(cmp_x, dtype=float)
        if cmp_x.shape[0] != K:
            raise ValueError(
                f"cmp_x length ({cmp_x.shape[0]}) != picks length ({K})"
            )
        if K == 0:
            raise ValueError("picks must contain at least one PickedDispersionCurve")
        if initmodels is not None and len(initmodels) != K:
            raise ValueError(
                f"initmodels length ({len(initmodels)}) != picks length ({K})"
            )

        if initmodels is None:
            tmpl = build_default_love_init_model(picks[0], n_layers=n_layers)
            initmodels = [tmpl.copy_with_vs(tmpl.vs.copy()) for _ in range(K)]

        results: List[RayleighInversionResult] = []
        for i in range(K):
            r = self.love_vs_inversion_1d(
                target=float(cmp_x[i]),
                picked=picks[i],
                init_model=initmodels[i],
                forward=forward,
                misfit=misfit,
                engine=engine,
                higher_mode_max=higher_mode_max,
                **engine_opts,
            )
            results.append(r)

        return Pseudo2DSectionBuilder.build(
            cmpx=cmp_x,
            results=results,
            sensitivitycutoff=sensitivitycutoff,
            maxdepthfactor=maxdepthfactor,
        )

    # =================================================================
    #  Unified one-shot API (CMP 重合 -> 分散 -> QC -> PSO -> Pseudo2D)
    # =================================================================

    def love_inversion(
        self,
        target: Union[None, float, Sequence[float], Literal["all"]] = "all",
        *,
        axis: str = "y",
        freq_range: Tuple[float, float] = (1.0, 100.0),
        vel_range: Tuple[float, float] = (50.0, 800.0),
        df: float = 0.5,
        dc: float = 1.0,
        qc_min_energy_ratio: float = 0.05,
        qc_continuity_rel_jump: float = 0.15,
        qc_max_secondary_peak_ratio: float = 0.85,
        qc_min_valid_points: int = 8,
        n_freq_resample: Optional[int] = 30,
        init_model: Optional[LayeredEarthModel] = None,
        n_layers: int = 10,
        vp_vs_ratio: float = float(np.sqrt(3.0)),
        rho: float = 2000.0,
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell_love",
        misfit: Union[str, DispersionMisfit] = "determinant",
        engine: Union[str, InversionEngine, type] = "niching_pso",
        higher_mode_max: int = 4,
        pso_n_runs: int = 50,
        pso_population_size: Optional[int] = None,
        pso_max_iter: int = 300,
        pso_eta: float = 1e-4,
        pso_rng_seed: Optional[int] = None,
        sensitivitycutoff: Optional[float] = 0.1,
        depth_fraction: float = 0.5,
        cmp_kwargs: Optional[dict] = None,
        show: bool = False,
        save_name: Optional[str] = None,
        return_intermediate: bool = False,
    ) -> Union[
        Pseudo2DVsSection,
        RayleighInversionResult,
        Dict[str, Any],
    ]:
        """SH-Love wave Vs 逆解析を 1 メソッドで完結させる一元化 API。

        既存の cmp_gathering / dispersion_curve / QC / 初期モデル生成 /
        niching PSO inversion / Pseudo2DSectionBuilder を内部で自動オーケストレーション。

        target の解決:
          - None / "all"      : self.targets 全部 (未設定なら cmp_gathering 自動実行)
          - float             : 単一 CMP (最近傍の self.targets[i] にスナップ)
          - Sequence[float]   : 指定 CMP 群

        Returns
        -------
        target が単一 float                         -> RayleighInversionResult
        target が複数                               -> Pseudo2DVsSection
        return_intermediate=True                    -> 中間物を含む dict
        """
        is_single = isinstance(target, (int, float)) and not isinstance(target, bool)

        cmp_targets = self._resolve_targets_love(target, axis=axis, cmp_kwargs=cmp_kwargs)

        cmp_x_list, picks_list, qc_results = self._collect_picks_with_qc_love(
            cmp_targets,
            freq_range=freq_range,
            vel_range=vel_range,
            df=df,
            dc=dc,
            qc_min_energy_ratio=qc_min_energy_ratio,
            qc_continuity_rel_jump=qc_continuity_rel_jump,
            qc_max_secondary_peak_ratio=qc_max_secondary_peak_ratio,
            qc_min_valid_points=qc_min_valid_points,
            n_freq_resample=n_freq_resample,
        )

        if len(picks_list) == 0:
            raise RuntimeError(
                "No valid Love dispersion picks survived QC. "
                "Loosen QC thresholds (qc_*) or widen freq_range/vel_range."
            )

        init_models = self._build_love_init_models(
            picks_list,
            init_model=init_model,
            n_layers=n_layers,
            vp_vs_ratio=vp_vs_ratio,
            rho=rho,
        )

        engine_opts = dict(
            n_runs=pso_n_runs,
            population_size=pso_population_size,
            max_iter=pso_max_iter,
            eta=pso_eta,
            rng_seed=pso_rng_seed,
        )

        inv_result = self._run_love_inversion(
            cmp_x_list=cmp_x_list,
            picks_list=picks_list,
            init_models=init_models,
            is_single=is_single,
            forward=forward,
            misfit=misfit,
            engine=engine,
            higher_mode_max=higher_mode_max,
            engine_opts=engine_opts,
            sensitivitycutoff=sensitivitycutoff,
            depth_fraction=depth_fraction,
            picks_for_section=picks_list,
        )

        if (show or save_name) and isinstance(inv_result, Pseudo2DVsSection):
            self._run_love_with_plot(inv_result, show=show, save_name=save_name)

        if return_intermediate:
            return {
                "result": inv_result,
                "section": inv_result if isinstance(inv_result, Pseudo2DVsSection) else None,
                "single_result": (
                    inv_result if isinstance(inv_result, RayleighInversionResult) else None
                ),
                "cmp_x": cmp_x_list,
                "picks": picks_list,
                "qc": qc_results,
                "init_models": init_models,
            }
        return inv_result

    # =================================================================
    #  Internal helpers
    # =================================================================

    def _resolve_targets_love(
        self,
        target: Union[None, float, Sequence[float], Literal["all"]],
        *,
        axis: str,
        cmp_kwargs: Optional[dict],
    ) -> List[float]:
        if not hasattr(self, "targets"):
            defaults = dict(
                axis=axis,
                closs_corr=False,
                integrate=False,
                average=True,
                show=False,
                save_name=None,
            )
            if cmp_kwargs:
                defaults.update(cmp_kwargs)
            self.cmp_gathering(**defaults)

        all_targets = np.asarray(self.targets, dtype=float)
        if all_targets.size == 0:
            raise RuntimeError("self.targets is empty; cmp_gathering produced no CMPs")

        if target is None or (isinstance(target, str) and target == "all"):
            return [float(t) for t in all_targets]

        cmp_interval = (
            float(np.median(np.diff(all_targets))) if all_targets.size >= 2 else 0.0
        )
        snap_tol = 0.5 * cmp_interval if cmp_interval > 0 else float("inf")

        def _snap(x: float) -> float:
            idx = int(np.argmin(np.abs(all_targets - x)))
            snapped = float(all_targets[idx])
            if abs(snapped - x) > snap_tol:
                warnings.warn(
                    f"target={x} snapped to nearest CMP {snapped} "
                    f"(distance {abs(snapped - x):.3f} > half CMP interval {snap_tol:.3f})",
                    stacklevel=3,
                )
            return snapped

        if isinstance(target, (int, float)) and not isinstance(target, bool):
            return [_snap(float(target))]

        if isinstance(target, Sequence) and not isinstance(target, str):
            return [_snap(float(x)) for x in target]

        raise TypeError(
            f"target must be None / 'all' / float / Sequence[float]; "
            f"got {type(target).__name__}"
        )

    def _collect_picks_with_qc_love(
        self,
        cmp_targets: List[float],
        *,
        freq_range: Tuple[float, float],
        vel_range: Tuple[float, float],
        df: float,
        dc: float,
        qc_min_energy_ratio: float,
        qc_continuity_rel_jump: float,
        qc_max_secondary_peak_ratio: float,
        qc_min_valid_points: int,
        n_freq_resample: Optional[int],
    ) -> Tuple[List[float], List[PickedDispersionCurve], List[DispersionPickQCResult]]:
        cmp_x_list: List[float] = []
        picks_list: List[PickedDispersionCurve] = []
        qc_results: List[DispersionPickQCResult] = []

        for tgt in cmp_targets:
            res, f_mesh, c_mesh, dx_cmp, n_sensors_cmp = self.dispersion_curve(
                tgt,
                freq=list(freq_range),
                c=list(vel_range),
                df=df,
                dc=dc,
                show=False,
                cut_nyquist=True,
            )
            if res is None:
                warnings.warn(
                    f"CMP {tgt}: dispersion_curve failed (insufficient traces); skipped",
                    stacklevel=3,
                )
                continue

            dx_arg = dx_cmp if (not np.isnan(dx_cmp) and dx_cmp > 0) else None
            nr_arg = (
                int(n_sensors_cmp)
                if (n_sensors_cmp is not None and n_sensors_cmp > 0)
                else None
            )

            qc = quality_control_dispersion_pick(
                res,
                f_mesh,
                c_mesh,
                dx=dx_arg,
                n_receivers=nr_arg,
                min_energy_ratio=qc_min_energy_ratio,
                continuity_rel_jump=qc_continuity_rel_jump,
                max_secondary_peak_ratio=qc_max_secondary_peak_ratio,
                min_valid_points=qc_min_valid_points,
            )

            if qc.picked is None:
                warnings.warn(
                    f"CMP {tgt}: QC produced no valid Love points; skipped",
                    stacklevel=3,
                )
                continue

            if n_freq_resample is not None:
                f_raw = np.asarray(qc.picked.f, dtype=float)
                c_raw = np.asarray(qc.picked.c, dtype=float)
                n_rs = min(int(n_freq_resample), f_raw.size)
                f_rs = np.linspace(f_raw[0], f_raw[-1], n_rs)
                c_rs = np.interp(f_rs, f_raw, c_raw)
                cstd_rs = (
                    np.interp(f_rs, f_raw, qc.picked.c_std)
                    if qc.picked.c_std is not None
                    else None
                )
                qc.picked = PickedDispersionCurve(
                    f=f_rs, c=c_rs, c_std=cstd_rs, mode=qc.picked.mode
                )

            cmp_x_list.append(float(tgt))
            picks_list.append(qc.picked)
            qc_results.append(qc)

        return cmp_x_list, picks_list, qc_results

    def _build_love_init_models(
        self,
        picks_list: List[PickedDispersionCurve],
        *,
        init_model: Optional[LayeredEarthModel],
        n_layers: int,
        vp_vs_ratio: float,
        rho: float,
    ) -> List[LayeredEarthModel]:
        K = len(picks_list)
        if init_model is not None:
            return [
                init_model.copy_with_vs(init_model.vs.copy()) for _ in range(K)
            ]
        template = build_default_love_init_model(
            picks_list[0],
            n_layers=n_layers,
            vp_vs_ratio=vp_vs_ratio,
            rho=rho,
        )
        return [template.copy_with_vs(template.vs.copy()) for _ in range(K)]

    def _run_love_inversion(
        self,
        *,
        cmp_x_list: List[float],
        picks_list: List[PickedDispersionCurve],
        init_models: List[LayeredEarthModel],
        is_single: bool,
        forward: Union[str, RayleighForwardSolver],
        misfit: Union[str, DispersionMisfit],
        engine: Union[str, InversionEngine, type],
        higher_mode_max: int,
        engine_opts: Dict[str, Any],
        sensitivitycutoff: Optional[float],
        depth_fraction: float,
        picks_for_section: List[PickedDispersionCurve],
    ) -> Union[Pseudo2DVsSection, RayleighInversionResult]:
        if is_single:
            return self.love_vs_inversion_1d(
                target=cmp_x_list[0],
                picked=picks_list[0],
                init_model=init_models[0],
                forward=forward,
                misfit=misfit,
                engine=engine,
                higher_mode_max=higher_mode_max,
                **engine_opts,
            )

        results: List[RayleighInversionResult] = []
        for x, pick, init_m in zip(cmp_x_list, picks_list, init_models):
            r = self.love_vs_inversion_1d(
                target=x,
                picked=pick,
                init_model=init_m,
                forward=forward,
                misfit=misfit,
                engine=engine,
                higher_mode_max=higher_mode_max,
                **engine_opts,
            )
            results.append(r)

        return Pseudo2DSectionBuilder.build(
            cmpx=np.asarray(cmp_x_list, dtype=float),
            results=results,
            sensitivitycutoff=sensitivitycutoff,
            picked_curves=picks_for_section,
            depth_fraction=depth_fraction,
        )

    def _run_love_with_plot(
        self,
        section: Pseudo2DVsSection,
        *,
        show: bool,
        save_name: Optional[str],
    ) -> None:
        """vs_section の描画を PlotProcesser 経由で実行する (遅延 import で循環回避)。

        Vs 断面の表示は wave 種別非依存のため、Rayleigh と同じ ``vs_section``
        plotter を再利用する (新 backend method は追加しない)。
        """
        from src.processor.plot_processor import PlotProcesser

        plotter = PlotProcesser(backend="mpl")
        plotter.vs_section(
            section=section,
            show=show,
            save_name=save_name,
        )
