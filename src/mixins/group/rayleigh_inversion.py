"""RayleighInversion mixin for GroupProcesser.

Step2 の責務 (既存):
  - inversion core (src/inversion/rayleigh/) を呼び出すための薄い orchestration layer
  - 文字列 -> クラス解決 (forward / misfit / engine の registry)
  - default 初期モデル生成の委譲
  - GroupProcesser 文脈で使えるよう public method を 1 つ提供

Step3 の追加 (既存挙動は変更しない):
  - rayleigh_vs_inversion_profile: 複数 CMP の 1D 逆解析 + Pseudo2DSectionBuilder

  [LOG-1] Step ?-3 で rayleigh_vs_inversion_profile の picks を
  dict[float, PickedDispersionCurve] から
  List[PickedDispersionCurve] に変更した。
  cmp_x と同じ順序で対応する並列配列として扱う。

inversion core (model / forward / misfit / engine / init_model) は
本ファイルから一切編集していない。本ファイルはあくまで wiring layer。

将来的な Vs section plotting の接続のため PlotterWrapperMixin を継承するが、
本スライスでは plotting メソッドを呼ばない (One slice per path)。
"""

from __future__ import annotations

import os
import warnings
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

from src.inversion.rayleigh.engine.base import InversionEngine
from src.inversion.rayleigh.engine.damped_lsq import DampedLeastSquaresEngine
from src.inversion.rayleigh.forward.base import RayleighForwardSolver
from src.inversion.rayleigh.forward.toy import ToyForwardSolver
from src.inversion.rayleigh.init_model import build_default_init_model
from src.inversion.rayleigh.misfit.base import DispersionMisfit
from src.inversion.rayleigh.misfit.weighted_l2 import WeightedL2Misfit
from src.inversion.rayleigh.model import (
    LayeredEarthModel,
    PickedDispersionCurve,
    RayleighInversionResult,
)
from src.inversion.rayleigh.model import LCIProfileResult, Pseudo2DVsSection
from src.inversion.rayleigh.picking_qc import (
    DispersionPickQCResult,
    quality_control_dispersion_pick,
)
from src.inversion.rayleigh.section import Pseudo2DSectionBuilder
from src.plotting.wrapper import PlotterWrapperMixin


_MISFIT_REGISTRY = {"weighted_l2": WeightedL2Misfit}


def _lazy_lci_engine_cls():
    """LCIEngine を遅延 import で返す (scipy.sparse 依存の遅延読込のため)。"""
    from src.inversion.rayleigh.engine.lci import LCIEngine
    return LCIEngine


def _lazy_niching_pso_engine_cls():
    """NichingPSOEngine を遅延 import で返す。

    Niching PSO は wave 種別に依存しない multimodal optimizer であり、
    Rayleigh / Love 両方の inversion で再利用される。本 registry へは
    Love wave inversion 導入時 (Phase: Love wave inversion Slice L5) に
    追加されたが、既存 ``damped_lsq`` / ``lci`` の挙動には影響しない。

    Reference: Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380.
    """
    from src.inversion.rayleigh.engine.pso import NichingPSOEngine
    return NichingPSOEngine


_ENGINE_REGISTRY = {
    "damped_lsq": DampedLeastSquaresEngine,
    "lci": _lazy_lci_engine_cls,  # callable (lazy)
    "niching_pso": _lazy_niching_pso_engine_cls,  # callable (lazy)
}


def _resolve_forward(spec: Union[str, RayleighForwardSolver]) -> RayleighForwardSolver:
    """文字列なら registry 経由で実体化、インスタンスならそのまま返す。

    ThomsonHaskellSolver は disba 依存のため遅延 import。
    """
    if isinstance(spec, RayleighForwardSolver):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"forward must be str or RayleighForwardSolver, got {type(spec).__name__}"
        )
    if spec == "toy":
        return ToyForwardSolver()
    if spec == "thomson_haskell":
        # disba 不在時はここで案内付き ImportError が上がる
        from src.inversion.rayleigh.forward.thomson_haskell import ThomsonHaskellSolver
        return ThomsonHaskellSolver()
    raise ValueError(
        f"unknown forward name {spec!r}; available: ['toy', 'thomson_haskell']"
    )


def _resolve_misfit(spec: Union[str, DispersionMisfit]) -> DispersionMisfit:
    if isinstance(spec, DispersionMisfit):
        return spec
    if not isinstance(spec, str):
        raise TypeError(
            f"misfit must be str or DispersionMisfit, got {type(spec).__name__}"
        )
    cls = _MISFIT_REGISTRY.get(spec)
    if cls is None:
        raise ValueError(
            f"unknown misfit name {spec!r}; available: {sorted(_MISFIT_REGISTRY)}"
        )
    return cls()


def _resolve_engine_cls(spec: Union[str, InversionEngine, type]):
    """engine は class または instance または str を許す。

    str registry のエントリは class または class を返す callable factory のいずれか。
    factory 形式は scipy.sparse 等の重い依存を遅延 import するためのもの。
    """
    if isinstance(spec, InversionEngine):
        return ("instance", spec)
    if isinstance(spec, type) and issubclass(spec, InversionEngine):
        return ("class", spec)
    if isinstance(spec, str):
        entry = _ENGINE_REGISTRY.get(spec)
        if entry is None:
            raise ValueError(
                f"unknown engine name {spec!r}; available: {sorted(_ENGINE_REGISTRY)}"
            )
        if isinstance(entry, type) and issubclass(entry, InversionEngine):
            return ("class", entry)
        if callable(entry):
            cls = entry()
            if not (isinstance(cls, type) and issubclass(cls, InversionEngine)):
                raise TypeError(
                    f"engine factory for {spec!r} did not return an InversionEngine subclass"
                )
            return ("class", cls)
        raise TypeError(f"engine registry entry for {spec!r} is invalid: {entry!r}")
    raise TypeError(
        f"engine must be str, InversionEngine subclass, or instance; "
        f"got {type(spec).__name__}"
    )


class RayleighInversion(PlotterWrapperMixin):
    """GroupProcesser に Rayleigh-wave Vs 1D 逆解析メソッドを提供する mixin。"""

    def rayleigh_vs_inversion_1d(
        self,
        target,
        picked: PickedDispersionCurve,
        *,
        init_model: Optional[LayeredEarthModel] = None,
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell",
        misfit: Union[str, DispersionMisfit] = "weighted_l2",
        engine: Union[str, InversionEngine, type] = "damped_lsq",
        n_layers: int = 10,
        **engine_opts,
    ) -> RayleighInversionResult:
        """1 CMP 分の Rayleigh-wave Vs 逆解析を実行する。

        Parameters
        ----------
        target : Any
            CMP 識別子 (metadata タグとしてのみ保持)。
            self.cmp[target] へのアクセスは行わない。
        picked : PickedDispersionCurve
            観測ピッキング済み分散曲線。逆解析の入力データ。
        init_model : LayeredEarthModel, optional
            初期モデル。None の場合 build_default_init_model() で生成。
        forward, misfit, engine : str or instance
            inversion core への dispatch。文字列 registry または instance を受ける。
            forward 既定 "thomson_haskell" (Step ?-1b の disba ラッパー)。
            engine は class でも instance でも str でも可。
        n_layers : int
            init_model 自動生成時の層数 (init_model 明示時は無視)。
        **engine_opts
            engine class に転送する option (lambda0 / max_iter / rms_tol 等)。
            engine が instance の場合は **無視** される (再構成しない)。

        Returns
        -------
        RayleighInversionResult
            metadata に cmp_target / forward_name / misfit_name / engine_name を付与。
        """
        fwd_obj = _resolve_forward(forward)
        mis_obj = _resolve_misfit(misfit)
        engine_kind, engine_arg = _resolve_engine_cls(engine)

        if engine_kind == "instance":
            eng = engine_arg
        else:
            eng = engine_arg(forward=fwd_obj, misfit=mis_obj, **engine_opts)

        if init_model is None:
            init_model = build_default_init_model(picked, n_layers=n_layers)

        result = eng.run_single(picked, init_model)

        result.metadata["cmp_target"] = target
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

    def rayleigh_vs_inversion_profile(
        self,
        cmp_x: np.ndarray,
        picks: List[PickedDispersionCurve],
        initmodels: Optional[List[LayeredEarthModel]] = None,
        *,
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell",
        misfit: Union[str, DispersionMisfit] = "weighted_l2",
        engine: Union[str, InversionEngine, type] = "damped_lsq",
        n_layers: int = 10,
        sensitivitycutoff: Optional[float] = None,
        maxdepthfactor: float = 2.5,
        **engine_opts,
    ) -> Pseudo2DVsSection:
        """複数 CMP の Rayleigh-wave Vs 1D 逆解析を実行し、pseudo-2D section を返す。

        各 CMP に対して rayleigh_vs_inversion_1d を呼び出し、
        結果を Pseudo2DSectionBuilder で pseudo-2D section に組み立てる。

        initmodels を省略した場合、最初の CMP の picks から共通初期モデルを
        生成し全 CMP で共有する (層構造 h の整合を保証するため)。
        各 CMP には独立したコピーが渡される (同じオブジェクト参照にはならない)。

        Parameters
        ----------
        cmp_x : ndarray, shape=(K,)
            CMP x 位置 [m]。
        picks : list[PickedDispersionCurve], length K
            各 CMP の観測ピッキング済み分散曲線。
            cmp_x と同じ順序で対応する並列配列として扱う。
        initmodels : list[LayeredEarthModel] or None, length K
            各 CMP の初期モデル。None の場合 build_default_init_model で
            最初の CMP の picks から共通モデルを生成し、各 CMP に独立コピーを渡す。
        forward, misfit, engine : str or instance
            rayleigh_vs_inversion_1d に転送。
        n_layers : int
            initmodels 省略時の初期モデル層数。
            init_model が自動生成される場合にのみ使用される。
        sensitivitycutoff : float or None
            Pseudo2DSectionBuilder に転送。vsgrid の低感度セルを NaN マスクする閾値。
        maxdepthfactor : float
            Pseudo2DSectionBuilder に転送。半空間深さ推定の係数。
        **engine_opts
            engine に転送。

        Returns
        -------
        Pseudo2DVsSection
            per-CMP 1D inversion 結果を深度グリッドに写像した pseudo-2D section。

        Raises
        ------
        ValueError
            cmp_x と picks の長さが不一致。
            initmodels と picks の長さが不一致。
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

        # [BUG-2 FIX] 独立コピーで渡す。同一オブジェクト参照を回避。
        # initmodels が省略された場合、最初の CMP の picks から共通初期モデルを生成。
        # 各 CMP に独立コピーが渡され、engine が model.vs を in-place 変更しても
        # 他の CMP は汚染されない。
        if initmodels is None:
            init_model_template = build_default_init_model(picks[0], n_layers=n_layers)
            initmodels = [
                init_model_template.copy_with_vs(init_model_template.vs.copy())
                for _ in range(K)
            ]

        results: List[RayleighInversionResult] = []
        for i in range(K):
            target = float(cmp_x[i])
            # [DESIGN-3] init_model を明示的に渡しているため、
            # n_layers は rayleigh_vs_inversion_1d 側で無視される。
            # n_layers は init_model が未指定の場合にのみ使用されるパラメータである。
            result = self.rayleigh_vs_inversion_1d(
                target,
                picks[i],
                init_model=initmodels[i],
                forward=forward,
                misfit=misfit,
                engine=engine,
                **engine_opts,
            )
            results.append(result)

        return Pseudo2DSectionBuilder.build(
            cmpx=cmp_x,
            results=results,
            sensitivitycutoff=sensitivitycutoff,
            maxdepthfactor=maxdepthfactor,
        )

    def rayleigh_vs_inversion_lci(
        self,
        cmp_x: np.ndarray,
        picks: List[PickedDispersionCurve],
        init_models: Optional[List[LayeredEarthModel]] = None,
        *,
        lambda_v: float = 1.0,
        lambda_h: float = 1.0,
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell",
        misfit: Union[str, DispersionMisfit] = "weighted_l2",
        n_layers: int = 10,
        sensitivitycutoff: Optional[float] = None,  # noqa: ARG002 (kept for API parity)
        **engine_opts,
    ) -> LCIProfileResult:
        """全 K CMP の同時逆解析 (Laterally Constrained Inversion)。

        rayleigh_vs_inversion_profile (各 CMP 独立 1D inversion) とは別に、
        本メソッドは LCIEngine を用いて全 CMP を結合した目的関数を解く。
        隣接 CMP 間の Vs 列方向への滑らかさ拘束により、
        地質学的整合性のとれた 2D 構造を得る。

        Parameters
        ----------
        cmp_x : ndarray, shape=(K,)
            CMP x 位置 [m]。
        picks : list[PickedDispersionCurve], length K
            各 CMP のピッキング済み分散曲線。
        init_models : list[LayeredEarthModel] or None
            各 CMP の初期モデル。None の場合 picks[0] から共通モデルを生成し
            各 CMP に独立コピーを渡す (層厚 h の整合を保証するため)。
        lambda_v : float
            垂直ラフネス正則化重み (>= 0)。
        lambda_h : float
            横方向 (CMP 間) スムージング正則化重み (>= 0)。
            0 に近づけると独立 1D inversion (DampedLSQ pseudo-2D) と同等になる。
        forward, misfit : str or instance
            inversion core への dispatch (engine は "lci" 固定)。
        n_layers : int
            init_models 省略時の初期モデル層数。
        sensitivitycutoff : float or None
            API 互換のためのみ受ける (LCI は section を直接返さない)。
        **engine_opts
            LCIEngine に転送する option (max_iter, rms_tol, use_sparse 等)。

        Returns
        -------
        LCIProfileResult
            per_cmp_results / joint_rms_history / lambda_v / lambda_h /
            converged / n_iter / metadata を含む。
        """
        del sensitivitycutoff  # API parity のため受けるが本メソッドでは未使用

        K = len(picks)
        cmp_x = np.asarray(cmp_x, dtype=float)
        if cmp_x.shape[0] != K:
            raise ValueError(
                f"cmp_x length ({cmp_x.shape[0]}) != picks length ({K})"
            )
        if K == 0:
            raise ValueError("picks must contain at least one PickedDispersionCurve")
        if init_models is not None and len(init_models) != K:
            raise ValueError(
                f"init_models length ({len(init_models)}) != picks length ({K})"
            )

        if init_models is None:
            tmpl = build_default_init_model(picks[0], n_layers=n_layers)
            init_models = [
                tmpl.copy_with_vs(tmpl.vs.copy()) for _ in range(K)
            ]

        fwd_obj = _resolve_forward(forward)
        mis_obj = _resolve_misfit(misfit)
        _, engine_cls = _resolve_engine_cls("lci")

        eng = engine_cls(
            forward=fwd_obj,
            misfit=mis_obj,
            lambda_v=lambda_v,
            lambda_h=lambda_h,
            **engine_opts,
        )
        result = eng.run_profile(
            cmp_x=cmp_x,
            picks=list(picks),
            init_models=list(init_models),
        )
        result.metadata["forward_name"] = (
            forward if isinstance(forward, str) else type(fwd_obj).__name__
        )
        result.metadata["misfit_name"] = (
            misfit if isinstance(misfit, str) else type(mis_obj).__name__
        )
        result.metadata["cmp_x"] = cmp_x.tolist()
        return result

    # =================================================================
    #  Unified one-shot API
    # =================================================================

    def rayleigh_inversion(
        self,
        target: Union[None, float, Sequence[float], Literal["all"]] = "all",
        *,
        axis: str = "z",
        freq_range: Tuple[float, float] = (1.0, 300.0),
        vel_range: Tuple[float, float] = (1.0, 400.0),
        df: float = 0.5,
        dc: float = 1.0,
        qc_min_energy_ratio: float = 0.05,
        qc_continuity_rel_jump: float = 0.15,
        qc_max_secondary_peak_ratio: float = 0.85,
        qc_min_valid_points: int = 8,
        n_freq_resample: Optional[int] = 50,
        init_model: Optional[LayeredEarthModel] = None,
        n_layers: int = 10,
        vp_vs_ratio: float = float(np.sqrt(3.0)),
        rho: float = 2000.0,
        mode: Literal["independent", "lci"] = "independent",
        forward: Union[str, RayleighForwardSolver] = "thomson_haskell",
        misfit: Union[str, DispersionMisfit] = "weighted_l2",
        engine: Union[str, InversionEngine, type] = "damped_lsq",
        max_iter: int = 50,
        rms_tol: float = 0.5,
        lambda0: float = 1e-3,
        lambda_v: float = 1.0,
        lambda_h: float = 1.0,
        sensitivitycutoff: Optional[float] = 0.1,
        depth_fraction: float = 0.5,
        cmp_kwargs: Optional[dict] = None,
        show: bool = False,
        save_name: Optional[str] = None,
        plot_dispersion_fit: bool = False,
        dispersion_fit_dir: Optional[str] = None,
        return_intermediate: bool = False,
    ) -> Union[
        Pseudo2DVsSection,
        RayleighInversionResult,
        LCIProfileResult,
        Dict[str, Any],
    ]:
        """Rayleigh 波 Vs 逆解析を 1 メソッドで完結させる一元化 API。

        既存の cmp_gathering / dispersion_curve / QC / 初期モデル生成 /
        rayleigh_vs_inversion_* の各ステップを内部で自動オーケストレーションする。

        target の解決:
          - None / "all"      : self.targets 全部 (未設定なら cmp_gathering 自動実行)
          - float             : 単一 CMP (最近傍の self.targets[i] にスナップ)
          - Sequence[float]   : 指定 CMP 群 (各値を最近傍にスナップ)

        分散フィット画像 (任意・既定 OFF):
          plot_dispersion_fit=True かつ (show または dispersion_fit_dir) のとき、
          各 CMP の位相速度ー周波数エネルギー像にピッキング結果を重畳した
          分散フィット画像を描画する。観測ピックは最適化の成功・失敗に関わらず
          常に埋め込まれ、理論曲線は当該 CMP が収束した場合のみ重畳される。
          dispersion_fit_dir を指定すると各 CMP を disp_cmp_<x>m.png として保存する。

        Parameters (抜粋)
        ----------
        plot_dispersion_fit : bool, default False
            True で per-CMP 分散フィット画像 (ピック重畳) を描画する。
            既定 False では従来挙動 (vs_section のみ) を維持する。
        dispersion_fit_dir : str or None, default None
            分散フィット画像の保存先ディレクトリ。指定時は exist_ok で作成し
            各 CMP を disp_cmp_<x>m.png として保存する。None なら保存しない
            (show=True なら画面表示のみ)。

        Returns
        -------
        target が単一 float                         → RayleighInversionResult
        target が複数 + mode="independent"          → Pseudo2DVsSection
        target が複数 + mode="lci"                  → LCIProfileResult
        return_intermediate=True                    → 中間物を含む dict
            (cmp_x / picks / qc / init_models に加え、各 CMP の分散エネルギー像
             res_maps / f_meshes / c_meshes を並列リストで含む)
        """
        is_single = isinstance(target, (int, float)) and not isinstance(target, bool)

        cmp_targets = self._resolve_targets(target, axis=axis, cmp_kwargs=cmp_kwargs)

        (
            cmp_x_list,
            picks_list,
            qc_results,
            res_maps,
            f_meshes,
            c_meshes,
        ) = self._collect_picks_with_qc(
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
                "No valid dispersion picks survived QC. "
                "Loosen QC thresholds (qc_*) or widen freq_range/vel_range."
            )

        init_models = self._build_init_models(
            picks_list,
            init_model=init_model,
            n_layers=n_layers,
            vp_vs_ratio=vp_vs_ratio,
            rho=rho,
        )

        inv_result = self._run_inversion(
            cmp_x_list=cmp_x_list,
            picks_list=picks_list,
            init_models=init_models,
            is_single=is_single,
            mode=mode,
            forward=forward,
            misfit=misfit,
            engine=engine,
            max_iter=max_iter,
            rms_tol=rms_tol,
            lambda0=lambda0,
            lambda_v=lambda_v,
            lambda_h=lambda_h,
            sensitivitycutoff=sensitivitycutoff,
            depth_fraction=depth_fraction,
        )

        if (show or save_name) and isinstance(inv_result, Pseudo2DVsSection):
            self._run_with_plot(inv_result, show=show, save_name=save_name)

        if plot_dispersion_fit and (show or dispersion_fit_dir):
            self._run_with_dispersion_fit_plot(
                cmp_x_list=cmp_x_list,
                qc_results=qc_results,
                res_maps=res_maps,
                f_meshes=f_meshes,
                c_meshes=c_meshes,
                inv_result=inv_result,
                show=show,
                dispersion_fit_dir=dispersion_fit_dir,
            )

        if return_intermediate:
            return {
                "result": inv_result,
                "section": inv_result if isinstance(inv_result, Pseudo2DVsSection) else None,
                "lci_result": inv_result if isinstance(inv_result, LCIProfileResult) else None,
                "single_result": (
                    inv_result if isinstance(inv_result, RayleighInversionResult) else None
                ),
                "cmp_x": cmp_x_list,
                "picks": picks_list,
                "qc": qc_results,
                "init_models": init_models,
                "res_maps": res_maps,
                "f_meshes": f_meshes,
                "c_meshes": c_meshes,
            }
        return inv_result

    # -----------------------------------------------------------------
    #  Internal helpers for rayleigh_inversion()
    # -----------------------------------------------------------------

    def _resolve_targets(
        self,
        target: Union[None, float, Sequence[float], Literal["all"]],
        *,
        axis: str,
        cmp_kwargs: Optional[dict],
    ) -> List[float]:
        """target 引数を CMP x 位置のリストに正規化する。

        self.targets が未設定なら cmp_gathering を自動実行する。
        float / Sequence[float] は self.targets への最近傍スナップを行い、
        スナップ差が CMP 間隔の半分を超える場合は警告を出す。
        """
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

    def _collect_picks_with_qc(
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
    ) -> Tuple[
        List[float],
        List[PickedDispersionCurve],
        List[DispersionPickQCResult],
        List[np.ndarray],
        List[np.ndarray],
        List[np.ndarray],
    ]:
        """各 target に dispersion_curve + QC を適用し、有効ピックのみ蓄積する。

        n_freq_resample が非 None の場合、QC 後の picked.f/c を均等周波数グリッドに
        線形リサンプリングする (高周波帯で QC により疎になったピックを正規化)。

        cmp_x_list / picks_list / qc_results に加え、各 CMP の分散エネルギー像
        res_maps と対応する周波数・速度軸 f_meshes / c_meshes を並列リストで返す。
        これらは分散フィット画像 (rayleigh_dispersion_fit_image) の入力として用いる。
        4 リストは同一の有効 CMP 順で整合する。
        """
        cmp_x_list: List[float] = []
        picks_list: List[PickedDispersionCurve] = []
        qc_results: List[DispersionPickQCResult] = []
        res_maps: List[np.ndarray] = []
        f_meshes: List[np.ndarray] = []
        c_meshes: List[np.ndarray] = []

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
                    f"CMP {tgt}: QC produced no valid points; skipped",
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
            res_maps.append(res)
            f_meshes.append(f_mesh)
            c_meshes.append(c_mesh)

        return cmp_x_list, picks_list, qc_results, res_maps, f_meshes, c_meshes

    def _build_init_models(
        self,
        picks_list: List[PickedDispersionCurve],
        *,
        init_model: Optional[LayeredEarthModel],
        n_layers: int,
        vp_vs_ratio: float,
        rho: float,
    ) -> List[LayeredEarthModel]:
        """全 CMP 分の独立した初期モデル配列を構築する。

        - init_model 指定あり: そのモデルを各 CMP に独立コピー
        - init_model 未指定 : 先頭 picks から build_default_init_model でテンプレート生成し、
                                vp_vs_ratio / rho を適用したうえで各 CMP に独立コピー
        各コピーは Vs / Vp / rho / h / vs_bounds が独立配列となり、engine の
        in-place 更新による相互汚染を防ぐ。
        """
        K = len(picks_list)
        if init_model is not None:
            return [
                init_model.copy_with_vs(init_model.vs.copy()) for _ in range(K)
            ]

        template = build_default_init_model(
            picks_list[0],
            n_layers=n_layers,
            vp_vs_ratio=vp_vs_ratio,
            rho=rho,
        )
        return [template.copy_with_vs(template.vs.copy()) for _ in range(K)]

    def _run_inversion(
        self,
        *,
        cmp_x_list: List[float],
        picks_list: List[PickedDispersionCurve],
        init_models: List[LayeredEarthModel],
        is_single: bool,
        mode: Literal["independent", "lci"],
        forward: Union[str, RayleighForwardSolver],
        misfit: Union[str, DispersionMisfit],
        engine: Union[str, InversionEngine, type],
        max_iter: int,
        rms_tol: float,
        lambda0: float,
        lambda_v: float,
        lambda_h: float,
        sensitivitycutoff: Optional[float],
        depth_fraction: float,
    ) -> Union[Pseudo2DVsSection, RayleighInversionResult, LCIProfileResult]:
        """mode 分岐で 1D / profile / LCI のいずれかを実行する。

        depth_fraction を Pseudo2DSectionBuilder.build に直接渡す必要があるため、
        independent モードでは rayleigh_vs_inversion_1d を per-CMP ループした後に
        builder を直接呼び出す (rayleigh_vs_inversion_profile は depth_fraction を
        受けないため迂回する。既存 _profile/_1d のシグネチャは不変)。
        """
        engine_opts = dict(max_iter=max_iter, rms_tol=rms_tol, lambda0=lambda0)

        if mode == "lci":
            if engine != "damped_lsq":
                warnings.warn(
                    "engine argument is ignored in mode='lci' (LCIEngine is fixed)",
                    stacklevel=3,
                )
            return self.rayleigh_vs_inversion_lci(
                cmp_x=np.asarray(cmp_x_list, dtype=float),
                picks=picks_list,
                init_models=init_models,
                lambda_v=lambda_v,
                lambda_h=lambda_h,
                forward=forward,
                misfit=misfit,
                max_iter=max_iter,
                rms_tol=rms_tol,
            )

        if mode != "independent":
            raise ValueError(
                f"mode must be 'independent' or 'lci'; got {mode!r}"
            )

        if is_single:
            return self.rayleigh_vs_inversion_1d(
                target=cmp_x_list[0],
                picked=picks_list[0],
                init_model=init_models[0],
                forward=forward,
                misfit=misfit,
                engine=engine,
                **engine_opts,
            )

        results: List[RayleighInversionResult] = []
        for x, pick, init_m in zip(cmp_x_list, picks_list, init_models):
            r = self.rayleigh_vs_inversion_1d(
                target=x,
                picked=pick,
                init_model=init_m,
                forward=forward,
                misfit=misfit,
                engine=engine,
                **engine_opts,
            )
            results.append(r)

        return Pseudo2DSectionBuilder.build(
            cmpx=np.asarray(cmp_x_list, dtype=float),
            results=results,
            sensitivitycutoff=sensitivitycutoff,
            picked_curves=picks_list,
            depth_fraction=depth_fraction,
        )

    def _run_with_plot(
        self,
        section: Pseudo2DVsSection,
        *,
        show: bool,
        save_name: Optional[str],
    ) -> None:
        """vs_section の描画を PlotProcesser 経由で実行する (遅延 import で循環回避)。"""
        from src.processor.plot_processor import PlotProcesser

        plotter = PlotProcesser(backend="mpl")
        plotter.vs_section(
            section=section,
            show=show,
            save_name=save_name,
        )

    def _run_with_dispersion_fit_plot(
        self,
        *,
        cmp_x_list: List[float],
        qc_results: List[DispersionPickQCResult],
        res_maps: List[np.ndarray],
        f_meshes: List[np.ndarray],
        c_meshes: List[np.ndarray],
        inv_result: Union[
            Pseudo2DVsSection, RayleighInversionResult, LCIProfileResult
        ],
        show: bool,
        dispersion_fit_dir: Optional[str],
    ) -> None:
        """各 CMP の分散フィット画像 (ピック重畳) を描画する。

        観測ピックは最適化の成功・失敗に関わらず常に埋め込まれ (backend が
        qc_result.picked から描画)、理論曲線は当該 CMP が収束 (converged) した
        場合にのみ重畳される。dispersion_fit_dir 指定時は disp_cmp_<x>m.png として
        保存し、未指定なら show=True による画面表示のみとなる。

        inv_result の型に応じて per-CMP 結果を cmpx で解決する:
          - RayleighInversionResult : 単一 CMP (cmp_x_list[0] に対応)
          - Pseudo2DVsSection       : section.percmpresults を cmpx で対応付け
          - LCIProfileResult        : per_cmp_results を cmp_x_list 順で対応付け
        """
        result_by_cmpx: Dict[float, RayleighInversionResult] = {}
        if isinstance(inv_result, RayleighInversionResult):
            if cmp_x_list:
                result_by_cmpx[float(cmp_x_list[0])] = inv_result
        elif isinstance(inv_result, Pseudo2DVsSection):
            result_by_cmpx = {
                float(inv_result.cmpx[i]): inv_result.percmpresults[i]
                for i in range(len(inv_result.cmpx))
            }
        elif isinstance(inv_result, LCIProfileResult):
            n = min(len(cmp_x_list), len(inv_result.per_cmp_results))
            result_by_cmpx = {
                float(cmp_x_list[i]): inv_result.per_cmp_results[i]
                for i in range(n)
            }

        if dispersion_fit_dir:
            os.makedirs(dispersion_fit_dir, exist_ok=True)

        for cmp_x, qc, f_mesh, c_mesh, res in zip(
            cmp_x_list, qc_results, f_meshes, c_meshes, res_maps
        ):
            r = result_by_cmpx.get(float(cmp_x))
            theory_c = None
            theory_f = None
            if r is not None and bool(r.converged) and qc.picked is not None:
                theory_c = np.asarray(r.predicted_c, dtype=float)
                theory_f = np.asarray(qc.picked.f, dtype=float)

            save_name = (
                os.path.join(dispersion_fit_dir, f"disp_cmp_{cmp_x:.1f}m.png")
                if dispersion_fit_dir
                else None
            )
            self.rayleigh_dispersion_fit_image(
                res=res,
                f_mesh=f_mesh,
                c_mesh=c_mesh,
                qc_result=qc,
                theory_c=theory_c,
                theory_f=theory_f,
                target=f"{cmp_x:.1f} m",
                show=show,
                save_name=save_name,
            )