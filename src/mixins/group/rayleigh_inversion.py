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

from typing import List, Optional, Union

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
from src.inversion.rayleigh.section import Pseudo2DSectionBuilder
from src.plotting.wrapper import PlotterWrapperMixin


_MISFIT_REGISTRY = {"weighted_l2": WeightedL2Misfit}


def _lazy_lci_engine_cls():
    """LCIEngine を遅延 import で返す (scipy.sparse 依存の遅延読込のため)。"""
    from src.inversion.rayleigh.engine.lci import LCIEngine
    return LCIEngine


_ENGINE_REGISTRY = {
    "damped_lsq": DampedLeastSquaresEngine,
    "lci": _lazy_lci_engine_cls,  # callable (lazy)
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