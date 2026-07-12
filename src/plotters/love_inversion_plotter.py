"""LoveInversionPlotter — Love wave 逆解析の可視化 dispatch クラス。

責務
----
- ``dispersion_fit_image()``: per-CMP 分散エネルギーマップ + 観測ピーク +
  理論 Love 曲線
- ``vs_section()``: Pseudo-2D Vs 断面図 (Rayleigh と共有)

backend method への配線
-----------------------
backend に新しい method は追加せず、既存の
``rayleigh_dispersion_fit_image`` / ``vs_section`` を再利用する。
両者とも描画内容は wave 種別に依存しない (f-v 画像 + 分散曲線オーバーレイ、
深度 x CMP の Vs グリッド) ため、Love wave inversion の可視化として
そのまま用いて公開挙動を変えない。

Step 3 ownership precedent との整合
-----------------------------------
- backend rendering body は backend 層に残置
- 本クラスは「stable plot-level presentation defaults」の owner として
  ``show`` / ``save_name`` などの薄い forwarding に留める
- 既存 ``RayleighInversionPlotter`` のシグネチャ・既定値を流用し、
  新規の semantically risky parameter は導入しない

References
----------
- Zhang, K. et al. (2023). DOI:10.1093/gji/ggac380. (Love wave determinant
  misfit inversion)
"""

from src.plotting.backend_base import PlotterBase


class LoveInversionPlotter:
    """Love wave inversion 結果の dispatch plotter (wave 非依存 backend method を再利用)。"""

    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def dispersion_fit_image(
        self,
        res,
        f_mesh,
        c_mesh,
        qc_result=None,
        theory_c=None,
        theory_f=None,
        *,
        target=None,
        show: bool = True,
        save_name: str | None = None,
        **kwargs,
    ):
        """f-v 画像 + 観測 Love ピック + 理論 Love 曲線。

        backend method ``rayleigh_dispersion_fit_image`` は描画内容が
        wave 種別非依存 (f-v 画像 + 観測ピーク + 任意の理論曲線オーバーレイ) のため
        そのまま再利用する。将来 backend 名称を ``surface_wave_dispersion_fit_image``
        に汎用化する slice で置き換える前提。
        """
        return self._backend.rayleigh_dispersion_fit_image(
            res,
            f_mesh,
            c_mesh,
            qc_result,
            theory_c,
            theory_f,
            target=target,
            show=show,
            save_name=save_name,
            **kwargs,
        )

    def vs_section(
        self,
        section,
        *,
        show: bool = True,
        save_name: str | None = None,
        **kwargs,
    ):
        """Pseudo-2D Vs 断面 (depth x CMP)。Rayleigh / Love で共通の backend method。"""
        return self._backend.vs_section(
            section,
            show=show,
            save_name=save_name,
            **kwargs,
        )
