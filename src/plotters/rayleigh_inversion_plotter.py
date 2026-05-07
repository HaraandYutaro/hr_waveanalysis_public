"""RayleighInversionPlotter — Rayleigh 波逆解析の可視化 dispatch クラス。

担当:
  - dispersion_fit_image(): per-CMP 分散エネルギーマップ + 観測ピーク + 理論曲線
  - vs_section(): Pseudo-2D Vs 断面図 (pcolormesh)

既存 DispersionPlotter (raw f-c 画像) とは責務が異なる。
DispersionPlotter の公開シグネチャは変更しない。
"""

from src.plotting.backend_base import PlotterBase


class RayleighInversionPlotter:
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
        return self._backend.vs_section(
            section,
            show=show,
            save_name=save_name,
            **kwargs,
        )
