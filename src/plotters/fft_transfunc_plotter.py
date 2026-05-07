"""
FftTransfuncPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「伝達関数スペクトルプロットのプレゼンテーション既定値の
canonical owner」と位置付け、color / xscale / xlabel / ylabel など
見た目に関する既定値の宣言をここで一元化する。描画ボディ自体は引き続き backend 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_fft_transfunc_image_impl` の本体は動かさない
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_COLOR = "blue"
_DEFAULT_XSCALE = "log"
_DEFAULT_XLABEL = "Frequency [Hz]"
_DEFAULT_YLABEL = "Transfunc"


class FftTransfuncPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        freq: np.ndarray,
        transfunc: np.ndarray,
        *,
        show: bool,
        save_name: str | None = None,
        title: str | None = None,
        color: str = _DEFAULT_COLOR,
        xscale: str = _DEFAULT_XSCALE,
        xlabel: str = _DEFAULT_XLABEL,
        ylabel: str = _DEFAULT_YLABEL,
        **kw,
    ):
        return self._backend._fft_transfunc_image_impl(
            freq,
            transfunc,
            show=show,
            save_name=save_name,
            title=title,
            color=color,
            xscale=xscale,
            xlabel=xlabel,
            ylabel=ylabel,
            **kw,
        )