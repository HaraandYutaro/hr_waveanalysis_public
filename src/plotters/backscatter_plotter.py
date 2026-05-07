"""
BackscatterPlotter — Step 3 プロトタイプの責務シフト起点。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「後方散乱振幅プロットのプレゼンテーション既定値の
canonical owner」と位置付け、ylabel など見た目に関する既定値の正規化を
ここで一元化する。描画ボディ自体は引き続き backend 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_backscatter_image_impl` の本体は動かさない
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_YLABEL = "amplitude"


class BackscatterPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        ind: np.ndarray,
        amp,
        *,
        title: str | None = None,
        ylabel: str = _DEFAULT_YLABEL,
        show: bool | None = False,
        save_name: str | None = None,
        **kw,
    ):
        return self._backend._backscatter_image_impl(
            ind,
            amp,
            title=title,
            ylabel=ylabel,
            show=show,
            save_name=save_name,
            **kw,
        )
