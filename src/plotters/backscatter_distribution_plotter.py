"""
BackscatterDistributionPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「後方散乱分布プロットのプレゼンテーション既定値の
canonical owner」と位置付け、ylabel など見た目に関する既定値の宣言を
ここで一元化する。描画ボディ自体は引き続き backend 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_backscatter_distribution_image_impl` の本体は動かさない
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_YLABEL = "averaged amplitude"


class BackscatterDistributionPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        distance: np.ndarray,
        amp: np.ndarray,
        *,
        count: np.ndarray | None = None,
        title: str | None = None,
        ylabel: str = _DEFAULT_YLABEL,
        show: bool | None = False,
        save_name: str | None = None,
        **kw,
    ):
        return self._backend._backscatter_distribution_image_impl(
            distance,
            amp,
            count=count,
            title=title,
            ylabel=ylabel,
            save_name=save_name,
            show=show,
            **kw,
        )