"""
ReflectionPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「反射断面プロットのプレゼンテーション既定値の
canonical owner」と位置付け、title / colorbar / num_ticks など
見た目に関する安定既定値の宣言をここで一元化する。描画ボディ自体は
引き続き backend (`_reflection_image_impl`) 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_reflection_image_impl` の本体は動かさない
- backend 固有オプション (cmap / figsize / aspect / interpolation 等) は
  引き続き `**kw` 経由で受け渡す

Step 3 対象外（risky / 純粋表示既定値ではない）:
- `plot_x_as_distance` は backend 内で xlabel 文字列だけでなく x_range の
  データ選択にも効くため、純粋な presentation default として
  定数化せず、現行の既定値 True をシグネチャに残す
- `xlabel` / `ylabel` は backend 描画ボディ内でハードコードされており、
  シグネチャ昇格には backend ボディの変更が必要なため対象外
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_TITLE = ""
_DEFAULT_COLORBAR = True
_DEFAULT_NUM_TICKS = 8


class ReflectionPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        stack_horiz: np.ndarray,
        cmp_pos: np.ndarray,
        elev_axis: np.ndarray,
        targets_sorted: np.ndarray,
        *,
        plot_x_as_distance: bool = True,
        title: str = _DEFAULT_TITLE,
        colorbar: bool = _DEFAULT_COLORBAR,
        num_ticks: int = _DEFAULT_NUM_TICKS,
        dense_x=None,
        hm_dense=None,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        show: bool | None = False,
        save_name: str | None = None,
        **kw,
    ):
        return self._backend._reflection_image_impl(
            stack_horiz,
            cmp_pos,
            elev_axis,
            targets_sorted,
            plot_x_as_distance=plot_x_as_distance,
            title=title,
            colorbar=colorbar,
            num_ticks=num_ticks,
            dense_x=dense_x,
            hm_dense=hm_dense,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            show=show,
            save_name=save_name,
            **kw,
        )
