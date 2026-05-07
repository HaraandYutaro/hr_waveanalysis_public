"""
SeismogramPlotter — Step 3 責務シフト適用済みディスパッチ層 (最終スライス)。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「地震波形 (wiggle) プロットの最も保守的な
プレゼンテーション既定値 (xlabel / ylabel) の canonical owner」と位置付ける。
描画ボディ自体は引き続き backend (`PlotterBase.seismogram` →
`_seismogram_impl`) 側に残す。

注意: PlotterBase.seismogram() 公開メソッド経由で委譲する。
PlotterBase.seismogram() が _seismogram_impl() を呼び出す構造を維持するため、
_impl ではなく seismogram() を呼ぶ。

互換性方針:
- 公開シグネチャ (keyword-only 引数列) は据え置き
- wrapper 側の orchestration 責務 (_build_title による processor-bound タイトル構築・
  _apply_agc による AGC 処理・_resolve_metadata・時間スライス) には触れない
- backend `_seismogram_impl` の trace-by-trace ループ・振幅スケーリング・
  source line・fill 分岐・xlim 自動計算などの本体は動かさない
- backend 固有オプション (figsize 等) は引き続き `**kw` 経由で受け渡す

Step 3 対象外（risky / 純粋表示既定値ではない）:
- `title` は wrapper の `_build_title` が processor-bound 情報
  (self.name / analysis_<axis> / AGC info) を組み合わせて構築し
  `kw["title"]` に強制注入する canonical owner であり、
  plotter での二重宣言は責務境界を曖昧化するため対象外
- `spacing` は backend で振幅スケーリング除数 (space = spacing * max_x;
  ratio = interval / space) として直接使用される rendering geometry
  parameter であり、純粋表示既定値ではない
- `fill` は `if fill: ax.fill_betweenx(...)` の描画分岐を制御する
  rendering mode toggle であり対象外
- `agc` は wrapper の `_apply_agc` で事前処理される signal-processing flag
  (backend body 未参照) であり対象外
- `xmin` / `xmax` は backend が `spacing_array` から自動計算する
  物理座標境界 (kxmin/kxmax 等の先例と同種) であり対象外
- `figsize` 等は backend 固有 (Matplotlib `(14, 10)` 既定) なので
  `**kw` に温存
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_XLABEL = "Distance [m]"
_DEFAULT_YLABEL = "Time [s]"


class SeismogramPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        sliced_data: np.ndarray,
        *,
        t_sec: np.ndarray,
        interval: float,
        distance,
        source_x,
        spacing: float,
        fill: bool,
        agc: bool,
        show: bool,
        save_name: str | None = None,
        xlabel: str = _DEFAULT_XLABEL,
        ylabel: str = _DEFAULT_YLABEL,
        **kw,
    ):
        return self._backend.seismogram(
            sliced_data,
            t_sec=t_sec,
            interval=interval,
            distance=distance,
            source_x=source_x,
            spacing=spacing,
            fill=fill,
            agc=agc,
            show=show,
            save_name=save_name,
            xlabel=xlabel,
            ylabel=ylabel,
            **kw,
        )
