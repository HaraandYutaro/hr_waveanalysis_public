# Step 4a-4 / 4b: Plotly backend implementation.
# Imports PlotterBase from src.plotting.backend_base.
"""
描画オプション kwargs 一覧 (Plotly backend)
===============================================

このファイルは PlotlyPlotter クラスを定義し、各プロットメソッドの
描画本体を実装している。大部分のメソッドは **kwargs を受け取り、
`kw.get(key, default)` の形式でオプションを取り出す。

ここに列挙するのは、現状コードが解釈する暗黙 kwargs の事実の一覧である。
コードの挙動は一切変更していない。

どの層で解釈されるか
--------------------

1. **wrapper 層 (PlotterWrapperMixin)** で解釈されるもの:
   - `show` / `save_name` — 描画の即時表示と保存パスの正規化
   - `agc` / `agc_batchsize` / `agc_clip` — seismogram の AGC 前処理
   - `mode` (旧称, cmp で plot_mode へリネーム)
   - `title` (seismogram では _build_title で構築・注入)

2. **plotter 層 (src/plotters/*.py)** でデフォルト値が宣言されるもの:
   - `xlabel` / `ylabel` — SeismogramPlotter, BackscatterPlotter 等
   - `title` — DispersionPlotter, FkPlotter, SpectraPlotter, FftTransfuncPlotter, Fft1dPlotter
   - `colorbar` / `num_ticks` — ReflectionPlotter
   - `color` / `plot_mode` — CmpPlotter
   - `color` / `xscale` / `xlabel` / `ylabel` — FftTransfuncPlotter
   - `xscale` / `time_xlabel` / `time_ylabel` / `freq_xlabel` / `freq_ylabel` — Fft1dPlotter

3. **backend 層 (このファイル)** でのみ解釈されるもの:
   以下の共通 kwargs およびメソッド固有 kwargs を参照。

ここに書いてあるもの以外の kwargs は、基本的に **Plotly の API には渡されず**
無視される（`go.Figure` / `go.Scatter` / `go.Heatmap` 等のコンストラクタ引数として
直接渡される形式ではないため）。

Plotly 固有の注意点
--------------------

- `figsize` は Matplotlib 専用の概念であり、Plotly 側では解釈されない。
  figsize が **kw に含まれていても Plotly 実装では使用されない。
- `cmap` は Matplotlib のカラーマップ名だが、Plotly 側では `colorscale` に
  変換して使用する（`kw.get("cmap", ...)` で取得後、Plotly の colorscale 引数に渡す）。
  デフォルト値は Matplotlib 側と必ずしも一致しない点に注意。
- `interpolation` は Plotly の Heatmap では `zsmooth` に対応する
  (`kw.get("interpolation", "best")` → `zsmooth`)。
  デフォルトが "best" のメソッドと False のメソッドがある。

共通 kwargs
----------

- `cmap` (str)
    colorscale 名。Plotly の colorscale 仕様（"RdBu", "Jet", "RdBu_r", "gray" 等）で
    指定する。メソッドごとのデフォルト:
    - `_cmap_impl`: "RdBu"
    - `_spectra_image_impl`: "Jet"
    - `_dispersion_image_impl`: "RdBu_r"
    - `fk_image_impl`: "Jet"
    - `_reflection_image_impl`: "gray"

- `interpolation` (bool | str)
    Heatmap の zsmooth 設定。メソッドごとのデフォルト:
    - `_cmap_impl`: False
    - `_spectra_image_impl`: "best"
    - `_dispersion_image_impl`: False
    - `fk_image_impl`: "best"
    - `_reflection_image_impl`: "best"

- `title` (str | None)
    図タイトル。メソッドによっては明示引数としても受け取る。

メソッド固有 kwargs
-------------------

_signal_impl
    `xlabel` (default: "time [s]") — x 軸ラベル
    `ylabel` (default: "amplitude") — y 軸ラベル
    `xmin` / `xmax` / `ymin` / `ymax` — 軸範囲の明示指定
    `color` (str | None) — 線色。None の時は Plotly の既定色
    `title` (str | None) — 図タイトル
    `grid` (bool, default: True) — グリッド表示

_cmap_impl
    `cmap` (→ colorscale, default: "RdBu")
    `interpolation` (→ zsmooth, default: False)
    `title`, `xlabel`, `ylabel`
    `invert_y` (bool, default: False) — Y 軸反転

_seismogram_impl
    `xmin` / `xmax` — x 軸範囲。指定なし時は spacing_array から自動計算。
    `xlabel` (default: "Distance [m]")
    `ylabel` (default: "Time [s]")
    `title` — 指定時のみタイトルを設定

_spectra_image_impl
    `cmap` (→ colorscale, default: "Jet")
    `interpolation` (→ zsmooth, default: "best")

_dispersion_image_impl
    `cmap` (→ colorscale, default: "RdBu_r")
    `interpolation` (→ zsmooth, default: False)
    `title` / `suptitle` — タイトル解決順序は Matplotlib 側と同一

_backscatter_image_impl
    追加 **kwargs なし（title, ylabel は明示引数）。

_backscatter_distribution_image_impl
    追加 **kwargs なし（title, ylabel は明示引数）。

fk_image_impl
    `cmap` (→ colorscale, default: "Jet")
    `interpolation` (→ zsmooth, default: "best")
    `title` — 指定時のみ使用。指定なし時は
    "f-k imaging {name} {axis} {analysis}" を生成。

_reflection_image_impl
    `cmap` (→ colorscale, default: "gray")
    `interpolation` (→ zsmooth, default: "best")
    `plot_x_as_distance`, `colorbar`, `num_ticks`,
    `dense_x`, `hm_dense`, `xmin`, `xmax`, `ymin`, `ymax` —
    いずれも明示引数として受け取る。

_cmp_impl
    `color` (default: "black") — トレースの色
    `plot_mode` (default: "fill") — "fill" / "plot" / "scatter"
    `mode` — 旧称。kw に含まれる場合 plot_mode を上書き（互換用途）
    `show` — `kw.get("show", show)` で上書き可能

_fft_transfunc_image_impl
    `complex_mode` (default: "abs") — "abs" / "real" / "imag" / "phase"
    `xlim` / `ylim` — 軸範囲の明示指定

_fft1d_image_impl
    `color` (default: "blue") — 線色
    `t_xlim` — 時系列パネルの x 軸範囲
    `f_xlim` — 周波数パネルの x 軸範囲
    `f_ylim` — 周波数パネルの y 軸範囲

attenuation_energy
    追加 **kwargs なし（show と save_name のみ）。

attenuation_freq
    追加 **kwargs なし（show と save_name のみ）。

attenuation_fit
    `velocity` (default: None) — None の時 xlabel が "distance from source[m]"、
    指定時は "wave number"。
    `dB` (default: False) — True の時 ylabel に "(dB)" を追加しタイトルに "[dB]" を付加。
"""
from typing import Sequence, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..backend_base import PlotterBase
from ..results import PlotlyBackendResult


class PlotlyPlotter(PlotterBase):
    def _setup_style(self, style: str | None):
        # Plotly はテーマを figure 生成時 or テンプレートで指定 (例: "plotly_white", "plotly_dark")
        self.template = style or "plotly_white"

    # --- 1D Graph ---
    def _signal_impl(
        self,
        x: Sequence[float],
        y: Sequence[float],
        *,
        save_name: str | None,
        # Baseから受け取るキーワード引数 (デフォルト値はBaseかここで設定)
        xlabel: str = "time [s]",
        ylabel: str = "amplitude",
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        color: str | None = None,
        title: str | None = None,
        grid: bool = True,
        show: bool = True,
        **kw,
    ):
        fig = go.Figure()

        # Plotlyの線色は "rgb(255, 0, 0)" や "red" などの形式
        line_dict = dict(color=color) if color else dict()

        fig.add_scatter(x=x, y=y, mode="lines", line=line_dict)

        fig.update_layout(
            template=self.template,
            title=title,
            xaxis=dict(
                title=xlabel,
                range=[xmin, xmax] if (xmin is not None and xmax is not None) else None,
                showgrid=grid,
            ),
            yaxis=dict(
                title=ylabel,
                range=[ymin, ymax] if (ymin is not None and ymax is not None) else None,
                showgrid=grid,
            ),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="signal",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, show)

    # --- 2D Color Map (旧 Heatmap) ---
    def _cmap_impl(
        self,
        data: np.ndarray,
        *,
        extent: Tuple[float, float, float, float] | None = None,
        save_name: str | None,
        **kw,
    ):
        """
        data: 2D array
        extent: (xmin, xmax, ymin, ymax)
        kw: cmap (colorscale), title, xlabel, ylabel 等
        """
        fig = go.Figure()

        # extent から x, y 座標を生成 (Plotly は extent 直接指定より x, y 配列を好むため)
        ny, nx = data.shape
        if extent:
            x0, x1, y0, y1 = extent
            # linspace で座標軸を作成
            x_axis = np.linspace(x0, x1, nx)
            y_axis = np.linspace(y0, y1, ny)
        else:
            x_axis = np.arange(nx)
            y_axis = np.arange(ny)

        # colorscale (Matplotlibのcmap名とは異なる場合があるため注意。 'Viridis', 'RdBu' など)
        colorscale = kw.get("cmap", "RdBu")

        fig.add_trace(
            go.Heatmap(
                z=data,
                x=x_axis,
                y=y_axis,
                colorscale=colorscale,
                zsmooth=kw.get("interpolation", False),  # 'best' or False
            )
        )

        fig.update_layout(
            template=self.template,
            title=kw.get("title", ""),
            xaxis_title=kw.get("xlabel", "Distance"),
            yaxis_title=kw.get("ylabel", "Time"),
            # 地震波断面図などではY軸反転が一般的
            yaxis=dict(autorange="reversed") if kw.get("invert_y", False) else None,
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="cmap",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, kw.get("show", True))

    # --- Backscatter Image (1D Line Plot) ---
    def _backscatter_image_impl(
        self,
        ind,
        amp,
        *,
        title,
        ylabel,
        show,
        save_name,
        **kw,
    ):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ind, y=amp, mode="lines"))

        fig.update_layout(
            template=self.template,
            title=title,
            xaxis=dict(
                title="index (last ch)",
                showgrid=True,
            ),
            yaxis=dict(
                title=ylabel,
                showgrid=True,
            ),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="backscatter_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def _seismogram_impl(
        self,
        data: np.ndarray,
        *,
        t_sec: np.ndarray,
        interval: float,
        distance: np.ndarray | None,
        source_x: float | None,
        spacing: float,
        fill: bool,
        show: bool,
        save_name: str | None,
        **kw,
    ):
        n_traces = data.shape[0]

        if distance is None:
            spacing_array = interval * np.arange(n_traces)
        elif len(distance) != n_traces:
            print(
                f"Warning: distance length ({len(distance)}) != channels ({n_traces}). Using index based."
            )
            spacing_array = interval * np.arange(n_traces)
        else:
            spacing_array = distance

        max_x = np.abs(data).max()
        if max_x == 0:
            max_x = 1.0

        space = spacing * max_x
        ratio = interval / space if space != 0 else 1.0

        fig = go.Figure()

        for i in range(n_traces):
            trace = (data[i] * ratio).astype(np.float32)
            offset = float(spacing_array[i])
            x_vals = offset + trace

            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=t_sec,
                    mode="lines",
                    line=dict(color="black", width=0.8),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

            if fill:
                x_pos = offset + np.maximum(trace, 0)

                fig.add_trace(
                    go.Scatter(
                        x=np.full(len(t_sec), offset),
                        y=t_sec,
                        mode="lines",
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_pos,
                        y=t_sec,
                        mode="lines",
                        line=dict(width=0),
                        fill="tonextx",
                        fillcolor="rgba(0,0,0,0.8)",
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        if source_x is not None:
            fig.add_shape(
                type="line",
                x0=source_x,
                x1=source_x,
                y0=float(t_sec[-1]),
                y1=float(t_sec[0]),
                line=dict(color="red", width=0.8, dash="dash"),
            )

        merge = interval / spacing
        xmin_val = kw.get("xmin", float(np.min(spacing_array) - merge))
        xmax_val = kw.get("xmax", float(np.max(spacing_array) + merge))

        xlabel = kw.get("xlabel", "Distance [m]")
        ylabel = kw.get("ylabel", "Time [s]")

        layout_kwargs = dict(
            template=self.template,
            xaxis=dict(
                title=xlabel,
                range=[xmin_val, xmax_val],
            ),
            yaxis=dict(
                title=ylabel,
                autorange="reversed",
            ),
        )
        if kw.get("title"):
            layout_kwargs["title"] = kw.get("title")

        fig.update_layout(**layout_kwargs)

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="seismogram",
            figure=fig,
            traces=list(fig.data[:6]),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, show)

    def _spectra_image_impl(
        self,
        data: np.ndarray,
        freq: list,
        title: str | None,
        save_name: str | None,
        show: bool,
        **kw,
    ) -> None:
        ny, nx = data.shape
        x_sensor = np.linspace(1, nx, nx)
        y_freq = np.linspace(freq[1], freq[0], ny)

        colorscale = kw.get("cmap", "Jet")

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=data,
                x=x_sensor,
                y=y_freq,
                colorscale=colorscale,
                zsmooth=kw.get("interpolation", "best"),
                colorbar=dict(title="Amplitude"),
            )
        )

        fig.update_layout(
            template=self.template,
            xaxis_title="sensor position",
            yaxis_title="frequency [Hz]",
        )
        if title:
            fig.update_layout(title=title)

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="spectra_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def _dispersion_image_impl(
        self,
        input: np.ndarray,
        ax_x: list[float],
        ax_y: list[float],
        show: bool,
        save_name: str,
        nyquist_k: float,
        d_maxdiff: float,
        mode: str,
        **kw,
    ):
        ax_x = np.asarray(ax_x)
        ax_y = np.asarray(ax_y)
        xmin, xmax = float(np.min(ax_x)), float(np.max(ax_x))
        ymin, ymax = float(np.min(ax_y)), float(np.max(ax_y))

        colorscale = kw.get("cmap", "RdBu_r")

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=input.T,
                x=ax_x,
                y=ax_y,
                colorscale=colorscale,
                zsmooth=kw.get("interpolation", False),
                colorbar=dict(title="Amplitude"),
            )
        )

        largest_k = 1.0 / (2.0 * d_maxdiff)

        if mode == "frequency-velocity":
            xlabel = "Frequency (Hz)"
            ylabel = "Phase Velocity (m/s)"
            x_line = np.array([xmin, xmax])
            y_line1 = x_line / nyquist_k
            y_line2 = x_line / largest_k
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line1,
                    mode="lines",
                    line=dict(color="yellow", dash="dash", width=2),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line2,
                    mode="lines",
                    line=dict(color="yellow", dash="dash", width=2),
                    showlegend=False,
                )
            )
        elif mode == "wavelength-velocity":
            xlabel = "Phase velocity (m/s)"
            ylabel = "Wavelength (m)"
            fig.add_hline(
                y=1.0 / nyquist_k,
                line=dict(color="yellow", dash="dash", width=2),
            )
            fig.add_hline(
                y=1.0 / largest_k,
                line=dict(color="yellow", dash="dash", width=2),
            )
        elif mode == "frequency-wavelength":
            xlabel = "Frequency (Hz)"
            ylabel = "Wavelength (m)"
            fig.add_hline(
                y=1.0 / nyquist_k,
                line=dict(color="yellow", dash="dash", width=2),
            )
            fig.add_hline(
                y=1.0 / largest_k,
                line=dict(color="yellow", dash="dash", width=2),
            )
        else:
            raise ValueError(
                "invalid mode. mode must be either frequency-velocity, wavelength-velocity, or frequency-wavelength"
            )

        title = kw.get("title", None)
        if title is None:
            title = kw.get("suptitle", None)
        if title is None:
            title = "Dispersion curve"

        fig.update_layout(
            template=self.template,
            title=title,
            xaxis=dict(title=xlabel, range=[xmin, xmax]),
            yaxis=dict(title=ylabel, range=[ymin, ymax]),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="dispersion_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    

    def _backscatter_distribution_image_impl(
        self,
        distance: np.ndarray,
        amp: np.ndarray,
        *,
        count: np.ndarray | None = None,
        title: str | None = None,
        ylabel: str = "averaged amplitude",
        show: bool,
        save_name: str | None = None,
        **kw,
    ):
        distance = np.asarray(distance)
        amp = np.asarray(amp)

        if count is not None and np.any(np.asarray(count) > 0):
            count = np.asarray(count)
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.75, 0.25],
            )

            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=amp,
                    mode="lines+markers",
                    line=dict(color="black", width=1.0),
                    marker=dict(size=4),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

            if len(distance) > 1:
                bar_width = float(np.median(np.diff(np.sort(distance)))) * 0.8
            else:
                bar_width = 0.5

            fig.add_trace(
                go.Bar(
                    x=distance,
                    y=count,
                    width=bar_width,
                    marker_color="gray",
                    marker_line=dict(color="black", width=0.3),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            fig.update_yaxes(title_text=ylabel, showgrid=True, row=1, col=1)
            fig.update_yaxes(title_text="count", showgrid=True, row=2, col=1)
            fig.update_xaxes(title_text="distance [m]", row=2, col=1)

            fig.update_layout(template=self.template)
            if title:
                fig.update_layout(title=title)

        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=distance,
                    y=amp,
                    mode="lines+markers",
                    line=dict(color="black", width=1.0),
                    marker=dict(size=4),
                    showlegend=False,
                ),
            )

            fig.update_layout(
                template=self.template,
                title=title,
                xaxis=dict(
                    title="distance [m]",
                    showgrid=True,
                ),
                yaxis=dict(
                    title=ylabel,
                    showgrid=True,
                ),
            )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="backscatter_distribution_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def _reflection_image_impl(
        self,
        stack_horiz: np.ndarray,
        cmp_pos: np.ndarray,
        elev_axis: np.ndarray,
        targets_sorted: np.ndarray,
        *,
        plot_x_as_distance: bool = True,
        title: str = "",
        colorbar: bool = True,
        num_ticks: int = 8,
        dense_x: np.ndarray | None = None,
        hm_dense: np.ndarray | None = None,
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        show: bool = True,
        save_name: str | None = None,
        **kw,
    ):
        x_range = targets_sorted if plot_x_as_distance else cmp_pos

        vmax = (
            np.nanmax(np.abs(stack_horiz)) * 0.6
            if np.any(~np.isnan(stack_horiz))
            else 1.0
        )

        ny, nx = stack_horiz.T.shape
        x_axis = np.linspace(x_range.min(), x_range.max(), nx)
        y_axis = np.linspace(elev_axis.min(), elev_axis.max(), ny)

        colorscale = kw.get("cmap", "gray")

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=stack_horiz.T,
                x=x_axis,
                y=y_axis,
                colorscale=colorscale,
                zmin=-vmax,
                zmax=vmax,
                zsmooth=kw.get("interpolation", "best"),
                colorbar=dict(title="Amplitude") if colorbar else None,
                showscale=colorbar,
            )
        )

        xlabel = "distance [m]" if plot_x_as_distance else "CMP position"

        _xmin = x_range.min() if xmin is None else xmin
        _xmax = x_range.max() if xmax is None else xmax
        xticks = np.linspace(_xmin, _xmax, num_ticks)
        ticktext_x = [f"{x:.1f}" for x in xticks]

        y_area = elev_axis.max() - elev_axis.min()
        base = (
            1.0
            if y_area < 10
            else 2.0
            if y_area < 30
            else 5.0
            if y_area < 100
            else 10.0
        )
        y_min_calc = np.floor(elev_axis.min() / base) * base
        y_max_calc = np.ceil(elev_axis.max() / base) * base
        yticks = np.arange(y_min_calc, y_max_calc + base, base)
        ticktext_y = [f"{y:.1f}" for y in yticks]

        x_range_layout = [xmin, xmax] if (xmin is not None and xmax is not None) else [_xmin, _xmax]
        y_range_layout = [ymin, ymax] if (ymin is not None and ymax is not None) else [y_min_calc, y_max_calc]

        if dense_x is not None and hm_dense is not None:
            fig.add_trace(
                go.Scatter(
                    x=dense_x,
                    y=hm_dense,
                    mode="lines",
                    line=dict(color="red", width=1.0),
                    showlegend=False,
                )
            )

        fig.update_layout(
            template=self.template,
            xaxis=dict(
                title=xlabel,
                tickvals=xticks,
                ticktext=ticktext_x,
                range=x_range_layout,
            ),
            yaxis=dict(
                title="Elevation [m]",
                tickvals=yticks,
                ticktext=ticktext_y,
                range=y_range_layout,
                autorange="reversed",
            ),
        )
        if title:
            fig.update_layout(title=title)

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="reflection_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def _cmp_impl(
        self,
        show,
        fs,
        data,
        off,
        target_cmp,
        t,
        *,
        save_name: str | None = None,
        color: str = "black",
        plot_mode: str = "fill",
        **kw,
    ) -> None:
        if "mode" in kw:
            plot_mode = kw.pop("mode")

        data = np.asarray(data, dtype=float)
        spacing_array = np.asarray(off, dtype=float)
        n_tr, n_samp = data.shape

        t_from_step = int(max(0.0, min(t)) * fs)
        t_to_step = min(int(max(t) * fs), n_samp)
        t_axis = np.arange(t_from_step, t_to_step) / fs

        max_x = np.abs(data[:, t_from_step:t_to_step]).max()
        if max_x == 0:
            max_x = 1.0

        spacing = 0.8
        interval = spacing
        space = spacing * max_x
        ratio = interval / space if space != 0 else 1.0

        fig = go.Figure()

        for i in range(n_tr):
            trace = (data[i, t_from_step:t_to_step] * ratio).astype(np.float32)
            offset = float(spacing_array[i])

            if plot_mode == "plot":
                fig.add_trace(go.Scatter(
                    x=offset + trace,
                    y=t_axis,
                    mode="lines",
                    line=dict(color=color, width=0.8),
                    showlegend=False,
                ))

            elif plot_mode == "fill":
                pos_trace = np.maximum(trace, 0)

                fig.add_trace(go.Scatter(
                    x=[offset] * len(t_axis),
                    y=t_axis,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=offset + pos_trace,
                    y=t_axis,
                    mode="lines",
                    line=dict(color="rgba(0,0,0,0)", width=0),
                    fill="tonextx",
                    fillcolor=color,
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=offset + trace,
                    y=t_axis,
                    mode="lines",
                    line=dict(color=color, width=0.8),
                    showlegend=False,
                ))

            elif plot_mode == "scatter":
                fig.add_trace(go.Scatter(
                    x=offset + trace,
                    y=t_axis,
                    mode="markers",
                    marker=dict(color=color, size=2.0),
                    showlegend=False,
                ))

            else:
                raise ValueError("Invalid plot_mode. select plot, fill or scatter.")

        fig.update_layout(
            template=self.template,
            title=f"CMP target={target_cmp:.4f}",
            xaxis_title="Offset (m)",
            yaxis=dict(
                title="Time (s)",
                autorange="reversed",
            ),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="cmp",
            figure=fig,
            traces=list(fig.data[:6]),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, kw.get("show", show))

    def fk_image_impl(
        self,
        axis,
        res,
        kxmin,
        kxmax,
        fmin,
        fmax,
        analysis,
        name,
        show,
        save_name,
        **kw,
    ):
        amp = np.abs(res)
        ny, nx = amp.shape
        k_axis = np.linspace(kxmin, kxmax, nx)
        f_axis = np.linspace(fmin, fmax, ny)

        colorscale = kw.get("cmap", "Jet")
        title = kw.get("title", None)
        if title is None:
            title = f"f-k imaging {name} {axis} {analysis}"

        fig = go.Figure()
        fig.add_trace(
            go.Heatmap(
                z=amp,
                x=k_axis,
                y=f_axis,
                colorscale=colorscale,
                zsmooth=kw.get("interpolation", "best"),
                colorbar=dict(title="Amplitude"),
            )
        )

        fig.update_layout(
            template=self.template,
            title=title,
            xaxis_title="Wavenumber [1/m]",
            yaxis_title="Frequency [Hz]",
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="fk_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def _fft_transfunc_image_impl(
        self,
        freq: np.ndarray,
        transfunc: np.ndarray,
        *,
        show: bool,
        save_name: str | None = None,
        title: str | None = None,
        color: str = "blue",
        xscale: str = "log",
        xlabel: str = "Frequency [Hz]",
        ylabel: str = "Transfunc",
        **kw,
    ) -> None:
        y = transfunc
        if np.iscomplexobj(y):
            mode = kw.get("complex_mode", "abs")
            if mode == "real":
                y = np.real(y)
            elif mode == "imag":
                y = np.imag(y)
            elif mode == "phase":
                y = np.angle(y)
            else:
                y = np.abs(y)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=freq, y=y, mode="lines", line=dict(color=color)))

        xaxis_dict = dict(title=xlabel, showgrid=True)
        if xscale == "log":
            xaxis_dict["type"] = "log"
        if kw.get("xlim") is not None:
            xaxis_dict["range"] = list(kw["xlim"])

        yaxis_dict = dict(title=ylabel, showgrid=True)
        if kw.get("ylim") is not None:
            yaxis_dict["range"] = list(kw["ylim"])

        fig.update_layout(
            template=self.template,
            title=title,
            xaxis=xaxis_dict,
            yaxis=yaxis_dict,
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="fft_transfunc_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def _fft1d_image_impl(
        self,
        t: np.ndarray,
        signal: np.ndarray,
        freq: np.ndarray,
        spec: np.ndarray,
        *,
        show: bool,
        save_name: str | None = None,
        title: str | None = None,
        xscale: str = "log",
        time_xlabel: str = "Time [s]",
        time_ylabel: str = "Signal",
        freq_xlabel: str = "Frequency [Hz]",
        freq_ylabel: str = "Amplitude",
        **kw,
    ) -> None:
        color = kw.get("color", "blue")

        fig = make_subplots(rows=2, cols=1, shared_xaxes=False)

        fig.add_trace(
            go.Scatter(x=t, y=signal, mode="lines", line=dict(color=color), showlegend=False),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(x=freq, y=spec, mode="lines", line=dict(color=color), showlegend=False),
            row=2,
            col=1,
        )

        fig.update_xaxes(title_text=time_xlabel, showgrid=True, row=1, col=1)
        fig.update_yaxes(title_text=time_ylabel, showgrid=True, row=1, col=1)

        freq_xaxis = dict(title_text=freq_xlabel, showgrid=True)
        if xscale == "log":
            freq_xaxis["type"] = "log"
        if kw.get("f_xlim") is not None:
            freq_xaxis["range"] = list(kw["f_xlim"])
        fig.update_xaxes(**freq_xaxis, row=2, col=1)

        freq_yaxis = dict(title_text=freq_ylabel, showgrid=True)
        if kw.get("f_ylim") is not None:
            freq_yaxis["range"] = list(kw["f_ylim"])
        fig.update_yaxes(**freq_yaxis, row=2, col=1)

        if kw.get("t_xlim") is not None:
            fig.update_xaxes(range=list(kw["t_xlim"]), row=1, col=1)

        fig.update_layout(template=self.template)
        if title is not None:
            fig.update_layout(title=title)

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="fft1d_image",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )

        self._finalize(fig, save_name, show)

    def attenuation_energy(
        self,
        distance: np.ndarray,
        energy: np.ndarray,
        noise_mean: float,
        *,
        show: bool,
        save_name: str | None = None,
        **kw,
    ) -> None:
        if not show and not save_name:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distance,
            y=energy,
            mode="markers",
            name="attenuation",
        ))
        fig.add_trace(go.Scatter(
            x=distance,
            y=np.full_like(energy, noise_mean),
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="noise_level",
        ))

        fig.update_layout(
            template=self.template,
            title="attenuation S[hc]^2",
            xaxis=dict(title="distance from source [m]", type="log"),
            yaxis=dict(title="attenuation", type="log"),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="attenuation_energy",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, show)

    def attenuation_freq(
        self,
        distance: np.ndarray,
        attenuation: np.ndarray,
        noise_mean: float,
        freq: float,
        *,
        show: bool,
        save_name: str | None = None,
        **kw,
    ) -> None:
        if not show and not save_name:
            return

        freq_t = str(np.round(freq, 3)).replace(".", "_")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distance,
            y=attenuation,
            mode="markers",
            name="attenuation",
        ))
        fig.add_trace(go.Scatter(
            x=distance,
            y=np.full_like(attenuation, noise_mean),
            mode="lines",
            line=dict(color="red", dash="dash"),
            name="noise_level",
        ))

        fig.update_layout(
            template=self.template,
            title=f"attenuation analysis at {freq_t}Hz",
            xaxis=dict(title="distance from source [m]", type="log"),
            yaxis=dict(title="attenuation", type="log"),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="attenuation_freq",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, show)

    def attenuation_fit(
        self,
        distance_ax: np.ndarray,
        geo_att: np.ndarray,
        a: float,
        target_freq: float,
        *,
        velocity: float | None = None,
        dB: bool = False,
        show: bool = True,
        save_name: str | None = None,
        **kw,
    ) -> None:
        if not show and not save_name:
            return

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distance_ax,
            y=geo_att,
            mode="markers",
            name="attenuation",
        ))

        x_fit = [0, float(distance_ax.max())]
        y_fit = [0, a * float(distance_ax.max())]
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode="lines",
            line=dict(dash="dash"),
            name="fitting line",
        ))

        xlabel = "wave number" if velocity is not None else "distance from source[m]"
        ylabel = "amp ratio (dB)" if dB else "amp ratio (log)"
        title = f"freq={round(target_freq, 1)}Hz damp ratio α:{round(a, 3)}[1/m]"
        if dB:
            title += " [dB]"

        fig.update_layout(
            template=self.template,
            title=title,
            xaxis=dict(title=xlabel),
            yaxis=dict(title=ylabel, type="log"),
        )

        _result = PlotlyBackendResult(
            backend_name="plotly",
            plot_type="attenuation_fit",
            figure=fig,
            traces=list(fig.data),
            layout=fig.layout,
        )
        self._finalize(fig, save_name, show)

    def _finalize(self, fig, save_name, show):
        if save_name:
            fp = self.outdir / save_name
            # scale=2 で高解像度保存
            fig.write_image(fp, scale=2)

        if show:
            fig.show()

    def bandpass_response(
        self,
        w: np.ndarray,
        h: np.ndarray,
        fpass: np.ndarray,
        fstop: np.ndarray,
        *,
        show: bool = True,
        save_name: str | None = None,
        **kw,
    ) -> None:
        raise NotImplementedError(
            "bandpass_response は Plotly backend では未実装です。"
            " Matplotlib backend を使用してください。"
        )

    def _traveltime_tomo_impl(
        self,
        initial_velocity,
        final_velocity,
        rms,
        seis_data,
        dt,
        synthetic_tt,
        picks,
        grid_nx,
        grid_nz,
        grid_dx,
        grid_dz,
        *,
        show,
        save_name,
        velocity_xlabel,
        velocity_ylabel,
        rms_xlabel,
        rms_ylabel,
        seis_xlabel,
        seis_ylabel,
        **kw,
    ):
        raise NotImplementedError(
            "traveltime_tomo is not yet implemented for the Plotly backend"
        )

    def _rayleigh_dispersion_fit_image_impl(self, *args, **kwargs):
        raise NotImplementedError(
            "PlotlyPlotter は rayleigh_dispersion_fit_image を未実装。"
            "backend='matplotlib' を使用してください。"
        )

    def _vs_section_impl(self, *args, **kwargs):
        raise NotImplementedError(
            "PlotlyPlotter は vs_section を未実装。"
            "backend='matplotlib' を使用してください。"
        )
