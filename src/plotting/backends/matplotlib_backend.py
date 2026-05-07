# Step 4a-3 / 4b: Matplotlib backend implementation.
# Imports PlotterBase from src.plotting.backend_base.
"""
描画オプション kwargs 一覧 (Matplotlib backend)
=================================================

このファイルは MatplotlibPlotter クラスを定義し、各プロットメソッドの
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

ここに書いてあるもの以外の kwargs は、基本的に **Matplotlib の API には渡されず**
無視される（一部メソッドを除き、`_default_imshow_kw` 経由で `imshow` に渡されるものを除く）。

共通 kwargs
----------

以下の kwargs は、`_default_imshow_kw` ヘルパーメソッド経由で
imshow 系メソッド共通のデフォルト値として抽出される。
各メソッドはこれに独自のデフォルト cmap を渡す。

- `cmap` (str)
    カラーマップ名。メソッドごとのデフォルト:
    - `_cmap_impl`: "seismic"
    - `_spectra_image_impl`: "jet"
    - `_dispersion_image_impl`: "seismic"
    - `fk_image`: "jet"
    - `_reflection_image_impl`: "gray"

- `interpolation` (str, default: "nearest")
    imshow の補間モード。

- `aspect` (str, default: "auto")
    imshow のアスペクト比設定。

- `figsize` (tuple, default: (10, 6))
    図のサイズ (幅, 高さ) [インチ]。
    imshow 系以外のメソッドでも個別に kw.get("figsize", ...) として
    取り出されることがある（デフォルト値はメソッドごとに異なる）。

メソッド固有 kwargs
-------------------

_signal_impl
    `figsize` (default: (8, 4))
    `color` (default: "k") — 線色
    `xlabel` (default: "x") — x 軸ラベル
    `ylabel` (default: "y") — y 軸ラベル

_cmap_impl
    `_default_imshow_kw(kw, cmap="seismic")` 経由で cmap / interpolation /
    aspect / figsize を解決。
    x/ylabel, colorbar label はハードコード。

_seismogram_impl
    `figsize` (default: (14, 10))
    `xmin` / `xmax` — x 軸範囲。指定なし時は spacing_array から自動計算。
    `xlabel` (default: "Distance [m]")
    `ylabel` (default: "Time [s]")
    `title` — 指定時のみタイトルを設定

_spectra_image_impl
    `_default_imshow_kw(kw, cmap="jet")` 経由で解決。

_dispersion_image_impl
    `_default_imshow_kw(kw, cmap="seismic")` 経由で解決。
    `title` (default: None) — タイトル。None の場合は `suptitle` を試し、
    どちらも None なら "Dispersion curve {name}" を生成。
    `suptitle` — title が None の場合のフォールバック。
    `mode` / `nyquist_k` / `d_maxdiff` — 位置引数として渡される
    （mode は描画分岐、nyquist_k/d_maxdiff は補助線計算に使用）。

_backscatter_image_impl
    `figsize` (default: (8, 4))

_backscatter_distribution_image_impl
    `figsize` (default: (10, 6) または (10, 4))
    count の有無でサブプロット構成が変わる。

fk_image
    `_default_imshow_kw(kw, cmap="jet")` 経由で解決。
    `title` — 指定時のみタイトルを設定。指定なし時は
    "f-k imaging {name} {axis} {analysis}" を生成。
    `fs` (default: 1.0) — Panel 2 の時間軸計算に使用。

attenuation_energy
    追加 **kwargs なし（show と save_name のみ）。

attenuation_freq
    追加 **kwargs なし（show と save_name のみ）。

attenuation_fit
    `velocity` (default: None) — None の時 xlabel が "distance from source[m]"、
    指定時は "wave number"。
    `dB` (default: False) — True の時 ylabel に "(dB)" を追加しタイトルに "[dB]" を付加。

_reflection_image_impl
    `_default_imshow_kw(kw, cmap="gray")` 経由で解決。
    `plot_x_as_distance` (default: True) — True の時 x 軸に距離、False の時 CMP position。
    `colorbar` (default: True) — カラーバー表示の有無。
    `num_ticks` (default: 8) — x 軸の目盛り数。
    `dense_x` / `hm_dense` — 地表ライン描画用データ。
    `xmin` / `xmax` / `ymin` / `ymax` — 軸範囲の明示指定。

_cmp_impl
    `color` (default: "black") — トレースの色。
    `plot_mode` (default: "fill") — "fill" / "plot" / "scatter"。
    `mode` — 旧称。kw に含まれる場合 plot_mode を上書き（互換用途）。
    `show` — `kw.get("show", show)` で上書き可能。

_fft_transfunc_image_impl
    `figsize` (default: (8, 4))
    `complex_mode` (default: "abs") — "abs" / "real" / "imag" / "phase"。
    `xlim` / `ylim` — 軸範囲の明示指定。

_fft1d_image_impl
    `figsize` (default: (10, 8))
    `color` (default: "blue") — 線色。
    `t_xlim` — 時系列パネルの x 軸範囲。
    `f_xlim` — 周波数パネルの x 軸範囲。
    `f_ylim` — 周波数パネルの y 軸範囲。
"""
import os

import matplotlib

os.environ["XCURSOR_THEME"] = "Adwaita"
os.environ["XCURSOR_SIZE"] = "24"

matplotlib.use("qtagg")
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ..backend_base import PlotterBase
from ..results import MatplotlibBackendResult

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "sans-serif",
    }
)


class MatplotlibPlotter(PlotterBase):
    def _setup_style(self, style: str | None):
        if style:
            plt.style.use(style)

    @staticmethod
    def _default_imshow_kw(kw: dict, *, cmap: str = "jet") -> dict:
        """
        imshow 系メソッド共通のデフォルト設定を返す。
        kw で渡された値があればそちらを優先する。
        cmap のデフォルト値はメソッドごとに異なるため引数で指定する。

        例:
                opts = self._default_imshow_kw(kw, cmap='gray')
                fig, ax = plt.subplots(figsize=opts.pop('figsize'))
                ax.imshow(data, **opts)
        """
        return {
            "cmap": kw.get("cmap", cmap),
            "interpolation": kw.get("interpolation", "nearest"),
            "aspect": kw.get("aspect", "auto"),
            "figsize": kw.get("figsize", (10, 6)),
        }

    def _signal_impl(self, x, y, *, show, save_name, **kw):
        fig, ax = plt.subplots(figsize=kw.get("figsize", (8, 4)))
        line, = ax.plot(x, y, color=kw.get("color", "k"))
        ax.set_xlabel(kw.get("xlabel", "x"))
        ax.set_ylabel(kw.get("ylabel", "y"))
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="signal",
            figure=fig,
            axes=[ax],
            artists=[line],
        )
        self._finalize(fig, save_name, show)

    def _cmap_impl(self, data, *, extent, show, save_name, **kw):
        opts = self._default_imshow_kw(kw, cmap="seismic")
        fig, ax = plt.subplots(figsize=opts.pop("figsize"))
        im = ax.imshow(data.T, extent=extent, **opts)
        plt.ylabel("time (s)")
        plt.xlabel("channel")
        plt.colorbar(im, ax=ax, label="Amplitude [m/s]")
        plt.subplots_adjust(
            left=0.11, right=0.98, bottom=0.075, top=0.95, hspace=0.023, wspace=0
        )
        cb = [c for c in fig.axes if c is not ax]
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="cmap",
            figure=fig,
            axes=[ax],
            artists=[im],
            colorbars=cb,
        )
        self._finalize(fig, save_name, show)

    # --- Seismogram (Wiggle Plot) ---
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
        """
        data shape: (n_traces, n_samples)、t_sec はスライス済み時間配列
        """
        figsize = kw.get("figsize", (14, 10))
        fig, ax = plt.subplots(figsize=figsize)

        n_traces = data.shape[0]

        # 距離配列の確定
        if distance is None:
            spacing_array = interval * np.arange(n_traces)
        elif len(distance) != n_traces:
            print(
                f"Warning: distance length ({len(distance)}) != channels ({n_traces}). Using index based."
            )
            spacing_array = interval * np.arange(n_traces)
        else:
            spacing_array = distance

        # 振幅スケーリング
        max_x = np.abs(data).max()
        if max_x == 0:
            max_x = 1.0

        space = spacing * max_x
        ratio = interval / space if space != 0 else 1.0

        # ループ描画
        for i in range(n_traces):
            trace = (data[i] * ratio).astype(np.float32)
            offset = spacing_array[i]

            ax.plot(offset + trace, t_sec, color="k", lw=0.8)

            if fill:
                pos_mask = trace > 0
                ax.fill_betweenx(
                    t_sec,
                    offset,
                    offset + trace,
                    where=pos_mask,
                    facecolor="k",
                    alpha=0.8,
                )

        # 起振点ライン
        if source_x is not None:
            ax.axvline(
                x=source_x, color="red", linestyle="--", lw=0.8, label="source line"
            )

        # 軸設定
        merge = interval * 1.0 / spacing
        ax.set_ylim(t_sec[-1], t_sec[0])

        # xmax, xminの指定,指定値があればそれを採用する
        xmin = kw.get('xmin', np.min(spacing_array) - merge)
        xmax = kw.get('xmax', np.max(spacing_array) + merge)
        ax.set_xlim(xmin,xmax)

        ax.set_xlabel(kw.get("xlabel", "Distance [m]"))
        ax.set_ylabel(kw.get("ylabel", "Time [s]"))

        if kw.get("title"):
            ax.set_title(kw.get("title"))

        plt.tight_layout()
        plt.subplots_adjust(
            left=0.08, right=0.98, bottom=0.15, top=0.92, hspace=0.023, wspace=0
        )

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="seismogram",
            figure=fig,
            axes=[ax],
            artists=ax.lines[:3],
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
        # TODO: res / n_sensor が未定義。wrapper 側で log 変換とデータ整形を完結させ、
        #       ここには整形済み配列とセンサー数を明示的に渡す設計へ要修正。
        opts = self._default_imshow_kw(kw, cmap="jet")
        fig, ax = plt.subplots(figsize=opts.pop("figsize"))
        ax.imshow(
            data,
            extent=(1, data.shape[1] + 1, freq[1], freq[0]),
            **opts,
        )
        ax.set_xlabel("sensor position")
        ax.set_ylabel("frequency [Hz]")
        plt.colorbar(ax.images[0], ax=ax, label="Amplitude")
        if title:
            fig.suptitle(title)

        plt.tight_layout()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="spectra_image",
            figure=fig,
            axes=[ax],
            artists=[ax.images[0]],
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
        # Plotting
        xmin, xmax = np.min(ax_x), np.max(ax_x)
        ymin, ymax = np.min(ax_y), np.max(ax_y)

        opts = self._default_imshow_kw(kw, cmap="seismic")
        plt.figure(figsize=opts.pop("figsize"))
        im = plt.imshow(input.T, extent=[xmin, xmax, ymin, ymax], origin="lower", **opts)
        cb = plt.colorbar(label="Amplitude")

        x_line = np.array([xmin, xmax])  # --- ナイキスト境界線の計算と描画 ---
        largest_k = 1 / (2 * d_maxdiff)

        # modeによるラベル, ナイキスト境界線の描写
        if mode == "frequency-velocity":
            plt.ylabel("Phase Velocity (m/s)")
            plt.xlabel("Frequency (Hz)")

            # k = f / c  =>  c = f / k_nyquist
            y_line1 = x_line / nyquist_k
            y_line2 = x_line / largest_k

            # グラフの表示範囲内でのみ描画されるよう、c の範囲でプロット
            plt.plot(
                x_line,
                y_line1,
                color="yellow",
                linestyle="--",
                linewidth=2,
                label="nyquist",
            )
            plt.plot(
                x_line,
                y_line2,
                color="yellow",
                linestyle="--",
                linewidth=2,
                label="largest",
            )

        elif mode == "wavelength-velocity":
            plt.ylabel("Wavelength (m)")
            plt.xlabel("Phase velocity (m/s)")
            plt.hlines(
                y=1 / nyquist_k,
                xmin=xmin,
                xmax=xmax,
                color="yellow",
                linestyle="--",
                linewidth=2,
                label="nyquist",
            )
            plt.hlines(
                y=1 / largest_k,
                xmin=xmin,
                xmax=xmax,
                color="yellow",
                linestyle="--",
                linewidth=2,
                label="largest",
            )

        elif mode == "frequency-wavelength":
            plt.ylabel("Wavelength (m)")
            plt.xlabel("Frequency (Hz)")
            plt.hlines(
                y=1 / nyquist_k,
                xmin=xmin,
                xmax=xmax,
                color="yellow",
                linestyle="--",
                linewidth=2,
                label="nyquist",
            )
            plt.hlines(
                y=1 / largest_k,
                xmin=xmin,
                xmax=xmax,
                color="yellow",
                linestyle="--",
                linewidth=2,
                label="largest",
            )

        else:
            raise ValueError(
                "invalid mode. mode must be either frequency-velocity, wavelength-velocity, or frequency-wavelength"
            )

        # titleの設定
        title = kw.get("title", None)
        title = kw.get("suptitle", None) if title is None else title
        if title == None:
            if hasattr(self, "name"):
                title = f"Dispersion curve {self.name}"
            else:
                title = "Dispersion curve"
        plt.title(title)

        # 描画の大枠の設定
        plt.subplots_adjust(
            left=0.085, right=1.0, bottom=0.085, top=0.945, hspace=0.023, wspace=0
        )
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)

        fig = plt.gcf()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="dispersion_image",
            figure=fig,
            axes=fig.axes,
            artists=[im],
            colorbars=[cb],
        )
        self._finalize(fig, save_name, show)

    def _backscatter_image_impl(
        self,
        ind: np.ndarray,
        amp,
        *,
        title: str | None = None,
        ylabel: str = "amplitude",
        show: bool,
        save_name: str | None = None,
        **kw,
    ):
        """後方散乱振幅の線プロット"""
        fig, ax = plt.subplots(figsize=kw.get("figsize", (8, 4)))
        ax.plot(ind, amp)
        ax.set_xlabel("index (last ch)")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="backscatter_image",
            figure=fig,
            axes=[ax],
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
        """後方散乱分布のプロット (上: 平均振幅, 下: カウント)"""
        distance = np.asarray(distance)
        amp = np.asarray(amp)

        if count is not None and np.any(np.asarray(count) > 0):
            count = np.asarray(count)
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=kw.get("figsize", (10, 6)),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax1.plot(distance, amp, color="k", lw=1.0, marker=".")
            ax1.set_ylabel(ylabel)
            ax1.grid(True)
            if title:
                ax1.set_title(title)

            if len(distance) > 1:
                width = float(np.median(np.diff(np.sort(distance)))) * 0.8
            else:
                width = 0.5
            ax2.bar(distance, count, width=width, color="gray", edgecolor="k", lw=0.3)
            ax2.set_xlabel("distance [m]")
            ax2.set_ylabel("count")
            ax2.grid(True)
        else:
            fig, ax = plt.subplots(figsize=kw.get("figsize", (10, 4)))
            ax.plot(distance, amp, color="k", lw=1.0, marker=".")
            ax.set_xlabel("distance [m]")
            ax.set_ylabel(ylabel)
            ax.grid(True)
            if title:
                ax.set_title(title)

        plt.tight_layout()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="backscatter_distribution_image",
            figure=fig,
            axes=fig.axes,
        )
        self._finalize(fig, save_name, show)

    def fk_image(
        self,
        F2d: np.ndarray,
        f_axis: np.ndarray,
        k_axis: np.ndarray,
        data: np.ndarray,
        dist: np.ndarray,
        *,
        fs: float = 1.0,
        title: str | None = None,
        show: bool,
        save_name: str | None = None,
        **kw,
    ):
        """f-k スペクトルと地震波形の 2 パネル図"""
        opts = self._default_imshow_kw(kw, cmap="jet")
        fig, axes = plt.subplots(2, 1, figsize=opts.pop("figsize"))

        # Panel 1: f-k スペクトル
        im0 = axes[0].imshow(
            np.abs(F2d.T),
            extent=[k_axis[0], k_axis[-1], f_axis[0], f_axis[-1]],
            origin="lower",
            **opts,
        )
        plt.colorbar(im0, ax=axes[0], label="Amplitude")
        axes[0].set_ylabel("Frequency (Hz)")
        axes[0].set_xlabel("Wavenumber (1/m)")
        if title:
            axes[0].set_title(title)
        axes[0].grid(True)

        # TODO: Panel 2 は地震波形で cmap 意図が異なるため opts は適用せず現状維持
        # Panel 2: 地震波形
        t_max = data.shape[1] / fs if fs > 0 else data.shape[1]
        im1 = axes[1].imshow(
            data.T,
            aspect="auto",
            extent=[dist[0], dist[-1], 0.0, t_max],
            origin="lower",
            interpolation="nearest",
        )
        plt.colorbar(im1, ax=axes[1], label="Amplitude")
        axes[1].set_xlabel("Distance from source (m)")
        axes[1].set_ylabel("Time (s)")
        axes[1].grid(True)

        plt.tight_layout()
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
        """全周波数帯エネルギー減衰の散布図 (log-log)"""
        if not show and not save_name:
            return
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(distance, energy, label="attenuation")
        ax.plot(
            distance,
            np.ones_like(energy) * noise_mean,
            label="noise_level",
            color="red",
            linestyle="--",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True)
        ax.set_xlabel("distance from source [m]")
        ax.set_ylabel("attenuation")
        ax.set_title("attenuation S[hc]^2")
        ax.legend()

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="attenuation_energy",
            figure=fig,
            axes=[ax],
            artists=[ax.collections[0], ax.lines[0]],
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
        """周波数ごとの減衰散布図 (log-log)"""
        if not show and not save_name:
            return
        freq_t = str(np.round(freq, 3)).replace(".", "_")
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(distance, attenuation, label="attenuation")
        ax.plot(
            distance,
            np.ones_like(attenuation) * noise_mean,
            label="noise_level",
            color="red",
            linestyle="--",
        )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("distance from source [m]")
        ax.set_ylabel("attenuation")
        ax.grid(True)
        ax.set_title(f"attenuation analysis at {freq_t}Hz")
        ax.legend()

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="attenuation_freq",
            figure=fig,
            axes=[ax],
            artists=[ax.collections[0], ax.lines[0]],
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
        show: bool,
        save_name: str | None = None,
        **kw,
    ) -> None:
        """幾何減衰補正済み振幅と近似直線のプロット"""
        if not show and not save_name:
            return
        fig, ax_plot = plt.subplots(figsize=(4, 5))
        ax_plot.plot(
            distance_ax,
            geo_att,
            "o",
            [0, distance_ax.max()],
            [0, a * distance_ax.max()],
            "--",
        )
        title = f"freq={round(target_freq, 1)}Hz damp ratio α:{round(a, 3)}[1/m]"
        ax_plot.set_title(title)
        ax_plot.set_xlabel(
            "wave number" if velocity is not None else "distance from source[m]"
        )
        ax_plot.set_ylabel("amp ratio (dB)" if dB else "amp ratio (log)")
        ax_plot.set_yscale("log")
        ax_plot.grid(True)
        ax_plot.legend(["attenuation", "fitting line"])
        if dB:
            ax_plot.set_title(title + " [dB]")

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="attenuation_fit",
            figure=fig,
            axes=[ax_plot],
            artists=list(ax_plot.lines[:2]),
        )
        self._finalize(fig, save_name, show)

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
        """Butterworth バンドパスフィルタの周波数応答を表示する。"""
        if not show and not save_name:
            return
        fig, ax1 = plt.subplots(sharex=True)
        ax1.set_title("Digital filter frequency response")
        ax1.plot(w, 20 * np.log10(np.abs(h)), "b")
        ax1.set_ylabel("Amplitude [dB]", color="b")
        ax1.set_xlabel("Frequency [Hz]")
        ax2 = ax1.twinx()
        ax2.plot(w, np.angle(h) / 2 * np.pi * 360, "r")
        ax2.set_ylabel("Angles (degree)", color="r")
        ax2.grid()
        ax2.axis("tight")
        fmax = max(fstop) * 10
        fmin = min(fstop) * 0.1
        ax1.set_xlim([fmin, fmax])
        ax2.set_xlim([fmin, fmax])
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        ax1.set_ylim([-100, 10])
        ax2.set_ylim([-360, 360])
        ax1.axvspan(fpass[0], fpass[1], color="orange", alpha=0.5)
        ax1.axvspan(fmin, fstop[0], color="purple", alpha=0.5)
        ax1.axvspan(fstop[1], fmax, color="purple", alpha=0.5)

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="bandpass_response",
            figure=fig,
            axes=[ax1, ax2],
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
        show: bool,
        save_name: str | None = None,
        **kw,
    ):
        """反射断面の imshow プロット"""
        x_range = targets_sorted if plot_x_as_distance else cmp_pos

        opts = self._default_imshow_kw(kw, cmap="gray")
        fig, ax = plt.subplots(figsize=opts.pop("figsize"))
        vmax = (
            np.nanmax(np.abs(stack_horiz)) * 0.6
            if np.any(~np.isnan(stack_horiz))
            else 1.0
        )
        im = ax.imshow(
            stack_horiz.T,
            extent=[x_range.min(), x_range.max(), elev_axis.min(), elev_axis.max()],
            vmin=-vmax,
            vmax=vmax,
            **opts,
        )
        ax.invert_yaxis()

        ax.set_xlabel("distance [m]" if plot_x_as_distance else "CMP position")
        _xmin = x_range.min() if xmin is None else xmin
        _xmax = x_range.max() if xmax is None else xmax
        xticks = np.linspace(_xmin, _xmax, num_ticks)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f"{x:.1f}" for x in xticks])

        ax.set_ylabel("Elevation [m]")
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
        y_min = np.floor(elev_axis.min() / base) * base
        y_max = np.ceil(elev_axis.max() / base) * base
        yticks = np.arange(y_min, y_max + base, base)
        ax.set_ylim([y_min, y_max])
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:.1f}" for y in yticks])

        if title:
            ax.set_title(title)
        cb = None
        if colorbar:
            cb = fig.colorbar(im, ax=ax, label="Amplitude")
        if dense_x is not None and hm_dense is not None:
            ax.plot(dense_x, hm_dense, "r-", lw=1.0, label="Surface")

        if ymin is not None and ymax is not None:
            ax.set_ylim([ymin, ymax])
        if xmin is not None and xmax is not None:
            ax.set_xlim([xmin, xmax])

        plt.tight_layout()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="reflection_image",
            figure=fig,
            axes=[ax],
            artists=[im],
            colorbars=[cb] if cb is not None else [],
        )
        self._finalize(fig, save_name, show)
        return fig, ax, im

    # ---------------GrounpProcesser---------------------
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
        """
        Plot CMP gather as a single wiggle plot:
        x = offset [m], y = time [s].
        plot_mode: 'fill', 'plot', or 'scatter'

        save_name は呼び出し元 (mixin_wrapper.cmp_image) で
        target ごとのサフィックス付与済みのフルパスとして渡される。
        """
        if "mode" in kw:
            plot_mode = kw.pop("mode")

        n_tr, n_samp = data.shape

        # 時間範囲に合わせてトレースを切り出し
        t_from_step = int(max(0.0, min(t)) * fs)
        t_to_step = min(int(max(t) * fs), n_samp)
        t_axis = np.arange(t_from_step, t_to_step) / fs

        # 振幅スケーリング
        max_x = np.abs(data[:, t_from_step:t_to_step]).max()
        if max_x == 0:
            max_x = 1.0

        # spacing: オフセット間隔（適宜チューニング）
        # interval: 各トレースの「横幅」をどの程度にするかのスケール
        spacing_array = np.asarray(off, dtype=float)
        # spacing = np.median(np.diff(np.sort(spacing_array)))  # 必要ならこう推定
        spacing = 0.8
        interval = spacing  # シンプルに同じにしておく
        space = spacing * max_x
        ratio = interval / space if space != 0 else 1.0

        fig, ax = plt.subplots(figsize=(10, 8))

        for i in range(n_tr):
            trace = (data[i, t_from_step:t_to_step] * ratio).astype(np.float32)
            offset = float(spacing_array[i])

            if plot_mode == "plot":
                # x = offset + trace, y = t_axis
                ax.plot(offset + trace, t_axis, color=color, lw=0.8)

            elif plot_mode == "fill":
                ax.plot(offset + trace, t_axis, color=color, lw=0.8)
                pos_mask = trace > 0
                ax.fill_betweenx(
                    t_axis,
                    offset,
                    offset + trace,
                    where=pos_mask,
                    facecolor=color,
                    alpha=0.8,
                )

            elif plot_mode == "scatter":
                ax.scatter(offset + trace, t_axis, color=color, s=2.0)
            else:
                raise ValueError("Invalid plot_mode. select plot, fill or scatter.")

        ax.set_xlabel("Offset (m)")
        ax.set_ylabel("Time (s)")
        ax.invert_yaxis()  # seismic っぽく上を浅くしたい場合
        ax.set_title(f"CMP target={target_cmp:.4f}")

        # 余白調整
        fig.tight_layout()

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="cmp",
            figure=fig,
            axes=[ax],
            artists=ax.lines[:3],
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

        opts = self._default_imshow_kw(kw, cmap="jet")
        extent = [kxmin, kxmax, fmin, fmax]

        fig = plt.figure(figsize=opts.pop("figsize"))
        amp = np.abs(res)
        plt.imshow(amp, extent=extent, **opts)
        plt.ylabel("Frequency [Hz]")
        plt.xlabel("Wavenumber [1/m]")
        plt.xlim(kxmin, kxmax)
        plt.ylim(fmin, fmax)
        plt.colorbar()
        plt.tight_layout()
        
        title = kw.get('title', None)
        if title is None:
            plt.title(f"f-k imaging {name} {axis} {analysis}")
        else:
            plt.title(title)
        plt.subplots_adjust(
            left=0.103, right=1.0, bottom=0.084, top=0.95, hspace=0.02, wspace=0.02
        )

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="fk_image",
            figure=fig,
            axes=fig.axes,
        )
        self._finalize(fig, save_name, show)

    def cmp_old(
        self,
        show,
        fs,
        data,
        off,
        target_cmp,
        time_range,
        *,
        save_name: str | None = None,
        color: str = "black",
        mode: str = "fill",
        **kw,
    ) -> None:
        """
        Helper to plot one CMP gather.
        mode: 'fill', 'plot', or 'scatter'"""

        n_tr, n_samp = data.shape

        # time_rangeに合わせて整形
        t_from_step = int(max(0, min(time_range)) * fs)
        t_to_step = min(int(max(time_range) * fs), n_samp)
        t_axis = np.arange(t_from_step, t_to_step, 1) / fs

        fig, axes = plt.subplots(1, n_tr, figsize=(10, 8), sharey=True)
        for i, ax in enumerate(np.atleast_1d(axes)):
            datus = data[i, t_from_step:t_to_step]
            if mode == "fill":
                ax.fill_betweenx(t_axis, 0, datus, color=color)
            elif mode == "plot":
                ax.plot(datus, t_axis, color=color)
            elif mode == "scatter":
                ax.scatter(datus, t_axis, color=color)
            else:
                raise ValueError("Invalid plot_mode. select plot, fill or scatter.")
            ax.set_title(f"d={off[i]:.2f}m", fontsize="x-small")
        plt.subplots_adjust(
            left=0.08, right=0.98, bottom=0.15, top=0.92, hspace=0.023, wspace=0
        )

        self._finalize(fig, save_name, kw.get("show", show))

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
        figsize = kw.get("figsize", (8, 4))
        fig, ax = plt.subplots(figsize=figsize)

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

        ax.plot(freq, y, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True)

        if xscale == "log":
            ax.set_xscale("log")
        if kw.get("xlim") is not None:
            ax.set_xlim(kw["xlim"])
        if kw.get("ylim") is not None:
            ax.set_ylim(kw["ylim"])
        if title is not None:
            ax.set_title(title)

        plt.tight_layout()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="fft_transfunc_image",
            figure=fig,
            axes=[ax],
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
        figsize = kw.get("figsize", (10, 8))
        color = kw.get("color", "blue")

        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)

        if title is not None:
            fig.suptitle(title)

        ax1.plot(t, signal, color=color)
        ax1.set_xlabel(time_xlabel)
        ax1.set_ylabel(time_ylabel)
        ax1.grid(True)

        ax2.plot(freq, spec, color=color)
        ax2.set_xlabel(freq_xlabel)
        ax2.set_ylabel(freq_ylabel)
        ax2.grid(True)
        if xscale == "log":
            ax2.set_xscale("log")

        if "t_xlim" in kw:
            ax1.set_xlim(kw["t_xlim"])
        if "f_xlim" in kw:
            ax2.set_xlim(kw["f_xlim"])
        if "f_ylim" in kw:
            ax2.set_ylim(kw["f_ylim"])

        plt.tight_layout()
        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="fft1d_image",
            figure=fig,
            axes=[ax1, ax2],
        )
        self._finalize(fig, save_name, show)

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
        """走時トモグラフィ結果の 2x2 描画 (初期速度 / 最終速度 / RMS 履歴 / セイモグラム)."""
        figsize = kw.get("figsize", (12, 9))
        cmap_velocity = kw.get("cmap", "viridis")
        cmap_seis = kw.get("seis_cmap", "gray_r")
        init_title = kw.get("init_title", "initial velocity model [m/s]")
        final_title = kw.get("final_title", "final velocity model [m/s]")
        rms_title = kw.get("rms_title", "RMS misfit history")
        seis_title = kw.get("seis_title", "seismogram and synthetic first-break times")
        suptitle = kw.get("suptitle", "traveltime tomography")

        extent_x = [0, grid_nx * grid_dx]
        extent_z = [grid_nz * grid_dz, 0]

        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # --- initial velocity model ---
        ax_init = axes[0, 0]
        im_init = ax_init.imshow(
            initial_velocity,
            aspect="auto",
            extent=[extent_x[0], extent_x[1], extent_z[0], extent_z[1]],
            cmap=cmap_velocity,
        )
        ax_init.set_xlabel(velocity_xlabel)
        ax_init.set_ylabel(velocity_ylabel)
        ax_init.set_title(init_title)
        fig.colorbar(im_init, ax=ax_init)

        # --- final velocity model ---
        ax_final = axes[0, 1]
        im_final = ax_final.imshow(
            final_velocity,
            aspect="auto",
            extent=[extent_x[0], extent_x[1], extent_z[0], extent_z[1]],
            cmap=cmap_velocity,
        )
        ax_final.set_xlabel(velocity_xlabel)
        ax_final.set_ylabel(velocity_ylabel)
        ax_final.set_title(final_title)
        fig.colorbar(im_final, ax=ax_final)

        # --- RMS misfit history ---
        ax_rms = axes[1, 0]
        if len(rms) > 0:
            ax_rms.plot(range(len(rms)), rms, "o-", linewidth=1.5, markersize=4)
        ax_rms.set_xlabel(rms_xlabel)
        ax_rms.set_ylabel(rms_ylabel)
        ax_rms.set_title(rms_title)
        ax_rms.grid(True, linestyle="--", alpha=0.6)

        # --- seismogram + synthetic/picked traveltimes ---
        ax_seis = axes[1, 1]
        n_traces, n_samples = seis_data.shape
        t_max = n_samples * dt

        ax_seis.imshow(
            seis_data.T,
            aspect="auto",
            extent=[0, n_traces, t_max, 0.0],
            cmap=cmap_seis,
            vmin=np.percentile(seis_data, 2),
            vmax=np.percentile(seis_data, 98),
        )

        if synthetic_tt is not None:
            for itrace in range(len(synthetic_tt)):
                t = synthetic_tt[itrace]
                if np.isfinite(t) and t > 0:
                    ax_seis.plot(itrace + 0.5, t, "r.", markersize=3)

        if picks is not None:
            for itrace in range(len(picks)):
                t = picks[itrace]
                if np.isfinite(t) and t > 0:
                    ax_seis.plot(itrace + 0.5, t, "b.", markersize=2)

        ax_seis.set_xlabel(seis_xlabel)
        ax_seis.set_ylabel(seis_ylabel)
        ax_seis.set_title(seis_title)
        ax_seis.legend(
            handles=[
                plt.Line2D([0], [0], marker="o", color="red", linestyle="None", label="Synthetic travel times", markersize=5),
                plt.Line2D([0], [0], marker="o", color="blue", linestyle="None", label="Picked travel times", markersize=5),
            ],
            loc="upper right",
            fontsize=8,
        )

        fig.suptitle(suptitle, fontsize=14)
        plt.tight_layout()

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="traveltime_tomo",
            figure=fig,
            axes=[ax_init, ax_final, ax_rms, ax_seis],
        )
        self._finalize(fig, save_name, show)

    def _rayleigh_dispersion_fit_image_impl(
        self,
        res,
        f_mesh,
        c_mesh,
        qc_result,
        theory_c,
        theory_f,
        *,
        target,
        show,
        save_name,
        **kwargs,
    ):
        """per-CMP 分散エネルギーマップ + 観測ピーク + 理論曲線の重畳描画."""
        figsize = kwargs.get("figsize", (8, 5))
        cmap = kwargs.get("cmap", "jet")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        f_arr = np.asarray(f_mesh, dtype=float)
        c_arr = np.asarray(c_mesh, dtype=float)
        f_vec = f_arr[:, 0] if f_arr.ndim == 2 else f_arr.ravel()
        c_vec = c_arr[0, :] if c_arr.ndim == 2 else c_arr.ravel()
        n_f = f_vec.size
        n_c = c_vec.size

        res_arr = np.asarray(res, dtype=float)
        res_max = float(res_arr.max()) if res_arr.size else 0.0
        res_norm = res_arr / res_max if res_max > 0 else res_arr

        if res_norm.shape == (n_f, n_c):
            grid = res_norm.T
        elif res_norm.shape == (n_c, n_f):
            grid = res_norm
        else:
            grid = res_norm

        im = ax.pcolormesh(
            f_vec,
            c_vec,
            grid,
            cmap=cmap,
            shading="auto",
            vmin=0,
            vmax=1 if res_max > 0 else None,
        )
        cb = fig.colorbar(im, ax=ax, label="Normalized energy")

        has_overlay = False
        if qc_result is not None:
            picked = getattr(qc_result, "picked", None)
            if picked is not None:
                f_valid = np.asarray(picked.f, dtype=float)
                c_valid = np.asarray(picked.c, dtype=float)
                fin_mask = np.isfinite(f_valid) & np.isfinite(c_valid)
                if np.any(fin_mask):
                    ax.scatter(
                        f_valid[fin_mask],
                        c_valid[fin_mask],
                        marker="o",
                        s=20,
                        color="white",
                        edgecolor="black",
                        linewidths=0.4,
                        label="Observed picks",
                        zorder=5,
                    )
                    has_overlay = True

        if theory_c is not None and theory_f is not None:
            tf = np.asarray(theory_f, dtype=float)
            tc = np.asarray(theory_c, dtype=float)
            fin = np.isfinite(tf) & np.isfinite(tc)
            if np.any(fin):
                ax.plot(
                    tf[fin],
                    tc[fin],
                    "w--",
                    lw=1.5,
                    label="Theoretical (mode 0)",
                    zorder=6,
                )
                has_overlay = True

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Phase velocity (m/s)")
        if target is not None:
            ax.set_title(f"CMP: {target}")
        else:
            ax.set_title("Dispersion fit")

        if has_overlay:
            ax.legend(loc="upper right", fontsize=8, framealpha=0.7)

        plt.tight_layout()

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="rayleigh_dispersion_fit_image",
            figure=fig,
            axes=[ax],
            artists=[im],
            colorbars=[cb],
        )
        self._finalize(fig, save_name, show)

    def _vs_section_impl(
        self,
        section,
        *,
        show,
        save_name,
        **kwargs,
    ):
        """Pseudo-2D Vs section の pcolormesh 描画."""
        figsize = kwargs.get("figsize", (10, 4))
        cmap_name = kwargs.get("cmap", "jet_r")

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        vsgrid = np.asarray(section.vsgrid, dtype=float).copy()
        vsgrid[vsgrid == 0] = np.nan

        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="white", alpha=0.0)

        im = ax.pcolormesh(
            section.cmpx,
            section.depthz,
            vsgrid,
            cmap=cmap,
            shading="auto",
        )
        ax.invert_yaxis()
        ax.set_xlabel("CMP position (m)")
        ax.set_ylabel("Depth (m)")
        ax.set_title("Pseudo-2D Vs Section")
        cb = fig.colorbar(im, ax=ax, label="Vs (m/s)")

        confidence_grid = getattr(section, "confidence_grid", None)
        if confidence_grid is not None:
            mask = np.asarray(confidence_grid, dtype=float) < 0.1
            ax.pcolormesh(
                section.cmpx,
                section.depthz,
                np.where(mask, 1.0, np.nan),
                cmap="Greys",
                alpha=0.5,
                shading="auto",
            )

        plt.tight_layout()

        _result = MatplotlibBackendResult(
            backend_name="matplotlib",
            plot_type="vs_section",
            figure=fig,
            axes=[ax],
            artists=[im],
            colorbars=[cb],
        )
        self._finalize(fig, save_name, show)

    def _finalize(self, fig, save_name, show):
        if save_name:
            fp = Path(save_name)
            fp.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(fp, dpi=self.dpi, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)