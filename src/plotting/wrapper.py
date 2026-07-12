"""
Canonical plotting wrapper facade location.

New code should import ``PlotterWrapperMixin`` from this module
(``src.plotting.wrapper``).
"""
import os
import warnings
from pathlib import Path
from typing import Any, Literal

import numpy as np

from src.plotting.backends.matplotlib_backend import MatplotlibPlotter
from src.plotting.backends.plotly_backend import PlotlyPlotter
from src.plotters import BackscatterDistributionPlotter, BackscatterPlotter, CmpPlotter, DispersionPlotter, Fft1dPlotter, FftTransfuncPlotter, FkPlotter, ManualPickPlotter, RayleighInversionPlotter, ReflectionPlotter, SeismogramPlotter, SpectraPlotter, TraveltimeTomoPlotter
from src.utils import utils

# 旧 API で使われていた保存系キーワード。新 API では save_name に統一されたため
# これらを渡された場合は TypeError を即時発生させる。
_LEGACY_SAVE_KWARGS = ("save", "outdir", "dir", "savename", "save_dir")


def _reject_legacy_save_kwargs(kw: dict) -> None:
    """**kw に旧保存系引数が含まれていたら TypeError を投げる。"""
    bad = [k for k in _LEGACY_SAVE_KWARGS if k in kw]
    if bad:
        raise TypeError(
            f"以下の引数は廃止されました: {bad}. "
            f"代わりに save_name(str|Path|None) を使用してください。"
        )


def _normalize_save_name(save_name) -> str | None:
    """save_name を正規化する。'' / None / 空白のみは None として扱う。"""
    if save_name is None:
        return None
    s = str(save_name)
    if not s.strip():
        return None
    return s


def _append_save_suffix(save_name, suffix: str) -> str | None:
    """
    save_name の拡張子の前に '_<suffix>' を挿入する。
    例: '/figs/atten.png', 'allfreq' -> '/figs/atten_allfreq.png'
    save_name が None/空 の場合は None を返す。
    """
    save_name = _normalize_save_name(save_name)
    if save_name is None:
        return None
    p = Path(save_name)
    return str(p.with_name(f"{p.stem}_{suffix}{p.suffix}"))


class PlotterWrapperMixin:
    """
    Processor に継承させるためのラッパー。
    plot_seismogram() を実装し、内部で描画クラス (MatplotlibPlotter, PlotlyPlotter) に処理を委譲する。

    **メソッドはいずれも show/save_name のどちらかが呼ばれているときに呼ばれるコマンドだが、各プロッタークラスでの読みやすさのために、showは残す。
    """

    _plotter: Any

    # =================================================================
    #  Plotter 初期化
    # =================================================================

    # ★ デフォルト backend。変更する場合はこの1行を書き換える。
    # ★ 実行コード上で backend を指定したい場合は PlotProcesser(backend='plotly') を使うこと。
    DEFAULT_PLOT_BACKEND: str = "mpl"

    def init_plotter(self, backend: Literal["mpl", "plotly"], **kwargs):
        if backend == "mpl":
            self._plotter = MatplotlibPlotter(**kwargs)
        elif backend == "plotly":
            self._plotter = PlotlyPlotter(**kwargs)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def _ensure_plotter(self, backend: str | None = None) -> None:
        """
        _plotter が未初期化の場合、指定 backend で自動初期化する
        (backend=None のとき DEFAULT_PLOT_BACKEND を使用)。
        各描画メソッドの冒頭で呼ぶことで、init_plotter() の明示呼び出しを不要にする。

        backend を切り替えたい場合は、描画メソッドを呼ぶ前に
            plot = PlotProcesser(backend='plotly', dpi=600)
        のように PlotProcesser を使って明示的に初期化すること。
        """
        if not hasattr(self, "_plotter") or self._plotter is None:
            self.init_plotter(backend=backend or self.DEFAULT_PLOT_BACKEND)

    # =================================================================
    #  Public API
    # =================================================================

    def seismogram(
        self,
        axis: str,
        t: list[float] | None = None,
        spacing: float = 0.8,
        title: str | None = None,
        fill: bool = True,
        agc: bool = False,
        show: bool = True,
        save_name: str | None = None,
        input: str = None,
        data: np.ndarray = None,
        **kw,
    ):
        """
        地震波形 (wiggle) プロット。

        Parameters
        ----------
        input : str | SingleProcesser
            NPZファイルパス または SingleProcesser インスタンス
        axis : 'x' | 'y' | 'z' |
            プロットする成分。
        title : str | None
            図タイトル。None のとき自動生成。
        t : [t_start, t_end] | None
            表示時間範囲 [s]。None のとき全区間。
        spacing : float
            wiggle の振幅スケール係数。
        fill : bool
            正振幅の塗りつぶし。
        agc : bool
            Automatic Gain Control を適用するか。
            kw: agc_batchsize (int, default=100), agc_clip (int, default=10)
        show : bool
            True のとき plt.show() を呼ぶ。
        save_name : str | Path | None
            保存ファイル名 (パス込み)。None / '' のとき保存しない。
        data : np.adarray
            特定のデータを見たいときに用いる

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``figsize``, ``xmin``, ``xmax``, ``xlabel``, ``ylabel``, ``title``。
        ``spacing`` / ``fill`` / ``agc`` は本メソッドの明示引数としても渡せる。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        # input 未指定なら self 自身をデータソースとして使う
        target = input if input is not None else self
        self._load_input(target)

        if data is None:
            axis = self._resolve_axis(axis)
            target_data = getattr(self, axis)
        else:
            print("special key. Showing special input data")
            target_data = data

        if agc:
            batchsize = kw.get("agc_batchsize", 100)
            clip = kw.get("agc_clip", 10)
            target_data = self._apply_agc(target_data, batchsize=batchsize, clip=clip)

        kw["title"] = self._build_title(
            axis,
            title,
            agc,
            batchsize=kw.get("agc_batchsize", 100) if agc else None,
            clip=kw.get("agc_clip", 10) if agc else None,
        )

        fs, dt, interval, source_x, distance = self._resolve_metadata()

        # 時刻切り出し（ここで一元管理）
        n_samples = target_data.shape[1]
        if t is None:
            step_from, step_to = 0, n_samples
        else:
            step_from = int(max(0, t[0]) * fs)
            step_to = int(min(n_samples, t[1] * fs))

        sliced_data = target_data[:, step_from:step_to]
        t_sec = np.arange(step_from, step_to) * dt

        self._ensure_plotter()
        return SeismogramPlotter(self._plotter).image(
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
            **kw,
        )

    def spectra_image(
        self,
        data: np.ndarray,
        axis: str,
        freq: list[float],
        name: str | None = None,
        analysis: str | None = None,
        show: bool | None = False,
        save_name: str | None = None,
        log: bool | None = False,
        **kw,
    ) -> None:
        """
        周波数スペクトル画像 (imshow) のプロット。

        Parameters
        ----------
        data      : (N_freq, N_sensor) の振幅配列（log変換前）
        freq      : [fmin, fmax] Hz
        log       : True のとき log10 スケールで表示
        save_name : 保存先パス。None のとき保存しない

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``cmap``, ``interpolation``, ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        # データの加工
        ramda = min(1e-20, 1e-20*np.max(np.abs(data))) #ゼロ割を防止する小さい数
        res = np.log10(data + ramda) if log else data

        # TODO: 旧コードの suptitle / dir / savename / save 参照は未定義のため削除。
        #       suptitle 対応は kw 経由に再設計する予定。
        title = f"{name} {axis} {analysis}"

        self._ensure_plotter()
        return SpectraPlotter(self._plotter).image(
            res,
            freq=freq,
            axis=axis,
            title=title,
            show=show,
            save_name=save_name,
            log=log,
            **kw,
        )

    def dispersion_image(
        self,
        input: np.ndarray,
        ax_x: list[float],
        ax_y: list[float],
        axis: str,
        show: bool,
        save_name: str,
        interval: float,
        Num_sensor: int,
        source_x: float,
        sensor1_x: float,
        d_maxdiff: float,
        mode: str,
        **kw,
    ):
        """
        dispersion image (f-c plot) のプロット。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``cmap``, ``interpolation``, ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        nyquist_k = 1 / (2 * interval)
        if d_maxdiff is None:
            d = np.zeros(Num_sensor)
            for i in range(Num_sensor):
                d[i] = np.abs(sensor1_x + i * interval - source_x)
            d_maxdiff = np.max(d) - np.min(d)

        self._ensure_plotter()
        return DispersionPlotter(self._plotter).image(
            input,
            ax_x=ax_x,
            ax_y=ax_y,
            axis=axis,
            show=show,
            save_name=save_name,
            nyquist_k=nyquist_k,
            d_maxdiff=d_maxdiff,
            mode=mode,
            **kw,
        )

    def backscatter_image(
        self,
        ind: np.ndarray,
        amp,
        title: str | None = None,
        ylabel: str = "amplitude",
        show: bool | None = False,
        save_name: str | None = None,
        **kw,
    ):
        """
        後方散乱振幅の線プロット。

        Parameters
        ----------
        ind       : インデックス配列
        amp       : 振幅リスト
        title     : 図タイトル
        ylabel    : y 軸ラベル
        save_name : 保存ファイル名。None のとき保存しない
        show      : True のとき plt.show() を呼ぶ

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return BackscatterPlotter(self._plotter).image(
            ind,
            amp,
            title=title,
            ylabel=ylabel,
            show=show,
            save_name=save_name,
            **kw,
        )

    def backscatter_distribution_image(
        self,
        distance: np.ndarray,
        amp: np.ndarray,
        *,
        count: np.ndarray | None = None,
        title: str | None = None,
        ylabel: str = "averaged amplitude",
        show: bool | None = False,
        save_name: str | None = None,
        **kw,
    ):
        """
        後方散乱分布（平均振幅 + 出現回数）のプロット。

        Parameters
        ----------
        distance  : ユニーク距離 (all_distance) 配列
        amp       : 各距離に対応する平均振幅
        count     : 各距離に対応する加算カウント（None のとき下段は描画しない）
        title     : 図タイトル
        ylabel    : 上段のy軸ラベル
        save_name : 保存ファイル名。None のとき保存しない
        show      : True のとき plt.show() を呼ぶ

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return BackscatterDistributionPlotter(self._plotter).image(
            distance,
            amp,
            count=count,
            title=title,
            ylabel=ylabel,
            show=show,
            save_name=save_name,
            **kw,
        )

    def attenuation(
        self,
        distance: np.ndarray,
        energy: np.ndarray,
        noise_mean: float,
        per_freq_data: list,
        name: str = "",
        *,
        show: bool = False,
        save_name: str | None = None,
    ) -> None:
        """
        減衰解析結果の描画。データ整形を担当し、実描画は MatplotlibPlotter に委譲する。

        Parameters
        ----------
        distance      : センサーごとの震源距離 [m]
        energy        : 全周波数帯エネルギー配列
        noise_mean    : ノイズエネルギーの平均値
        per_freq_data : 周波数ごとの描画データリスト。各要素は以下のキーを持つ dict:
                        'attenuation', 'noise_mean', 'freq',
                        'distance_ax', 'geo_att', 'a', 'dB', 'velocity'
        name          : (互換用パラメータ。新 API では save_name に統合済み)
        show          : True のとき plt.show() を呼ぶ
        save_name     : 保存先パス (拡張子込み)。None / '' のとき保存しない。
                        各サブ画像は拡張子の前に以下サフィックスを付加して保存される:
                          - allfreq      : 全周波数エネルギー散布図
                          - {freq}Hz     : 周波数別散布図
                          - {freq}Hz_damp_ratio : 周波数別 fit 図

        Notes
        -----
        attenuation_fit サブ画像では ``velocity`` / ``dB`` が描画に影響する。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        save_name = _normalize_save_name(save_name)
        if not show and save_name is None:
            return

        self._ensure_plotter()
        self._plotter.attenuation_energy(
            distance,
            energy,
            noise_mean,
            show=show,
            save_name=_append_save_suffix(save_name, "allfreq"),
        )

        for d in per_freq_data:
            freq = d["freq"]
            freq_t = str(freq).replace(".", "_")
            self._plotter.attenuation_freq(
                distance,
                d["attenuation"],
                d["noise_mean"],
                freq,
                show=show,
                save_name=_append_save_suffix(save_name, f"{freq_t}Hz"),
            )
            self._plotter.attenuation_fit(
                d["distance_ax"],
                d["geo_att"],
                d["a"],
                freq,
                velocity=d.get("velocity"),
                dB=d.get("dB", False),
                show=show,
                save_name=_append_save_suffix(save_name, f"{freq_t}Hz_damp_ratio"),
            )

    def cmap(
        self,
        input,
        axis,
        t: list | None = None,
        show: bool | None = False,
        save_name: str | None = None,
        title: str | None = None,
        agc: bool | None = False,
        **kw,
    ):
        """
        カラーマップ (imshow) プロット。

        Parameters
        ----------
        input : str | SingleProcesser
            NPZファイルパス または SingleProcesser インスタンス
        axis : 'x' | 'y' | 'z'
            プロットする成分
        t : [t_start, t_end] | None
            表示時間範囲 [s]。None のとき全区間。
        show : bool
            True のとき plt.show() を呼ぶ
        save_name : str | Path | None
            保存ファイル名。None / '' のとき保存しない。
        title : str | None
            図タイトル。None のとき自動生成。
        agc : bool
            Automatic Gain Control を適用するか。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``cmap``, ``interpolation``, ``aspect``, ``figsize``, ``agc``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        self._load_input(input)
        axis = self._resolve_axis(axis)
        target_data = getattr(self, axis)

        M, N = target_data.shape
        fs = self.fs

        # dataをself.distanceの降順に並び替える
        sort_indices = np.argsort(self.distance)
        # 2. 1D配列の distance 自体を並び替える
        sorted_distance = self.distance[sort_indices]
        # 3. 2D配列の target_data (n_traces, n_samples) を並び替える
        sorted_data = target_data[sort_indices, :]

        t_start = t[0] if t is not None and t[0] is not None else 0.0
        t_end = t[1] if t is not None and t[1] is not None else N / fs
        step_from = max(int(t_start * fs), 0)
        step_to = min(int(t_end * fs), N)

        # 表示するChも編集できるようにしよう
        # remoce_chs=[3,4]-> 4番目と5番目のchは摘出する　とか

        ax_modified = sorted_data[:, step_from:step_to]
        t_from = step_to / fs
        t_to = step_from / fs
        extent = (0, M, t_from, t_to)

        if agc:
            batchsize = kw.get("agc_batchsize", 100)
            clip = kw.get("agc_clip", 10)
            ax_modified = self._apply_agc(ax_modified, batchsize=batchsize, clip=clip)

        self._ensure_plotter()
        return self._plotter.cmap(
            data=ax_modified,
            extent=extent,
            show=show,
            save_name=save_name,
            title=title,
            **kw,
        )

    def reflection_image(
        self,
        stack_horiz: np.ndarray,
        cmp_pos: np.ndarray,
        elev_axis: np.ndarray,
        targets_sorted: np.ndarray,
        *,
        plot_x_as_distance: bool = True,
        title: str = "",
        num_ticks: int = 8,
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
        """
        反射断面プロット。描画データの整形は reflection() が担い、
        実描画は MatplotlibPlotter.reflection_image() に委譲する。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``cmap``, ``interpolation``, ``colorbar``, ``num_ticks``, ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return ReflectionPlotter(self._plotter).image(
            stack_horiz,
            cmp_pos,
            elev_axis,
            targets_sorted,
            plot_x_as_distance=plot_x_as_distance,
            title=title,
            colorbar=True,
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

    def cmp_image(
        self,
        cmp,
        offsets,
        fs: float,
        target: float | list[float],
        t: list[float, float] = [0.0, 1.7],
        show: bool | None = False,
        save_name: str | None = None,
        color: str = "black",
        plot_mode: str = "fill",
        **kw,
    ) -> None:
        """
        Display CMP gather(s) as 'fill', 'plot', or 'scatter' over time.

        save_name は CMP target ごとに拡張子の前へ
        '_<round(target,4)を _ で繋いだ文字列>' を挿入して保存する。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``color``, ``plot_mode``, ``mode``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)
        if "mode" in kw:
            plot_mode = kw.pop("mode")

        self._ensure_plotter()
        tgts = [target] if isinstance(target, float) else target
        for tgt in tgts:
            data = cmp[tgt]
            off = offsets[tgt]
            target_suffix = str(round(float(tgt), 4)).replace(".", "_")
            target_save = _append_save_suffix(save_name, target_suffix)
            CmpPlotter(self._plotter).image(
                show,
                fs,
                data,
                off,
                tgt,
                t,
                save_name=target_save,
                color=color,
                plot_mode=plot_mode,
            )

    def fft_transfunc_image(
        self,
        freq: np.ndarray,
        transfunc: np.ndarray,
        *,
        show: bool,
        save_name: str | None = None,
        **kw,
    ) -> None:
        """伝達関数スペクトルプロットを backend に委譲する。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``complex_mode``, ``xlim``, ``ylim``, ``color``, ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return FftTransfuncPlotter(self._plotter).image(
            freq,
            transfunc,
            show=show,
            save_name=save_name,
            **kw,
        )

    def fft1d_image(
        self,
        t: np.ndarray,
        signal: np.ndarray,
        freq: np.ndarray,
        spec: np.ndarray,
        *,
        show: bool,
        save_name: str | None = None,
        **kw,
    ) -> None:
        """FFT 1D プロット (時系列 + スペクトル 2 パネル) を backend に委譲する。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``color``, ``t_xlim``, ``f_xlim``, ``f_ylim``, ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return Fft1dPlotter(self._plotter).image(
            t,
            signal,
            freq,
            spec,
            show=show,
            save_name=save_name,
            **kw,
        )

    def fk_image(
        self,
        axis,
        res,
        kxmin,
        kxmax,
        fmin,
        fmax,
        analysis,
        show,
        save_name,
        **kw,
    ):
        """
        f-k スペクトル画像のプロット。

        Notes
        -----
        描画調整用の ``**kw`` を受け付ける。
        代表例: ``cmap``, ``interpolation``, ``title``, ``figsize``。
        完全一覧と backend ごとの差異は
        ``src/plotting/backends/matplotlib_backend.py`` /
        ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        name = self.name

        self._ensure_plotter()
        return FkPlotter(self._plotter).image(
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
        )

    def manual_pick_gui(
        self,
        traces: np.ndarray,
        fs: float,
        *,
        distance=None,
        **kw,
    ) -> np.ndarray:
        """
        手動初動ピッキング GUI を起動し、各 trace の初動時刻 [s] を返す。

        ``traveltime_tomography(picking_mode="manual")`` から、手動 pick 引数
        (manual_picks / manual_pick_times / manual_pick_indices) が一切与えられ
        なかった場合に呼ばれる thin な convenience 入口。GUI 本体は backend 側
        (``_manual_pick_impl``) が担い、本メソッドは時間軸の解決と plotter への
        委譲のみを行う。

        Parameters
        ----------
        traces : ndarray, shape=(n_traces, n_samples)
            ピッキング対象の波形 (trace 順は呼び出し側の index と一致)。
        fs : float
            サンプリング周波数 [Hz]。
        distance : array-like, optional
            各 trace のラベル表示用 (記録される pick index には影響しない)。

        Returns
        -------
        pick_times : ndarray, shape=(n_traces,), float [s]
        """
        traces = np.asarray(traces, dtype=float)
        n_t = traces.shape[1]
        t_sec = np.arange(n_t, dtype=float) / float(fs)
        self._ensure_plotter()
        return ManualPickPlotter(self._plotter).pick(
            traces=traces,
            t_sec=t_sec,
            distance=distance,
            **kw,
        )

    def traveltime_tomo_image(
        self,
        result: dict,
        axis: str,
        *,
        show: bool = True,
        save_name: str | None = None,
        use_topography_overlay: bool = False,
        show_ray_paths: bool = False,
        **kw,
    ):
        """
        走時トモグラフィ結果の 2x2 描画。

        Parameters
        ----------
        result : dict
            traveltime_tomography() の戻り値。
            'initial_velocity', 'velocity_model', 'history',
            'synthetic_tt', 'picks', 'grid' を含む必要がある。
        axis : str
            セイモグラムパネルに表示する波形成分 ('x', 'y', 'z')。
        show : bool
            True の場合 plt.show() を呼ぶ。
        save_name : str | Path | None
            保存先パス。None / '' のとき保存しない。
        **kw
            描画調整用の keyword 引数。
            代表例: ``figsize``, ``cmap``, ``seis_cmap``, ``init_title``,
            ``final_title``, ``rms_title``, ``seis_title``, ``suptitle``。
            完全一覧と backend ごとの差異は
            ``src/plotting/backends/matplotlib_backend.py`` /
            ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
        """
        _reject_legacy_save_kwargs(kw)
        save_name = _normalize_save_name(save_name)

        # Topography overlay: inject z_distance, distance, source_x, grid_x0 into kw
        if use_topography_overlay:
            if not hasattr(self, "z_distance"):
                warnings.warn(
                    "use_topography_overlay=True が指定されましたが self.z_distance が存在しません。"
                    "overlay を描画しません。",
                    UserWarning,
                    stacklevel=2,
                )
            elif np.asarray(self.z_distance).shape != np.asarray(self.distance).shape:
                warnings.warn(
                    "z_distance と distance の shape が一致しません。overlay を描画しません。",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                kw.setdefault("z_distance", np.asarray(self.z_distance, dtype=float))
                kw.setdefault("distance",   np.asarray(self.distance,   dtype=float))
                kw.setdefault("source_x",   float(self.source_x))
                kw.setdefault("grid_x0",    float(result["grid"].get("x0", 0.0)))
                kw.setdefault("velocity_ylabel", "Elevation [m]")

        # Ray paths overlay: inject ray_paths into kw if available and requested
        if show_ray_paths:
            if "ray_paths" not in result:
                warnings.warn(
                    "show_ray_paths=True が指定されましたが result に 'ray_paths' キーがありません。"
                    " traveltime_tomography(..., store_ray_paths=True) で実行してください。"
                    " ray overlay を描画しません。",
                    UserWarning,
                    stacklevel=2,
                )
            else:
                kw.setdefault("ray_paths", result["ray_paths"])

        # T3: pass elevation_correction_info to backend via **kw
        _elev_info = result.get("elevation_correction")
        if _elev_info is not None:
            kw.setdefault("elevation_correction_info", _elev_info)
            kw.setdefault("velocity_ylabel", "Elevation [m]")

        initial_velocity = result["initial_velocity"]
        final_velocity = result["velocity_model"]
        history = result.get("history", [])
        rms = np.array([h["rms"] for h in history]) if history else np.array([])
        synthetic_tt = result.get("synthetic_tt")
        picks = result.get("picks")
        grid = result["grid"]

        axis = self._resolve_axis(axis)
        seis_data = getattr(self, axis)
        dt = 1.0 / self.fs

        self._ensure_plotter()
        return TraveltimeTomoPlotter(self._plotter).image(
            initial_velocity=initial_velocity,
            final_velocity=final_velocity,
            rms=rms,
            seis_data=seis_data,
            dt=dt,
            synthetic_tt=synthetic_tt,
            picks=picks,
            grid_nx=grid["nx"],
            grid_nz=grid["nz"],
            grid_dx=grid["dx"],
            grid_dz=grid["dz"],
            show=show,
            save_name=save_name,
            **kw,
        )

    def rayleigh_dispersion_fit_image(
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
        """per-CMP 分散エネルギーマップ + 観測ピーク + 理論曲線の重畳描画。

        Notes
        -----
        既存 dispersion_image (raw f-c 画像) とは責務が異なる。Rayleigh 波 1D
        逆解析のフィット結果を CMP 単位で診断するためのオーバーレイ込み描画。
        """
        _reject_legacy_save_kwargs(kwargs)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return RayleighInversionPlotter(self._plotter).dispersion_fit_image(
            res,
            f_mesh,
            c_mesh,
            qc_result=qc_result,
            theory_c=theory_c,
            theory_f=theory_f,
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
        """Pseudo-2D Vs section の pcolormesh 描画。"""
        _reject_legacy_save_kwargs(kwargs)
        save_name = _normalize_save_name(save_name)

        self._ensure_plotter()
        return RayleighInversionPlotter(self._plotter).vs_section(
            section,
            show=show,
            save_name=save_name,
            **kwargs,
        )

    # =================================================================
    #  Private helpers — データ解決
    # =================================================================

    def _load_input(self, input):
        """input の型に応じてデータを self に読み込む。"""
        if isinstance(input, str):
            self._load_npz(input)
        elif type(input).__name__ == "SingleProcesser":
            self._load_single_processer(input)
        else:
            raise TypeError(
                f"入力された型 '{type(input).__name__}' は無効です。"
                "ファイルパス(str) または SingleProcesser インスタンスを指定してください。"
            )

    def _resolve_axis(self, axis: str) -> str:
        """axis を確定する。None のときエラーを表示"""
        if axis is not None:
            if axis not in ("x", "y", "z"):
                raise ValueError(
                    f"無効な軸指定: '{axis}'. 'x', 'y', 'z' のいずれかを指定してください。"
                )
            return axis
        raise ValueError(
            f"{axis}が未設定です。'x', 'y', 'z' のいずれかを指定してください"
        )

    def _resolve_metadata(self):
        """プロット用のメタデータをまとめて取得する。"""
        fs = getattr(self, "fs", 1.0)
        dt = 1.0 / fs
        interval = getattr(self, "interval", 1.0)
        source_x = getattr(self, "source_x", None)
        distance = getattr(self, "distance", None)
        return fs, dt, interval, source_x, distance

    def _build_title(
        self,
        axis: str,
        title: str | None,
        agc: bool,
        *,
        batchsize: int | None,
        clip: int | None,
    ) -> str:
        """タイトルを確定して返す。save_name は呼び出し元が直接管理する。"""
        if title is not None:
            return title

        proc_name = getattr(self, "name", "")
        analysis_text = getattr(self, f"analysis_{axis}", "")
        if agc:
            analysis_text += f" + AGC(batch={batchsize}, clip={clip})"

        return f"{proc_name} [{axis.upper()}] {analysis_text}"

    # =================================================================
    #  Private helpers — AGC
    # =================================================================

    def _apply_agc(self, data: np.ndarray, *, batchsize: int, clip: int) -> np.ndarray:
        """
        Automatic Gain Control をバッチ単位で適用して返す。

        Parameters
        ----------
        data      : shape (N, M)
        batchsize : サンプル単位のバッチ幅
        clip      : ゲインの上限倍率
        """
        data = data.copy()  # ←元のデータの汚染を防ぐ
        N, M = data.shape
        maxamp = np.abs(data).max()

        gain_full = self._batch_gain(data, N, M, batchsize, maxamp, clip)
        utils.multiple(data, gain_full)
        return data

    @staticmethod
    def _batch_gain(
        data: np.ndarray,
        N: int,
        M: int,
        batchsize: int,
        maxamp: float,
        clip: int,
    ) -> np.ndarray:
        """バッチ毎のゲイン行列 (shape: N×M) を作成する。"""
        pad_len = (-M) % batchsize
        abs_data = (
            np.pad(np.abs(data), ((0, 0), (0, pad_len)), "constant")
            if pad_len
            else np.abs(data)
        )

        B = abs_data.shape[1] // batchsize
        batch_max = abs_data.reshape(N, B, batchsize).max(axis=2)  # (N, B)

        gain_batch = maxamp / (batch_max + 1e-3 * maxamp)  # (N, B)
        gain_batch = np.clip(gain_batch, 1.0, clip)

        gain_full = np.repeat(gain_batch[:, :, None], batchsize, axis=2).reshape(N, -1)[
            :, :M
        ]  # (N, M)
        return gain_full

    # =================================================================
    #  Private helpers — データ読み込み
    # =================================================================

    def _load_npz(self, file_path: str):
        """NPZファイルを読み込んで self に属性をセットする。"""
        self.name = os.path.splitext(os.path.basename(file_path))[0]

        with np.load(file_path, allow_pickle=True) as file:
            for ax in ("x", "y", "z"):
                if ax in file:
                    setattr(self, ax, file[ax].copy())
                    setattr(self, f"analysis_{ax}", "")

            self.fs = file["fs"] if "fs" in file else getattr(self, "fs", 1.0)
            self.interval = (
                file["interval"]
                if "interval" in file
                else getattr(self, "interval", 1.0)
            )
            self.source_x = (
                file["source_x"]
                if "source_x" in file
                else getattr(self, "source_x", None)
            )
            self.distance = (
                file["distance"]
                if "distance" in file
                else getattr(self, "distance", None)
            )

    def _load_single_processer(self, sp):
        """SingleProcesser インスタンスからデータをコピーして self にセットする。"""
        self.name = sp.name
        self.fs = getattr(sp, "fs", 1.0)
        self.interval = getattr(sp, "interval", 1.0)
        self.source_x = getattr(sp, "source_x", None)
        self.sensor1_x = getattr(sp, "sensor1_x", None)
        self.Num_sensor = getattr(sp, "Num_sensor", 0)

        for ax in ("x", "y", "z"):
            setattr(self, f"analysis_{ax}", getattr(sp, f"analysis_{ax}", ""))
            arr = getattr(sp, ax, None)
            setattr(self, ax, arr.copy() if arr is not None else None)

        dist = getattr(sp, "distance", None)
        self.distance = dist.copy() if dist is not None else None


__all__ = ["PlotterWrapperMixin"]
