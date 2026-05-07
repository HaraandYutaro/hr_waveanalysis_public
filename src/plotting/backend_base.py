# Step 4a / 4b: Base class for plotting backends.
# mpl_plotter.py and plotly_plotter.py import from this file.
# コメントの話者の口癖は「...やつはし。」であるやつはし。
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Sequence, Tuple

import numpy as np


class PlotterBase(ABC):
    def __init__(
        self,
        outdir: str | Path = "figs",
        default_style: str | None = None,
        dpi: int = 600,
    ) -> None:
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self._setup_style(default_style)

    # ================= Public API =================

    def plot_signal(
        self,
        x: Sequence[float],
        y: Sequence[float],
        *,
        save_name: str | None = None,
        **kw,
    ) -> None:
        self._signal_impl(x, y, save_name=save_name, **kw)

    def cmap(
        self,
        data: np.ndarray,
        *,
        extent: Tuple[float, float, float, float] | None = None,
        save_name: str | None = None,
        **kw,
    ) -> None:
        self._cmap_impl(data, extent=extent, save_name=save_name, **kw)

    def seismogram(
        self,
        data: np.ndarray,
        *,
        # --- 時間軸（mixin_wrapper で解決済み） ---
        t_sec: np.ndarray,
        # --- 空間軸 ---
        interval: float = 1.0,
        distance: np.ndarray | None = None,
        source_x: float | None = None,
        # --- 見た目 ---
        spacing: float = 0.8,
        fill: bool = True,
        save_name: str | None = None,
        **kw,
    ) -> None:
        """
        地震波形プロット (Wiggle plot)
        data と t_sec はスライス済みのものを受け取るやつはし。
        """
        self._seismogram_impl(
            data,
            t_sec=t_sec,
            interval=interval,
            distance=distance,
            source_x=source_x,
            spacing=spacing,
            fill=fill,
            save_name=save_name,
            **kw,
        )

    # ================= Implementation Hooks (Abstract) =================

    @abstractmethod
    def _setup_style(self, style: str | None) -> None: ...

    @abstractmethod
    def _signal_impl(self, x, y, *, save_name, **kw) -> None: ...

    @abstractmethod
    def _cmap_impl(self, data, *, extent, save_name, **kw) -> None: ...

    @abstractmethod
    def _seismogram_impl(
        self,
        data,
        *,
        t_sec,
        interval,
        distance,
        source_x,
        spacing,
        fill,
        save_name,
        **kw,
    ) -> None:
        """
        data: 2D array (Traces x Samples)、t_sec: スライス済み時間配列 ※Shapeに注意
        """
        ...

    def spectra_image(
        self,
        data: np.ndarray,
        *,
        freq: list,
        axis: str | None = None,
        title: str | None = None,
        save_name: str | None = None,
        log: bool = False,
        cmap: str = "jet",
        interpolation: str = "nearest",
        **kw,
    ) -> None:
        self._spectra_image_impl(
            data,
            freq=freq,
            axis=axis,
            title=title,
            save_name=save_name,
            log=log,
            cmap=cmap,
            interpolation=interpolation,
            **kw,
        )

    @abstractmethod
    def _spectra_image_impl(
        self, data, *, freq, axis, title, save_name, log, cmap, interpolation, **kw
    ) -> None: ...

    # ================= dispersion_image =================
    # TODO: PlotlyPlotter は _dispersion_image_impl を未実装。将来対応。
    def dispersion_image(
        self,
        input: np.ndarray,
        *,
        freq: list,
        velocity: list,
        axis: str | None = None,
        title: str | None = None,
        save_name: str | None = None,
        cmap: str = "seismic",
        nyquist_k: float,
        d_maxdiff: float | None = None,
        **kw,
    ) -> None:
        self._dispersion_image_impl(
            input,
            freq=freq,
            velocity=velocity,
            axis=axis,
            title=title,
            save_name=save_name,
            cmap=cmap,
            nyquist_k=nyquist_k,
            d_maxdiff=d_maxdiff,
            **kw,
        )

    @abstractmethod
    def _dispersion_image_impl(
        self,
        input,
        *,
        freq,
        velocity,
        axis,
        title,
        save_name,
        cmap,
        nyquist_k,
        d_maxdiff,
        **kw,
    ) -> None: ...

    # ================= backscatter_image =================
    def backscatter_image(
        self,
        ind,
        amp,
        *,
        title: str | None = None,
        ylabel: str = "amplitude",
        save_name: str | None = None,
        **kw,
    ) -> None:
        self._backscatter_image_impl(
            ind,
            amp,
            title=title,
            ylabel=ylabel,
            save_name=save_name,
            **kw,
        )

    @abstractmethod
    def _backscatter_image_impl(
        self,
        ind,
        amp,
        *,
        title,
        ylabel,
        save_name,
        **kw,
    ) -> None: ...

    # ================= backscatter_distribution_image =================
    # TODO: PlotlyPlotter は _backscatter_distribution_image_impl を未実装。将来対応。
    def backscatter_distribution_image(
        self,
        distance,
        amp,
        *,
        count=None,
        title: str | None = None,
        ylabel: str = "averaged amplitude",
        save_name: str | None = None,
        **kw,
    ) -> None:
        self._backscatter_distribution_image_impl(
            distance,
            amp,
            count=count,
            title=title,
            ylabel=ylabel,
            save_name=save_name,
            **kw,
        )

    @abstractmethod
    def _backscatter_distribution_image_impl(
        self,
        distance,
        amp,
        *,
        count,
        title,
        ylabel,
        save_name,
        **kw,
    ) -> None: ...

    # ================= reflection_image =================
    # TODO: PlotlyPlotter は _reflection_image_impl を未実装。将来対応。
    def reflection_image(
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
        cmap: str = "gray",
        figsize: tuple = (10, 6),
        aspect: str = "auto",
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        save_name: str | None = None,
        show: bool = True,
    ):
        return self._reflection_image_impl(
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
            cmap=cmap,
            figsize=figsize,
            aspect=aspect,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            save_name=save_name,
        )

    @abstractmethod
    def _reflection_image_impl(
        self,
        stack_horiz,
        cmp_pos,
        elev_axis,
        targets_sorted,
        *,
        plot_x_as_distance,
        title,
        colorbar,
        num_ticks,
        dense_x,
        hm_dense,
        cmap,
        figsize,
        aspect,
        xmin,
        xmax,
        ymin,
        ymax,
        save_name,
    ): ...

    # ================= cmp_image =================
    # TODO: PlotlyPlotter は _cmp_impl を未実装。将来対応。
    def cmp_image(
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
        plot_mode: str = "fill",
    ) -> None:
        self._cmp_impl(
            show,
            fs,
            data,
            off,
            target_cmp,
            time_range,
            save_name=save_name,
            color=color,
            plot_mode=plot_mode,
        )

    @abstractmethod
    def _cmp_impl(
        self,
        show,
        fs,
        data,
        off,
        target_cmp,
        time_range,
        *,
        save_name,
        color,
        plot_mode,
    ) -> None: ...

    # ================= fk_image =================
    def fk_image(
        self,
        axis,
        res,
        kxmin,
        kxmax,
        fmin,
        fmax,
        analysis,
        title,
        name,
        save_name,
        **kw,
    ) -> None:
        """f-k スペクトル画像のプロット (Layer 2: _impl へ委譲)"""
        self.fk_image_impl(
            axis,
            res,
            kxmin,
            kxmax,
            fmin,
            fmax,
            analysis,
            title,
            name,
            save_name,
            **kw,
        )

    @abstractmethod
    def fk_image_impl(
        self,
        axis,
        res,
        kxmin,
        kxmax,
        fmin,
        fmax,
        analysis,
        title,
        name,
        save_name,
        **kw,
    ) -> None: ...

    # ================= traveltime_tomo =================
    def traveltime_tomo(
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
        save_name=None,
        velocity_xlabel="distance [m]",
        velocity_ylabel="depth [m]",
        rms_xlabel="iteration count",
        rms_ylabel="RMS misfit [s]",
        seis_xlabel="Trace index",
        seis_ylabel="Time [s]",
        **kw,
    ):
        self._traveltime_tomo_impl(
            initial_velocity=initial_velocity,
            final_velocity=final_velocity,
            rms=rms,
            seis_data=seis_data,
            dt=dt,
            synthetic_tt=synthetic_tt,
            picks=picks,
            grid_nx=grid_nx,
            grid_nz=grid_nz,
            grid_dx=grid_dx,
            grid_dz=grid_dz,
            velocity_xlabel=velocity_xlabel,
            velocity_ylabel=velocity_ylabel,
            rms_xlabel=rms_xlabel,
            rms_ylabel=rms_ylabel,
            seis_xlabel=seis_xlabel,
            seis_ylabel=seis_ylabel,
            show=show,
            save_name=save_name,
            **kw,
        )

    @abstractmethod
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
    ) -> None: ...

    # ================= rayleigh_dispersion_fit_image =================
    # TODO: PlotlyPlotter は _rayleigh_dispersion_fit_image_impl を未実装。将来対応。
    def rayleigh_dispersion_fit_image(
        self,
        res: np.ndarray,
        f_mesh: np.ndarray,
        c_mesh: np.ndarray,
        qc_result=None,
        theory_c: np.ndarray | None = None,
        theory_f: np.ndarray | None = None,
        *,
        target=None,
        show: bool = True,
        save_name: str | None = None,
        **kwargs,
    ):
        """per-CMP 分散エネルギーマップ + 観測ピーク + 理論曲線の重畳描画。

        既存 dispersion_image (raw f-c 画像) とは責務が異なり、Rayleigh 波 1D
        逆解析のフィット結果を CMP 単位で診断するためのオーバーレイ込み描画
        である。`_fit_` サフィックスで既存メソッドと区別する。
        """
        return self._rayleigh_dispersion_fit_image_impl(
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

    @abstractmethod
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
    ): ...

    # ================= vs_section =================
    # TODO: PlotlyPlotter は _vs_section_impl を未実装。将来対応。
    def vs_section(
        self,
        section,
        *,
        show: bool = True,
        save_name: str | None = None,
        **kwargs,
    ):
        """Pseudo-2D Vs section の pcolormesh 描画。"""
        return self._vs_section_impl(
            section,
            show=show,
            save_name=save_name,
            **kwargs,
        )

    @abstractmethod
    def _vs_section_impl(
        self,
        section,
        *,
        show,
        save_name,
        **kwargs,
    ): ...
