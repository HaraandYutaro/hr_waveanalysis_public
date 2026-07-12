"""TraveltimeTomoPlotter — Step 9 責務シフト適用済みディスパッチ層。

走時トモグラフィ 2x2 描画 (初期速度モデル / 最終速度モデル /
RMS misfit 履歴 / セイモグラム+合成到達時刻) の stable presentation
defaults の canonical owner として位置付ける。
描画ボディ自体は引き続き backend 側に残す。

Step 9 対象外 (risky / 純粋表示既定値ではない):
- panel title / suptitle は描画文脈に依存しうるため **kw 透過とする
- extents, vmin/vmax はデータ依存の物理座標境界
- figsize は Matplotlib 固有
- cmap は rendering 設定
- grid linestyle / alpha / marker size は rendering detail
"""

import numpy as np
from src.plotting.backend_base import PlotterBase

_DEFAULT_VELOCITY_XLABEL = "distance [m]"
_DEFAULT_VELOCITY_YLABEL = "depth [m]"
_DEFAULT_RMS_XLABEL = "iteration count"
_DEFAULT_RMS_YLABEL = "RMS misfit [s]"
_DEFAULT_SEIS_XLABEL = "Trace index"
_DEFAULT_SEIS_YLABEL = "Time [s]"
_DEFAULT_SURFACE_LINE_LABEL = "ground surface"
_DEFAULT_SENSOR_MARKER_LABEL = "receivers"
_DEFAULT_SOURCE_MARKER_LABEL = "source"
_DEFAULT_RAY_LINE_LABEL = "ray paths"


class TraveltimeTomoPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        initial_velocity: np.ndarray,
        final_velocity: np.ndarray,
        rms: np.ndarray,
        seis_data: np.ndarray,
        dt: float,
        synthetic_tt: np.ndarray | None,
        picks: np.ndarray | None,
        grid_nx: int,
        grid_nz: int,
        grid_dx: float,
        grid_dz: float,
        *,
        show: bool,
        save_name: str | None = None,
        velocity_xlabel: str = _DEFAULT_VELOCITY_XLABEL,
        velocity_ylabel: str = _DEFAULT_VELOCITY_YLABEL,
        rms_xlabel: str = _DEFAULT_RMS_XLABEL,
        rms_ylabel: str = _DEFAULT_RMS_YLABEL,
        seis_xlabel: str = _DEFAULT_SEIS_XLABEL,
        seis_ylabel: str = _DEFAULT_SEIS_YLABEL,
        **kw,
    ):
        kw.setdefault("surface_line_label",  _DEFAULT_SURFACE_LINE_LABEL)
        kw.setdefault("sensor_marker_label", _DEFAULT_SENSOR_MARKER_LABEL)
        kw.setdefault("source_marker_label", _DEFAULT_SOURCE_MARKER_LABEL)
        kw.setdefault("ray_line_label",      _DEFAULT_RAY_LINE_LABEL)
        return self._backend._traveltime_tomo_impl(
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