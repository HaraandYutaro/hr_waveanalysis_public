"""
Fft1dPlotter — Step 3 責務シフト適用済みディスパッチ層。

Step 2 ではこのクラスは backend への純粋な pass-through だった。
Step 3 ではこのクラスを「FFT 1D プロットのプレゼンテーション既定値の
canonical owner」と位置付け、title / xscale / time_xlabel / time_ylabel /
freq_xlabel / freq_ylabel など見た目に関する既定値の宣言をここで一元化する。
描画ボディ自体は引き続き backend 側に残す。

互換性方針:
- 公開シグネチャ・既定値は据え置き
- wrapper 側の既定値・互換ヘルパ呼び出し順序にも触れない
- backend `_fft1d_image_impl` の本体は動かさない
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_TITLE = None
_DEFAULT_XSCALE = "log"
_DEFAULT_TIME_XLABEL = "Time [s]"
_DEFAULT_TIME_YLABEL = "Signal"
_DEFAULT_FREQ_XLABEL = "Frequency [Hz]"
_DEFAULT_FREQ_YLABEL = "Amplitude"


class Fft1dPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def image(
        self,
        t: np.ndarray,
        signal: np.ndarray,
        freq: np.ndarray,
        spec: np.ndarray,
        *,
        show: bool,
        save_name: str | None = None,
        title: str | None = _DEFAULT_TITLE,
        xscale: str = _DEFAULT_XSCALE,
        time_xlabel: str = _DEFAULT_TIME_XLABEL,
        time_ylabel: str = _DEFAULT_TIME_YLABEL,
        freq_xlabel: str = _DEFAULT_FREQ_XLABEL,
        freq_ylabel: str = _DEFAULT_FREQ_YLABEL,
        **kw,
    ):
        return self._backend._fft1d_image_impl(
            t,
            signal,
            freq,
            spec,
            show=show,
            save_name=save_name,
            title=title,
            xscale=xscale,
            time_xlabel=time_xlabel,
            time_ylabel=time_ylabel,
            freq_xlabel=freq_xlabel,
            freq_ylabel=freq_ylabel,
            **kw,
        )