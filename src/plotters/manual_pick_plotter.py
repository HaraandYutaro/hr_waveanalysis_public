"""ManualPickPlotter — 手動初動ピッキング GUI のディスパッチ層。

走時トモグラフィの ``picking_mode="manual"`` で手動 pick 引数が一切与えられ
なかった場合に起動する、対話的初動ピッキング GUI の stable presentation
defaults (軸ラベル / マーカー線種・色) の canonical owner として位置付ける。
GUI 本体 (figure / canvas / widget / event handling) は backend 側
(``_manual_pick_impl``) に委譲する。

他の plotter (TraveltimeTomoPlotter 等) と同じく wrapper → plotter → backend
の 3 層分離に従う。GUI 実装詳細はここには持ち込まない。
"""

import numpy as np

from src.plotting.backend_base import PlotterBase

_DEFAULT_TIME_YLABEL = "Time [s]"
_DEFAULT_AMP_XLABEL = "Amplitude"
_DEFAULT_MARKER_LINESTYLE = ":"
_DEFAULT_MARKER_COLOR = "red"


class ManualPickPlotter:
    def __init__(self, backend: PlotterBase) -> None:
        self._backend = backend

    def pick(
        self,
        traces: np.ndarray,
        t_sec: np.ndarray,
        *,
        distance=None,
        time_ylabel: str = _DEFAULT_TIME_YLABEL,
        amp_xlabel: str = _DEFAULT_AMP_XLABEL,
        marker_linestyle: str = _DEFAULT_MARKER_LINESTYLE,
        marker_color: str = _DEFAULT_MARKER_COLOR,
        **kw,
    ) -> np.ndarray:
        """
        手動初動ピッキング GUI を起動し、各 trace の初動時刻 [s] を返す。

        Parameters
        ----------
        traces : ndarray, shape=(n_traces, n_samples)
            ピッキング対象の波形 (trace 順は呼び出し側の index と一致)。
        t_sec : ndarray, shape=(n_samples,)
            時間軸 [s]。
        distance : array-like, optional
            各 trace のラベル表示用 (記録される pick index には影響しない)。

        Returns
        -------
        pick_times : ndarray, shape=(n_traces,), float [s]
        """
        return self._backend._manual_pick_impl(
            traces=traces,
            t_sec=t_sec,
            distance=distance,
            time_ylabel=time_ylabel,
            amp_xlabel=amp_xlabel,
            marker_linestyle=marker_linestyle,
            marker_color=marker_color,
            **kw,
        )
