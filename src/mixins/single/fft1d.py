import os

import numpy as np


class FFT1D:
    def fft1d_spectrum(
        self,
        axis: str,
        ch: int,
        t: list[float] = [0, 10],
        freq: list[float] = [1, 200],
        *,
        show: bool = False,
        save_name: str | None = None,
        key: str = "amp",
        backend: str | None = None,
        xscale: str = "log",
        **kw,
    ):
        """
        1 チャンネルの FFT スペクトルを計算し、plotter に描画を委譲する。

        Parameters
        ----------
        axis    : 'x', 'y', 'z'
        ch      : チャンネル番号 (1-indexed)
        t       : [t_min, t_max] 時間範囲 [s]
        f       : [f_min, f_max] 周波数表示範囲 [Hz]
        key     : 'amp' | 'phase' | 'real' | 'imag' | 'real_abs'
        backend : 'mpl' など (None のとき DEFAULT_PLOT_BACKEND)
        xscale  : 'log' or 'linear'

        Returns
        -------
        freq, real, imag, amp, phase : np.ndarray
        """
        _ax, _analysis = self.getax_analysis(axis)
        fs = self.fs
        data = _ax[ch - 1]

        N = len(data)
        t_from = int(max(0.0, t[0]) * fs)
        t_to = int(min(N, t[1] * fs))

        data_win = data[t_from:t_to]
        N_win = len(data_win)
        dt = 1.0 / fs
        t_axis = np.arange(0, N_win * dt, dt)

        _freq = np.linspace(0.0, fs, N_win)
        fft_result = np.fft.fft(data_win)
        real = np.real(fft_result)
        imag = np.imag(fft_result)
        amp = np.abs(fft_result)
        phase = np.angle(fft_result)

        key_map = {
            "amp": (amp, "Amplitude"),
            "phase": (phase, "Phase [rad]"),
            "real": (real, "Real"),
            "imag": (imag, "Imag"),
            "real_abs": (np.abs(real), "|Real|"),
        }
        if key not in key_map:
            raise ValueError(
                f"key が不正です: {key!r} amp, phase, real, imag, real_abs のどれかを入れてください"
            )
        spec, y_label = key_map[key]

        if show or save_name:
            self._ensure_plotter(backend=backend)

            f_min, f_max = float(freq[0]), float(freq[1])
            mask = (_freq >= f_min) & (_freq <= f_max)
            spec_slice = spec[mask]
            if spec_slice.size > 0:
                ymin, ymax = spec_slice.min(), spec_slice.max()
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                f_ylim = (ymin * 0.95, ymax * 1.05)
            else:
                f_ylim = None

            title = kw.pop("title", None)
            if title is None:
                title = f"{getattr(self, 'name', '')} ch{ch} FFT ({key})"

            self.fft1d_image(
                t_axis,
                data_win,
                _freq,
                spec,
                show=show,
                save_name=save_name,
                title=title,
                xscale=xscale,
                time_xlabel="Time [s]",
                time_ylabel="Signal",
                freq_xlabel="Frequency [Hz]",
                freq_ylabel=y_label,
                t_xlim=(t_axis[0], t_axis[-1]),
                f_xlim=(f_min, f_max),
                f_ylim=f_ylim,
                **kw,
            )

        return _freq, real, imag, amp, phase

    def _fft_spectrum_core(self, axis: str, ch: int, t: list, key: str):
        """
        1ch の FFT を計算して (t_axis, data_win, freq, spec) を返す内部ヘルパ。
        描画は行わない。
        """
        _ax, _ = self.getax_analysis(axis)
        fs = self.fs
        data = _ax[ch - 1]

        N = len(data)
        t_from = int(max(0.0, t[0]) * fs)
        t_to = int(min(N, t[1] * fs))
        data_win = data[t_from:t_to]

        N_win = len(data_win)
        dt = 1.0 / fs
        t_axis = np.arange(0, N_win * dt, dt)

        freq = np.linspace(0.0, fs, N_win)
        fft_result = np.fft.fft(data_win)

        key_map = {
            "amp": np.abs(fft_result),
            "phase": np.angle(fft_result),
            "real": np.real(fft_result),
            "imag": np.imag(fft_result),
            "real_abs": np.abs(np.real(fft_result)),
        }
        if key not in key_map:
            raise ValueError(
                f"key が不正です: {key!r} amp, phase, real, imag, real_abs のどれかを入れてください"
            )

        return t_axis, data_win, freq, key_map[key]

    def fft_transfunc(
        self,
        axis: str,
        ch: int,
        source_ch: int | None = None,
        t: list[float] = [0, 10],
        freq: list[float] = [1, 200],
        *,
        show: bool = False,
        save_name: str | None = None,
        title: str | None = None,
        color: str = "blue",
        key: str = "amp",
        xscale: str = "log",
        backend: str | None = None,
        **kw,
    ):
        """
        伝達関数 (対象 ch / 基準 ch) を計算し、plotter に描画を委譲する。

        Parameters
        ----------
        axis      : 'x', 'y', 'z'
        ch        : 対象チャンネル (1-indexed)
        source_ch : 基準チャンネル (None のとき self.get_source_ch() + 1)
        t         : [t_min, t_max] 時間範囲 [s]
        f         : [f_min, f_max] 周波数表示範囲 [Hz]
        key       : 'amp' | 'phase' | 'real' | 'imag' | 'real_abs'
        xscale    : 'log' or 'linear'

        Returns
        -------
        freq : np.ndarray
        div  : np.ndarray  (実数スペクトル比)
        """
        if source_ch is None:
            source_ch = int(self.get_source_ch()) + 1

        _, _, _freq, spec_target = self._fft_spectrum_core(axis, ch, t, key)
        _, _, _freq_s, spec_source = self._fft_spectrum_core(axis, source_ch, t, key)

        if len(_freq) != len(_freq_s):
            raise ValueError("target と source の周波数軸長が一致しません")

        eps = kw.get("eps", 1e-12)
        div = spec_target / (spec_source + eps)

        if show or save_name:
            self._ensure_plotter(backend=backend)

            f_min, f_max = float(freq[0]), float(freq[1])
            mask = (_freq >= f_min) & (_freq <= f_max)
            div_slice = div[mask]
            if div_slice.size > 0:
                ymin = float(np.nanmin(div_slice))
                ymax = float(np.nanmax(div_slice))
                if ymin == ymax:
                    ymin -= 1.0
                    ymax += 1.0
                y_lim = (ymin * 0.95, ymax * 1.05)
            else:
                y_lim = None

            if title is None:
                title = f"{getattr(self, 'name', '')} ch{ch}/ch{source_ch} Transfunc ({key})"

            self.fft_transfunc_image(
                _freq,
                div,
                show=show,
                save_name=save_name,
                title=title,
                color=color,
                xscale=xscale,
                xlabel="Frequency [Hz]",
                ylabel=f"Transfunc {key}",
                xlim=(f_min, f_max),
                ylim=y_lim,
                **kw,
            )

        return _freq, div

    def FFT_transfunc(
        self,
        axis,
        ch,
        source_ch=None,
        t=[0, 10],
        f=[1, 100],
        show=False,
        suptitle=None,
        color="blue",
        key="amp",
        xscale="log",
    ):
        """後方互換ラッパー。新規コードは fft_transfunc() を使うこと。"""
        freq, div = self.fft_transfunc(
            axis,
            ch,
            source_ch=source_ch,
            t=t,
            freq=f,
            show=show,
            save_name=None,
            title=suptitle,
            color=color,
            key=key,
            xscale=xscale,
        )
        return freq, div
