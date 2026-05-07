"""
Group用 分散曲線計算モジュール

SingleProcesser 側の Dispersion mixin (src.mixins.single.dispersion) に概念的に対応。
Single 側は単一トレースデータ (self.axis, self.distance 等) を用いるのに対し、
Group 側は CMP gather 構築済みのデータ (self.cmp, self.offsets 等) を前提とする。

前提属性 (GroupProcesser.__init__ または CmpGathering.cmp_gather で設定):
  - self.cmp      : dict[float, ndarray]  — 各 CMP target → (n_traces, n_samples) 波形
  - self.offsets  : dict[float, ndarray]  — 各 CMP target → (n_traces,) ハーフオフセット
  - self.fs       : float                 — サンプリング周波数
  - self.interval : float                 — レシーバ間隔
  - self.axis     : str                   — 解析軸 ('x', 'y', 'z')
  - self.name     : str (optional)        — データ名 (プロットタイトル用)
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class GroupDispersion(PlotterWrapperMixin):
    """
    Mixin for computing and plotting dispersion curves on CMP-gathered data.

    Corresponds to the single-side Dispersion mixin (src.mixins.single.dispersion),
    but operates on CMP gather dictionaries (self.cmp, self.offsets) rather than
    a single-sensor axis array.

    Design note:
        Single 側 Dispersion は self.axis (2D ndarray), self.distance, self.fs 等を用い、
        method='fft'/'cc'/'nsc' の切替や bandpass filter を持つ多機能クラス。
        Group 側 GroupDispersion は CMP gather (self.cmp, self.offsets) を入力とし、
        method='fft' のみに対応する簡易版。将来の拡張で cc/nsc 法を追加する場合、
        この mixin 内に _calc_cc_core / _calc_nsc_core を追加することが想定される。
    """

    def dispersion_curve(
        self,
        target,
        t=[0, 1.7],
        freq=[1, 200],
        c: list[float] = [1, 500],
        save_name=None,
        df=0.5,
        dc=1.0,
        debug=False,
        rawtrace_normalize=False,
        cut_nyquist=False,
        zeropadding=0,
        title: str | None = None,
        show: bool = True,
        **kw,
    ):
        """
        Compute the dispersion spectrum for a single CMP target using the FFT method.

        Requires :meth:`GroupProcesser.cmp_gathering` to have been called
        first so that ``self.cmp``, ``self.offsets``, ``self.fs``,
        ``self.interval``, and ``self.axis`` are set.

        Parameters
        ----------
        target : float
            CMP midpoint position key in ``self.cmp`` / ``self.offsets``.
        t : list of float, default [0, 1.7]
            Time window [t_start, t_end] in seconds to extract from the CMP
            gather before analysis.
        freq : list of float, default [1, 200]
            Frequency range [fmin, fmax] in Hz.  If *fmax* exceeds the
            Nyquist frequency it is silently clipped.
        c : list of float, default [1, 500]
            Phase-velocity range [cmin, cmax] in m/s.
        save_name : str or None, optional
            File path used to save the figure.  If None, the figure is not
            saved.
        df : float, default 0.5
            Nominal frequency step in Hz.  In practice overridden by
            ``n_samples / fs`` inside :meth:`_initialize_meshgrid`.
        dc : float, default 1.0
            Phase-velocity sampling interval in m/s.
        debug : bool, default False
            If True, display intermediate waveform plots.
        rawtrace_normalize : bool, default False
            If True, normalise each trace to its maximum absolute amplitude
            before analysis.
        cut_nyquist : bool, default False
            If True, zero out spectral components beyond the spatial Nyquist
            wavenumber and below the minimum resolvable wavenumber.
        zeropadding : int, default 0
            Number of zero samples appended to each trace along the time
            axis before the FFT.
        title : str or None, optional
            Plot title.  Defaults to ``"Dispersion curve <name>"`` when
            None.
        show : bool, default True
            If True, display the dispersion image after computation.

        Returns
        -------
        res : numpy.ndarray or None
            Dispersion spectrum with shape ``(len(f_mesh), len(c_mesh))``.
            Returns ``None`` if the CMP gather contains three or fewer
            traces.
        f_mesh : numpy.ndarray or None
            Frequency axis in Hz.  ``None`` on early return.
        c_mesh : numpy.ndarray or None
            Phase-velocity axis in m/s.  ``None`` on early return.
        """

        # tの選択範囲でデータを取得
        cmp, d, axis = self._get_data(target, t)

        # 描画時に必要な引数の取得
        maxdiff = max(d)*2
        num_sensor = cmp.shape[0]
        interval = np.average(np.diff(2*d))

        # cmpのサイズが1の時、失敗メッセージを出しreturn None
        if num_sensor <= 3:
            print(f"cmp.shape:{cmp.shape}. cannot calc dispersion curve in only 1ax.")
            return None, None, None, float("nan"), 0

        # ナイキスト周波数の確認
        freq = self._check_nyquist(freq)

        # 正規化コマンドがTrueの場合、データの正規化
        if rawtrace_normalize:
            cmp = self._normalize_trace(cmp)

        # 必要に応じてゼロパディングして、周波数成分を補正
        if zeropadding != 0:
            cmp = self._zero_padding(cmp, zeropadding)

        # デバック用: 正規化後のデータを表示
        if debug:
            self._show_fordebug(
                cmp, f"Normalized data rawtrace_normalize = {rawtrace_normalize}"
            )

        # Create meshgrid
        f_mesh, c_mesh, res = self._initialize_meshgrid(cmp, freq, c, df, dc)
        _fft_at_fi_normalized = self._compute_fft(cmp, f_mesh)

        # Compute omega and phy
        omega, phy = self._compute_omega_phy(f_mesh, c_mesh)

        # Compute k and nyquist_k
        k, nyquist_k = self._compute_k_nyquist(f_mesh, c_mesh, interval)

        # Compute res
        res = self._calc_res_dc(phy, d, _fft_at_fi_normalized)

        # optionally cut nyquist components:
        if cut_nyquist:
            res = self._cut_nyquist(res, k, nyquist_k, d)

        
        # Plotting and saving
        if show or save_name:
            self._plot_dispersion_curve(
                res,
                f_mesh,
                c_mesh,
                maxdiff,
                num_sensor,
                interval,
                save_name,
                title,
                show,
                **kw,
            )

        # return result array
        return res, f_mesh, c_mesh, interval, num_sensor

    # ------------------------------------------------------------------
    # dispersion_curve の内部メソッド
    # ------------------------------------------------------------------

    def _show_fordebug(self, cmp, suptitle):
        n_d, m_d = cmp.shape
        fig, axes = plt.subplots(1, n_d, figsize=(16, 10), sharex=True)
        fig.suptitle(suptitle)
        Y = np.arange(0, m_d)
        for i in range(n_d):
            axes[i].fill_betweenx(
                Y / self.fs, 0, cmp[i, Y], color="black", label=str(i + 1)
            )
            max_x = np.max(np.abs(cmp[i]))
            axes[i].set_xlim([-1.1 * max_x, +1.1 * max_x])
            axes[i].set_ylim(m_d / self.fs, 0)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            if i == 0:
                axes[i].set_ylabel("time [s]")
        plt.show()

    def _get_data(self, target, t):
        """
        Extract and time-clip the CMP gather for a given target.

        Parameters
        ----------
        target : float
            CMP midpoint position key in ``self.cmp`` / ``self.offsets``.
        t : list of float
            Time window [t_start, t_end] in seconds.  Converted to sample
            indices via ``self.fs``; indices are clamped to valid bounds.

        Returns
        -------
        cmp : numpy.ndarray
            Waveform array of shape ``(n_traces, n_samples)`` clipped to
            the requested time window.
        d : numpy.ndarray
            Half-offset array of shape ``(n_traces,)`` from
            ``self.offsets[target]``.
        axis : str
            Analysis axis string from ``self.axis``.
        """
        cmp = self.cmp[target]
        d = self.offsets[target]
        axis = self.axis

        # tの値でaxデータを切り取る（秒 → サンプル数変換は self.fs を使用）
        cmp = cmp[
            :, max(int(t[0] * self.fs), 0) : min(cmp.shape[1], int(t[1] * self.fs))
        ]
        return cmp, d, axis

    def _check_nyquist(self, freq):
        # fmaxがナイキスト周波数より大きい場合、修正する
        if 2.0 * freq[1] > self.fs / 2:
            freq[1] = int(self.fs / 2.0 * 100) / 100
            print(
                "fmax is over Nyquist frequency. fmax is set to Nyquist frequency:"
                + str(freq[1])
            )
        return freq

    @staticmethod
    def _normalize_trace(cmp):
        temp = np.zeros_like(cmp)
        for i in range(cmp.shape[0]):
            temp[i] = np.float64(cmp[i] / np.max(np.abs(cmp[i])))
        return temp

    @staticmethod
    def _zero_padding(cmp, zeropadding):
        """ゼロパディングして、fft時の周波数分解能を向上させる"""
        padding = np.zeros((len(cmp), zeropadding))
        return np.concatenate([cmp, padding], axis=1)

    def _initialize_meshgrid(self, cmp, freq, c, df, dc):
        fmin, fmax = freq[0], freq[1]
        cmin, cmax = c[0], c[1]

        # FFT 周波数分解能 Δf = fs / n_samples = 1 / T [Hz]
        df = self.fs / cmp.shape[1]
        f_mesh = np.arange(fmin, fmax, df)
        c_mesh = np.arange(cmin, cmax, dc)
        res = np.zeros((len(f_mesh), len(c_mesh)))
        return f_mesh, c_mesh, res

    def _compute_fft(self, cmp, f_mesh):
        _fft = np.fft.fft(cmp, axis=1)
        N = cmp.shape[1]
        dt = 1 / self.fs
        freqs = np.fft.fftfreq(N, d=dt)

        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        _fft_pos = _fft[:, pos_mask]

        fi_indices = np.array([np.argmin(np.abs(pos_freqs - fi)) for fi in f_mesh])
        _fft_at_fi = _fft_pos[:, fi_indices].T
        _fft_at_fi_normalized = _fft_at_fi / np.abs(_fft_at_fi)
        return _fft_at_fi_normalized

    @staticmethod
    def _compute_omega_phy(f_mesh, c_mesh):
        omega = 2 * np.pi * f_mesh
        phy = omega[:, np.newaxis] / c_mesh[np.newaxis, :]
        return omega, phy

    def _compute_k_nyquist(self, f_mesh, c_mesh, interval):
        k = f_mesh[:, np.newaxis] / c_mesh[np.newaxis, :]
        nyquist_k = 1 / (2 * interval)
        return k, nyquist_k

    @staticmethod
    def _calc_res_dc(phy, d, fft_at_fi_normalized):
        exp_term = np.exp(1j * phy[:, :, np.newaxis] * d[np.newaxis, np.newaxis, :])
        product = exp_term * fft_at_fi_normalized[:, np.newaxis, :]
        res = np.abs(np.sum(product, axis=2))
        return res

    @staticmethod
    def _cut_nyquist(res, k, nyquist_k, d):
        """
        ナイキスト波数外・空間波長以内の成分をゼロ埋めする。

        NOTE: 元の CmpGathering 内部関数では res, k, nyquist_k を
        クロージャーから参照していたが、明示的な引数として受け取るよう修正。
        """
        d_maxdiff = np.max(d) - np.min(d)
        res[np.abs(k) >= nyquist_k] = 0
        res[np.abs(k) <= 1 / (2 * d_maxdiff)] = 0
        return res

    def _plot_dispersion_curve(self, res, f_mesh, c_mesh, maxdiff, num_sensor, interval, save_name, title, show, **kw):
        """
        分散曲線を描画する。3層構造 (wrapper → base → impl) に乗せるため、
        PlotterWrapperMixin.dispersion_image() に委譲する。

        source_x / sensor1_x は定義できない。d_maxdiff は 最大オフセットとする。
        nyquist / *max-wavelength 補助線を正確に描くには別途設定が必要
        """
        self._ensure_plotter()

        if title is None:
            title = (
                f"Dispersion curve {self.name}"
                if hasattr(self, "name")
                else "Dispersion curve"
            )

        self.dispersion_image(
            input=res,
            ax_x=[float(f_mesh[0]), float(f_mesh[-1])],
            ax_y=[float(c_mesh[0]), float(c_mesh[-1])],
            axis = self.axis,
            show=show,
            interval=interval,
            Num_sensor=num_sensor,
            source_x=None,
            sensor1_x=None,
            d_maxdiff=maxdiff,
            mode="frequency-velocity",
            title=title,
            save_name=save_name,
            **kw,
        )

if __name__ == "__main__":
    pass
