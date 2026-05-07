import numpy as np
from scipy import signal, special

from src.plotting.wrapper import PlotterWrapperMixin


class Dispersion(PlotterWrapperMixin):
    def dispersion_curve(
        self,
        axis,
        freq: list[float] = [1, 200],
        c: list[float] = [1, 500],
        df=0.5,
        dc=1.0,
        method="fft",
        sigma_forgraph=0.01,
        bandpass_method="sosfiltfilt",
        eps=1e-10,
        debug=False,
        rawtrace_regularize=False,
        filter_resularize="all",
        band_ratio=2,
        cut_nyquist=False,
        zeropadding=0,
        show: bool = True,
        save_name: str | None = None,
        mode: str = "frequency-velocity",
        wavelength: list | None = None,
        dramda: float = 0.001,
        backend: str | None = "mpl",
        ch: list[int] | None = None,
        **kw,
    ):
        """
        Compute the dispersion spectrum and optionally display or save it.

        Parameters
        ----------
        axis : {"x", "y", "z"}
            Analysis axis.
        freq : list of float, default [1, 200]
            Frequency range [fmin, fmax] in Hz.  If *fmax* exceeds the
            Nyquist frequency it is silently clipped.
        c : list of float, default [1, 500]
            Phase-velocity range [cmin, cmax] in m/s.  In
            ``"wavelength-velocity"`` mode this is also used as the
            velocity axis.
        df : float, default 0.5
            Frequency sampling interval in Hz.  Ignored when
            ``method="fft"``; in that case the step is determined by
            ``n_samples / fs``.
        dc : float, default 1.0
            Phase-velocity sampling interval in m/s.
        method : {"fft", "cc", "nsc"}, default "fft"
            Dispersion analysis method.  Only applies when
            ``mode="frequency-velocity"``.
        sigma_forgraph : float, default 0.01
            Width parameter used in the ``"nsc"`` method.
        bandpass_method : {"sosfiltfilt", "filtfilt", "fft", "high_low", "low_high"}, default "sosfiltfilt"
            Bandpass filter implementation used by ``"cc"`` and ``"nsc"``
            methods.
        eps : float, default 1e-10
            Small constant added for numerical stability in normalisation.
        debug : bool, default False
            If True, display intermediate waveform plots at several
            processing stages.
        rawtrace_regularize : bool, default False
            If True, normalise each trace to its maximum absolute amplitude
            before analysis.
        filter_resularize : {"all", "each", "none"}, default "all"
            Amplitude normalisation applied after bandpass filtering in
            ``"cc"`` and ``"nsc"`` methods.
        band_ratio : float, default 2
            Ratio defining the bandpass width around each target frequency
            for ``"cc"`` and ``"nsc"`` methods: band = [f/band_ratio,
            f*band_ratio].
        cut_nyquist : bool, default False
            If True, zero out spectral components beyond the spatial Nyquist
            wavenumber and below the minimum resolvable wavenumber.
        zeropadding : int, default 0
            Number of zero samples appended to each trace along the time
            axis before the FFT to improve frequency interpolation.
        show : bool, default True
            If True, display the dispersion image after computation.
        save_name : str or None, optional
            File path used to save the figure.  If None or empty, the
            figure is not saved.
        mode : {"frequency-velocity", "wavelength-velocity", "frequency-wavelength"}, default "frequency-velocity"
            Output domain of the dispersion spectrum.
        wavelength : list of float or None, optional
            Wavelength range [kmin, kmax] in metres for wavelength-domain
            modes.  If None, kmin is set to the spatial Nyquist wavelength
            (2 × interval) and kmax to half the maximum sensor separation.
        dramda : float, default 0.001
            Wavelength sampling interval in metres.  Used only in
            wavelength-domain modes.
        backend : str or None, default "mpl"
            Plotting backend passed to the plotter wrapper.
        ch : list of int or None, optional
            Indices of channels to use.  Must contain at least 3 elements.
            If None, all channels are used.

        Returns
        -------
        res : numpy.ndarray
            Computed dispersion spectrum with shape
            ``(len(axis_x), len(axis_y))``.
        axis_x : numpy.ndarray
            First axis of the spectrum (frequency [Hz] in
            ``"frequency-velocity"`` and ``"frequency-wavelength"`` modes;
            phase velocity [m/s] in ``"wavelength-velocity"`` mode).
        axis_y : numpy.ndarray
            Second axis of the spectrum (phase velocity [m/s] in
            ``"frequency-velocity"`` and ``"wavelength-velocity"`` modes;
            wavelength [m] in ``"frequency-wavelength"`` and
            ``"wavelength-velocity"`` modes).

        Raises
        ------
        ValueError
            If *mode* is not one of the accepted strings.
        ValueError
            If *method* is not ``"fft"``, ``"cc"``, or ``"nsc"`` when
            ``mode="frequency-velocity"``.
        ValueError
            If *ch* contains fewer than 3 indices.
        """
        VALID_MODES = (
            "frequency-velocity",
            "frequency-wavelength",
            "wavelength-velocity",
        )
        if mode not in VALID_MODES:
            raise ValueError(
                f"mode='{mode}' は無効です。"
                f"{VALID_MODES} のいずれかを指定してください。"
            )

        # 1. 前処理 -------------------------------------------------------

        # ナイキスト周波数チェック
        if 2.0 * freq[1] > self.fs / 2:
            freq[1] = int(self.fs / 2.0 * 100) / 100
            print(
                "fmax is over Nyquist frequency. fmax is set to Nyquist frequency:"
                + str(freq[1])
            )

        # axの設定
        _ax, _analysis = self.getax_analysis(axis)
        num_sensor = self.Num_sensor
        interval = self.interval
        sensor1_x = min(self.distance)
        source_x = self.source_x

        # distance arrayの設定
        if hasattr(self, "distance"):
            d = self.distance
        else:
            d = np.zeros(num_sensor)
            for i in range(num_sensor):
                d[i] = np.abs(sensor1_x + i * interval - source_x)
        d_maxdiff = np.max(d) - np.min(d)

        # 使用するchが指定されているときは、axとdを切り出す
        if ch is not None:
            if len(ch) < 3:
                raise ValueError(
                    "chの数が少ないへち。。。設定するなら3つ以上のchを設定してやつはし。。。"
                )
            _ax = _ax[ch]
            d = d[ch]
            num_sensor = len(ch)
            interval = np.diff(d).mean()
            sensor1_x = min(d)
            source_x = source_x
            d_maxdiff = np.max(d) - np.min(d)

        if rawtrace_regularize:
            temp = np.zeros_like(_ax)
            for i in range(_ax.shape[0]):
                temp[i] = np.float64(_ax[i] / np.max(np.abs(_ax[i])))
            _ax = temp

        if zeropadding != 0:
            _ax = np.concatenate([_ax, np.zeros((len(_ax), zeropadding))], axis=1)

        if debug:
            self._show_fordebug(
                _ax, f"Normalized data rawtrace_regularize = {rawtrace_regularize}"
            )

        # 2. ディスパッチ --------------------------------------------------

        if mode == "frequency-velocity":
            fmin, fmax_val = freq[0], freq[1]
            cmin, cmax_val = c[0], c[1]
            df_actual = _ax.shape[1] / self.fs if method == "fft" else df
            f_mesh = np.arange(fmin, fmax_val, df_actual)
            c_mesh = np.arange(cmin, cmax_val, dc)

            if method == "fft":
                res = self._calc_fft_core(
                    _ax, f_mesh, c_mesh, d, d_maxdiff, cut_nyquist
                )
            elif method == "cc":
                res = self._calc_cc_core(
                    _ax,
                    f_mesh,
                    c_mesh,
                    d,
                    band_ratio=band_ratio,
                    bandpass_method=bandpass_method,
                    freq=freq,
                    filter_resularize=filter_resularize,
                    debug=debug,
                    eps=eps,
                )
            elif method == "nsc":
                res = self._calc_nsc_core(
                    _ax,
                    f_mesh,
                    c_mesh,
                    d,
                    band_ratio=band_ratio,
                    bandpass_method=bandpass_method,
                    freq=freq,
                    filter_resularize=filter_resularize,
                    sigma_forgraph=sigma_forgraph,
                    debug=debug,
                    eps=eps,
                )
            else:
                raise ValueError("dispersion method, set method = fft or cc or nsc")

            axis_x = f_mesh
            axis_y = c_mesh

        else:
            # wavelength 系モード共通の波長範囲解決
            nyquist_wl = 2 * interval
            _wl = wavelength if wavelength is not None else [0, None]
            kmin = max(_wl[0], nyquist_wl)
            kmax = _wl[1] if _wl[1] is not None else d_maxdiff / 2

            if mode == "wavelength-velocity":
                res, axis_x, axis_y = self._calc_wavelength_velocity_core(
                    _ax, d, kmin, kmax, c, dramda
                )  # 速度, 波長でresが出力

            elif mode == "frequency-wavelength":
                res, axis_x, axis_y = self._calc_frequency_wavelength_core(
                    _ax, d, kmin, kmax, freq, dramda
                )  # 周波数,波長でresが出力

        # 3. 共通後処理（描画） --------------------------------------------

        if show or save_name:
            self._ensure_plotter(backend=backend)
            self.dispersion_image(
                res,
                ax_x=[float(axis_x[0]), float(axis_x[-1])],
                ax_y=[float(axis_y[0]), float(axis_y[-1])],
                axis=axis,
                interval=interval,
                Num_sensor=num_sensor,
                source_x=source_x,
                sensor1_x=sensor1_x,
                save_name=save_name,
                d_maxdiff=d_maxdiff,
                show=show,
                mode=mode,
                **kw,
            )

        return res, axis_x, axis_y

    def get_negative_phasevel_amp(
        self,
        axis,
        N=6,
        freq: list[float] = [1, 200],
        pick_c_range=[20, 120],
        zeropadding=0,
    ):
        """
        Get negative phase velocity amplitude by dispersion curve analysis
        return: ind:list, amp:list ->  各センサーの負の位相速度成分の振幅リスト
        """
        ax, analysis = self.getax_analysis(axis)

        negative_c_amp = []

        for i in range(N, len(ax), 1):
            ## Dispersion curve analysis
            ax_i = ax[i - N : i]  # 最新のNチャンネル分を取得
            d = np.zeros(self.Num_sensor)  # ch(i+1)の起振点からの距離を出す。
            d = (
                self.distance[i - N : i] - self.source_x
                if self.distance is not None
                else [
                    np.abs(self.sensor1_x + j * self.interval - self.source_x)
                    for j in range(i - N, i)
                ]
            )
            d_maxdiff = np.max(d) - np.min(d)
            nyquist_wavelength = 2 * self.interval  # ナイキスト波長[m]
            nyquist_time = 2 * (1.0 / self.interval)  # ナイキスト時間間隔[s]
            kmin, kmax = nyquist_wavelength, d_maxdiff / 2
            fmin, fmax = min(freq), max(freq)

            if zeropadding > 0:
                padding = np.zeros((len(ax_i), zeropadding))
                ax_i = np.concatenate([ax_i, padding], axis=1)

            df = ax_i.shape[1] / self.fs
            dramda = 1 / (d_maxdiff)

            # Create mesh grid for the dispersion curve
            c_mesh = np.arange(kmax, kmin, -dramda)  # Wavenumber [1/m]
            f_mesh = np.arange(fmin, fmax, df)  # Frequency [Hz]
            res = np.zeros((len(c_mesh), len(f_mesh)))

            # pick up dispersion image
            _fft_i = np.fft.fft(ax_i, axis=1)
            Mi, Ni = ax_i.shape  # Compute frequencies corresponding to the FFT bins
            dt = 1 / self.fs  # Replace with your actual sampling interval
            freqs = np.fft.fftfreq(Ni, d=dt)

            pos_mask = freqs >= 0  # Keep only the positive frequencies
            pos_freqs = freqs[pos_mask]
            _fft_pos = _fft_i[:, pos_mask]

            fii_indices = np.array(
                [np.argmin(np.abs(pos_freqs - fii)) for fii in f_mesh]
            )  # Find the indices of frequencies in f_mesh within pos_freqs
            _fft_at_fii = _fft_pos[:, fii_indices].T  # Shape: (len(f_mesh), len(_ax))
            _fft_at_fii_normalized = _fft_at_fii / np.abs(_fft_at_fii)

            # Compute omega and phy
            phy = (
                np.ones((len(f_mesh), len(c_mesh))) / c_mesh[np.newaxis, :]
            )  # Shape: (len(f_mesh), len(c_mesh))

            # Compute the exponential term
            exp_term = np.exp(
                1j * phy[:, :, np.newaxis] * d[np.newaxis, np.newaxis, :]
            )  # Shape: (len(f_mesh), len(r_mesh), len(_ax))

            # Multiply and sum over k to get res
            product = (
                exp_term * _fft_at_fii_normalized[:, np.newaxis, :]
            )  # Shape: (len(f_mesh), len(r_mesh), len(_ax))

            # where k > nyquist_k, res =  np.real(np.sum(product, axis=2)), else res = 0
            res = np.abs(np.sum(product, axis=2))

            # pickup negative phase velocity
            neg_col_idx = np.where(
                (c_mesh < -pick_c_range[0]) & (c_mesh > -pick_c_range[1])
            )[0]  # 1-D 配列（列番号）

            # その列だけ抜き出した res
            res_neg = res[:, neg_col_idx]
            amp_neg_c = np.abs(np.sum(res_neg)) / N

            # store amp
            negative_c_amp.append(amp_neg_c)

        ind = np.arange(N, len(ax), 1)

        # return negative phase velocity amplitude list
        return ind, negative_c_amp

    def get_reflectioin_amplitude(self, axis, N=6):
        """
        f-k スペクトルから反射波の振幅を取得する
        return
                amplitude: list -> 各センサーの反射波成分の振幅リスト
        """
        ax, analysis = self.getax_analysis(axis)

        refletction_amp = []

        for i in range(N, len(ax), 1):
            reflection_amp_i = 0
            ax_i = ax[i - N : i]  # 最新のNチャンネル分を取

            distance_from_source_i = (
                self.distance[i - N : i] - self.source_x
                if self.distance is not None
                else np.array(
                    [
                        np.abs(self.sensor1_x + j * self.interval - self.source_x)
                        for j in range(i - N, i)
                    ]
                )
            )

            positive_d = distance_from_source_i[distance_from_source_i > 0]
            negative_d = distance_from_source_i[distance_from_source_i < 0]

            ax_i_positive = (
                ax_i[distance_from_source_i > 0]
                if len(positive_d) > 0
                else np.array([[]])
            )
            ax_i_negative = (
                ax_i[distance_from_source_i < 0]
                if len(negative_d) > 0
                else np.array([[]])
            )

            positive_d = np.abs(positive_d)
            negative_d = np.abs(negative_d)

            if len(positive_d) > 1:
                sort_positive_indices = np.argsort(positive_d)
                positive_d = positive_d[sort_positive_indices]
                ax_i_positive = ax_i_positive[sort_positive_indices]

                F2d_pos = np.fft.fftshift(np.fft.fft2(ax_i_positive))
                f_pos = np.fft.fftshift(
                    np.fft.fftfreq(ax_i_positive.shape[1], d=1 / self.fs)
                )
                k_pos = np.fft.fftshift(
                    np.fft.fftfreq(ax_i_positive.shape[0], d=self.interval)
                )
                k_pos = -k_pos  # 正の距離なので、wavenumberは負にする

                # 周波数成分の正の部分のみ抽出
                mask_f_pos = f_pos > 0  # shape: (len(f_pos))
                F2d_pos = (
                    2 * F2d_pos[:, mask_f_pos]
                )  # shape: (len(k_pos), len(f_pos_positive))
                f_pos = f_pos[mask_f_pos]  # shape: (len(f_pos_positive))

                neg_col_idx = np.where(k_pos < 0)[0]  # 1-D 配列（列番号）
                F2d_pos_neg = F2d_pos[
                    :, neg_col_idx
                ]  # shape: (len(neg_col_idx), len(f_pos_positive))
                reflection_amp_i += np.sum(np.abs(F2d_pos_neg))
            else:
                pass

            if len(negative_d) > 1:
                sort_negative_indices = np.argsort(negative_d)
                negative_d = negative_d[sort_negative_indices]
                ax_i_negative = ax_i_negative[sort_negative_indices]

                F2d_neg = np.fft.fftshift(np.fft.fft2(ax_i_negative))
                f_neg = np.fft.fftshift(
                    np.fft.fftfreq(ax_i_negative.shape[1], d=1 / self.fs)
                )
                k_neg = np.fft.fftshift(
                    np.fft.fftfreq(ax_i_negative.shape[0], d=self.interval)
                )
                k_neg = -k_neg  # 負の距離なので、wavenumberは正にする

                # 周波数成分の正の部分のみ抽出
                mask_f_neg = f_neg > 0  # shape: (len(f_neg))
                F2d_neg = (
                    2 * F2d_neg[:, mask_f_neg]
                )  # shape: (len(k_neg), len(f_neg_positive))
                f_neg = f_neg[mask_f_neg]  # shape: (len(f_neg_positive))

                pos_col_idx = np.where(k_neg > 0)[0]  # 1-D 配列（列番号）
                F2d_neg_pos = F2d_neg[
                    :, pos_col_idx
                ]  # shape: (len(pos_col_idx), len(f_neg_positive))
                reflection_amp_i += np.sum(np.abs(F2d_neg_pos))
            else:
                pass

            # store amp
            refletction_amp.append(reflection_amp_i / N)

        ind = np.arange(N, len(ax), 1)

        # return reflection amplitude list
        return ind, refletction_amp

    def _calc_wavelength_velocity_core(self, _ax, d, kmin, kmax, c, dramda):
        """
        位相速度-波長 分散曲線の計算コア（波長方向ループ、速度方向ベクトル化）

        Parameters
        ----------
        _ax : (n_traces, n_samples) ゼロパディング済みデータ
        d   : (n_traces,) 起振点からの距離 [m]
        kmin : 最小波長 [m]
        kmax : 最大波長 [m]
        c   : [vmin, vmax] 速度範囲 [m/s]
        dramda : 波長刻み [m]

        Returns
        -------
        res   : (len(v_mesh), len(r_mesh))
        v_mesh : 速度軸
        r_mesh : 波長軸（kmax → kmin の降順）
        """
        vmin, vmax = c[0], c[1]
        df_val = _ax.shape[1] / self.fs

        r_mesh = np.arange(kmax, kmin, -dramda)  # wavelength [m]
        v_mesh = np.arange(vmin, vmax, df_val)  # phase velocity [m/s]
        n_v = len(v_mesh)
        n_r = len(r_mesh)

        res = np.zeros((n_v, n_r), dtype=float)

        # FFT と正の周波数部分
        _fft = np.fft.fft(_ax, axis=1)
        N = _ax.shape[1]
        freqs = np.fft.fftfreq(N, d=1 / self.fs)

        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        _fft_pos = _fft[:, pos_mask]  # (n_traces, n_pos_freqs)

        # trace 方向の距離を (1, n_traces) に揃えておく
        d_tr = d[np.newaxis, :]  # (1, n_traces)

        for j, lam in enumerate(r_mesh):
            # この波長 λ に対応する各速度での周波数 f = v / λ
            f_vec = v_mesh / lam  # (n_v,)

            # 各 f に最も近い FFT ビンを一括で選ぶ
            fi_idx = np.array(
                [np.argmin(np.abs(pos_freqs - fi)) for fi in f_vec]
            )  # (n_v,)

            # (n_v, n_traces) のスペクトル行列
            _fft_at_fi = _fft_pos[:, fi_idx].T  # (n_v, n_traces)
            _fft_at_fi_normed = _fft_at_fi / np.abs(_fft_at_fi)

            # Park 法と同じ位相項 ω/c = 2πf/v = 2π/λ → 周波数に依存せず λ のみ
            phase_slowness = (2 * np.pi) / lam  # スカラー
            exp_term = np.exp(1j * phase_slowness * d_tr)  # (1, n_traces)

            # 各 v について trace 方向に和
            product = _fft_at_fi_normed * exp_term  # (n_v, n_traces)
            res[:, j] = np.abs(np.sum(product, axis=1))

        return res, v_mesh, r_mesh

    def _calc_frequency_wavelength_core(self, _ax, d, kmin, kmax, freq, dramda):
        """
        周波数-波長 分散曲線の計算コア (Park 型の位相項に合わせた版)

        Parameters
        ----------
        _ax : (n_traces, n_samples) ゼロパディング済みデータ
        d   : (n_traces,) 起振点からの距離 [m]
        kmin : 最小波長 [m]
        kmax : 最大波長 [m]
        freq : [fmin, fmax] 周波数範囲 [Hz]
        dramda : 波長刻み [m]

        Returns
        -------
        res   : (len(f_mesh), len(r_mesh))
        f_mesh : 周波数軸
        r_mesh : 波長軸（kmax → kmin の降順）
        """
        fmin, fmax = freq[0], freq[1]
        df_val = _ax.shape[1] / self.fs

        r_mesh = np.arange(kmax, kmin, -dramda)  # wavelength [m]
        f_mesh = np.arange(fmin, fmax, df_val)  # frequency [Hz]

        _fft = np.fft.fft(_ax, axis=1)
        N = _ax.shape[1]
        freqs = np.fft.fftfreq(N, d=1 / self.fs)

        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        _fft_pos = _fft[:, pos_mask]  # (n_traces, n_pos_freqs)

        # f_mesh に対応する周波数ビンを抽出（Park と同じ）
        fi_indices = np.array([np.argmin(np.abs(pos_freqs - fi)) for fi in f_mesh])
        _fft_at_fi = _fft_pos[:, fi_indices].T  # (len(f_mesh), n_traces)
        _fft_at_fi_normed = _fft_at_fi / np.abs(_fft_at_fi)

        # 位相項: 2π / λ （Park の ω/c = 2π/λ に対応）
        phase_slowness = (2 * np.pi) / r_mesh  # (len(r_mesh),)
        phy = phase_slowness[np.newaxis, :]  # (len(f_mesh), len(r_mesh)) と同次元扱い
        exp_term = np.exp(1j * phy[:, :, np.newaxis] * d[np.newaxis, np.newaxis, :])
        product = exp_term * _fft_at_fi_normed[:, np.newaxis, :]
        res = np.abs(np.sum(product, axis=2))  # (len(f_mesh), len(r_mesh))

        return res, f_mesh, r_mesh

    # -----------------------------------------------------------------
    # method='fft' 実装
    # -----------------------------------------------------------------

    def _calc_fft_core(self, _ax, f_mesh, c_mesh, d, d_maxdiff, cut_nyquist):
        """FFT法による分散曲線計算 (Park et al.)"""
        _fft = np.fft.fft(_ax, axis=1)
        N = _ax.shape[1]
        freqs = np.fft.fftfreq(N, d=1 / self.fs)

        pos_mask = freqs >= 0
        pos_freqs = freqs[pos_mask]
        _fft_pos = _fft[:, pos_mask]

        fi_indices = np.array([np.argmin(np.abs(pos_freqs - fi)) for fi in f_mesh])
        _fft_at_fi = _fft_pos[:, fi_indices].T  # (len(f_mesh), n_traces)
        _fft_at_fi_normed = _fft_at_fi / np.abs(_fft_at_fi)

        omega = 2 * np.pi * f_mesh  # (len(f_mesh),)
        phy = omega[:, np.newaxis] / c_mesh[np.newaxis, :]  # (len(f_mesh), len(c_mesh))
        k = f_mesh[:, np.newaxis] / c_mesh[np.newaxis, :]  # (len(f_mesh), len(c_mesh))
        nyquist_k = 1 / (2 * self.interval)

        exp_term = np.exp(1j * phy[:, :, np.newaxis] * d[np.newaxis, np.newaxis, :])
        product = exp_term * _fft_at_fi_normed[:, np.newaxis, :]
        res = np.abs(np.sum(product, axis=2))

        if cut_nyquist:
            res[np.abs(k) >= nyquist_k] = 0
            res[np.abs(k) <= 1 / (2 * d_maxdiff)] = 0

        return res

    # -----------------------------------------------------------------
    # method='cc' 実装
    # -----------------------------------------------------------------

    def _calc_cc_core(
        self,
        _ax,
        f_mesh,
        c_mesh,
        d,
        *,
        band_ratio,
        bandpass_method,
        freq,
        filter_resularize,
        debug,
        eps,
    ):
        """相互相関法 (cc) による分散曲線計算"""
        res = np.zeros((len(f_mesh), len(c_mesh)))

        for i, fi in enumerate(f_mesh):
            if i % 10 == 0 and i != 0:
                print("freq..." + str(fi))
            fpass = np.array([fi / band_ratio, fi * band_ratio])
            fstop = np.array([0.5 * fpass[0], 2.0 * fpass[1]])
            filtered_fi = self._bandpass_filter(
                _ax,
                fpass,
                fstop,
                gpass=3,
                gstop=40,
                bandpass_method=bandpass_method,
                freq=freq,
                regularize=filter_resularize,
                eps=eps,
            )
            src_fi = filtered_fi[self.get_source_ch()]

            if debug and i % 5 == 0:
                self._show_fordebug(filtered_fi, "filtered data: f=" + str(fi))

            for j, cj in enumerate(c_mesh):
                res_ij = 0.0
                for k_ch in range(self.Num_sensor):
                    t_step = int(self.fs * d[k_ch] / cj)
                    if t_step >= filtered_fi.shape[1]:
                        continue
                    t_maxstep = filtered_fi.shape[1] - t_step - 1
                    trace_x = filtered_fi[k_ch, t_step : t_step + t_maxstep]
                    trace_1 = src_fi[0:t_maxstep]
                    res_ij += self._disp_cc_pair(trace_x, trace_1)
                res[i, j] = res_ij

        return res

    # -----------------------------------------------------------------
    # method='nsc' 実装
    # -----------------------------------------------------------------

    def _calc_nsc_core(
        self,
        _ax,
        f_mesh,
        c_mesh,
        d,
        *,
        band_ratio,
        bandpass_method,
        freq,
        filter_resularize,
        sigma_forgraph,
        debug,
        eps,
    ):
        """非線形信号比較法 (nsc) による分散曲線計算"""
        res = np.zeros((len(f_mesh), len(c_mesh)))

        for i, fi in enumerate(f_mesh):
            if i % 10 == 0 and i != 0:
                print("freq..." + str(fi))
            fpass = np.array([fi / band_ratio, fi * band_ratio])
            fstop = np.array([0.5 * fpass[0], 2.0 * fpass[1]])
            filtered_fi = self._bandpass_filter(
                _ax,
                fpass,
                fstop,
                gpass=3,
                gstop=40,
                bandpass_method=bandpass_method,
                freq=freq,
                regularize=filter_resularize,
                eps=eps,
            )
            src_fi = filtered_fi[self.get_source_ch()]

            if debug and i % 5 == 0:
                self._show_fordebug(filtered_fi, "filtered data: f=" + str(fi))

            for j, cj in enumerate(c_mesh):
                res_ij = 0.0
                for k_ch in range(self.Num_sensor):
                    t_step = int(self.fs * d[k_ch] / cj)
                    if t_step >= filtered_fi.shape[1] - 1:
                        continue
                    t_maxstep = filtered_fi.shape[1] - t_step - 1
                    trace_x = filtered_fi[k_ch, t_step : t_step + t_maxstep]
                    trace_1 = src_fi[0:t_maxstep]
                    res_ij += self._disp_nsc_pair(
                        trace_x, trace_1, fi, sigma_forgraph, eps
                    )
                res[i, j] = res_ij

        return res

    # -----------------------------------------------------------------
    # フィルタ helpers
    # -----------------------------------------------------------------

    def _bandpass_filter(
        self,
        data,
        fpass,
        fstop,
        gpass,
        gstop,
        *,
        bandpass_method,
        freq,
        regularize,
        eps,
    ):
        """バンドパスフィルタを適用して正規化済みデータを返す"""
        fn = self.fs / 2
        wp = fpass / fn
        ws = fstop / fn
        N, Wn = signal.buttord(wp, ws, gpass, gstop)
        temp = np.zeros_like(data, dtype=np.float64)

        if bandpass_method == "filtfilt":
            b, a = signal.butter(N, Wn, "band")
            for i in range(len(data)):
                temp[i] = signal.filtfilt(b, a, data[i])

        elif bandpass_method == "sosfiltfilt":
            sos = signal.butter(N, Wn, btype="band", output="sos")
            for i in range(len(data)):
                temp[i] = signal.sosfiltfilt(sos, data[i])

        elif bandpass_method == "fft":
            min_N = self.fs / freq[0]
            fftN = 2 ** int(np.ceil(np.log2(min_N)))
            d_pad = (
                np.pad(data, ((0, 0), (0, fftN - data.shape[1])), mode="constant")
                if fftN > data.shape[1]
                else data
            )
            d_fft = np.fft.fft(d_pad, axis=1, n=fftN)
            fftfreq = np.fft.fftfreq(fftN, d=1 / self.fs)
            d_fft[:, (fftfreq < fpass[0]) | (fftfreq > fpass[1])] = 0
            temp = np.fft.ifft(d_fft, axis=1).real

        elif bandpass_method == "high_low":
            temp = self._lowpass_filter(data.copy(), fpass[1], fstop[1], gpass, gstop)
            temp = self._highpass_filter(temp, fpass[0], fstop[0], gpass, gstop)
            return self._regularize(temp, regularize, eps)

        elif bandpass_method == "low_high":
            temp = self._highpass_filter(data.copy(), fpass[0], fstop[0], gpass, gstop)
            temp = self._lowpass_filter(temp, fpass[1], fstop[1], gpass, gstop)
            return self._regularize(temp, regularize, eps)

        else:
            raise ValueError(
                "bandpass_method には filtfilt, sosfiltfilt, fft, high_low, low_high のいずれかを指定して"
            )

        return self._regularize(temp, regularize, eps)

    def _lowpass_filter(self, data, fpass, fstop, gpass, gstop):
        """ローパスフィルタ (in-place)"""
        fn = self.fs / 2
        N, Wn = signal.buttord(fpass / fn, fstop / fn, gpass, gstop)
        b, a = signal.butter(N, Wn, "low")
        for i in range(len(data)):
            data[i] = signal.filtfilt(b, a, data[i])
        return data

    def _highpass_filter(self, data, fpass, fstop, gpass, gstop):
        """ハイパスフィルタ (in-place)"""
        fn = self.fs / 2
        N, Wn = signal.buttord(fpass / fn, fstop / fn, gpass, gstop)
        b, a = signal.butter(N, Wn, "high")
        for i in range(len(data)):
            data[i] = signal.filtfilt(b, a, data[i])
        return data

    @staticmethod
    def _regularize(data, mode, eps):
        """振幅正規化"""
        if mode == "all":
            return np.float64(data / (np.max(np.abs(data)) + eps))
        elif mode == "each":
            out = data.copy()
            for i in range(data.shape[0]):
                out[i] = np.float64(data[i] / (np.max(np.abs(data[i])) + eps))
            return out
        return data  # 'none'

    # -----------------------------------------------------------------
    # per-pair 計算 helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _disp_cc_pair(trace_x, trace_1):
        """相互相関法: 1チャンネル分の相関係数"""
        sxx = np.sqrt(np.sum(trace_x**2))
        s11 = np.sqrt(np.sum(trace_1**2))
        s1x = np.sum(trace_x * trace_1)
        return s1x / s11 / sxx

    @staticmethod
    def _disp_nsc_pair(trace_x, trace_1, fi, sigma_forgraph, eps):
        """非線形信号比較法: 1チャンネル分の類似度"""
        sxx = np.sqrt(np.correlate(trace_x, trace_x))
        s11 = np.sqrt(np.correlate(trace_1, trace_1))
        trace_x = trace_x / (sxx + eps)
        trace_1 = trace_1 / (s11 + eps)
        Snv_ij = np.sum(
            np.exp(-((trace_1 - trace_x) ** 2) / (16 * fi**2 * sigma_forgraph**2))
        ) / len(trace_1)
        b = np.pi**2 / sigma_forgraph**2 / (2 * np.pi * fi) ** 2 / len(trace_1)
        S_pi = special.iv(0, b) * np.exp(-b)
        return (Snv_ij - S_pi) / (1 - S_pi)

    # -----------------------------------------------------------------
    # デバッグ用ヘルパー
    # -----------------------------------------------------------------

    def _show_fordebug(self, _ax, suptitle):
        import matplotlib.pyplot as plt
        """デバッグ用: 全チャンネルの波形を並べて表示"""
        n_d, m_d = _ax.shape
        fig, axes = plt.subplots(1, n_d, figsize=(16, 10), sharex=True)
        fig.suptitle(suptitle)
        Y = np.arange(0, m_d)
        for i in range(n_d):
            axes[i].fill_betweenx(
                Y / self.fs, 0, _ax[i, Y], color="black", label=str(i + 1)
            )
            max_x = np.max(np.abs(_ax[i]))
            axes[i].set_xlim([-1.1 * max_x, +1.1 * max_x])
            axes[i].set_ylim(m_d / self.fs, 0)
            axes[i].set_xticklabels([])
            axes[i].set_yticklabels([])
            if i == 0:
                axes[i].set_ylabel("time [s]")
        plt.show()
