import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


class Filter:
    """
    フィルターの基底クラス
    """

    def _get_params_for_filter(self, fpass, fstop, gpass, gstop):
        """
        -将来的にはsignalを使わずにできないか検討中-
        フィルタのパラメータを計算する
        :fpass: 通過域端周波数[Hz]
        :fstop: 阻止域端周波数[Hz]
        :gpass: 通過域の最大損失[dB]
        :gstop: 阻止域の最小損失[dB]
        :return: N, Wn
        """
        fn = self.fs / 2
        wp = fpass / fn  # ナイキスト周波数で通過域端周波数を正規化
        ws = fstop / fn  # ナイキスト周波数で阻止域端周波数を正規化
        N, Wn = signal.buttord(wp, ws, gpass, gstop)
        return N, Wn

    def highpass(self, axis, fpass, fstop, gpass=3, gstop=40):
        """
        axis == 'x', 'y', 'z'
        fpass: float
        fstop: float
        gpass:float default 3
        gstop:float default 40
        """
        _ax, _analysis = self.getax_analysis(axis)
        N, Wn = self._get_params_for_filter(fpass, fstop, gpass, gstop)
        b, a = signal.butter(N, Wn, "high")  # フィルタ伝達関数の分子と分母を計算
        _ax = signal.filtfilt(b, a, _ax, axis=1)

        self.putax_analysis(axis, _ax, _analysis, add=f"_Hp(p{fpass}-s{fstop}Hz)")

    def lowpass(self, axis, fpass, fstop, gpass=3, gstop=40):
        """
        axis:str 'x','y','z'
        fpass: float
        fstop: float
        gpass:float default 3
        gstop:float default 40
        """
        _ax, _analysis = self.getax_analysis(axis)
        N, Wn = self._get_params_for_filter(fpass, fstop, gpass, gstop)
        b, a = signal.butter(N, Wn, "low")  # フィルタ伝達関数の分子と分母を計算
        _ax = signal.filtfilt(b, a, _ax, axis=1)

        self.putax_analysis(axis, _ax, _analysis, add=f"_Lp(p{fpass}-s{fstop}Hz)")

    def bandpass(
        self,
        axis,
        fpass,
        fstop,
        gpass=3,
        gstop=40,
        method="sosfiltfilt",
        show=False,
        worN=8000,
    ):
        """
        Butterworth bandpass filter をゼロ位相で適用する。

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            解析対象軸。
        fpass : array-like, shape (2,)
            通過域端周波数 [Hz]。
        fstop : array-like, shape (2,)
            阻止域端周波数 [Hz]。
        gpass : float, default 3
            通過域最大損失 [dB]。
        gstop : float, default 40
            阻止域最小減衰量 [dB]。
        method : {'sosfiltfilt'}, default 'sosfiltfilt'
            フィルタリング手法。数値安定性の観点から SOS ゼロ位相のみに統一。
        show : bool, default False
            True のときフィルタ応答を表示する。
        worN : int, default 8000
            周波数応答表示に用いるサンプル数。

        Returns
        -------
        self
            bandpass 適用後の self。

        Notes
        -----
        理論背景:
        本関数は Butterworth バンドパスフィルタを設計し、
        forward-backward filtering によりゼロ位相で適用する。
        これにより地震波形の到達時刻や波形ピーク位置の位相歪みを避けやすい。

        ただし、伝達関数の多項式係数 (b, a) による高次 IIR 実装は、
        狭帯域条件や高次数条件で数値不安定になりうるため、
        本実装では second-order sections (SOS) 形式のみを採用している。
        これは SciPy 系の実務上も推奨される構成である。

        Examples
        --------
        self.bandpass('z', fpass=np.array([1000, 3000]), fstop=np.array([500, 6000]))
        self.bandpass('z', fpass=[10, 50], fstop=[5, 80], show=True)
        """

        _ax, _analysis = self.getax_analysis(axis)

        fpass = np.asarray(fpass, dtype=float)
        fstop = np.asarray(fstop, dtype=float)

        N, Wn = self._get_params_for_filter(fpass, fstop, gpass, gstop)

        if method != "sosfiltfilt":
            raise ValueError("method must be 'sosfiltfilt'")

        sos = signal.butter(N, Wn, btype="band", output="sos")
        filtered_ax = signal.sosfiltfilt(sos, _ax, axis=1)

        self.putax_analysis(
            axis,
            filtered_ax,
            _analysis + f"_bandpass({fpass[0]},{fpass[1]})Hz"
        )

        if show:
            w, h = signal.sosfreqz(sos, worN=worN, fs=self.fs)
            self._show_bandpass_response_sos(
                w=w,
                h=h,
                N=N,
                Wn=Wn,
                fpass=fpass,
                fstop=fstop,
                worN=worN,
            )

    def _show_bandpass_response_sos(self, w, h, N, Wn, fpass, fstop, worN):
        """
        SOS バンドパスフィルタの周波数応答を表示する (thin wrapper)。

        描画本体は plotter 層の bandpass_response に委譲する。
        N, Wn, worN は委譲先では使用しないが、シグネチャ互換のため残置。

        Parameters
        ----------
        w : array-like
            sosfreqz で計算済みの周波数配列 [Hz]（fs 指定時）。
        h : array-like
            sosfreqz で計算済みの複素周波数応答。
        N : int
            フィルタ次数（互換のため残置、委譲先では未使用）。
        Wn : array-like
            正規化遮断周波数（互換のため残置、委譲先では未使用）。
        fpass : array-like
            通過域端周波数 [Hz]。
        fstop : array-like
            阻止域端周波数 [Hz]。
        worN : int
            周波数応答のサンプル数（互換のため残置、委譲先では未使用）。
        """
        self._ensure_plotter()
        self._plotter.bandpass_response(w, h, fpass, fstop, show=True)

    def _bandpass_filtfilt(self, axis, _ax, _analysis, b, a, fpass, fstop):
        _ax = signal.filtfilt(b, a, _ax, axis=1)
        self.putax_analysis(
            axis,
            _ax,
            _analysis,
            add="_Bp(filtfilt, fpass" + str(fpass) + "Hz fstop=" + str(fstop) + "Hz)",
        )

    def _bandpass_sosfiltfilt(self, axis, _ax, _analysis, N, Wn, fpass, fstop):
        sos = signal.butter(N, Wn, btype="band", output="sos")
        _ax = signal.sosfiltfilt(sos, _ax, axis=1)
        self.putax_analysis(
            axis,
            _ax,
            _analysis,
            add="_Bp(filtfilt, fpass=" + str(fpass) + ", fstop=" + str(fstop) + ")",
        )

    def _show_bandpass_response(self, N, Wn, fpass, fstop, worN):
        sos = signal.butter(N, Wn, btype="band", output="sos")
        w, h = signal.sosfreqz(sos, worN=worN, fs=self.fs)

        # wを周波数に変換
        w = w / np.pi * self.fs / 2
        fig, ax1 = plt.subplots(sharex=True)
        ax1.set_title("Digital filter frequency response")
        ax1.plot(w, 20 * np.log10(abs(h)), "b")
        ax1.set_ylabel("Amplitude [dB]", color="b")
        ax1.set_xlabel("Frequency [Hz]")
        ax2 = ax1.twinx()
        ax2.plot(w, np.angle(h) / 2 * np.pi * 360, "r")
        ax2.set_ylabel("Angles (degree))", color="r")
        ax2.grid()
        ax2.axis("tight")
        # x の範囲をバンドパス遮断周波数の大きい側、の10倍までに設定
        fmax = max(fstop) * 10
        fmin = min(fstop) * 0.1
        ax1.set_xlim([fmin, fmax])
        ax2.set_xlim([fmin, fmax])
        # x軸を対数スケールにする
        ax1.set_xscale("log")
        ax2.set_xscale("log")
        # y の範囲は-100dB から 10dB に設定
        ax1.set_ylim([-100, 10])
        ax2.set_ylim([-360, 360])
        # 通過周波数帯を橙色で表示
        ax1.axvspan(fpass[0], fpass[1], color="orange", alpha=0.5)
        # 阻止周波数帯を紫色で表示
        ax1.axvspan(fmin, fstop[0], color="purple", alpha=0.5)
        ax1.axvspan(fstop[1], fmax, color="purple", alpha=0.5)
        plt.show()

    def fk_filter(
        self,
        axis,
        v_range=None,
        k_range=None,
        f_range=None,
        gain=0,
        remove_source_ch=True,
        include_inf_velocity=False,
        taper_percent=0.10,
        crossfade_ch=3,
        epsilon=1e-12,
        edge_taper_percent=0.0,
        min_taper_fraction_for_zero_vmin=0.10,
    ):
        """
        起振点基準の方向分離付き F-K フィルタ

        Parameters
        ----------
        axis : {'x', 'y', 'z'}
            解析対象軸。
        v_range : list[float, float] or None
            除去または増減衰したい位相速度範囲 [vmin, vmax] [m/s]。
            ここでの位相速度は「起振点から遠ざかる方向を正」として扱う。
        k_range : list[float, float] or None
            除去または増減衰したい波数範囲 [kmin, kmax] [1/m]。
        f_range : list[float, float] or None
            除去または増減衰したい周波数範囲 [fmin, fmax] [Hz]。
        gain : float, default 0
            対象範囲に掛けるゲイン。0 で完全抑制、1 で無変更、1超で増幅。
        remove_source_ch : bool, default True
            起振点近傍チャネルを F-K フィルタ処理から除外する。
        include_inf_velocity : bool, default False
            k=0 に対応する無限大速度成分も位相速度フィルタ対象に含めるかどうか。
        taper_percent : float, default 0.10
            各カットオフ近傍の遷移帯幅の割合。
        crossfade_ch : int, default 3
            起振点境界で左右波形を混合する半幅チャネル数。
        epsilon : float, default 1e-12
            ゼロ除算回避用の微小値。
        edge_taper_percent : float, default 0.0
            F-K 領域外縁（Nyquist 近傍）に掛ける弱い2次元テーパー幅。
        min_taper_fraction_for_zero_vmin : float, default 0.10
            vmin=0 のとき、低速側テーパー幅が epsilon に潰れないように
            vmax * taper_percent * min_taper_fraction_for_zero_vmin を下限として与える係数。

        Returns
        -------
        self
            fk_filter を適用後の self。

        Notes
        -----
        理論背景:
        地震探査やアレイ信号処理で広く用いられる F-K
        (frequency-wavenumber) フィルタリングに基づく。平面波
        u(x,t)=exp(i 2π (k x - f t)) を仮定すると、F-K 領域では
        位相速度は v = f / k で与えられる。NumPy の FFT 規約に合わせると、
        本コードでは v = -f_t / f_x を用いる実装になっている。
        これは実装中の _velocity_matrix() に対応する。

        また、freq_x * freq_t の符号により進行方向を左右に分離し、
        起振点から遠ざかる成分を正方向速度として扱う構成を採っている。
        この考え方は、地震探査における F-K フィルタの基本的な方向分離と整合する。

        参考となる理論・先行研究:
        - F-K フィルタリングは、線形イベントの傾き（見掛け速度）に応じた
          ノイズ抑制・波動分離手法として地震探査処理で広く用いられている。
        - F-K 領域でのマスク処理は、時間-空間領域での見掛け速度選別に対応する。
        - Capon 法や高分解能 F-K 推定との比較研究では、古典的 F-K 法は
          実装が簡潔で頑健な基準法として位置づけられている。

        実装上の注意:
        - 本実装は F-K 領域のマスクにはコサインテーパーを導入しているが、
          入力データそのものには空間窓関数を掛けていない。
        - センサー数が 24〜96 ch 程度あれば多くのケースで実用的だが、
          アレイ端で強い不連続がある場合にはスペクトル漏洩が残る可能性がある。
        - そのため、本関数は実行時に note としてこの注意を表示する。
        """

        def _cosine_step(x):
            x_clip = np.clip(x, 0.0, 1.0)
            return 0.5 - 0.5 * np.cos(np.pi * x_clip)

        def _smooth_range_mask(value_matrix, value_range, taper_ratio, mode="generic"):
            if value_range is None:
                return np.ones_like(value_matrix, dtype=float)

            vmin, vmax = sorted(value_range)
            if np.isclose(vmin, vmax):
                return np.isclose(value_matrix, vmin).astype(float)

            taper_ratio = max(float(taper_ratio), 0.0)
            if taper_ratio == 0.0:
                return ((value_matrix >= vmin) & (value_matrix <= vmax)).astype(float)

            if mode == "velocity":
                w_ref = max(abs(vmin), abs(vmax)) * taper_ratio
                min_floor = max(
                    abs(vmax) * taper_ratio * float(min_taper_fraction_for_zero_vmin),
                    epsilon
                )
                wmin = max(abs(vmin) * taper_ratio, min_floor, epsilon)
                wmax = max(abs(vmax) * taper_ratio, max(w_ref * 0.1, epsilon), epsilon)
            else:
                wmin = max(abs(vmin) * taper_ratio, epsilon)
                wmax = max(abs(vmax) * taper_ratio, epsilon)

            low_start = vmin - wmin
            low_end = vmin + wmin
            high_start = vmax - wmax
            high_end = vmax + wmax

            up = _cosine_step((value_matrix - low_start) / max(low_end - low_start, epsilon))
            down = _cosine_step((high_end - value_matrix) / max(high_end - high_start, epsilon))
            return np.minimum(up, down)

        def _edge_taper_2d(freq_x, freq_t, taper_ratio):
            taper_ratio = max(float(taper_ratio), 0.0)
            if taper_ratio == 0.0:
                return np.ones_like(freq_x, dtype=float)

            k_nyq = np.max(np.abs(freq_x))
            f_nyq = np.max(np.abs(freq_t))
            if k_nyq <= 0 or f_nyq <= 0:
                return np.ones_like(freq_x, dtype=float)

            nk = np.abs(freq_x) / k_nyq
            nf = np.abs(freq_t) / f_nyq
            radius = np.sqrt((nk ** 2 + nf ** 2) / 2.0)

            start = max(0.0, 1.0 - taper_ratio)
            edge_prog = _cosine_step((radius - start) / max(taper_ratio, epsilon))
            edge_strength = 0.15
            return 1.0 - edge_strength * edge_prog

        def _velocity_matrix(freq_x, freq_t):
            safe_k = np.where(
                np.abs(freq_x) < epsilon,
                np.where(freq_x >= 0, epsilon, -epsilon),
                freq_x
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                return -(freq_t / safe_k)

        def _directional_mask(vr, kr, fr, freq_x, freq_t):
            mask = np.ones_like(freq_x, dtype=float)

            if vr is not None:
                vmat = _velocity_matrix(freq_x, freq_t)
                v_mask = _smooth_range_mask(vmat, vr, taper_percent, mode="velocity")
                finite_mask = np.isfinite(vmat)
                v_mask *= finite_mask.astype(float)
                if include_inf_velocity:
                    v_mask = np.maximum(v_mask, (np.abs(freq_x) < epsilon).astype(float))
                mask *= v_mask

            if kr is not None:
                kmin, kmax = sorted(kr)
                k_mask = _smooth_range_mask(
                    np.abs(freq_x), [abs(kmin), abs(kmax)], taper_percent, mode="generic"
                )
                mask *= k_mask

            if fr is not None:
                fmin, fmax = sorted(fr)
                f_mask = _smooth_range_mask(
                    np.abs(freq_t), [abs(fmin), abs(fmax)], taper_percent, mode="generic"
                )
                mask *= f_mask

            if edge_taper_percent > 0.0:
                mask *= _edge_taper_2d(freq_x, freq_t, edge_taper_percent)

            return np.clip(mask, 0.0, 1.0)

        if v_range is None and k_range is None and f_range is None:
            raise ValueError("v_range, k_range, f_range のいずれかを指定してください。")

        warnings.warn(
            "note: fk_filter は F-K 領域マスクにテーパーを導入していますが、"
            "入力データ自体には空間窓関数を適用していません。"
            "24〜96ch 程度では多くの実務で有効ですが、アレイ端で波形不連続が強い場合は"
            "スペクトル漏れが生じる可能性があります。",
            UserWarning
        )

        _ax, _analysis = self.getax_analysis(axis)

        if remove_source_ch:
            source_ch = self.get_source_ch()

            remove_indices = [source_ch]
            if source_ch > 0:
                remove_indices.append(source_ch - 1)
            remove_indices = sorted(remove_indices)

            original_ax = _ax.copy()
            original_distance = self.distance.copy()
            original_num_sensor = self.Num_sensor

            for idx in sorted(remove_indices, reverse=True):
                _ax = np.delete(_ax, idx, axis=0)
                self.distance = np.delete(self.distance, idx)
                self.Num_sensor = _ax.shape[0]
        else:
            original_ax = _ax.copy()
            original_distance = self.distance.copy()
            original_num_sensor = self.Num_sensor
            remove_indices = []

        dt = 1 / self.fs
        dx = self.interval

        fftfreq_t = np.fft.fftfreq(_ax.shape[1], d=dt)
        fftfreq_x = np.fft.fftfreq(_ax.shape[0], d=dx)
        freq_x, freq_t = np.meshgrid(fftfreq_x, fftfreq_t, indexing="ij")
        fft2_all = np.fft.fft2(_ax)

        prod_fk = freq_x * freq_t
        zero_axis = (np.isclose(freq_x, 0.0) | np.isclose(freq_t, 0.0)).astype(float)
        left_weight = (prod_fk > 0).astype(float) + 0.5 * zero_axis
        right_weight = (prod_fk < 0).astype(float) + 0.5 * zero_axis

        spec_left = fft2_all * left_weight
        spec_right = fft2_all * right_weight

        if v_range is not None:
            vmin, vmax = sorted(v_range)
            vr_right = [vmin, vmax]
            vr_left = [-vmax, -vmin]
        else:
            vr_right = None
            vr_left = None

        mask_left = _directional_mask(vr_left, k_range, f_range, freq_x, freq_t)
        mask_right = _directional_mask(vr_right, k_range, f_range, freq_x, freq_t)

        gain_map_left = 1.0 + (float(gain) - 1.0) * mask_left
        gain_map_right = 1.0 + (float(gain) - 1.0) * mask_right

        spec_left_filtered = spec_left * gain_map_left
        spec_right_filtered = spec_right * gain_map_right

        ax_left_raw = np.fft.ifft2(spec_left).real
        ax_right_raw = np.fft.ifft2(spec_right).real
        ax_left_filtered = np.fft.ifft2(spec_left_filtered).real
        ax_right_filtered = np.fft.ifft2(spec_right_filtered).real

        rel_dist = self.distance - self.source_x
        source_idx = int(np.argmin(np.abs(rel_dist)))

        if v_range is None:
            new_ax = ax_left_filtered + ax_right_filtered
            crossfade_ch = 0 if crossfade_ch is None else max(int(crossfade_ch), 0)
        else:
            right_side_mix = ax_left_raw + ax_right_filtered
            left_side_mix = ax_right_raw + ax_left_filtered

            if crossfade_ch is None:
                crossfade_ch = 0
            crossfade_ch = max(int(crossfade_ch), 0)

            idx_offset = np.arange(_ax.shape[0]) - source_idx
            if crossfade_ch == 0:
                right_blend = (idx_offset >= 0).astype(float)
            else:
                right_blend = _cosine_step((idx_offset + crossfade_ch) / (2.0 * crossfade_ch))

            left_blend = 1.0 - right_blend
            new_ax = left_side_mix * left_blend[:, None] + right_side_mix * right_blend[:, None]

        if remove_source_ch:
            self.distance = original_distance
            self.Num_sensor = original_num_sensor

            full_ax = np.zeros((original_num_sensor, new_ax.shape[1]), dtype=new_ax.dtype)
            j = 0
            for i in range(original_num_sensor):
                if i not in remove_indices:
                    full_ax[i] = new_ax[j]
                    j += 1
                else:
                    full_ax[i] = original_ax[i]
            new_ax = full_ax

        filter_info = []
        if v_range is not None:
            vmin, vmax = sorted(v_range)
            filter_info.append(f"v({vmin},{vmax})")
        if k_range is not None:
            kmin, kmax = sorted(k_range)
            filter_info.append(f"k({kmin},{kmax})")
        if f_range is not None:
            fmin, fmax = sorted(f_range)
            filter_info.append(f"f({fmin},{fmax})")
        filter_info.append(f"taper({taper_percent})")
        filter_info.append(f"cf({crossfade_ch})")
        if edge_taper_percent > 0:
            filter_info.append(f"edge({edge_taper_percent})")

        # TODO: fk_filter のデバッグ可視化を plotter 層に委譲する場合、
        # ここに self._ensure_plotter() → self._plotter.fk_filter_response(...)
        # を呼び出す show パラメータ経路を追加する。
        self.putax_analysis(axis, new_ax, _analysis + f"_fkfilter{'-'.join(filter_info)}")
        return self

    def denoise_upgoing_wave(
        self, axis, v_range=[100, 200], gain=0, remove_source_ch=True
    ):
        """
        起振点から遠ざかる方向の波を除去する関数
        ---parameters---
        axis: 'x' or 'y' or 'z'
        v_range = [float, float]: 除去したい位相速度範囲([vmin, vmax],[m/s]
        gain: 対象のv範囲において書けるゲイン(デフォルトは0->denoise)
        ---return---
        self: denoise_upgoing_waveを実行した後のselfオブジェクト
        """

        # axを取得
        _ax, _analysis = self.getax_analysis(axis)

        if remove_source_ch:
            # 起振点のchを除去
            source_ch = self.get_source_ch()
            _ax = np.delete(_ax, source_ch, axis=0)
            self.distance = np.delete(self.distance, source_ch)  # 起振点のchを除去
            self.Num_sensor = _ax.shape[0]  # チャンネル数を更新

            if source_ch > 0:
                _ax = np.delete(_ax, source_ch - 1, axis=0)
                self.distance = np.delete(
                    self.distance, source_ch - 1
                )  # 起振点の前のchを除去
                self.Num_sensor = _ax.shape[0]  # チャンネル数を更新

        mask_plus = (self.distance - self.source_x) >= 0  # True ↔︎ 起振点よりプラス側
        # 反転マスクは ~ を付ける
        _ax_plus = _ax[mask_plus]  # = _ax[mask_plus, :]
        _ax_minus = _ax[~mask_plus]  # = _ax[~mask_plus, :]

        vmin, vmax = sorted(v_range)
        vr_minus = [-vmax, -vmin]
        vr_plus = [vmin, vmax]

        if len(_ax_plus) != 0:  # 起振点から正方向に遠ざかるchがある場合
            _ax_plus_denoised = self._denoise_v_fft(
                _ax_plus, vr_plus, 1 / self.fs, self.interval, gain
            )  # 正方向の波を除去

        if len(_ax_minus) != 0:  # 起振点から負方向に遠ざかるchがある場合
            _ax_minus_denoised = self._denoise_v_fft(
                _ax_minus, vr_minus, 1 / self.fs, self.interval, gain
            )  # 負方向の波を除去

        if (
            len(_ax_plus) != 0 and len(_ax_minus) != 0
        ):  # 正方向と負方向の両方の波がある場合
            new_ax = (
                np.vstack([_ax_plus_denoised, _ax_minus_denoised])
                if mask_plus[0]
                else np.vstack([_ax_minus_denoised, _ax_plus_denoised])
            )
        elif len(_ax_plus) != 0:  # 正方向の波のみがある場合
            new_ax = _ax_plus_denoised
        elif len(_ax_minus) != 0:  # 負方向の波のみがある場合
            new_ax = _ax_minus_denoised
        else:  # 正方向と負方向の波がない場合
            raise ValueError(
                "正方向と負方向の波がない場合は、denoise_upgoing_waveは使えません"
            )

        if remove_source_ch:
            # 起振点のchを復元
            new_ax = (
                np.insert(new_ax, source_ch, _ax[self.get_source_ch()], axis=0)
                if source_ch < new_ax.shape[0]
                else np.vstack((new_ax, _ax[self.get_source_ch()]))
            )
            self.distance = (
                np.insert(self.distance, source_ch, self.source_x)
                if source_ch < self.distance.shape[0]
                else np.concatenate([self.distance, [self.source_x]])
            )  # 起振点のchを復元
            self.Num_sensor = new_ax.shape[0]  # チャンネル数を更新
            if source_ch > 0:
                new_ax = (
                    np.insert(new_ax, source_ch - 1, _ax[source_ch - 1], axis=0)
                    if source_ch < new_ax.shape[0]
                    else np.vstack((new_ax, _ax[self.get_source_ch() - 1]))
                )
                self.distance = (
                    np.insert(
                        self.distance, source_ch - 1, self.distance[source_ch - 1]
                    )
                    if source_ch < self.distance.shape[0]
                    else np.concatenate([self.distance, [self.source_x]])
                )
                self.Num_sensor = new_ax.shape[0]  # チャンネル数を更新

        self.putax_analysis(
            axis, new_ax, _analysis + f"_denoise_upgoing_wave({vmin},{vmax})"
        )

    def _denoise_v_fft(self, ax, v_range, dt, dx, gain):
        """
        denoise_upgoing_wave() の純粋計算部 (描画なし)。
        v_range で指定された位相速度範囲の f-k 成分にゲインを掛けて逆 FFT する。
        """
        # FFT変換
        fftfreq_t = np.fft.fftfreq(ax.shape[1], d=dt)  # 共通
        fftfreq_x = np.fft.fftfreq(ax.shape[0], d=dx)
        fft2 = np.fft.fft2(ax)

        # メッシュグリッドによる周波数行列を作成
        freq_x, freq_t = np.meshgrid(fftfreq_x, fftfreq_t, indexing="ij")

        # ゼロ除算防止のための処理（freq_xが0の場合は無限速度として扱いマスク適用外）
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity_matrix = freq_t / freq_x

        # マスクの作成（条件に合致する部分をゼロに）
        vmin, vmax = sorted(v_range)

        mask = (velocity_matrix > vmin) & (velocity_matrix < vmax)

        # ゼロ除算などによるNaNやinfを除外
        mask &= np.isfinite(velocity_matrix)

        # フィルタリングの適用
        fft2[mask] *= gain  # gainをかける

        # 逆FFTで元の領域に戻す
        new_ax = np.fft.ifft2(fft2).real
        return new_ax

    def denoise_peakwave_v(
        self, axis, v_range=[100, 200], reverse=False, debug=False, gain=0
    ):
        """
        主にピーク波を除去するための関数
        ---parameters---
        axis: 'x' or 'y' or 'z'
        v_range = [float, float]: 除去したい位相速度範囲([vmin, vmax],[m/s]
        reverse: True or False: 除去したい位相速度範囲を逆伝播方向にも適用させるか
                True: 逆伝播方向にも適用 -> 反射波方向も除去
                False: 逆伝播方向に適用しない -> 反射波方向は除去しない
        debug: True or False: debug用のプロットを表示するか
        gain: 対象のv範囲において書けるゲイン(デフォルトは0->denoise)
        """
        # axを取得
        _ax, _analysis = self.getax_analysis(axis)

        # FFT変換
        fft2 = np.fft.fft2(_ax)
        fftfreq_t = np.fft.fftfreq(_ax.shape[1], d=1 / self.fs)  # 時間周波数 f [1/s]
        fftfreq_x = np.fft.fftfreq(_ax.shape[0], d=self.interval)  # 空間周波数 k [1/m]

        if debug:
            plt.figure(figsize=(10, 6))
            plt.subplot(2, 1, 1)
            plt.imshow(
                (np.abs(fft2) / np.max(np.abs(fft2))).T,
                aspect="auto",
                origin="lower",
                cmap="seismic",
                extent=[fftfreq_x[0], fftfreq_x[-1], fftfreq_t[0], fftfreq_t[-1]],
            )
            # plt.colorbar(label='Amplitude')
            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Frequency (1/m)")
            plt.title("before fkkfFFT2")

        new_ax, vmin, vmax = self._denoise_peakwave_v_apply_mask(
            fft2, fftfreq_t, fftfreq_x, v_range, reverse, gain
        )

        if debug:
            plt.subplot(2, 1, 2)
            plt.imshow(
                np.abs(fft2) / np.max(np.abs(fft2)),
                aspect="auto",
                origin="lower",
                cmap="seismic",
                extent=[fftfreq_x[0], fftfreq_x[-1], fftfreq_t[0], fftfreq_t[-1]],
            )
            # plt.colorbar(label='Amplitude')
            plt.ylabel("Frequency (Hz)")
            plt.xlabel("Frequency (1/m)")
            plt.title("after fkkfFFT2")
            plt.show()

        # 結果を保存
        new_analysis = _analysis + f"_fkkf({vmin},{vmax})"
        if reverse:
            new_analysis += "[rvs=True]"
        self.putax_analysis(axis, new_ax, new_analysis)

    def _denoise_peakwave_v_apply_mask(
        self, fft2, fftfreq_t, fftfreq_x, v_range, reverse, gain
    ):
        """
        denoise_peakwave_v() の純粋計算部 (描画なし)。

        引数 fft2 を **in-place で書き換える** ことに注意。呼び出し側は事前/事後の
        fft2 をそれぞれ "before"/"after" の debug 描画に利用するため、本仕様は意図的。

        Returns
        -------
        new_ax : np.ndarray, ifft2 後の実数化 2D 波形
        vmin   : float, sorted(v_range)[0]
        vmax   : float, sorted(v_range)[1]
        """
        # メッシュグリッドによる周波数行列を作成
        freq_x, freq_t = np.meshgrid(fftfreq_x, fftfreq_t, indexing="ij")

        # ゼロ除算防止のための処理（freq_xが0の場合は無限速度として扱いマスク適用外）
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity_matrix = freq_t / freq_x

        # マスクの作成（条件に合致する部分をゼロに）
        vmin, vmax = sorted(v_range)

        mask = (velocity_matrix > vmin) & (velocity_matrix < vmax)

        if reverse:
            reverse_mask = (-velocity_matrix > vmin) & (-velocity_matrix < vmax)
            mask |= reverse_mask

        # ゼロ除算などによるNaNやinfを除外
        mask &= np.isfinite(velocity_matrix)

        # フィルタリングの適用
        fft2[mask] *= gain  # gainをかける

        # 逆FFTで元の領域に戻す
        new_ax = np.fft.ifft2(fft2).real
        return new_ax, vmin, vmax

    def denoise_peakwave(self, axis, f=[10, 400], c=[-500, 500], df=1, dc=1):
        """
        主要波を除去して、反射波を見やすくする
        ---parameters-----------------------------------
        axis: 'x' or 'y' or 'z'

        f: [fmin, fmax] (default is [10, 400]) 単位[Hz]
        c: [cmin, cmax] (default is [-500, 500])単位[m/s]
        df: frequency step (default is 1) 単位[Hz]
        dc: phase velocity step (default is 1) 単位[m/s]
        ------------------------------------------------
        *手法*
        表面波の分散曲線からピーク位相速度を読み取り、二次元フーリエ変換上でピーク部分を除去して反射波を見やすくする
        1. 表面波の分散曲線を計算し、角周波数でのピーク位相速度を導出
        2. ピーク速度に対応する波数kを出し、(f,k)平面上でfft2のピーク部分を除去
        3. 逆フーリエ変換して実部をとり、波形を格納する

        ~for loopで書くと以下~
        for j, freq_t in enumerate(fftfreq_t):
                indices = np.abs(fmesh - freq_t) < 0.1*np.abs(freq_t) #幅をつけて
                freq_x = k_mesh[indices] # その周波数に対応する波数を取得
                for freq in freq_x:
                        k_cut_range = [min(freq*0.9, freq*1.1), max(freq*0.9, freq*1.1)]
                        k_cut_index = np.where(np.logical_and(np.abs(fftfreq_x) > k_cut_range[0], np.abs(fftfreq_x) < k_cut_range[1]))
                        fft2[k_cut_index, j] = 0

        """
        ###############
        # データの取得 #
        ###############
        _ax, _analysis = self.getax_analysis(axis)
        new_ax = self._denoise_peakwave_core(_ax, f, c, df, dc)
        new_analysis = _analysis + "_fkkf_filter"
        self.putax_analysis(axis, new_ax, new_analysis)

    def _denoise_peakwave_core(self, _ax, f, c, df, dc):
        """
        denoise_peakwave() の純粋計算部 (描画なし)。

        分散曲線的に各周波数のピーク位相速度を求め、(f,k) 平面上で対応する波数帯を
        ゼロにマスクし、逆 FFT で時空間に戻した波形 (実数化済み) を返す。

        Returns
        -------
        new_ax : np.ndarray, ifft2 後の実数化 2D 波形
        """
        fmin, fmax = f[0], f[1]
        cmin, cmax = c[0], c[1]

        #################
        # 位相速度の導出 #
        #################
        distance_from_source = np.abs(
            (self.distance - np.ones_like(self.distance) * self.source_x)
        )
        d_maxdiff = np.max(distance_from_source) - np.min(distance_from_source)
        f_mesh, c_mesh = np.arange(fmin, fmax, df), np.arange(cmin, cmax, dc)

        fft_t = np.fft.fft(_ax, axis=1)
        freq_t = np.fft.fftfreq(_ax.shape[1], d=1 / self.fs)
        pos_mask = freq_t >= 0  # Keep only the positive frequencies
        pos_freqs = freq_t[pos_mask]
        _fft_pos = fft_t[:, pos_mask]

        fi_indices = np.array(
            [np.argmin(np.abs(pos_freqs - fi)) for fi in f_mesh]
        )  # Find the indices of frequencies in f_mesh within pos_freqs
        _fft_at_fi = _fft_pos[:, fi_indices].T  # Shape: (len(f_mesh), len(_ax))
        _fft_at_fi_normalized = _fft_at_fi / np.abs(_fft_at_fi)

        # Compute omega and phy
        omega = 2 * np.pi * f_mesh  # Shape: (len(f_mesh),)
        phy = (
            omega[:, np.newaxis] / c_mesh[np.newaxis, :]
        )  # Shape: (len(f_mesh), len(c_mesh))
        k = (
            f_mesh[:, np.newaxis] / c_mesh[np.newaxis, :]
        )  # Shape: (len(f_mesh), len(c_mesh))
        nyquist_k = 1 / (2 * self.interval)
        # Compute the exponential term
        exp_term = np.exp(
            1j * phy[:, :, np.newaxis] * distance_from_source[np.newaxis, np.newaxis, :]
        )  # Shape: (len(f_mesh), len(c_mesh), len(_ax))

        # Multiply and sum over k to get res
        product = (
            exp_term * _fft_at_fi_normalized[:, np.newaxis, :]
        )  # Shape: (len(f_mesh), len(c_mesh), len(_ax))

        # where k > nyquist_k, res =  np.real(np.sum(product, axis=2)), else res = 0
        res = np.abs(np.sum(product, axis=2))

        res[np.abs(k) > nyquist_k] = 0
        res[np.abs(k) < 1 / (2 * d_maxdiff)] = 0

        #######################################
        # 二次元フーリエ変換上でのフィルタリング #
        #######################################
        c = c_mesh[np.argmax(res, axis=1)]
        k_mesh = f_mesh / c
        # filter前のデータの二次元FFTを取得
        fft2 = np.fft.fft2(_ax)  # (distance, timestep) array
        fftfreq_t = np.fft.fftfreq(_ax.shape[1], d=1 / self.fs)
        fftfreq_x = np.fft.fftfreq(_ax.shape[0], d=self.interval)

        ##### 以下、chatgptのコードを参考 #####
        # 必要な絶対値を計算
        fftfreq_x_abs = np.abs(fftfreq_x)  # 形状: (Nx,)
        fftfreq_t_abs = np.abs(fftfreq_t)  # 形状: (Nt,)
        k_mesh_abs = np.abs(k_mesh)  # 形状: (Nk,)

        # fft2 の形状が(Nx, Nt)であることを確認
        assert fft2.shape == (len(fftfreq_x), len(fftfreq_t)), (
            "fft2 の形状が期待と異なります"
        )

        # 外側のループを開始
        for j, freq_t in enumerate(fftfreq_t):
            # freq_t の絶対値
            freq_t_abs = fftfreq_t_abs[j]

            indices = np.abs(f_mesh - freq_t) < 0.1 * freq_t_abs  # 形状: (Nk,)
            freq_x = k_mesh[indices]  # 形状: (N_indices,)

            if freq_x.size == 0:
                continue  # 該当する freq_x がない場合は次へ

            # freq_x に対する k_cut_range を計算
            k_cut_range_lower = np.minimum(
                freq_x * 0.9, freq_x * 1.1
            )  # 形状: (N_indices,)
            k_cut_range_upper = np.maximum(
                freq_x * 0.9, freq_x * 1.1
            )  # 形状: (N_indices,)

            # fftfreq_x_abs を拡張して形状を合わせる
            fftfreq_x_abs_expanded = fftfreq_x_abs[np.newaxis, :]  # 形状: (1, Nx)
            k_cut_range_lower_expanded = k_cut_range_lower[
                :, np.newaxis
            ]  # 形状: (N_indices, 1)
            k_cut_range_upper_expanded = k_cut_range_upper[
                :, np.newaxis
            ]  # 形状: (N_indices, 1)

            # 条件を評価してマスクを作成
            k_cut_mask = np.logical_and(
                fftfreq_x_abs_expanded > k_cut_range_lower_expanded,
                fftfreq_x_abs_expanded < k_cut_range_upper_expanded,
            )  # 形状: (N_indices, Nx)

            # 各 freq_x に対するマスクを統合
            combined_mask = np.any(k_cut_mask, axis=0)  # 形状: (Nx,)

            # マスクを適用して fft2 の該当部分を 0 に設定
            fft2[combined_mask, j] = 0

        #########################
        # filter後のデータを格納 #
        #########################
        new_ax = np.fft.ifft2(fft2).real
        return new_ax
