"""
後方散乱（バックスキャッター）解析モジュール

後方散乱振幅を取得するアルゴリズム:
- 'fft'        : f-k スペクトル解析による反射波振幅
- 'dispersion' : 分散イメージングによる負位相速度振幅

"""

import copy

import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class BackscatterAnalysis(PlotterWrapperMixin):
    """
    後方散乱振幅解析 Mixin。

    SingleProcesser に継承させることで hr.backscatter() が使えるようになる。
    """

    def backscatter(
        self,
        axis: str,
        mode: str = "fft",
        N: int = 6,
        freq: list | None = None,
        c: list | None = None,
        df: float = 1.0,
        dc: float = 1.0,
        method: str = "fft",
        zeropadding: int = 0,
        pick_c_range: list | None = None,
        show: bool = True,
        save_name: str | None = None,
        show_fk: bool = False,
        show_fk_index: int = 10,
        **kw,
    ):
        """
        後方散乱振幅を計算し、必要に応じて描画・保存する。

        Parameters
        ----------
        axis          : 'x' | 'y' | 'z'
        mode          : 'fft' | 'dispersion'
        N             : スライドするウィンドウのチャンネル数
        freq          : [fmin, fmax] Hz  (dispersion モードで使用)
        c             : [cmin, cmax] m/s (dispersion モードで使用)
        df            : 周波数刻み [Hz]
        dc            : 速度刻み [m/s]
        method        : dispersion 計算の手法 ('fft' | 'cc' | 'nsc')
        zeropadding   : ゼロパディングサンプル数
        pick_c_range  : [c_abs_min, c_abs_max]  負位相速度の抽出範囲
        show          : True のとき最終結果を表示
        save_name     : 保存ファイル名 (None のとき自動生成)
        show_fk       : True のとき特定インデックスの f-k スペクトルを表示 (fft モードのみ)
        show_fk_index : f-k スペクトルを表示するインデックス

        Returns
        -------
        ind : np.ndarray  センサーインデックス配列
        amp : list        各インデックスの振幅リスト
        """
        if freq is None:
            freq = [100, 1000]
        if c is None:
            c = [-300, 300]
        if pick_c_range is None:
            pick_c_range = [20, 120]

        if mode == "fft":
            ind, amp = self._backscatter_fft_core(
                axis,
                N=N,
                show_fk=show_fk,
                show_fk_index=show_fk_index,
            )
            title = f"reflection_amp (N={N})"
            ylabel = "reflection_amp"

        elif mode == "dispersion":
            ind, amp = self._backscatter_dispersion_core(
                axis,
                N=N,
                freq=freq,
                c=c,
                df=df,
                dc=dc,
                method=method,
                zeropadding=zeropadding,
                pick_c_range=pick_c_range,
            )
            title = f"negative_c_amp (N={N})"
            ylabel = "negative_c_amp"

        else:
            raise ValueError(
                f"mode='{mode}' は無効です。'fft' または 'dispersion' を指定してください。"
            )

        # ---- 後処理（描画） ----
        if show or save_name:
            self._ensure_plotter()
            self.backscatter_image(
                ind,
                amp,
                title=title,
                ylabel=ylabel,
                show=show,
                save_name=save_name,
                **kw,
            )

        return ind, amp

    # ------------------------------------------------------------------
    # コア計算: dispersion モード
    # ------------------------------------------------------------------

    def _backscatter_dispersion_core(
        self, axis, N, freq, c, df, dc, method, zeropadding, pick_c_range
    ):
        """
        分散イメージングを用いた後方散乱振幅の計算。

        Parameters
        ----------
        axis        : データ軸 ('x' | 'y' | 'z')
        N           : スライドウィンドウのチャンネル数
        freq        : [fmin, fmax]
        c           : [cmin, cmax]
        df          : 周波数刻み
        dc          : 速度刻み
        method      : 'fft' | 'cc' | 'nsc'
        zeropadding : ゼロパディングサンプル数
        pick_c_range: [c_abs_min, c_abs_max] 負速度抽出範囲

        Returns
        -------
        ind            : np.ndarray
        negative_c_amp : list
        """
        _ax, hr_analysis = self.getax_analysis(axis)
        negative_c_amp = []

        for i in range(N, len(_ax), 1):
            hr_i = copy.deepcopy(self)
            hrax, _ = hr_i.getax_analysis(axis)
            hr_iax = hrax[i - N : i]
            hr_i.putax_analysis(axis, hr_iax, hr_analysis)
            hr_i.distance = self.distance[i - N : i]
            hr_i.Num_sensor = len(hr_iax)

            res, f_mesh, c_mesh = hr_i.dispersion_curve(
                axis,
                freq=freq,
                c=c,
                df=df,
                dc=dc,
                method=method,
                show=False,
                zeropadding=zeropadding,
            )

            # 負の位相速度成分の抽出
            neg_col_idx = np.where(
                (c_mesh < -pick_c_range[0]) & (c_mesh > -pick_c_range[1])
            )[0]
            res_neg = res[:, neg_col_idx]
            amp_neg_c = np.abs(np.sum(res_neg)) / hr_i.Num_sensor
            negative_c_amp.append(amp_neg_c)

        ind = np.arange(N, len(_ax), 1)
        return ind, negative_c_amp

    # ------------------------------------------------------------------
    # コア計算: fft モード
    # ------------------------------------------------------------------

    def _backscatter_fft_core(self, axis, N, show_fk=False, show_fk_index=10):
        """
        f-k スペクトル解析による後方散乱振幅の計算。

        Parameters
        ----------
        axis          : データ軸 ('x' | 'y' | 'z')
        N             : スライドウィンドウのチャンネル数
        show_fk       : True のとき特定インデックスの f-k 画像を表示
        show_fk_index : f-k 表示対象のインデックス

        Returns
        -------
        ind            : np.ndarray
        reflection_amp : list
        """
        _ax, hr_analysis = self.getax_analysis(axis)
        reflection_amp = []

        for i in range(N, len(_ax), 1):
            hr_i = copy.deepcopy(self)
            hr_iax = hr_i.getax_analysis(axis)[0]
            hr_iax = hr_iax[i - N : i]
            hr_i.putax_analysis(axis, hr_iax, hr_analysis)
            hr_i.distance = self.distance[i - N : i]
            hr_i.Num_sensor = len(hr_iax)
            reflection_amp_i = 0

            distance_from_src_i = hr_i.distance

            positive_dfs = distance_from_src_i[distance_from_src_i > 0]
            negative_dfs = distance_from_src_i[distance_from_src_i < 0]

            hr_i_ax = hr_i.getax_analysis(axis)[0]
            hr_i_positive = (
                hr_i_ax[distance_from_src_i > 0] if len(positive_dfs) > 0 else None
            )
            hr_i_negative = (
                hr_i_ax[distance_from_src_i < 0] if len(negative_dfs) > 0 else None
            )

            positive_dfs = np.abs(positive_dfs) if len(positive_dfs) > 0 else None
            negative_dfs = np.abs(negative_dfs) if len(negative_dfs) > 0 else None

            if hr_i_positive is not None and len(hr_i_positive) > 1:
                sort_idx = np.argsort(positive_dfs)
                hr_i_positive = hr_i_positive[sort_idx]
                positive_dfs = positive_dfs[sort_idx]

                F2d_pos = np.fft.fftshift(np.fft.fft2(hr_i_positive))
                f_pos = np.fft.fftshift(
                    np.fft.fftfreq(hr_i_positive.shape[1], d=1 / hr_i.fs)
                )
                k_pos = np.fft.fftshift(
                    np.fft.fftfreq(hr_i_positive.shape[0], d=self.interval)
                )
                k_pos = -k_pos  # np.fft.fftfreq は正値が先頭なので符号反転

                mask_f_pos = f_pos > 0
                F2d_pos = 2 * F2d_pos[:, mask_f_pos]
                f_pos = f_pos[mask_f_pos]

                if show_fk and (i == show_fk_index):
                    self._ensure_plotter()
                    self.fk_image(
                        F2d_pos,
                        f_pos,
                        k_pos,
                        hr_i_positive,
                        positive_dfs,
                        fs=hr_i.fs,
                        title=f"Positive offset: {i}",
                        save_name=None,
                        show=True,
                    )

                neg_col_idx = np.where(k_pos < 0)[0]  #
                F2d_pos_neg = F2d_pos[:, neg_col_idx]
                reflection_amp_i += np.sum(np.abs(F2d_pos_neg))

            if hr_i_negative is not None and len(hr_i_negative) > 1:
                sort_idx = np.argsort(negative_dfs)
                hr_i_negative = hr_i_negative[sort_idx]
                negative_dfs = negative_dfs[sort_idx]

                F2d_neg = np.fft.fftshift(np.fft.fft2(hr_i_negative))
                f_neg = np.fft.fftshift(
                    np.fft.fftfreq(hr_i_negative.shape[1], d=1 / hr_i.fs)
                )
                k_neg = np.fft.fftshift(
                    np.fft.fftfreq(hr_i_negative.shape[0], d=hr_i.interval)
                )
                k_neg = -k_neg

                mask_f_neg = f_neg > 0
                F2d_neg = 2 * F2d_neg[:, mask_f_neg]
                f_neg = f_neg[mask_f_neg]

                if show_fk and (i == show_fk_index):
                    self._ensure_plotter()
                    self.fk_image(
                        F2d_neg,
                        f_neg,
                        k_neg,
                        hr_i_negative,
                        negative_dfs,
                        fs=hr_i.fs,
                        title=f"Negative offset: {i}",
                        save_name=None,
                        show=True,
                    )

                neg_col_idx = np.where(k_neg < 0)[0]
                F2d_neg_neg = F2d_neg[:, neg_col_idx]
                reflection_amp_i += np.sum(np.abs(F2d_neg_neg))

            reflection_amp.append(reflection_amp_i)

        ind = np.arange(N, len(_ax), 1)
        return ind, reflection_amp
