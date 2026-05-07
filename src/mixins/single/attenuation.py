import copy
import os

import numpy as np
from scipy.optimize import minimize_scalar

from src.plotting.wrapper import PlotterWrapperMixin


class Attenuation(PlotterWrapperMixin):
    """
    Attenuationの基底クラス
    """

    def amp(self, axis, method="fft", unit="v", f=[1, 200]):
        """
        method = 'fft' or 'energy'
        unit = 'a' or 'v' or 'u' choose unit of data.
        f = [fmin, fmax] choose frequency range
        """
        _ax, _analysis = self.getax_analysis(axis)  # (24, 16384)
        maxamplist = np.zeros(_ax.shape[0])
        res = None

        if method == "fft":
            freq = np.fft.fftfreq(_ax.shape[1], d=1 / self.fs)
            fmin_ind = np.argmin(np.abs(freq - f[0]))
            fmax_ind = np.argmin(np.abs(freq - f[1]))
            maxfreq = np.zeros(_ax.shape[0])

            for i in range(_ax.shape[0]):
                ampi = np.abs(np.fft.fft(_ax[i]))
                maxamp = np.max(ampi[fmin_ind:fmax_ind])
                maxamp_ind = np.argmax(ampi[fmin_ind:fmax_ind])
                maxfreq[i] = freq[maxamp_ind]
                maxamplist[i] = maxamp
        elif method == "energy":
            if unit == "a":
                self.integral(axis)
            elif unit == "u":
                self.differential(axis)
            else:
                raise ValueError(
                    "unitはu(変位),v(速度),a(加速度)のどれかを入力してください"
                )
            maxamplist, res = self._amp_E(_ax, f)
        return maxamplist, res

    def _amp_E(self, data, f=[1, 200]):
        fmin = f[0]
        Tmax = 1 / fmin
        Tmax_step = int(Tmax * self.fs)
        res = np.zeros((data.shape[0], data.shape[1] - Tmax_step), dtype=np.float64)
        for i in range(data.shape[0]):
            for j in range.sum((data[i, j : j + Tmax_step] / self.fs) ** 2):
                res[i, j] = data.shape[1] - Tmax_step
        maxamplist = np.max(res, axis=1)
        return maxamplist, res

    def attenuation_analysis(
        self,
        axis,
        nd=2,
        dB=False,
        freq=[None, None],
        velocity=None,
        velocity_estimation=False,
        noise_level=None,
        show=False,
        save_name=None,
        overflow_value=8388607,
    ):
        """
        振幅の減衰を解析する
        振幅の取り方: パワースペクトルによるfftの振幅を使用する。(unit^2の単位)
        ・震源からの距離
        ・各chのfft abs
        近似曲線としての幾何減衰定数と内部減衰定数を求める

        est_nd, est_mu

        Attenuation = R^(-nd) * exp(-mu * k) = R(-nd) * exp(-mu * R*f/v) の形で近似する。
        Q = Q:float [ramda/%] -> 2π/Q = ΔE/E = 1 - 10^a として計算されている。

        parameter:
                axis: 'x' or 'y' or 'z'
                velocity: 伝播速度 : デフォルト:None -> None の場合は 波数を考慮せずに解析する
                freq = [fmin, fmax]: 解析する周波数範囲. デフォルトは[len(data)/fs, fs/2]
                noise_level: ノイズレベルの推定値(デフォルトはNone) -> 二乗平均(分散)を用いる

                nd: float 幾何減衰定数(R^-nd)の導入 デフォルトは2(実体波を想定。表面波の場合はn=0.5)
                dB: bool(default:False) 結果をdB表示にするかどうか
                show: bool plt.show() を呼ぶか
                save_name: str|Path|None
                    画像保存先パス (拡張子込み)。None / '' のとき画像も Excel も保存しない。
                    画像はサフィックス付与で複数枚 (allfreq / {freq}Hz / {freq}Hz_damp_ratio) として、
                    Excel は同 stem の .xlsx として書き出される。

                velocity_estimation: 速度推定を行うかどうか
                        true: 速度推定を分散曲線(dispersion_curve, method = 'fft')によって行い、そのピークを推定位相速度として利用する

        ##追記
        velocity推定は dispersion curveのcc法によって出すのでも良いかも(位相速度基準の場合)

        """
        from pathlib import Path

        _ax, _analysis = self.getax_analysis(axis)

        # 震源からの距離: distance_from_src
        distance_from_src = copy.deepcopy(self.distance)
        distance_from_src += -copy.deepcopy(self.source_x)
        distance_from_src = np.abs(distance_from_src)

        # オーバーフローしているばあい、_ax[i]と対応するdを除外する
        ax_temp = _ax
        distance_from_src_temp = distance_from_src
        overflow_index = []
        chs = np.arange(len(_ax))
        for i in range(len(_ax)):
            if np.max(np.abs(_ax[i])) > overflow_value:  # オーバーフローしている場合
                overflow_index.append(i)
        ax_temp = np.delete(ax_temp, overflow_index, axis=0)
        distance_from_src_temp = np.delete(distance_from_src_temp, overflow_index)
        chs = np.delete(chs, overflow_index)

        # noise_level が設定されてない時、最期の500サンプル分のデータを用いて行う
        ax_noise = _ax[:, -500:]
        if noise_level is None:
            noise_level = np.var(ax_noise)

        # 各chのfft abs と　freq index を取得
        amp_fft = np.abs(np.fft.fft(ax_temp, axis=1)) ** 2
        noise_fft = np.abs(np.fft.fft(ax_noise, axis=1, n=_ax.shape[1])) ** 2
        freq_fft = np.linspace(0, self.fs, ax_temp.shape[1])

        # fmin<f<fmaxのインデックスを取得
        if freq == [None, None]:
            freq = [
                len(_ax[0]) / self.fs,
                self.fs / 2,
            ]  # fftの最小種波数とナイキスト周波数
        fmin = freq[0]
        fmax = freq[1]
        fmin_idx = np.argmin(np.abs(freq_fft - fmin))
        fmax_idx = np.argmin(np.abs(freq_fft - fmax))

        amp_fft = amp_fft[:, fmin_idx:fmax_idx]
        noise_fft = noise_fft[:, fmin_idx:fmax_idx]
        freq_fft = freq_fft[fmin_idx:fmax_idx]

        # 速度推定を行う場合,分散曲線から速度を推定する
        if velocity_estimation:
            res, f_mesh, c_mesh = self.dispersion_curve(
                axis, method="fft", freq=freq, c=[0, 500], show=False
            )
            estimate_velocity_array = np.zeros_like(freq_fft, dtype=np.float64)

        # 幾何減衰定数、内部減衰定数、Q値の結果を格納するリストを初期化
        est_nd_array = np.zeros_like(freq_fft, dtype=np.float64)
        est_a_array = np.zeros_like(freq_fft, dtype=np.float64)
        Q_array = np.zeros_like(freq_fft, dtype=np.float64)
        attenuation_matrix = np.zeros(
            (len(freq_fft) + 1, len(distance_from_src_temp) + 1)
        )  # 最初の列はfftではない

        energy = np.sum((ax_temp / np.sqrt(ax_temp.shape[1])) ** 2, axis=1)
        attenuation_matrix[0, :-1] = energy
        noise_mean = np.mean(
            np.sum((ax_noise / np.sqrt(ax_noise.shape[1])) ** 2, axis=1)
        )
        attenuation_matrix[0, len(distance_from_src_temp)] = noise_mean

        per_freq_data = []

        for j, freq in enumerate(freq_fft):
            # print(f'freq: {freq}Hz')
            amp_atfreq = amp_fft[:, j]  # (len(data),)

            if velocity_estimation:  # 速度推定を行う場合:　該当周波数でのピーク位相速度を推定速度として利用する
                f_mesh_idx = np.argmin(np.abs(f_mesh - freq))
                velocity = c_mesh[np.argmax(res[f_mesh_idx])]
                estimate_velocity_array[j] = velocity

            attenuation = np.zeros(len(distance_from_src_temp))
            for i in range(len(distance_from_src_temp)):
                attenuation[i] = amp_atfreq[i] / amp_atfreq[0]
            noise_mean_j = np.mean(noise_fft[:, j] / amp_atfreq[0])
            attenuation_matrix[j + 1, :-1] = attenuation
            attenuation_matrix[j + 1, len(distance_from_src_temp)] = noise_mean_j

            est_nd, est_a = self.get_geometatt_internalatt_dis(
                attenuation, distance_from_src_temp, velocity=velocity, freq=freq
            )

            Q_value, distance_ax, geo_att, a = self._geo_inter_att(
                freq,
                attenuation,
                distance_from_src_temp,
                est_nd,
                velocity,
                dB,
            )

            per_freq_data.append(
                {
                    "attenuation": attenuation,
                    "noise_mean": noise_mean_j,
                    "freq": freq,
                    "distance_ax": distance_ax,
                    "geo_att": geo_att,
                    "a": a,
                    "dB": dB,
                    "velocity": velocity,
                }
            )

            est_nd_array[j] = est_nd
            est_a_array[j] = est_a
            Q_array[j] = Q_value

        self.attenuation(
            distance_from_src_temp,
            energy,
            noise_mean,
            per_freq_data,
            name=getattr(self, "name", ""),
            show=show,
            save_name=save_name,
        )

        if save_name:
            import pandas as pd

            print("save excel file...")
            attenuation_columns = [f"{f}" for f in freq_fft]
            attenuation_columns.insert(0, "S[ch]^2")
            attenuation_matrix_t = attenuation_matrix.T
            sheet1_df = pd.DataFrame(attenuation_matrix_t, columns=attenuation_columns)

            distance_for_sheet = np.append(distance_from_src_temp, 100)
            sheet1_df.insert(0, "distance", distance_for_sheet)

            sheet2_data = {
                "freq[Hz]": freq_fft,
                "Q_value[%/λ]": Q_array,
                "est_nd": est_nd_array,
                "est_mu[]": est_a_array,
            }
            if velocity_estimation:
                sheet2_data["est_velocity"] = estimate_velocity_array

            sheet2_df = pd.DataFrame(sheet2_data)

            # save_name は画像用フルパス。Excel は同 stem の .xlsx に書き換える。
            excel_path = Path(save_name).with_suffix(".xlsx")
            self._export_excel(
                sheet1_df,
                sheet2_df,
                save_dir=str(excel_path.parent),
                save_name=excel_path.stem,
            )

    # -----------------------------------------------------------------
    # Excel export private helper
    # -----------------------------------------------------------------

    def _export_excel(
        self,
        sheet1_df,
        sheet2_df,
        *,
        save_dir: str | None = None,
        save_name: str | None = None,
    ) -> str:
        """減衰解析結果を Excel ファイルに書き出す。"""
        try:
            import openpyxl  # noqa: F401  (ExcelWriter の engine として使用)
            import pandas as pd
        except ImportError:
            raise ImportError("Excel出力には pandas と openpyxl が必要です。\n")

        save_dir = save_dir or ""
        save_name = save_name or getattr(self, "name", "attenuation_res")

        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        filepath = os.path.join(save_dir, save_name + ".xlsx")
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
            sheet1_df.to_excel(writer, sheet_name="Sheet1", index=False)
            sheet2_df.to_excel(writer, sheet_name="Sheet2", index=False)

        print(f"saved: {filepath}")
        return filepath

    # -----------------------------------------------------------------

    def get_geometatt_internalatt_dis(
        self, da, dl, velocity=None, freq=None, bounds_n=(0.0, 5.0)
    ):
        """
        幾何減衰定数 n を探索的に決定し、それに基づき内部減衰定数 mu を推定する。
        Simultaneous optimization の不安定性を解消するための分離アプローチ。

        2025/12/14
        Model: ln(A) = -n * ln(r) - mu * r + C
        変形:  ln(A) + n * ln(r) = -mu * r + C
                ~~~~~ Y ~~~~~~~~   ~~ slope ~~

        Parameters
        ----------
        da : array
                Attenuation (振幅)
        dl : array
                Distance (距離)
        bounds_n : tuple, optional
                幾何減衰定数 n の探索範囲 (min, max). デフォルトは (0.0, 2.0).
                実体波(1.0)や表面波(0.5)を含む範囲に設定。

        Returns
        -------
        n_estimated : float
        mu_estimated : float
        """
        # 0や無効値の除外
        valid_idx = (dl > 0) & (da > 0)
        da = da[valid_idx]
        dl = dl[valid_idx]

        # 正規化（基準化）
        r0 = dl[0]
        A0 = da[0]

        # 計算用配列
        # Bornitz: A = A0 * (r/r0)^-n * exp(-mu*(r-r0))
        # 対数化: ln(A/A0) = -n * ln(r/r0) - mu * (r-r0)

        log_amp_ratio = np.log(da / A0)
        log_dist_ratio = np.log(dl / r0)
        delta_r = dl - r0

        # --- 目的関数: 任意の n についての線形回帰のフィッティング誤差 ---
        def objective_func(n_val):
            # 幾何減衰分を補正した振幅の対数ー＞対数上では直線として近似する
            ## Y = ln(A/A0) + n * ln(r/r0) = -mu * (r-r0)
            y_corrected = log_amp_ratio + n_val * log_dist_ratio

            # 原点を通る回帰 (Y = aX) , X = delta_r, slope a = -mu
            # np.linalg.lstsq または dot積で slope を計算
            # slope = sum(x*y) / sum(x^2)
            slope = np.dot(delta_r, y_corrected) / np.dot(delta_r, delta_r)

            # mu は減衰なので正の値であることを期待するが、ここでは自由な slope を許容
            # 予測値
            y_pred = slope * delta_r

            # 残差二乗和 (Residual Sum of Squares)
            rss = np.sum((y_corrected - y_pred) ** 2)
            return rss

        # --- n の最適化 (Bounded Optimization) ---
        # bounds_n の範囲内で、最も直線回帰の誤差が小さくなる n を探す
        res = minimize_scalar(objective_func, bounds=bounds_n, method="bounded")

        n_estimated = res.x

        # --- 最適な n を用いて mu を確定 ---
        y_final = log_amp_ratio + n_estimated * log_dist_ratio
        slope_final = np.dot(delta_r, y_final) / np.dot(delta_r, delta_r)

        # slope = -mu なので mu = -slope
        mu_estimated = -slope_final

        # --- 単位変換 (必要な場合) ---
        if velocity is not None and freq is not None:
            # mu [1/m] -> loss factor eta (または 1/Q 関連)
            # ここでは元のコードに従い、k で割って無次元化などを実施
            k = 2 * np.pi * freq / velocity
            mu_estimated = mu_estimated / k

        return n_estimated, mu_estimated

    def _geo_inter_att(self, freq, attenuation, distance, nd, velocity, dB):
        """
        幾何減衰を考慮して内部減衰をQ値で計算する。
        return (Q_value, distance_ax, geo_att, a)

        Q:float [ramda/%] -> 2π/Q = ΔE/E = 1 - 10^a
        Q値が大きいほど減衰が小さいことを示す
        """
        distance_ax = distance
        if velocity is not None:
            print(f"弾性波速度{velocity}m/s")
            distance_ax = distance * freq / velocity

        geo_att = (distance**nd) * attenuation

        geo_att = geo_att[distance_ax != 0]
        distance_ax = distance_ax[distance_ax != 0]

        if dB:
            geo_att = 20 * np.log10(geo_att)

        inclination = np.dot(distance_ax, np.log(geo_att)) / (distance_ax**2).sum()
        a = inclination * (-1 / 8.68) if dB else inclination
        Q_value = 2 * np.pi / (1 - np.exp(inclination))

        return Q_value, distance_ax, geo_att, a

    def _estimate_geo_inter_att(self, attenuation, distance, show):
        x1 = np.sum(np.exp(distance) * attenuation) * np.sum(
            np.exp(distance) * distance
        ) - np.sum(np.exp(distance) * np.exp(distance)) * np.sum(
            attenuation * distance
        ) / (
            np.sum(distance * distance) * np.sum(np.exp(distance) * np.exp(distance))
            - np.sum(distance * np.exp(distance)) * sigma(distance * np.exp(distance))
        )
        x2 = np.sum(attenuation * distance) * np.sum(
            np.exp(distance) * distance
        ) - np.sum(distance * distance) * np.sum(attenuation * np.exp(attenuation)) / (
            np.sum(distance * distance) * np.sum(np.exp(distance) * np.exp(distance))
            - np.sum(distance * np.exp(distance)) * np.sum(distance * np.exp(distance))
        )
        if show:
            print(
                "近似曲線: 幾何減衰定数 n = "
                + str(x1)
                + ", 内部減衰定数 η = "
                + str(x2)
            )
        return x1, x2
