import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class backscatter_distribution(PlotterWrapperMixin):
    def backscatter_distribution(
        self,
        axis: str = "y",
        N: int = 6,
        mode: str = "fft",
        freq: list | None = None,
        c: list | None = None,
        df: float = 1.0,
        dc: float = 1.0,
        method: str = "fft",
        zeropadding: int = 0,
        pick_c_range: list | None = None,
        show: bool = True,
        save_name: str | None = None,
        **kw,
    ):
        """
        一つの探査側線で得られたデータに対し、各計測値点での後方散乱分布を平均化する

        for sp in splist
            sp.distance: センサーのx地点が格納されたデータ
            sp.backscatter() -> (ind, reflection_amp)
              reflection_amp[i] は sp.distance[ind[i]] 位置における後方散乱振幅

        Returns
        -------
        averaged_amp_distribution : np.ndarray
            self.all_distance の各位置に対応する平均振幅
        all_distance : np.ndarray
            self.all_distance (ユニークな距離値の昇順配列)
        """
        averaged_amp_distribution, all_distance, count_distribution = (
            self._backscatter_distribution_core(
                axis,
                N,
                mode,
                freq,
                c,
                df,
                dc,
                method,
                zeropadding,
                pick_c_range,
            )
        )

        if show or save_name:
            self._ensure_plotter()
            self.backscatter_distribution_image(
                all_distance,
                averaged_amp_distribution,
                count=count_distribution,
                title=f"backscatter_distribution (N={N}, mode={mode})",
                ylabel="averaged amplitude",
                save_name=save_name,
                show=show,
                **kw,
            )

        return averaged_amp_distribution, all_distance

    def _backscatter_distribution_core(
        self,
        axis,
        N,
        mode,
        freq,
        c,
        df,
        dc,
        method,
        zeropadding,
        pick_c_range,
    ):
        """
        backscatter_distribution() の純粋計算部 (描画なし)。

        Returns
        -------
        averaged_amp_distribution : np.ndarray
            self.all_distance の各位置に対応する平均振幅
        all_distance : np.ndarray
            ユニークな距離値の昇順配列 (float キャスト済み)
        count_distribution : np.ndarray
            各距離位置での累積カウント (描画用)
        """
        splist = self.datalist  # SingleProcesser インスタンスのリスト
        all_distance = np.asarray(self.all_distance, dtype=float)

        amp_distribution = np.zeros_like(all_distance, dtype=float)
        count_distribution = np.zeros_like(all_distance, dtype=int)

        for sp in splist:
            ind, reflection_amp = sp.backscatter(
                axis=axis,
                mode=mode,
                N=N,
                freq=freq,
                c=c,
                df=df,
                dc=dc,
                method=method,
                zeropadding=zeropadding,
                pick_c_range=pick_c_range,
                show=False,
                show_fk=False,
            )

            sp_distance = np.asarray(sp.distance, dtype=float)

            for i in range(len(reflection_amp)):
                pos = sp_distance[ind[i]]
                j = int(np.argmin(np.abs(all_distance - pos)))
                amp_distribution[j] += reflection_amp[i]
                count_distribution[j] += 1

        averaged_amp_distribution = np.zeros_like(amp_distribution, dtype=float)
        mask = count_distribution > 0
        averaged_amp_distribution[mask] = (
            amp_distribution[mask] / count_distribution[mask]
        )

        return averaged_amp_distribution, all_distance, count_distribution
