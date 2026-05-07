
import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class FFT2D(PlotterWrapperMixin):
    """
    2D FFTの基底クラス
    """

    def fk(
        self,
        axis,
        freq=[0, 200],
        show=True,
        save_name: str | None = None,
        **kw,
    ):
        """
        f-k imaging by fft2d

        Parameters
        ----------
        save_name : str | Path | None
            保存先パス。None / '' のとき保存しない。
        """
        res, kxmin, kxmax, fmin, fmax, analysis = self._fk_core(axis, freq)

        if show or save_name:
            self.fk_image(
                axis,
                res,
                kxmin,
                kxmax,
                fmin,
                fmax,
                analysis,
                show=show,
                save_name=save_name,
                **kw,
            )

    def _fk_core(self, axis, freq):
        """
        fk() の純粋計算部 (描画なし)。

        Returns
        -------
        res     : np.ndarray, f-k 切り出し後の振幅 (周波数 x 波数)
        kxmin   : float, 表示用波数下限
        kxmax   : float, 表示用波数上限
        fmin    : float, クリップ後の周波数下限
        fmax    : float, クリップ後の周波数上限
        analysis: str, 解析履歴文字列
        """
        ax, analysis = self.getax_analysis(axis)
        freq = np.sort(freq)
        fmin, fmax = max(freq[0], 0), min(freq[1], self.fs / 2)
        kxmax, kxmin = 1 / self.interval / 2, -1 / self.interval / 2
        fxt = np.fft.fft2(ax)
        fxt = np.fft.fftshift(fxt)
        res = fxt.T  # 結果を転置
        freq = np.fft.fftfreq(ax.shape[1], d=1 / self.fs)
        kx = np.fft.fftfreq(ax.shape[0], d=self.interval)
        freq = np.fft.fftshift(freq)
        kx = np.fft.fftshift(kx)

        # resのf<0の部分を切り取る
        freq_indices = np.where((freq >= fmin) & (freq <= fmax))[0]
        freq_indices = freq_indices[::-1]
        freq = freq[freq_indices]
        kx_indices = np.where(
            (kx >= -1 / self.interval / 2) & (kx <= 1 / self.interval / 2)
        )[0]
        res = res[np.ix_(freq_indices, kx_indices)]

        return res, kxmin, kxmax, fmin, fmax, analysis
