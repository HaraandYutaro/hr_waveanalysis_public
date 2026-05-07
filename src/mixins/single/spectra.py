
import numpy as np

from src.plotting.wrapper import PlotterWrapperMixin


class Spectra(PlotterWrapperMixin):
    def spectra(
        self,
        axis: str,
        freq: list[float] = [1, 200],
        show: bool = True,
        save_name: str | None = None,
        log: bool = False,
        transfunc=False,
        zeropadding=0,
        **kw,
    ):
        """
        axis: 'x', 'y', 'z'
        各センサーの周波数スペクトルを表示する。
        return : array of amplitude (Num_sensor, N), freq_axis

        parameters:
                axis: 'x', 'y', 'z'
                freq: list of two elements [fmin, fmax] or [fmin, fmax, fstep]
                show: bool, plt.show() を呼ぶか
                save_name: str|Path|None, 保存先パス。None / '' のとき保存しない
                log: bool, if True, use log scale for amplitude
                cmap: str, colormap for the figure
                interpolation: str, interpolation method for the image
                title: str, title of the figure
                transfunc: bool, if True, normalize the spectra by the source channel
                zeropadding: int, add zero to ax which conducts FFT, so that spacing of discrete fft range[Hz] shrinks and results are mode deteiled.
        """

        res, freq_axis, f, _analysis = self._spectra_core(
            axis, freq, transfunc, zeropadding
        )

        # 描画
        if show or save_name:
            name = self.name
            analysis = _analysis

            self.spectra_image(
                res,
                freq=f,
                axis=axis,
                name=name,
                analysis=analysis,
                show=show,
                save_name=save_name,
                log=log,
                **kw,
            )
        return res, freq_axis

    def _spectra_core(self, axis, freq, transfunc, zeropadding):
        """
        spectra() の純粋計算部 (描画なし)。

        Returns
        -------
        res        : np.ndarray, shape (N_freq, N_sensor)
        freq_axis  : np.ndarray, 周波数軸
        f          : list[float], クリップ後の [fmin, fmax]
        _analysis  : str, 解析履歴文字列
        """
        fmin, fmax = sorted(freq[:2])
        fmin = 0 if fmin < 0 else fmin
        fmax = self.fs / 2 if fmax > self.fs / 2 else fmax
        f = [fmin, fmax]

        # spectra
        _ax, _analysis = self.getax_analysis(axis)

        # zeropadding _ax
        if zeropadding != 0:
            padding = np.zeros((len(_ax), zeropadding))
            _ax = np.concatenate([_ax, padding], axis=1)

        for i in range(len(_ax)):
            axi = _ax[i]
            freq_axis = np.linspace(0, self.fs, len(axi))
            fft = np.fft.fft(axi)[(freq_axis > f[0]) & (freq_axis < f[1])]
            freq_axis = freq_axis[(freq_axis > f[0]) & (freq_axis < f[1])]
            amp = np.abs(fft)
            if i == 0:
                fftax = np.zeros((len(_ax), len(amp)))
            fftax[i] = amp

        if transfunc:
            src_ch = self.get_source_ch()
            fftax = fftax / fftax[src_ch]

        res = fftax.T  # (N_freq, N_sensor)

        return res, freq_axis, f, _analysis
