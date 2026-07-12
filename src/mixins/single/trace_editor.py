"""
トレースエディター

"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from scipy import signal
from scipy.fft import next_fast_len

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.utils import utils


class TraceEditor:
    def _pickup(self, axis, ch_form, ch_to):
        """
        axis:('x','y','z')
        ch_from:int -> 取り出すstart ch
        ch_to:int -> end ch

        :return: _ax, _analysis, distance, Num_sensor, sensor1_x
        _ax: array(int32) axis data
        _analysis: str analysis message
        distance: array(float32) x placements array of each sensors [m]
        Num_sensor: int32, number of sensors
        sensor1_x: float32, 1st sensor [m]
        """
        _ax, _analysis = self.getax_analysis(axis)
        chs = np.arange(ch_form, ch_to + 1)  # chsは1-indexed
        idx = np.asarray(chs, dtype=int) - 1  # ← ここが“for を無くす”ポイント
        distance = self.distance[idx]  # 高度インデックスでそのまま抽出
        new_ax = axis[idx]  # axis == _ax と同じ扱い
        Num_sensor = len(chs)
        sensor1_x = self.distance[chs[0]]
        _analysis += f"_pickup(ch{ch_form}-{ch_to})"
        return _ax, _analysis, distance, Num_sensor, sensor1_x

    def remove(self, axis, remove_chs):
        """
        :param axis: 'x', 'y', 'z'
        :param remove_chs: list of int, 1-indexed channel numbers to remove, ex:[12,78]-> 13ch, 79chが消える
        """
        _ax, _analysis = self.getax_analysis(axis)
        
        # remove_chsに-1以下が入っている場合、0からにする
        if any(ch < 0 for ch in remove_chs):
            remove_chs = [ch if ch >= 0 else 0 for ch in remove_chs]
        # _ax.shape[0] と　remove_chsの値を照らし合わせて、remove_chsの値が_axのチャンネル数を超えている場合、warningを出して_axのチャンネル数に収まるようにremove_chsを修正する
        if max(remove_chs) >= _ax.shape[0]:
            print(
                f"Warning: remove_chs contains channel numbers that exceed the number of channels in _ax. Adjusting remove_chs to fit the valid range."
            )
            remove_chs = [ch for ch in remove_chs if ch < _ax.shape[0]]
            if not remove_chs:
                print("No valid channels to remove after adjustment. Exiting remove function.")
                return
        # 重複を削除してソートする
        remove_chs = list(set(remove_chs))
        remove_chs.sort()

        chs = np.asarray(remove_chs, dtype=int)
        _ax = np.delete(_ax, chs, axis=0)
        self.distance = np.delete(self.distance, chs, axis=0)  # Remove corresponding distances
        if hasattr(self, "x_distance"):
            self.x_distance = np.delete(self.x_distance, chs, axis=0)
        else:
            pass
        if hasattr(self, "z_distance"):
            self.z_distance = np.delete(self.z_distance, chs, axis=0)
        else:
            pass
        self.Num_sensor = len(self.distance)
        self.putax_analysis(axis, _ax, _analysis + f"_rm({remove_chs})")

    def sort_distance(self):
        """
        距離の昇順に並び替える
        self.distanceを基準にx, y, zを並び替える
        """

        def has_valid_values(x) -> bool:
            """
            x が None           → False
            x が 0 次元で None   → False
            それ以外            → True
            """
            if x is None:
                return False

            if isinstance(x, np.ndarray):  # ndarray かどうかを念のため確認
                # 0-D 配列 (shape == ()) で要素が None
                if x.shape == () and x.dtype == object and x.item() is None:
                    return False
                # それ以外（N_ch × N_timesteps など）
                return True

            return True

        sorted_indices = np.argsort(self.distance)
        if hasattr(self, "x") and has_valid_values(self.x):
            self.x = self.x[sorted_indices]
            self.analysis_x += "_sorted"
        if hasattr(self, "y") and has_valid_values(self.y):
            self.y = self.y[sorted_indices]
            self.analysis_y += "_sorted"
        if hasattr(self, "z") and has_valid_values(self.z):
            self.z = self.z[sorted_indices]
            self.analysis_z += "_sorted"
        self.distance = self.distance[sorted_indices]

    def save(self, dir, savename):
        """
        npzファイルとして保存
        :param dir: 保存先ディレクトリ
        :param savename: 保存ファイル名(拡張子なし)
        """
        # 属性を辞書として収集
        # npz として保存できる型のみを対象とする。None / bool / 数値スカラー
        # (fs, Num_sensor, source_x, metadata_source, metadata_is_fallback 等) も
        # 含める。np.savez_compressed は内部で np.asanyarray を呼ぶため、これらは
        # 0次元配列として保存され、allow_pickle=True で読み戻せる。
        saveable = (
            np.ndarray,
            np.generic,
            list,
            tuple,
            str,
            bytes,
            int,
            float,
            bool,
            type(None),
        )
        data_dict = {}
        for key, value in self.__dict__.items():
            # 描画プロッタなどアンダースコア始まりの実行時オブジェクト
            # (例: _plotter) は保存対象外
            if key.startswith("_"):
                continue
            if isinstance(value, saveable):
                data_dict[key] = value
            else:
                raise ValueError(
                    f"Unsupported data type for attribute '{key}': {type(value).__name__}"
                )

        # ディレクトリが存在しない場合は作成
        if not os.path.exists(dir):
            os.makedirs(dir)

        # 辞書を.npzファイルとして保存
        filepath = os.path.join(dir, savename + ".npz")
        # 辞書を.npzファイルとして保存
        np.savez_compressed(filepath, **data_dict)

    def getax_analysis(self, axis):
        """
        :param axis:'x', 'y', 'z'
        :return: _ax, _analysis

        if self.x, self.y, self.z is defined
                return self.x, self.analysis_x
        if not defined, and ax1, ax2 is defined -> prototype
                return self.ax1 (if axis == z), self.analysis_ax2 (if axis == x,y)
        if not defined, and ax1, ax2 is not defined
                return self.data, ''
        if not defined, and ax1 is defined, ax2 is not defined
                raise ValueError
        """

        if hasattr(self, axis):
            _ax = getattr(self, axis)
        elif hasattr(self, "ax1") or hasattr(self, "ax2"):
            _ax = self.ax1 if axis == "z" else self.ax2
        elif hasattr(self, "data"):
            print("dataを返します")
            _ax = self.data
        else:
            raise ValueError("x,y,z,ax1,ax2,dataのいずれも定義されていません。")

        if hasattr(self, "analysis_" + axis):
            _analysis = getattr(self, "analysis_" + axis)
        elif hasattr(self, "analysis_ax1") or hasattr(self, "analysis_ax2"):
            _analysis = self.analysis_ax1 if axis == "z" else self.analysis_ax2
        elif hasattr(self, "data"):
            _analysis = ""
        else:
            raise ValueError(
                "x,y,z,ax1,ax2,dataのいずれも定義されていません(このエラーAnalysisの部分で確認されました。)"
            )

        return _ax, _analysis

    def putax_analysis(self, axis, _ax, _analysis, add: str = ""):
        """
        axis: 'x', 'y', 'z'
        _ax: array(int32) axis data
        _analysis: str analysis message
        add: str, 追加のメッセージ
        """

        if axis == "x":  # z方向のデータなら_axはax1, analysis_ax1に追加
            self.x = _ax
            self.analysis_x = _analysis + add
        elif axis == "y":
            self.y = _ax
            self.analysis_y = _analysis + add
        elif axis == "z":
            self.z = _ax
            self.analysis_z = _analysis + add
        else:
            ValueError("axisにはx,y,zのいずれかを入れてください。(格納時)")

    def trace_amp_regularize(self, axis, amp=2**24):
        """
        振幅を正規化する
        :param axis: 'x', 'y', 'z'
        :param amp: int, 振幅の最大値 default 2**24
        :return: None
        """
        _ax, _analysis = self.getax_analysis(axis)
        scale = amp / np.max(np.abs(_ax), axis=1, keepdims=True)  # shape: (num_ch, 1)
        regularized_ax = (scale * _ax).astype(_ax.dtype)  # 振幅を正規化
        self.putax_analysis(axis, regularized_ax, _analysis, add="_amp_reg")

    def timepick(self, axis, t_from, t_to):
        _ax, _analysis = self.getax_analysis(axis)
        fs = self.fs
        _ax = np.delete(_ax, range(int(t_to * fs), _ax.shape[1]), 1)
        _ax = np.delete(_ax, range(int(t_from * fs)), 1)
        t_from = np.round(t_from, 3)
        _analysis += f"_picktime({t_from}"
        self.putax_analysis(axis, _ax, _analysis)

    ##deconvolution##
    def get_source_ch(self):
        """
        起振点に一番近いチャンネルインデックスをリスト型で返す
        :return: source_ch

        ~~例~~
        側線方向の座標 x: 0---1---2---3---4---5---6---7---8---9---10--11--12--13--14--15--16--17--18--19--20--21--[m]
        センサーの位置 R:     o   o   o   o   o   o   o   o   o   o   o   o   o   o   o   o   o   o
        起振震源の位置 S:                         ☆

        source_x = 6.0 [m] #起振点のx位置
        distance = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0] #各センサーの位置を示したアレイ
        -> source_ch = [5] (=np.min(np.abs(distance - source_x)))

        ***TODO:リスト型からarray型に変換して返す様にしたい(20241027)
                -> get_source_ch() を使っているところを今後修正したい
        """
        nearest_index = np.argmin(np.abs(np.array(self.distance) - self.source_x))
        return nearest_index

    def _wavelet(self, axis, n, window, show, ch=99):
        _ax, _analysis = self.getax_analysis(axis)
        fs = self.fs
        t_to = _ax.shape[1] / fs

        ###基本波形の抽出##############################
        if ch == 99:
            # chの指定が無ければ、起振店近傍のchを特定
            source_ch = self.get_source_ch()
            source = np.array(_ax[source_ch])
        else:
            source = np.array(_ax[ch - 1])

        # 予め起振データの範囲を指定
        wavelett_to = 30.0
        if wavelett_to > t_to:
            wavelett_to = t_to

        # 最大振幅値を取得
        Maxamp, Maxind = (
            np.max(np.abs(source[: int(fs * wavelett_to)])),
            np.abs(source[: int(fs * wavelett_to)]).argmax(),
        )
        # 閾値の設定
        boundary = Maxamp / n
        # 閾値を超えた範囲(waveletになる候補エリア)のインデックスを取得
        indlist = np.where(abs(source) > boundary)
        ind_min, ind_max = np.min(indlist), np.max(indlist)
        # 閾値範囲と最大値をとる個所から、ウェーブレットの方側をどれくらいとるか決める
        w_side = min(Maxind - ind_min, ind_max - Maxind)
        # ウェーブレットの決定(真ん中を最大値に設定している)
        wavelet = source[Maxind - w_side : Maxind + w_side + 1]
        # 奇数でとっているが。。。念のためwaveletは奇数にする
        if len(wavelet) % 2 == 0:
            print("wavelet size is not odd!! I modify...")
            wavelet = np.delete(wavelet, -1)
        # 最大値が1なるように正規化
        wavelet = np.float64(wavelet / np.max(np.abs(wavelet)))
        if source[Maxind] < 0:
            wavelet *= -1
        # 窓関数処理　あった方がFFTをかけたときに安定する
        if window == "hanning":
            wavelet = np.hanning(len(wavelet)) * wavelet
        elif window == "blackman":
            wavelet = np.blackman(len(wavelet)) * wavelet
        elif window == "hamming":
            wavelet = np.hamming(len(wavelet)) * wavelet
        else:
            pass
        return wavelet

    def window(self, wavelet: np.array, key: str) -> np.array:
        """
        window関数
        wavelet: np.array, waveletのデータ
        key: str, 'hanning', 'blackman', 'hamming'
        return: np.array, window関数を適用したwaveletデータ
        """
        if key == "hanning":
            return np.hanning(len(wavelet)) * wavelet
        elif key == "blackman":
            return np.blackman(len(wavelet)) * wavelet
        elif key == "hamming":
            return np.hamming(len(wavelet)) * wavelet
        else:
            raise ValueError(
                f"Unknown window type: {key}. Use 'hanning', 'blackman', or 'hamming'."
            )

    def deconvolution_fft(
        self,
        axis,
        n=20,
        show=False,
        window="hanning",
        key="conj",
        ramda=1e-8,
        ch=99,
        wavelet=None,
        regularize=False,
    ):
        """
        fftを用いたデコンボリューション処理
        引数
        axis
        wavelet:None

        """
        wavelet_window = 0
        if wavelet is None:
            wavelet = self._wavelet(axis, n, window, show, ch)
            wavelet_window = 1

        wavelet = self.window(wavelet, window) if not wavelet_window else wavelet

        if wavelet.shape[0] == 1:  # wavelet shape = (1,M)
            wavelet = wavelet[0]

        if show == True:
            plt.plot(wavelet)
            plt.title("wavelet")
            plt.show()

        if regularize:
            wavelet = np.float64(wavelet)
            wavelet /= np.max(np.abs(wavelet))

        ax_deconv_ar, _analysis = self._freqdeconv(axis, wavelet, key, ramda, window)
        self.putax_analysis(axis, ax_deconv_ar, _analysis)

    def _freqdeconv(self, axis, wavelet, key="conj", ramda=1e-8, window="hanning"):
        _ax, _analysis = self.getax_analysis(axis)
        if wavelet.shape[0] != 1:
            side = (wavelet.shape[0] - 1) // 2  # 499
        else:
            side = (wavelet.shape[1] - 1) // 2  # 499
        N = _ax.shape[1]
        w_respond = np.zeros(N)
        for j in range(side):  # 0,1,2,3,...498
            if wavelet.shape[0] != 1:
                w_respond[j] = wavelet[side - j]
                w_respond[N - side + 1 :] = wavelet[1:side][::1]
            else:
                w_respond[j] = wavelet[:, j + side]
                w_respond[N - side + 1 :] = wavelet[0, 1:side][::1]

        ## deconvolution in freq domain
        hF = np.fft.fft(w_respond)
        w_respond[0] *= 0
        w_respond[-1] *= 0

        for i in range(len(_ax)):
            nF = np.fft.fft(_ax[i])
            if key == "conj":
                fnF = np.conj(hF) * nF / (np.conj(hF) * hF + ramda)
                ifft_sig = np.fft.ifft(fnF)
                ninv = np.real(ifft_sig)
            elif key == "scipy":
                # use scipy signal deconvolve
                import scipy.signal

                ninv, _ = scipy.signal.deconvolve(nF, hF)
            else:
                fnF = nF / (hF + ramda)  # ramdaは０で割り算するのを防ぐ微小な浮動点少数
                ifft_sig = np.fft.ifft(fnF)
                ninv = np.real(ifft_sig)

            if i == 0:  # 初回のみ initialize
                ax_deconv_ar = np.zeros((self.Num_sensor, ninv.shape[0]))

            ax_deconv_ar[i] = ninv
        _analysis += f"_deconv({key})"
        return ax_deconv_ar, _analysis

    def deconvolution_pred(self, axis, wavelet=None, n=128, alpha=8):
        """
        predictable deconvolution

        Parameters
                axis: str, 'x', 'y', 'z'
                n: int, number of operators, default 128
                alpha: int, prediction order, default 8

        Return
                None, but store the result in the analysis axis

        Reference: Deconvolution filter in seismic data processing
                           Sekiyu Gakkai, 31, (6), pp439-447, 1988
        """
        from scipy.linalg import solve_toeplitz
        from scipy.signal import lfilter

        def autocorr(x, order):
            n = len(x)
            return np.correlate(x, x, mode="full")[n - 1 : n + order] / n

        def predictive_decon(x, n, alpha):
            """
            Predictive deconvolution (Levinson–Durbin + Toeplitz solver)
            n     : オペレータ長 (サンプル)
            alpha : 予測距離 α   (サンプル)
            """
            if alpha <= 0:
                raise ValueError("alpha must be ≥ 1")

            # 自己相関 r(0)…r(n+alpha) を求める
            r_full = autocorr(x, n + alpha)

            # Toeplitz 行列 R の 1 列目 / 1 行目（長さ n）
            first_col = r_full[:n]
            first_row = first_col  # 対称 Toeplitz

            # 右辺ベクトル r(α)…r(α+n-1)（長さ n）
            rhs = r_full[alpha : alpha + n]

            # 予測係数 b′ を求める
            bp = solve_toeplitz((first_col, first_row), rhs)

            # フィルタ f = [1, 0×α, -b′]
            f = np.concatenate(([1.0], np.zeros(alpha), -bp))
            y = lfilter(f, [1.0], x)
            return y.astype(np.float32), f

        def levinson_durbin(r):
            """
            SciPy の Toeplitz ソルバで AR 係数 a[1:] を直接求める。
            r[0]…r[n] : 自己相関列
            """
            c = r[:-1]
            a_tail = solve_toeplitz((c, c), -r[1:])  # R a_tail = -r(1…n)
            return np.concatenate(([1.0], a_tail.astype(r.dtype)))

        # データの取得
        _ax, _analysis = self.getax_analysis(axis)
        N, M = _ax.shape

        # 基準ウェーブレットの設定
        wavelet = (
            _ax[self.get_source_ch()].astype(np.float32)
            if wavelet is None
            else wavelet.astype(np.float32)
        )
        wavelet -= np.mean(wavelet)  # 平均を引く
        # 窓関数処理
        wavelet = wavelet * np.hanning(len(wavelet))

        # 基準ウェーブレットからフィルタを作成する
        _, f = predictive_decon(wavelet, n=n, alpha=alpha)

        # フィルタの作用
        ret = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            tr = (_ax[i] - np.mean(_ax[i])).astype(np.float32)  # 平均を引く
            # predictive_deconvolution
            y = lfilter(f, [1.0], tr)
            y.astype(np.float32)
            ret[i] = y

        # データの格納
        _analysis += f"_deconv_pred(n={n}, alpha={alpha})"
        self.putax_analysis(axis, ret, _analysis)

    def closs_corr(
        self, axis, wavelet=None, show=False, save=False, dir=None, savename=None
    ):
        """
        波形の相互相関を実施する
        axis: 'x', 'y', 'z'
        ---parameter---
        axis: 'x', 'y', 'z'
        wavelet: np.array -> 相互相関で基準となる波形(ウェーブレット)
        show: bool -> グラフを表示するかどうか
        save: bool -> 保存するかどうか
                dir -> 保存先のディレクトリ
                savename -> 保存するファイル名

        **基準ウェーブレットを基に相互相関
        res[ch,j] = Σ(k=0 -> k=N) ax[ch,:] * flip(wavelet, k)  (ch = 0, 1, ..., M-1)
        """

        def pad_wavelet(wavelet, M):
            """
            ウェーブレットをaxの形状に合わせてパディングする
            """
            wavelet = np.atleast_1d(wavelet).flatten()
            pad_width = M - wavelet.size

            if pad_width > 0:
                wavelet = np.pad(wavelet, (0, pad_width), mode="constant")
            else:
                wavelet = wavelet[:M]  # waveletがMより長い場合は切り詰める
            return wavelet

        def flip_conj(wavelet):
            """
            フリップして畳み込む
            ウェーブレットをフリップして共役をとる

            # wavelet = np.flip(wavelet)
            # wavelet = np.conj(wavelet)

            # res = np.zeros_like(_ax)  # 結果を保存する配列
            # for i in range(len(_ax)):
            # 	temp = np.convolve(_ax[i], wavelet, mode='full')
            # 	# 結果を保存
            # 	if i == 0:
            # 		res = np.zeros((len(_ax), len(temp)))
            # 	res[i] = temp
            """
            # waveletをフリップして共役をとる
            wavelet = np.conj(np.flip(wavelet))

            # 配列のサイズを定義
            num_signals, signal_len = _ax.shape
            wavelet_len = len(wavelet)

            # 畳み込み後のサイズを計算
            conv_len = signal_len + wavelet_len - 1
            fast_len = next_fast_len(conv_len)

            # waveletのFFT
            wavelet_fft = np.fft.fft(wavelet, fast_len)

            # _axのFFT
            _ax_fft = np.fft.fft(_ax, fast_len, axis=1)

            # Broadcastingで一気に掛け算し、IFFTで戻す
            res_fft = _ax_fft * wavelet_fft  # 自動的にbroadcasting
            res = np.real(np.fft.ifft(res_fft, axis=1))[:, :conv_len]

            return res

        # _ax, _analysisを取得
        _ax, _analysis = self.getax_analysis(axis)

        # waveletの指定が無い場合、デフォルトのウェーブレットを生成
        wavelet = (
            self._wavelet(axis, 20, "hanning", False) if wavelet is None else wavelet
        )

        # ax, waveletを平均0にする
        """
		for i in range(len(_ax)):
			temp = _ax[i] - np.mean(_ax[i])
			_ax[i] = temp
		"""
        _ax = _ax - np.mean(_ax, axis=1, keepdims=True)
        wavelet = wavelet - np.mean(wavelet)

        # waveletをaxの形状に合わせる
        wavelet = pad_wavelet(wavelet, _ax.shape[1])

        # waveletをフリップして共役をとる
        res = flip_conj(wavelet)

        # 結果の格納
        _analysis += "_closscorr"
        self.putax_analysis(axis, res, _analysis)

        if show:
            plt.figure(figsize=(10, 5))
            plt.imshow(res, aspect="auto", cmap="jet")
            plt.colorbar()
            plt.title("Cross Correlation")
            plt.show()
        if save == True and dir != None:
            if not os.path.exists(dir):
                os.makedirs(dir)
            if savename is None:
                savename = dir + " Cross_Correlation" + ".png"
            else:
                savename = dir + savename.replace(".", "") + ".png"
            plt.savefig(savename)
            plt.close()
        elif save == True:
            ValueError("dirが指定されていません")
        else:
            pass

    def gain_geomet(self, axis, velocity, n=2, threshold_t=0.2):
        """
        幾何的な増幅(距離のn乗に比例した増幅)を行う
        後の時刻のデータは雑音も増幅されるため、一定時刻threshold_tで指定した時間で指数関数的増幅を打ち切る。
        axis: 'x', 'y', 'z'
        velocity: float -> 伝播速度
        n: int -> 距離のn乗に比例した増幅 default:2
        threshold_t: float -> 何秒までのデータを指数関数的に増幅するか.設定時刻[s]以降のデータは増幅率は一定 default:0.2
        """
        _ax, _analysis = self.getax_analysis(axis)
        fs = self.fs
        thoreshold = np.int32(threshold_t * fs)
        N, M = _ax.shape
        for it in range(M):
            if it < thoreshold:
                _ax[:, it] = (_ax[:, it] * (velocity * it / fs) ** n).astype(_ax.dtype)
            else:
                _ax[:, it] = (_ax[:, it] * (velocity * threshold_t) ** n).astype(
                    _ax.dtype
                )
        _analysis += "_Gain_geometlic(n=" + str(n) + ")"
        self.putax_analysis(axis, _ax, _analysis)

    def gain_comp_old(self, axis, criteria, maxgain, ch="each", show=False):
        """
		コンプレッサー増幅
		parameters-------------------
		criteria: float -> 最大振幅/criteria にて一番大きい増幅(maxgain倍)となるように増幅関数を作成する
		maxgain: float  -> 最大増幅倍率: 最大振幅は (maxamp / criteria)の振幅の maxgain倍されるので, maxgain > criteriaだと振幅の最大値が変わる
		ch: 'each' or 'all': defualt'each' -> maxampの取り方: each->各ch毎に最大値を設定する, all->データ全体の最大値を基準とし、全ch同様の増幅を行う
		show: bool -> 増幅の

		~~the image of compressor~~
		|   _............. max_gain
		|  / \
	gain| / : \
		|/  :  \ (max amp)
		|_______________ original amp
		o   maxamp/criteria -> the point which is gained the most

		"""
        if ch != "each" and ch != "all":
            raise ValueError("chにはeachかallを入れてください")

        _ax, _analysis = self.getax_analysis(axis)
        if maxgain > criteria:
            print("Maxamp is changed")
        threshold = np.max(np.abs(_ax)) / criteria
        log_c = np.abs(np.log(criteria))
        max_amp = np.max(np.abs(_ax))

        for i in range(_ax.shape[0]):
            if ch == "each":
                threshold = np.max(np.abs(_ax[i])) / criteria
            for it in range(_ax.shape[1]):
                gain = -np.abs((np.log(np.abs(_ax[i, it] + 1e-2) / threshold))) + log_c
                _ax[i, it] *= gain

        # eps = 1e-2                     # 0 除算・ log(0) 回避用の微小量
        # abs_ax = np.abs(_ax)           # |_ax| だけを先にまとめて計算
        # if ch == 'each':
        # 	# 行ごとに最大値を求めてしきい値を作る（shape = (Nrow, 1) なので列方向にブロードキャストされる）
        # 	threshold = np.max(abs_ax, axis=1, keepdims=True) / criteria

        # # ch != 'each' のときは threshold を外で与えておく（スカラー or (Nrow,1)/(1,Ncol)/ (Nrow,Ncol) いずれでも可）

        # gain = -np.abs(np.log((abs_ax + eps) / threshold)) + log_c  # shape = _ax と同じ
        # _ax *= gain                                                 # インプレースで更新

        if show:
            # show amp-gain curve
            amp_range = np.linspace(0.1, max_amp, 1000)
            gain_curve = -np.abs(np.log(amp_range / threshold)) + log_c
            plt.figure(figsize=(8, 6))
            plt.plot(amp_range, gain_curve, label="Gain Curve", color="blue")
            plt.axhline(y=maxgain, color="red", linestyle="--", label="Max Gain")
            plt.axvline(x=threshold, color="green", linestyle="--", label="Threshold")
            plt.axvline(
                x=max_amp, color="black", linestyle="--", label="Maximum Amplitude"
            )
            plt.title("Compressor Gain Curve")
            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Amplitude")
            plt.ylabel("Gain")
            plt.legend()
            plt.grid()
            plt.show()

        _analysis += (
            "_gaincomp(criteria=" + str(criteria) + ", maxgain=" + str(maxgain) + ")"
        )
        self.putax_analysis(axis, _ax, _analysis)

    def gain_comp(self, axis, criteria, maxgain, ch="each", show=False):
        """
		コンプレッサー増幅
		parameters-------------------
		criteria: float -> 最大振幅/criteria にて一番大きい増幅(maxgain倍)となるように増幅関数を作成する
		maxgain: float  -> 最大増幅倍率: 最大振幅は (maxamp / criteria)の振幅の maxgain倍されるので, maxgain > criteriaだと振幅の最大値が変わる
		ch: 'each' or 'all': defualt'each' -> maxampの取り方: each->各ch毎に最大値を設定する, all->データ全体の最大値を基準とし、全ch同様の増幅を行う
		show: bool -> 増幅の

		~~the image of compressor~~
		|   _............. max_gain
		|  / \
	gain| / : \
		|/  :  \ (max amp)
		|_______________ original amp
		o   maxamp/criteria -> the point which is gained the most
		山形コンプレッサ
		------------------------------------
		|x| = max_amp/criteria  で  gain = maxgain
		|x| ≥ max_amp           で  gain → 0
		|x| ≤ max_amp/criteria² で  gain → 0
		"""

        if ch not in ("each", "all"):
            raise ValueError("ch には 'each' か 'all' を指定してください")

        _ax, _analysis = self.getax_analysis(axis)

        eps = 1e-12  # log(0) 回避用
        abs_ax = np.abs(_ax) + eps  # shape = (_ax.shape)

        # ---------------- 最大振幅としきい値 ----------------
        if ch == "each":
            max_amp = abs_ax.max(axis=1, keepdims=True)  # shape = (Nrow,1)
        else:  # "all"
            max_amp = abs_ax.max()  # スカラー

        threshold = max_amp / criteria  # |x| がここで最大ゲイン
        log_c = np.abs(np.log(criteria))  # 正のスカラー

        # ---------------- ゲイン計算 ----------------
        ratio = abs_ax / threshold  # |x| を基準化
        gain = np.log(maxgain) * np.maximum(1 - np.abs(np.log(ratio)) / log_c, 0.0)
        #        ↑  peak=maxgain  ↓  負になった分は 0 で切り捨て
        gain = np.exp(
            gain
        )  # gain = maxgain * (1 - |log(|x| / threshold)| / log(criteria)) の形に変換

        # ---------------- 適用 ----------------
        _ax = (gain * _ax).astype(_ax.dtype)  # インプレースで更新

        _analysis += f"_gaincomp(criteria={criteria}, maxgain={maxgain}, ch={ch})"
        self.putax_analysis(axis, _ax, _analysis)

        # （任意）プロットして様子を確認
        if show:
            import matplotlib.pyplot as plt

            r = np.logspace(-np.log10(criteria), np.log10(criteria), 400)
            g = maxgain * np.maximum(1 - np.abs(np.log(r)) / log_c, 0.0)
            plt.plot(r, g)
            plt.axhline(y=maxgain, color="red", linestyle="--", label="Max Gain")
            plt.legend()
            plt.yscale("log")
            plt.xscale("log")
            plt.xlabel("|x| / threshold")
            plt.ylabel("gain")
            plt.title("Compressor gain curve")
            plt.grid(True)
            plt.show()

    def gain_AGC(self, axis, batchsize=100, clip=10):
        """
        automatic gain control by vector calculation
        --parameters--
        axis: 'x', 'y', 'z'
        batchsize: int -> 何サンプル毎に最大振幅を計算するか
        clip: int -> 最大振幅の何倍まで増幅するか
        """
        # 引数の axis からデータを取得
        _ax, _analysis = self.getax_analysis(axis)
        N, M = _ax.shape

        # バッチ単位の最大振幅をまとめて計算
        maxamp = np.max(np.abs(_ax))  # データ全体の最大振幅
        batch_max = self._batch_max(_ax, batchsize)

        # gain 行列を作成
        gain_full = self._gain_AGC_batch(M, N, maxamp, batch_max, batchsize, clip)

        # _ax -> _ax * gain_full
        utils.multiple(_ax, gain_full)  # gainを適用

        # 格納
        _analysis += "_AGC_mat(batch=" + str(batchsize) + ", clip=" + str(clip) + ")"
        self.putax_analysis(axis, _ax, _analysis)

    def _gain_AGC_batch(self, M, N, maxamp, batch_max, batchsize, clip):
        """
        gain 行列を作成する
        同じ値をバッチ幅だけくりかえす
        --return--
        np.array -> gain 行列 shape=(N, M)
        """
        gain_batch = maxamp / (batch_max + 1e-3 * maxamp)  # shape=(N,B)
        gain_batch = np.clip(gain_batch, 1.0, clip)  # [1, clip]

        # ブロードキャストで (N,B,1) → (N,B,batchsize) に広げてから戻す
        gain_full = np.repeat(gain_batch[:, :, None], batchsize, axis=2).reshape(N, -1)[
            :, :M
        ]  # shape=(N,M)
        return gain_full

    def _batch_max(self, ax: np.array, batchsize: int) -> np.array:
        """
        バッチ毎に最大振幅を計算する
        --parameters--
        ax: np.array -> 振幅データ
        batchsize: int -> 何サンプル毎に最大振幅を計算するか
        --return--
        np.array -> バッチ毎の最大振幅を格納した配列 shape=(N, ax.shape[1] // batchsize # バッチ数)
        """
        N, M = ax.shape
        # ---------- バッチ単位の最大振幅をまとめて計算 ----------
        pad_len = (-M) % batchsize  # batchsize の倍数にする 0〜batchsize-1
        if pad_len:
            _abs_ax = np.pad(np.abs(ax), ((0, 0), (0, pad_len)), "constant")
        else:
            _abs_ax = np.abs(ax)

        B = _abs_ax.shape[1] // batchsize  # バッチ数
        # (N, B, batchsize) へ reshape → 轴2 で最大値
        batch_max = _abs_ax.reshape(N, B, batchsize).max(axis=2)  # shape=(N,B)
        return batch_max

    def zerofill(self, axis, velocity, offset=0):
        """
        理論上の伝播時刻前のデータを0にする
        フィルター処理後のデータに対し、非因果的な部分に使用する

        --parameters--
        axis: 'x', 'y', 'z'
        velocity: float -> 伝播速度[m/s]
        offset: int -> 伝播時刻のオフセット(データの先頭から何サンプル分ずらすか) default:0
        """
        _ax, _analysis = self.getax_analysis(axis)

        for i in range(self.distance.shape[0]):
            d = self.source_x - (self.distance[i])  # 距離
            step = (
                self.fs * np.abs(d) / velocity + offset
            )  # 伝播時間[s]をサンプル数に変換
            step = min(step.astype(np.int32), _ax.shape[1])
            gain = np.exp(np.arange(step, 0, -1) ** 2 / (step**2) * -200)
            _ax[i, :step] = _ax[i, :step] * gain

        # データの格納
        _analysis += "_zerofill"
        self.putax_analysis(axis, _ax, _analysis)

    def envelope(self, axis):  # 包絡線処理
        """
        ヒルベルト変換を用いた包絡線処理と瞬時周波数の計算
        self.env_x, self.insfreq_x, に保存される
        ---parameters---
        axis: 'x' or 'y' or 'z'

        """
        _ax, _analysis = self.getax_analysis(axis)
        env_ax = np.zeros_like(_ax)
        insfreq_ax = np.zeros_like(_ax)
        for i in range(_ax.shape[0]):
            hilb_i = signal.hilbert(_ax[i])
            env_i = np.abs(hilb_i)
            instantaneous_phase = np.unwrap(np.angle(hilb_i))
            instantaneous_frequency = (
                np.diff(instantaneous_phase) / (2.0 * np.pi) * self.fs
            )
            instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), "edge")
            env_ax[i] = env_i
            insfreq_ax[i] = instantaneous_frequency

        if axis == "x":
            self.env_x = env_ax
            self.insfreq_x = insfreq_ax
        elif axis == "y":
            self.env_y = env_ax
            self.insfreq_y = insfreq_ax
        elif axis == "z":
            self.env_z = env_ax
            self.insfreq_z = insfreq_ax
        else:
            ValueError("x,y,zのどれかのaxisを選んで")

    def integral(self, axis, highpass=0.1, method="fourier"):
        """
        離散積分して、ずれをハイパスで調整
        ---parameters---
        axis: 'x' or 'y' or 'z'
        highpass: ハイパスフィルタのカットオフ周波数
        method: 'sum' or 'fourier' (default is fourier)
        """
        _ax, _analysis = self.getax_analysis(axis)  # (24, 16384)
        res = np.zeros_like(_ax)
        if method == "sum":
            for i in range(_ax.shape[0]):
                for j in range(_ax.shape[1]):
                    if j == 0:
                        res[i, j] = 0
                    else:
                        res[i, j] = res[i, j - 1] + _ax[i, j] * (1 / self.fs)  # 積分
            _analysis += "integrate(sum)"
            self.highpass(
                axis, fpass=highpass * 10, fstop=highpass
            )  # 積分した値をハイパスで調整
        elif method == "fourier":
            F_ax = np.fft.fft(_ax, axis=1)
            N = _ax.shape[1]
            freqs = np.fft.fftfreq(N, d=1 / self.fs)
            F_ax_integrated = F_ax / (1j * 2 * np.pi * freqs + 1e-24)
            F_ax_integrated[:, 0] = 0  # ω=0のところは0にする
            res = np.fft.ifft(F_ax_integrated, axis=1).real
            # 結果を対応する軸に保存
            _analysis += "integrate(fft)"
        else:
            ValueError("set method [sum] or [fourier] ")
        self.putax_analysis(axis, res, _analysis)

    def differential(self, axis, method="fourier"):
        """
        離散データを微分して、微分値を保存
        ---parameters---
        axis: 'x' or 'y' or 'z'
        method: 'diff' or 'fourier' (default is fourier)
        """
        _ax, _analysis = self.getax_analysis(axis)  # (24, 16384)
        res = np.zeros_like(_ax)
        if method == "diff":
            for i in range(_ax.shape[0]):
                for j in range(_ax.shape[1]):
                    if j == 0:
                        res[i, j] = _ax[j, j] * self.fs
                    else:
                        res[i, j] = (_ax[i, j] - _ax[i, j - 1]) * self.fs
            _analysis += "differential(diff)"
        elif method == "fourier":
            F_ax = np.fft.fft(_ax, axis=1)
            N = _ax.shape[1]
            freqs = np.fft.fftfreq(N, d=1 / self.fs)
            F_ax_integrated = F_ax * (1j * 2 * np.pi * freqs)
            res = np.fft.ifft(F_ax_integrated, axis=1).real
            _analysis += "differential(fft)"
        else:
            ValueError("set method [diff] or [fourier] ")

        self.putax_analysis(axis, res, _analysis)

    def wiener_deconv(
        self,
        axis,
        show=False,
        a=[1, 0.5],
        b=[1],
        Nit=100,
        lam=0.01,
        batch=True,
        batchsize=100,
    ):
        """
        ・wiener deconvolutionを実施する
                axis: 'x' or 'y' or 'z'
                show: True or False
                a, b: Wiener filterの係数
                Nit: 最適化の反復回数
                lam: 正則化パラメータ
                batch: True or False
                        True: バッチ処理を行う
                batchsize: バッチサイズ
        """
        if batch == True:
            self._wiener_deconv_batch(axis, batchsize, show, a, b, Nit, lam)
        else:
            self._wiener_deconv(axis, show, a, b, Nit, lam)

    def _wiener_deconv_batch(self, axis, batch_size, show, a, b, Nit, lam):
        _ax, _analysis = self.getax_analysis(axis)
        res = np.zeros_like(_ax)

        num_batches = _ax.shape[0] // batch_size
        if _ax.shape[0] % batch_size != 0:
            num_batches += 1

        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, _ax.shape[0])

            for i in range(start_idx, end_idx):
                y = _ax[
                    :, start_idx:end_idx
                ].flatten()  # 各トレースデータを1次元配列に変換
                N = len(y)
                Nb = len(b)
                Na = len(a)

                B = sp.diags(b, offsets=range(0, -Nb, -1), shape=(N, N), format="csr")
                A = sp.diags(a, offsets=range(0, -Na, -1), shape=(N, N), format="csr")
                H = lambda x: np.linalg.solve(A.toarray(), B @ x)
                AAT = A @ A.T
                x = y
                g = B.T @ splinalg.spsolve(A.T, y)
                cost = np.zeros(Nit)

                for k in range(Nit):
                    W_data = 1 / np.maximum(np.abs(x), lam)
                    if W_data.ndim == 1:
                        W = sp.diags(W_data, offsets=0, shape=(N, N), format="csr")
                    else:
                        raise ValueError("W_data is not a 1-dimensional array.")
                    F = AAT + B @ W @ B.T
                    x = W @ (g - B.T @ np.linalg.solve(F.toarray(), B @ W @ g))
                    cost[k] = 0.5 * np.sum(np.abs(y - H(x)) ** 2) + lam * np.sum(
                        np.abs(x)
                    )

                    if show:
                        print(
                            f"Batch {batch + 1}, Trace {i + 1}, Iteration {k + 1}/{Nit}, Cost: {cost[k]}"
                        )
                res[:, start_idx:end_idx] += x.reshape(_ax[:, start_idx:end_idx].shape)
        _analysis += "_wienerdeconv_batch"
        self.putax_analysis(axis, res, _analysis)

    def _wiener_deconv(self, axis, show, a, b, Nit, lam):
        _ax, _analysis = self.getax_analysis(axis)
        res = np.zeros_like(_ax)
        # for i in range(len(_ax)):
        y = _ax.flatten()
        N = len(y)
        Nb = len(b)
        Na = len(a)
        # B = sp.diags(b, offsets=range(0, -Nb, -1), shape=(N, N)).toarray()
        # A = sp.diags(a, offsets=range(0, -Na, -1), shape=(N, N)).toarray()
        B = sp.diags(b, offsets=range(0, -Nb, -1), shape=(N, N), format="csr")
        A = sp.diags(a, offsets=range(0, -Na, -1), shape=(N, N), format="csr")
        H = lambda x: np.linalg.solve(A, B @ x)
        AAT = A @ A.T
        x = y
        # g = B.T @ np.linalg.solve(A.T, y)
        g = B.T @ splinalg.spsolve(A.T, y)
        cost = np.zeros(Nit)

        for k in range(Nit):
            W = sp.diags(1 / np.maximum(np.abs(x), lam), offsets=0).toarray()
            F = AAT + B @ W @ B.T
            x = W @ (g - B.T @ np.linalg.solve(F, B @ W @ g))
            cost[k] = 0.5 * np.sum(np.abs(y - H(x)) ** 2) + lam * np.sum(np.abs(x))

            if show:
                print(f"Iteration {k + 1}/{Nit}, Cost: {cost[k]}")
        res = x
        _analysis += "_wienerdeconv"
        self.putax_analysis(axis, res, _analysis)

    def deconvolution_pred(self, axis, wavelet=None, n=128, alpha=8):
        """
        predictable deconvolution

        Parameters
                axis: str, 'x', 'y', 'z'
                n: int, number of operators, default 128
                alpha: int, prediction order, default 8

        Return
                None, but store the result in the analysis axis

        Reference: Deconvolution filter in seismic data processing
                           Sekiyu Gakkai, 31, (6), pp439-447, 1988
        """
        from scipy.linalg import solve_toeplitz
        from scipy.signal import lfilter

        def autocorr(x, order):
            n = len(x)
            return np.correlate(x, x, mode="full")[n - 1 : n + order] / n

        def predictive_decon(x, n, alpha):
            """
            Predictive deconvolution (Levinson–Durbin + Toeplitz solver)
            n     : オペレータ長 (サンプル)
            alpha : 予測距離 α   (サンプル)
            """
            if alpha <= 0:
                raise ValueError("alpha must be ≥ 1")

            # 自己相関 r(0)…r(n+alpha) を求める
            r_full = autocorr(x, n + alpha)

            # Toeplitz 行列 R の 1 列目 / 1 行目（長さ n）
            first_col = r_full[:n]
            first_row = first_col  # 対称 Toeplitz

            # 右辺ベクトル r(α)…r(α+n-1)（長さ n）
            rhs = r_full[alpha : alpha + n]

            # 予測係数 b′ を求める
            bp = solve_toeplitz((first_col, first_row), rhs)

            # フィルタ f = [1, 0×α, -b′]
            f = np.concatenate(([1.0], np.zeros(alpha), -bp))
            y = lfilter(f, [1.0], x)
            return y.astype(np.float32), f

        def levinson_durbin(r):
            """
            SciPy の Toeplitz ソルバで AR 係数 a[1:] を直接求める。
            r[0]…r[n] : 自己相関列
            """
            c = r[:-1]
            a_tail = solve_toeplitz((c, c), -r[1:])  # R a_tail = -r(1…n)
            return np.concatenate(([1.0], a_tail.astype(r.dtype)))

        # データの取得
        _ax, _analysis = self.getax_analysis(axis)
        N, M = _ax.shape

        # 基準ウェーブレットの設定
        wavelet = (
            _ax[self.get_source_ch()].astype(np.float32)
            if wavelet is None
            else wavelet.astype(np.float32)
        )
        wavelet -= np.mean(wavelet)  # 平均を引く
        # 窓関数処理
        wavelet = wavelet * np.hanning(len(wavelet))

        # 基準ウェーブレットからフィルタを作成する
        _, f = predictive_decon(wavelet, n=n, alpha=alpha)

        # フィルタの作用
        ret = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            tr = (_ax[i] - np.mean(_ax[i])).astype(np.float32)  # 平均を引く
            # predictive_deconvolution
            y = lfilter(f, [1.0], tr)
            y.astype(np.float32)
            ret[i] = y

        # データの格納
        _analysis += f"_deconv_pred(n={n}, alpha={alpha})"
        self.putax_analysis(axis, ret, _analysis)

    def WT(self, axis, mother="", dir=None, save=False, show=False):
        """
        ウェーブレット変換
        axis: 'x', 'y', 'z'
        mother: str -> 'Morlet'
                define orthogonal wavelet,
        """
        _ax, _analysis = self.getax_analysis(axis)
        fs = self.fs
        N = _ax.shape[1]
        # 母波形の定義
        if mother == "Morlet":
            # Morlet wavelet
            wavelet = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    wavelet[i, j] = np.exp(2j * np.pi * (i - j) / N) * np.exp(
                        -((i - j) ** 2) / (2 * (0.5 * N) ** 2)
                    )
            wavelet /= np.max(np.abs(wavelet))

    def _WT_motherwavelett(self, t, typ):
        if typ == "original":
            """ 定義されたマザーウェーブレット φ(t) """
            if 0 <= t < 0.5:
                return 1
            elif 0.5 <= t < 1:
                return -1
            else:
                return 0
