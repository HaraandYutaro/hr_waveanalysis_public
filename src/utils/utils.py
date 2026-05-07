"""
Utility functions for wave analysis

This module contains standalone utility functions used across the wave analysis package.
"""

import glob
import math
import os
import warnings
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import scipy
from numba import njit, prange

if TYPE_CHECKING:
    from src.converter.seg2 import SEG2Reader

# Import SEG2Reader for runtime usage in utility functions
try:
    from src.converter.seg2 import SEG2Reader
except ImportError:
    # Fallback if seg2 is not available
    SEG2Reader = None

warnings.filterwarnings("ignore")
plt.style.use("fast")


def multiple(a: np.ndarray, b: np.ndarray, *, out=None):
    """
    a *= b
    - a : 元データ (in/out)
    - b : ブロードキャスト可能なゲイン配列
    - out : None なら a をインプレース更新
            それ以外なら out に書き込み(shape は a と同じであること）

    returns:
    - None : a が更新される
    """

    # 1. 乗算に使う「安全な dtype」を決める
    # ------------------------------------------------------------
    #   * int と float の組合せ → 少なくとも float32
    #   * int × int は幅が足りないとき int64 へ
    #   * それ以外は NumPy の昇格ルールに任せる
    # ------------------------------------------------------------
    if out is None:
        out = a
    dt = np.result_type(a.dtype, b.dtype)
    if np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.floating):
        # 明示的に float32/64 へ（好みで切替）
        dt = np.float64
    elif np.issubdtype(a.dtype, np.integer) and np.issubdtype(b.dtype, np.integer):
        # 幅が足りなければ int64 に昇格
        if np.iinfo(a.dtype).bits < 64:
            dt = np.int64

    # 乗算（高精度で計算して仮バッファに入れる）
    tmp = np.multiply(a, b, dtype=dt)

    # ------------------------------------------------------------
    # 3. 必要ならクリップして元 dtype に戻す
    # ------------------------------------------------------------
    if out.dtype == tmp.dtype:  # 既に安全 dtype ならそのまま
        np.copyto(out, tmp)
    elif np.issubdtype(out.dtype, np.integer):
        info = np.iinfo(out.dtype)
        np.clip(tmp, info.min, info.max, out=tmp)
        np.copyto(out, tmp.astype(out.dtype))
    else:
        np.copyto(out, tmp.astype(out.dtype))


def stack(
    pathlist,
    endstep=5000,
    ch=1,
    highpass=True,
    print_log=True,
    kill_ratio=[0.2, 3.0],
    save=False,
    savename=None,
    dir=None,
    debug=False,
):
    """
    endstep = 5000: 事前にステップ以降を削除し、データを削減する
    ch = 1: 相互相関をとる基準のチャンネル
    highpass = True: highpassスタッキングする前1Hzのハイパスフィルターでノイズ除去を行う
    kill_ratio = [0.2, 3.]: 時間遅れを適用する際に、データの比率がこの範囲外のデータは外れ値とみなして、無視する
    save = True: スタッキング結果を保存する
    savename: 保存するファイル名
    dir: 保存するディレクトリ
    debug = True: デバッグモード
    print_log = True:時間遅れを表示する
    """

    if save == True:
        if dir != None and savename != None:
            pass
        else:
            ValueError("dir, savenameを指定してください")

    for i in range(len(npz_list)):
        path = pathlist[i]
        file = np.load(path)
        hr = HaraFormatNpzProcesser(path)
        if i == 0:
            fs = file["fs"]
            interval = file["interval"]
            Num_sensor = file["Num_sensor"]
            excitation = file["excitation"]
            receiver_x = file["sensor1_x"]
            condition = file["condition"]
        else:
            error = 0
            if fs != file["fs"]:
                error = 1
            if interval != file["interval"]:
                error = 1
            if Num_sensor != file["Num_sensor"]:
                error = 1
            if excitation != file["excitation"]:
                error = 1
            if receiver_x != file["sensor1_x"]:
                error = 1
            if condition != file["condition"]:
                error = 1

            if error == 1:
                name = path.split("/")[-1]
                ValueError("データの条件が異なります: " + name)

    ##同条件のデータをnoise除去してスタッキング
    for i in range(len(npz_list)):
        path = pathlist[i]
        file = np.load(path)
        hr = HaraFormatNpzProcesser(path)

        if highpass == True:
            hr.highpass(axis="x", fpass=1, fstop=0.1)  # highpassしてノイズ除去
            hr.highpass(axis="z", fpass=1, fstop=0.1)  # highpassしてノイズ除去

        ithdata = np.delete(
            hr.data, slice(endstep, hr.data.shape[1]), axis=1
        )  # 5000ステップ以降を削除

        if i == 0:  # 保存のため、ほかの情報は基本的に一番目のデータに準拠
            data_dict = {key: file[key] for key in file.files}
            data = ithdata  # 単純に足している　stacking
            data_sum = data
            data_counts = np.ones(len(data))
            time_delay = 0
            if print_log:
                hr.show(
                    axis="z",
                    t=[0, ithdata.shape[1] / hr.fs],
                    suptitle="base data Z",
                    save=False,
                    color="black",
                )
                hr.show(
                    axis="x",
                    t=[0, ithdata.shape[1] / hr.fs],
                    suptitle="base data X",
                    save=False,
                    color="blue",
                )

        else:
            ##dataとithdata との相互相関を出して、stacking
            # 相互相関を計算
            correlation = scipy.signal.correlate(
                np.float64(data_sum[ch - 1] / np.max(np.abs(data_sum[ch - 1]))),
                np.float64(ithdata[ch - 1] / np.max(np.abs(ithdata[ch - 1]))),
                mode="full",
            )
            max_index = np.argmax(correlation)

            ## for debug
            if debug:
                plt.plot(correlation)
                plt.title("cross-correlation")
                plt.show()

            time_delay = max_index - (ithdata.shape[1] - 1)

            if print_log:
                # 時間遅れを表示
                print(
                    str(i + 1) + "番目のdataは: {} stepsの時間遅れ".format(time_delay)
                )

            # 時間遅れを適用
            for k in range(len(ithdata)):
                max_ithk = np.max(ithdata[k])
                max_data_sumk = np.max(data_sum[k])
                ratio = max_ithk / max_data_sumk
                # 明らかに異常な値を持つデータ[k]は無視
                if ratio < kill_ratio[0] or ratio > kill_ratio[1]:
                    ithdata[k] = np.zeros_like(ithdata[k])

                if time_delay > 0:  # ithdata は dataよりも遅れている
                    for i in range(time_delay, data.shape[1]):
                        data_sum[k, i] = (
                            data_sum[k, i] * data_counts[k] + ithdata[k, i - time_delay]
                        ) / (data_counts[k] + 1)
                elif time_delay < 0:  # ithdata は dataよりも先行している
                    for i in range(-time_delay, data.shape[1]):
                        data_sum[k, i] = (
                            data_sum[k, i] * data_counts[k] + ithdata[k, i + time_delay]
                        ) / (data_counts[k] + 1)
                else:  # time_delay == 0
                    data_sum[k] = (data_sum[k] * data_counts[k] + ithdata[k]) / (
                        data_counts[k] + 1
                    )
                data_counts[k] += 1

        # for debug
        if debug:
            plt.figure(figsize=(9, 6))

            # data[0]とithdata[0]のプロット、time_delayの位置に目印
            plt.subplot(2, 1, 1)
            # plt.plot(data_sum[0], label='data_sum[0]', color='blue')
            # plt.plot(ithdata[0], label='ithdata[0]', color='orange')
            indc = 23
            # time_delayの位置に目印を付ける
            if time_delay > 0:  # ithdata は dataよりも遅れている
                plt.plot(
                    data_sum[indc, time_delay:],
                    color="red",
                    label="data_sum[" + str(indc + 1) + "]",
                )  # dataの位置
                plt.plot(
                    ithdata[indc, : ithdata.shape[1] - time_delay],
                    "--",
                    color="orange",
                    label="ithdata[" + str(indc + 1) + "]",
                )  # ithdataの位置
                plt.legend()
                # plt.scatter(i - time_delay, ithdata[0, i - time_delay], color='green', s=5)  # ithdataの位置
            elif time_delay < 0:  # ithdata は dataよりも先行している
                plt.plot(
                    data_sum[indc, : data.shape[1] + time_delay],
                    color="red",
                    label="data_sum[" + str(indc + 1) + "]",
                )  # dataの位置
                plt.plot(
                    ithdata[indc, -time_delay:],
                    "--",
                    color="orange",
                    label="ithdata[" + str(indc + 1) + "]",
                )  # ithdataの位置
                plt.legend()
                # plt.scatter(i + time_delay, ithdata[0, i + time_delay], color='green', s=5)  # ithdataの位置
            else:
                plt.plot(
                    data_sum[indc], color="red", label="data_sum[" + str(indc + 1) + "]"
                )  # dataの位置
                plt.plot(
                    ithdata[indc],
                    "--",
                    color="orange",
                    label="ithdata[" + str(indc + 1) + "]",
                )  # ithdataの位置
                plt.legend()
                # plt.scatter(range(data.shape[1]), ithdata[0], color='green', s=5)  # ithdataの位置

            plt.title(
                "data_sum["
                + str(indc + 1)
                + "] and ithdata["
                + str(indc + 1)
                + "] with time_delay={time_delay}"
            )

            # data_sum[0]のプロット
            plt.subplot(2, 1, 2)
            plt.plot(
                data_sum[indc], label="data_sum[" + str(indc + 1) + "]", color="purple"
            )
            plt.title("data_sum[" + str(indc + 1) + "]")
            plt.legend()

            plt.tight_layout()
            plt.show()

    N = hr.Num_sensor
    data_dict["ax1"] = data_sum[0:N]
    data_dict["ax2"] = data_sum[N : 2 * N]
    data_dict["data"] = data_sum
    data_dict["analysis_ax1"] = hr.analysis_ax1 + "_stacking"
    data_dict["analysis_ax2"] = hr.analysis_ax2 + "_stacking"

    if save == True:
        if dir != None and savename != None:
            if not os.path.exists(dir):
                os.makedirs(dir)
            else:
                _savename = dir + savename + ".npz"
            np.savez(_savename, **data_dict)
        else:
            ValueError(
                "dir, savenameを指定してください(ここのエラーが表示されるのは本来おかしいです。もっと前に表示されるべきだが??)"
            )
    else:
        hr3 = hr
        hr3.data = data_sum
        hr3.ax1 = data_dict["ax1"]
        hr3.ax2 = data_dict["ax2"]
        hr3.Num_sensor = len(hr3.ax1)
        hr3.analysis_ax1 = data_dict["analysis_ax1"]
        hr3.analysis_ax2 = data_dict["analysis_ax2"]
        hr3.show(axis="z", t=[0, 0.5], save=False, suptitle="Stacking Result_z")
        hr3.show(
            axis=hr.second_axis, t=[0, 0.5], save=False, suptitle="Stacking Result_"
        )


def target_cmplist(xl_path, sheet, Num_from, Num_to, npz_dir):
    xlfile = openpyxl.load_workbook(xl_path)
    st1 = xlfile[sheet]
    target_cmplist = []
    for row in range(Num_from, Num_to):
        datanum = st1.cell(row, 1).value  # 番号

        # npzファイルの取得
        name = "GEO_" + str(datanum)
        if len(str(datanum)) == 2:
            name = "GEO_00" + str(datanum)
        if len(str(datanum)) == 3:
            name = "GEO_0" + str(datanum)
        npz_path = glob.glob(npz_dir + name + ".npz")[0]

        # HaraFormatNpzProcesserを利用してCMPリストを取得
        cmplist = HaraFormatNpzProcesser(npz_path).get_all_CMP()
        for cmp in cmplist:
            if cmp in target_cmplist:
                pass
            else:
                target_cmplist.append(cmp)
    return target_cmplist


## Obspyで波形表示するためのソフト
def show_seg2file_by_obspy(path, step_meter=1):
    from obspy.core import read

    st = read(path)
    m = 0
    for s in st:
        s.stats.distance = m
        m += step_meter
    st.plot(type="section")


def setfft(fs, data, x_from, x_to, y_from, y_to):
    # データのパラメータ
    N = len(data)  # サンプル数
    dt = 1.0 / fs  # サンプリング間隔
    t = np.arange(0, N * dt, dt)  # 時間軸
    freq = np.linspace(0, 1.0 / dt, N)  # 周波数軸

    # 高速フーリエ変換
    F = np.fft.fft(data)

    # 振幅スペクトルを計算
    Amp = np.abs(F)

    # グラフ表示
    plt.figure()
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 17
    plt.plot()
    plt.plot(freq, Amp)
    plt.xlabel("Frequency", fontsize=20)
    plt.ylabel("Amplitude", fontsize=20)
    plt.xlim([x_from, x_to])
    plt.ylim([y_from, y_to])
    plt.xscale("log")
    plt.grid()
    # leg = plt.legend(loc=1, fontsize=25)
    # leg.get_frame().set_alpha(1)
    plt.show()


# #DFT解析
# def DFT(data,fs):
# 	N = len(data)
# 	Fourier = []
# 	Frequency = []
# 	Amplitude = []
# 	for k in range(N):
# 		#一定値のkの中でnを動かす
# 		F_k = 0 + 0j
# 		for n in range(N):
# 			F_k += data[n]*np.exp((-2*math.pi*1j)*(k/N)*n)
# 		Fourier.append(F_k)
# 		Frequency.append((k/N)*fs)
# 		Amplitude.append((2/N)*abs(F_k))
# 	return Frequency, Amplitude


def sigma(f, g):
    S = 0
    if len(f) == len(g):
        for i in range(len(f)):
            S += f[i] * g[i]
        return S
    else:
        print("fuck")
        return False


def get_geometatt_andinternalatt(da, dl):
    x1 = (
        sigma(np.exp(dl), da) * sigma(np.exp(dl), dl)
        - sigma(np.exp(dl), np.exp(dl)) * sigma(da, dl)
    ) / (
        sigma(dl, dl) * sigma(np.exp(dl), np.exp(dl))
        - sigma(dl, np.exp(dl)) * sigma(dl, np.exp(dl))
    )
    x2 = (
        sigma(da, dl) * sigma(np.exp(dl), dl) - sigma(dl, dl) * sigma(da, np.exp(dl))
    ) / (
        sigma(dl, dl) * sigma(np.exp(dl), np.exp(dl))
        - sigma(dl, np.exp(dl)) * sigma(dl, np.exp(dl))
    )
    print("近似曲線: 幾何減衰定数 n = " + str(x1) + ", 内部減衰定数 η = " + str(x2))
    return [x1, x2]


def attenuationrate(Maxamplist: list, Distancelist: list, name):
    print(name)
    M = Maxamplist[0]
    xmin = np.min(Distancelist)
    xmax = np.max(Distancelist)
    attenuationlist = []
    for i in range(len(Maxamplist)):
        a = Maxamplist[i] / M
        log_a = math.log10(a)
        attenuationlist.append(log_a)

    dlist = []
    for i in range(len(Distancelist)):
        x = Distancelist[i]
        # 出発点からの距離差
        x = x - xmin
        dlist.append(x)

    if len(dlist) == len(attenuationlist):
        # 近似曲線の表示
        # f = interp1d(dlist, attenuationlist, kind = 'slinear')
        # itpx = dlist
        # itpy = f(itpx)
        # plt.plot(dlist, attenuationlist, 'o', itpx, itpy, '--')
        arraydlist = np.array(dlist)
        array_attenuationlist = np.array(attenuationlist)
        a = np.dot(arraydlist, array_attenuationlist) / (arraydlist**2).sum()
        attenuation_ratio = round(100 * (1 - (10**a)), 1)
        # グラフ表示
        plt.plot(
            dlist,
            attenuationlist,
            "o",
            [0, arraydlist.max()],
            [0, a * arraydlist.max()],
            "--",
        )
        name = str(name) + "_減衰率" + str(attenuation_ratio) + "[%/m]"
        plt.title(
            name,
        )
        plt.xlabel("距離差[m]")
        plt.ylabel("振幅比 対数表示")
        # plt.yscale('log')

        print("減衰率..." + str(attenuation_ratio) + "[%/m]")
        print("\n")
        plt.show()

        get_geometatt_andinternalatt(attenuationlist, dlist)

    else:
        print("振幅表と距離差表の数があってないぞ！バカもの〜〜！！")


def attenuationrate_wavelength(Maxamplist: list, Distancelist: list, name):
    print(name)
    M = Maxamplist[0]
    xmin = min(Distancelist)
    # xmax = max(Distancelist)
    attenuationlist = []
    for i in range(len(Maxamplist)):
        a = Maxamplist[i] / M
        log_a = math.log10(a)
        attenuationlist.append(log_a)

    dlist = []
    for i in range(len(Distancelist)):
        x = Distancelist[i]
        # 出発点からの距離差
        x = x - xmin
        dlist.append(x)

    if len(dlist) == len(attenuationlist):
        # 近似曲線の表示
        # f = interp1d(dlist, attenuationlist, kind = 'slinear')
        # itpx = dlist
        # itpy = f(itpx)
        # plt.plot(dlist, attenuationlist, 'o', itpx, itpy, '--')
        arraydlist = np.array(dlist)
        array_attenuationlist = np.array(attenuationlist)
        a = (
            np.dot(arraydlist, array_attenuationlist) / (arraydlist**2).sum()
        )  # a = log10(An+1/An), 10^a = An+1/An
        print("傾きα = " + str(a))
        # Q値... 2π/Q = ΔE/E = 1 - (An+1/An)^2 = 1 - 10^2a
        Q_value = round(
            (2 * 3.14159265358979323846264338327950288419) / (1 - (10 ** (2 * a))), 3
        )
        attenuation_ratio = round(100 * (1 - (10**a)), 1)
        # グラフ表示
        plt.plot(
            dlist,
            attenuationlist,
            "o",
            [0, arraydlist.max()],
            [0, a * arraydlist.max()],
            "--",
        )
        name = (
            str(name)
            + "_減衰率"
            + str(attenuation_ratio)
            + "[%/波長]_Q値"
            + str(Q_value)
        )
        plt.title(name)
        plt.xlabel("波数")
        plt.ylabel("振幅比 対数表示")
        # plt.yscale('log')
        print("減衰率..." + str(attenuation_ratio) + "[%/波長]")
        h = 1 / Q_value
        print("減衰定数h=1/Q..." + str(round(h, 4)) + "[%/波長]")
        print("\n")
        # print(attenuationlist)
        # print(dlist)
        plt.show()
    else:
        print("振幅表と距離差表の数があってないぞ！バカもの〜〜！！")


def attenuationrate_geometric(Maxamplist: list, Distancelist: list, name):
    print(name)
    ##振幅と距離差リストの数が同じか確認
    if len(Maxamplist) == len(Distancelist):
        M = Maxamplist[0]
        # xmin = min(Distancelist)
        # xmax = max(Distancelist)
        dlist = []
        ax_list = []
        axratio_list = []
        dBlist = []
        for i in range(len(Maxamplist)):
            # 起振点からの距離差(手動入力)
            x = Distancelist[i]
            # 基準点からの距離差
            y = x - Distancelist[0]
            a = Maxamplist[i] / M
            ax = a * (x**1)  ## r*A0/A1 は指数関数的に下がるはず
            ax_list.append(ax)

            # 基準値と比を取得
            ax1 = ax_list[0]
            b = ax_list[i] / ax1
            log_b = math.log10(b)
            dB = 20 * log_b
            axratio_list.append(log_b)
            dBlist.append(dB)
            dlist.append(y)

        # 近似曲線の表示
        arraydlist = np.array(dlist)
        array_axratio_list = np.array(axratio_list)
        a = np.dot(arraydlist, array_axratio_list) / (arraydlist**2).sum()
        attenuation_ratio = round(100 * (1 - (10**a)), 1)
        # グラフ表示
        plt.plot(
            dlist,
            axratio_list,
            "o",
            [0, arraydlist.max()],
            [0, a * arraydlist.max()],
            "--",
        )
        name = str(name) + "_減衰率" + str(attenuation_ratio) + "[%/m]"
        plt.title(name)
        plt.xlabel("距離差[m]")
        plt.ylabel("振幅比＊伝播距離 対数表示")
        # plt.yscale('log')
        print("減衰率..." + str(attenuation_ratio) + "[%/m]")
        print("\n")
        plt.show()

        # dBグラフ表示
        arraydlist = np.array(dlist)
        array_dBlist = np.array(dBlist)
        dBmean = np.dot(arraydlist, array_dBlist) / (arraydlist**2).sum()
        # 内部減衰係数arufa
        arufa = dBmean / (-8.68)
        # 内部減衰定数h
        h_ramda = arufa / (2 * 3.141592635358979323846264338327950288419)
        plt.plot(
            dlist,
            dBlist,
            "o",
            [0, arraydlist.max()],
            [0, dBmean * arraydlist.max()],
            "--",
        )
        name = str(name) + "_内部減衰係数α" + str(arufa) + "[1/m]"
        plt.title(name)
        plt.xlabel("距離差[m]")
        plt.ylabel("振幅比　dB表示")
        # plt.yscale('log')
        print("内部減衰定数h/λ..." + str(h_ramda) + "[無次元]")
        print("\n")
        plt.show()

    else:
        print("振幅表と距離差表の数があってないぞ！バカもの〜〜！！")
    return attenuation_ratio


def attenuationrate_geometric_ndimention(
    Maxamplist: list, Distancelist: list, name, nd: int
):
    print(name)
    ##振幅と距離差リストの数が同じか確認
    if len(Maxamplist) == len(Distancelist):
        M = Maxamplist[0]
        # xmin = min(Distancelist)
        # xmax = max(Distancelist)
        dlist = []
        ax_list = []
        axratio_list = []
        for i in range(len(Maxamplist)):
            # 起振点からの距離差(手動入力)
            x = Distancelist[i]
            # 基準点からの距離差
            y = x - Distancelist[0]
            a = Maxamplist[i] / M
            ax = a * (x**nd)  ## r*A0/A1 は指数関数的に下がるはず
            ax_list.append(ax)

            # 基準値と比を取得
            ax1 = ax_list[0]
            b = ax_list[i] / ax1
            log_b = math.log10(b)
            axratio_list.append(log_b)
            dlist.append(y)

        # 近似曲線の表示
        arraydlist = np.array(dlist)
        array_axratio_list = np.array(axratio_list)
        a = np.dot(arraydlist, array_axratio_list) / (arraydlist**2).sum()
        attenuation_ratio = round(100 * (1 - (10**a)), 1)
        # グラフ表示
        plt.plot(
            dlist,
            axratio_list,
            "o",
            [0, arraydlist.max()],
            [0, a * arraydlist.max()],
            "--",
        )
        name = str(name) + "_減衰率" + str(attenuation_ratio) + "[%/m]"
        plt.title(name)
        plt.xlabel("距離差[m]")
        plt.ylabel("振幅比＊伝播距離 対数表示")
        # plt.yscale('log')

        print("減衰率..." + str(attenuation_ratio) + "[%/m]")
        print("\n")
        plt.show()
    else:
        print("振幅表と距離差表の数があってないぞ！バカもの〜〜！！")
    return attenuation_ratio


def dBattenuationrate_geometric_ndimention(
    Maxamplist: list, Distancelist: list, name, nd: int
):
    print(name)
    ##振幅と距離差リストの数が同じか確認
    if len(Maxamplist) == len(Distancelist):
        M = Maxamplist[0]
        # xmin = min(Distancelist)
        # xmax = max(Distancelist)
        dlist = []
        dB_list = []
        dB_andlogdistancelist = []
        for i in range(len(Maxamplist)):
            # 振幅比をエネルギー比(dB)で表示
            dB_i = 20 * math.log10(Maxamplist[i] / Maxamplist[0])
            dB_list.append(dB_i)
            # 基準点からの距離差
            y = Distancelist[i] - Distancelist[0]
            dlist.append(y)

            # 距離減衰まで考慮した減衰値の導入
            dB_and_Dis = dB_i + 20 * nd * math.log10(
                Distancelist[i] / Distancelist[0]
            )  # = -8.68α(R1-R0)
            dB_andlogdistancelist.append(dB_and_Dis)

        # 近似曲線の表示
        arraydlist = np.array(dlist)
        array_dBandDis_list = np.array(dB_andlogdistancelist)
        a = np.dot(arraydlist, array_dBandDis_list) / (arraydlist**2).sum()
        # 内部減衰定数
        arufa = -a / 8.68  # 8.68 = 20/loge(10)
        # グラフ表示
        plt.plot(
            dlist,
            dB_andlogdistancelist,
            "o",
            [0, arraydlist.max()],
            [0, a * arraydlist.max()],
            "--",
        )
        name = str(name) + "_減衰定数" + str(arufa) + "[1/m]"
        plt.title(name)
        plt.xlabel("距離差[m]")
        plt.ylabel("振幅比 dB表示")
        # plt.yscale('log')

        print("減衰定数α..." + str(arufa) + "[1/m]")
        print("\n")
        plt.show()
    else:
        print("振幅表と距離差表の数があってないぞ！バカもの〜〜！！")
    return arufa


def createcolormap_Amp_t1_to_t2_ch1_to_ch2(
    seg2: SEG2Reader, t1, t2, ch1, ch2, dirname=None, filename=None
):
    data = seg2.get_all_numpy_array()
    freq = seg2.get_frequency()
    Num_ch = seg2.get_max_ch()
    Num_sg2 = len(data)
    n1 = int(freq * t1)
    n2 = int(freq * t2)
    t = np.arange(0, Num_sg2 / freq, 1 / freq)
    data1 = np.zeros((n2 - n1, ch2 - ch1))
    # どこからどこまでのチャンネルか指定
    for i in range(ch1 - 1, ch2 - 1):
        # 何秒から何秒までか指定
        for j in range(n1, n2):
            abs_data = abs(data[j, i]) / 5
            data1[j, i] = abs_data
    plt.imshow(
        data1.transpose(),
        extent=(0, Num_sg2 / freq, 0, ch2 - ch1),
        interpolation="nearest",
        aspect="auto",
        cmap="jet",
    )
    plt.xlabel("time (s)")

    if dirname is not None:
        if filename is not None:
            os.makedirs(dirname, exist_ok=True)
            savepath = ""
            savepath += dirname + "/"
            savepath += filename
            plt.savefig(savepath)
    else:
        plt.show()


# 反射波fft2画像の出力
def FFT_2d(
    sg2_2: SEG2Reader,
    dx,
    del_ch=[],
    Hzrange: list = [1, 100],
    fftm=4096,
    fftn=256,
):
    npa = sg2_2.get_all_numpy_array()
    freq = sg2_2.get_frequency()
    dt = 1 / freq  # サンプリング間隔[s]
    dx = dx
    ## FFT処理
    F2d = np.fft.fftshift(np.fft.fft2(npa))
    mag_2d = np.abs(F2d)
    dB_2d = 20 * np.log10(mag_2d)  ##振幅をパワーと同次元にしてデシベル化
    ##画像処理
    umax = 1 / (2 * dx)
    vmax = 1 / (2 * dt)
    # value_max = 1000 ## colormapのvalue 最大値を設定する
    plt.imshow(
        mag_2d,
        [fftn, fftm],
        extent=(-umax, umax, -vmax, vmax),
        cmap="jet",
        interpolation="nearest",
        aspect="auto",
    )
    plt.xlim([-umax, umax])
    plt.xlabel("κ/2π : $m^{-1}$")
    plt.ylim([0, vmax / 16])
    plt.ylabel("f : Hz")
    plt.colorbar()
    plt.show()


def show24ch_2lines(DATfile):
    sg2_2 = SEG2Reader(DATfile)
    data = sg2_2.get_all_numpy_array()
    freq = sg2_2.get_frequency()
    # print(freq) #10000Hz
    Num_data = len(data)  # 8192
    print("size = " + str(data.shape) + "maxsize is " + str(data.shape[0]))
    ## 24行2列でのデータ表示
    max = np.max(data)
    t = np.arange(0, Num_data / freq, 1 / freq)
    fig, axes = plt.subplots(12, 2)
    for i in range(12):
        name = "Ch" + str(i)
        datai = data[:, i + 24]
        axes[i, 0].plot(t, datai, color="red", label=name)
        axes[i, 0].set_yticklabels([])
        axes[i, 0].set_xlim([0, 0.5])
        # axes[i, 0].set_ylim([-max, max])
    for i in range(12, 24):
        name = "Ch" + str(i)
        datai = data[:, i + 24]
        axes[i - 12, 1].plot(t, datai, color="red", label=name)
        axes[i - 12, 1].set_yticklabels([])
        axes[i - 12, 1].set_xlim([0, 0.5])
        # axes[i-12, 1].set_ylim([-max, max])
    plt.show()


def save2dFFT(DATfile2, Distance_sensor, t_from, t_to, del_chs: list, dirname_2dfft):
    datanumber = DATfile2.split("/")[-1].split(".")[0]
    fft2title = "FFT2 " + datanumber + ", t: " + str(t_from) + "s ~" + str(t_to) + "s"
    npa = SEG2Reader(DATfile2).get_all_numpy_array()
    freq = SEG2Reader(DATfile2).get_frequency()

    dx = Distance_sensor  # センサー間の距離
    dt = 1 / freq  # サンプリング間隔[s]
    ## FFT処理
    fftm = 8192
    fftn = 1024

    step_from = round(t_from * freq, 0)
    step_to = round(t_to * freq, 0)
    # Pandasを利用してデータ読み出しと不要データ削除
    npa = np.delete(npa, del_chs, 1)
    npa = np.delete(npa, range(int(step_to), npa.shape[0]), 0)
    npa = np.delete(npa, range(int(step_from)), 0)

    F2d = np.fft.fftshift(np.fft.fft2(npa, [fftm, fftn]))
    mag_2d = np.abs(F2d)
    dB_2d = 20 * np.log10(mag_2d)  ##振幅をパワーと同次元にしてデシベル化
    ##画像処理
    umax = 1 / (2 * dx)
    vmax = 1 / (2 * dt)
    # value_max = 1000 ## colormapのvalue 最大値を設定する
    plt.imshow(
        mag_2d,
        extent=(-umax, umax, -vmax, vmax),
        cmap="jet",
        interpolation="nearest",
        aspect="auto",
    )
    # plt.imshow(dB_2d,extent=(-umax, umax, -vmax, vmax), cmap='jet',interpolation='nearest',aspect = 'auto')
    plt.xlim([-umax, umax])
    # plt.xlim([0, umax])
    plt.xlabel("v = 1/λ: $m^{-1}$")
    plt.ylim([0, 200])  ##見たい周波数帯を設定
    plt.ylabel("f : Hz")
    plt.colorbar()
    plt.title(fft2title)
    if not os.path.exists(dirname_2dfft):
        os.makedirs(dirname_2dfft)
    savename = dirname_2dfft + fft2title + ".png"
    plt.savefig(savename)
    plt.close()
    return


def save2dFFT_fromnpa(
    npa, freq, Distance_sensor, t_from, t_to, del_chs: list, dirname_2dfft
):
    fft2title = "FFT2  , t: " + str(t_from) + "s ~" + str(t_to) + "s"
    dx = Distance_sensor  # センサー間の距離
    dt = 1 / freq  # サンプリング間隔[s]
    ## FFT処理
    fftm = 8192
    fftn = 1024

    step_from = round(t_from * freq, 0)
    step_to = round(t_to * freq, 0)
    # Pandasを利用してデータ読み出しと不要データ削除
    npa = np.delete(npa, del_chs, 1)
    npa = np.delete(npa, range(int(step_to), npa.shape[0]), 0)
    npa = np.delete(npa, range(int(step_from)), 0)
    F2d = np.fft.fftshift(np.fft.fft2(npa, [fftm, fftn]))
    mag_2d = np.abs(F2d)
    dB_2d = 20 * np.log10(mag_2d)  ##振幅をパワーと同次元にしてデシベル化
    ##画像処理
    umax = 1 / (2 * dx)
    vmax = 1 / (2 * dt)
    # value_max = 1000 ## colormapのvalue 最大値を設定する
    plt.imshow(
        mag_2d,
        extent=(-umax, umax, -vmax, vmax),
        cmap="jet",
        interpolation="nearest",
        aspect="auto",
    )
    # plt.imshow(dB_2d,extent=(-umax, umax, -vmax, vmax), cmap='jet',interpolation='nearest',aspect = 'auto')
    plt.xlim([-umax, umax])
    # plt.xlim([0, umax])
    plt.xlabel("v = 1/λ: $m^{-1}$")
    plt.ylim([0, 100])  ##見たい周波数帯を設定
    plt.ylabel("f : Hz")
    plt.colorbar()
    plt.title(fft2title)
    if not os.path.exists(dirname_2dfft):
        os.makedirs(dirname_2dfft)
    savename = dirname_2dfft + fft2title + ".png"
    plt.savefig(savename)
    plt.close()
    return


@njit(parallel=True)
def DFT_2d(data, fs, dx):  # 時間と空間方向でのDFT
    F2d = np.zeros(data.shape, dtype=np.complex128)
    F2d_f = np.zeros(data.shape[0])
    F2d_k = np.zeros(data.shape[1])
    (N, M) = data.shape
    for l in prange(N):  # サンプル数、f
        # print('sample ' + str(l+1))
        # サンプル値f = l　において
        for s in prange(M):  # センサー数、 kx
            # センサー値x = s において
            F2d_ls = 0 + 0j
            ###以下、積分操作
            for i in prange(N):  # f領域
                a = l * i / N
                for j in range(M):  # x領域
                    b = s * j / M
                    F2d_ls += data[i, j] * np.exp((-2j * np.pi) * (a + b))
            F2d[l, s] = F2d_ls
            F2d_f[l] = (l / N) * fs
            F2d_k[s] = (s / M) * dx - 1 / (2 * dx)
    return F2d, F2d_f, F2d_k


def f(datalist, dlist, xrange, yrange):
    for i in range(len(xrange)):
        for j in range(len(yrange)):
            ind = dlist.index(y)
    return datalist[x, ind]


def convlv(data, respns, isign):
    n = len(data)
    m = len(respns)  # 999なら

    NMAX = 16385  # Maximum anticipated size of FFT
    if m % 2 == 0:
        raise ValueError("response steps should be odd.")
    if n > NMAX:
        raise ValueError("n is greater than NMAX.")
    side = (m - 1) // 2  # 499

    # Put respns in array of length n
    convert = np.zeros(n)
    convert[side] = 1
    respns_extended = np.zeros(n)

    for i in range(side):  # 0,1,2,3,...498
        respns_extended[i] = respns[i + side]
    respns_extended[n - side + 1 :] = respns[1:side][::-1]
    # proc(fs,respns_extended)
    # Perform FFT
    data_fft = np.fft.fft(data)
    respns_fft = np.fft.fft(respns_extended)
    convert_fft = np.fft.fft(convert)

    # Convolution or Deconvolution
    if isign == 1:
        ans_fft = data_fft * respns_fft / n
    elif isign == -1:
        with np.errstate(divide="ignore", invalid="ignore"):
            ans_fft = data_fft / respns_fft  # * convert_fft # / n
            # ans_fft[np.isnan(ans_fft)] = 0
            # ans_fft[np.isinf(ans_fft)] = 0
    else:
        raise ValueError("isign must be 1 (convolution) or -1 (deconvolution).")

    # Inverse FFT to get the answer back in time domain
    ans = np.fft.ifft(ans_fft).real

    return ans[:n]


def v_estimate_from2dFFT(data, fs, Interval_sensor):  ##DFTを自分で作成
    # F2d, F2d_f, F2d_k = DFT_2d(data, fs, Interval_sensor)
    # print('2dDFT finished')

    ###以下、グラフ表示のための関数
    # データ数をそのままで二次元FFT
    F2d = np.fft.fftshift(np.fft.fft2(data))
    # 2次元FFTの軸
    F2d_f = np.linspace(-fs / 2, fs / 2, F2d.shape[0])  # 周波数軸
    F2d_k = np.linspace(
        -1 / (2 * Interval_sensor), 1 / (2 * Interval_sensor), F2d.shape[1]
    )  # 波数軸

    # 解析信号の出力　念の為別に設定
    # S_F2d = np.zeros(F2d.shape, dtype=np.complex128)
    # (N,M) = F2d.shape
    # for l in range(N): #サンプル数、f
    #     for s in range(M): #センサー数、 kx
    #         ## データの並び替え:空間周波数軸において、ナイキスト空間周波数以上はΩを-2πずらす
    #         if M % 2 == 1: #Mは奇数
    #             M_2 = int((M - 1)/2) #M=７の時は3になる
    #             if s <= M_2: #3以下ならsに+3して
    #                 S_F2d[l, s + M_2] = F2d[l,s]
    #             else: #3より大きいならsに−4して
    #                 S_F2d[l, s - M_2 - 1] = F2d[l,s]
    #         if M % 2 == 0: #Mは偶数
    #             M_2 = int(M/2) #M = 6の時は3となる
    #             if s < M_2: #3より小さいなら+3
    #                 S_F2d[l, s + M_2] = F2d[l,s]
    #             else:#3以上なら-3
    #                 S_F2d[l, s - M_2] = F2d[l,s]

    # for i in range(F2d.shape[0]):
    #     S_F2d[i,:] = F2d[i,:] + np.sign(F2d_f[i])*F2d[i,:]

    # S_F2dの最大のインデックスを取得
    # Maxindex_F2d = np.unravel_index(np.argmax(np.abs(S_F2d)), S_F2d.shape)
    Maxindex_F2d = np.unravel_index(np.argmax(np.abs(F2d)), F2d.shape)

    fmax = F2d_f[Maxindex_F2d[0]]
    kmax = F2d_k[Maxindex_F2d[1]]  # はじめが取り出されているので、どうしても

    # ピークの取得・・・f-k　マイグレーションに必要
    peak_Velosity = np.abs(fmax / kmax)
    # #print
    # print('max f is...' + str(fmax))
    # print('max k is...' + str(kmax))
    # print('peak velocity is...' + str(round(peak_Velosity,3)) + 'm/s')
    return peak_Velosity


def callback(x, xtrue, err):
    """callback to track error norm"""
    return err.append(np.linalg.norm(x - xtrue))


def create_convolution_matrix(wavelet, signal_length):
    wavelet_length = len(wavelet)
    # The size of the convolution matrix depends on the handling of the boundaries.
    # For simplicity, we assume zero-padding to keep the output size equal to the input signal length.
    matrix_size = signal_length

    # Initialize the convolution matrix with zeros
    convolution_matrix = np.zeros((matrix_size, wavelet_length))

    # Fill the convolution matrix with shifted versions of the wavelet
    # for i in range(matrix_size):
    #     start_index = i
    #     end_index = i + wavelet_length
    #     convolution_matrix[i, max(0, -start_index):min(wavelet_length, signal_length - start_index)] = wavelet[max(0, start_index):min(end_index, wavelet_length + start_index)]

    # 畳み込み行列にwaveletのシフトされたバージョンを埋め込む
    for i in range(matrix_size):
        for j in range(wavelet_length):
            if i + j < matrix_size:
                convolution_matrix[i + j, j] = wavelet[j]

    # return convolution_matrix
    return convolution_matrix
