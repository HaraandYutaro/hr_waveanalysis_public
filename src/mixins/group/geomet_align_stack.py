import copy
from typing import Literal, Optional, Sequence

import numpy as np
from scipy import signal

from src.plotting.wrapper import PlotterWrapperMixin
from src.processor.single_processor import SingleProcesser


class GeometryAlignedStacker(PlotterWrapperMixin):
    def align_stack_by_geometry(
        self,
        datalist: list,
        axis: Literal["x", "y", "z"],
        ref_distance: float,
        ref_height: float = None,
        use_cross_correlation: bool = True,
        time_lags: Optional[Sequence[int]] = None,
        show: bool = False,
        debug: bool = False,
        reverse_list: Optional[Sequence[int]] = None,
        name: Optional[str] = None,
    ) -> SingleProcesser:
        """
        GroupProcesser.datalist (list[SingleProcesser]) を入力として，
        (distance, height) が一致するチャンネルを基準に時間合わせ＆スタックする。

        Parameters
        ----------
        datalist : list[SingleProcesser]
            同一 source_x, 同一 axis, 同一 fs を前提としたデータ群。
        axis : {'x','y','z'}
            解析対象の軸。
        ref_distance : float
            基準受信点の distance。
        ref_height : float
            基準受信点の height (z_distance 等)。
        use_cross_correlation : bool, default True
            True の場合は基準トレースとの相互相関で time lag を推定。
            False の場合は time_lags 引数をそのまま使用。
        time_lags : Sequence[int] or None
            use_cross_correlation=False のとき，各データに適用するサンプルシフト量リスト。
            len(time_lags) == len(datalist) を要求。
        show : bool, default False
            True のとき，スタック結果を seismogram() で表示。
        debug : bool, default False
            True のとき，中間結果を色々表示・簡易プロット（PlotterWrapperMixin 依存）。
        reverse_list : Sequence[int] or None
            各 SingleProcesser に対して振幅を ±1, 0 などで反転／無視するための係数。
            None の場合はすべて 1 とみなす。
        name : str or None
            返却する SingleProcesser の name を上書きしたい場合に指定。

        Returns
        -------
        stacked_sp : SingleProcesser
            時間合わせ＋distance 統合済みの SingleProcesser インスタンス

        複数のファイルをスタックして、一つのファイルを生成する
        distanceとaxを統合する。axisとsource_xのあっていないdataは”大丈夫?”と確認する機能をつけたい。
        -> distanceの構成が異なるデータにおいても結合が可能。
        return Singleprocesser,

        ほしい機能、
        save == Trueなら、savenameで名前を指定して保存できる機能
        average == Trueなら、同一のdistanceを持っているデータは平均化する


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

        stacked_sp = self._align_stack_by_geometry_core(
            datalist,
            axis,
            ref_distance,
            ref_height,
            use_cross_correlation,
            time_lags,
            reverse_list,
            debug,
        )

        # --------------------------------------------------
        # 6. show=True のとき，スタック結果を可視化
        # --------------------------------------------------
        if show:
            # PlotterWrapperMixin 経由で seismogram を呼ぶ想定
            try:
                stacked_sp.seismogram(
                    axis,
                    distance_sort=True,
                    title=name or f"GeomAlignedStack_{stacked_sp.name}",
                )
            except AttributeError:
                # seismogram が SingleProcesser 側の mixin にある前提なので，
                # そこに委譲されることを期待
                pass

        # 名前の上書き
        if name is not None:
            stacked_sp.name = name

        return stacked_sp

    def _align_stack_by_geometry_core(
        self,
        datalist,
        axis,
        ref_distance,
        ref_height,
        use_cross_correlation,
        time_lags,
        reverse_list,
        debug,
    ):
        """
        align_stack_by_geometry() の純粋計算部 (描画なし)。

        Section 1: 入力検証 (datalist 空・fs/source_x/axis 整合性)
        Section 2: 各 sp の参照チャネル決定
        Section 3: time_lags の確定 (cross-correlation または引数)
        Section 4: 時間合わせ + (distance, height) ベースの蓄積
        Section 5: 平均化 + distance 統合 + stacked_sp 生成

        Returns
        -------
        stacked_sp : SingleProcesser
        """
        if len(datalist) == 0:
            raise ValueError("datalist is empty.")

        # --------------------------------------------------
        # 1. source_x, axis, fs の整合性チェック
        # --------------------------------------------------
        fs_list = []
        source_x_list = []
        axis_data_exist = []

        for sp in datalist:
            fs_list.append(float(sp.fs))
            source_x_list.append(float(sp.source_x))
            # SingleProcesser が axis を持っているかチェック
            ax, _ = sp.getax_analysis(axis)
            axis_data_exist.append(ax is not None)

        if len(set(fs_list)) != 1:
            raise ValueError(
                f"Inconsistent sampling frequency in datalist: {set(fs_list)}"
            )

        if len(set(source_x_list)) != 1:
            raise ValueError(f"Inconsistent source_x in datalist: {set(source_x_list)}")

        if not all(axis_data_exist):
            raise ValueError(f"Some data do not have requested axis '{axis}'.")

        fs = fs_list[0]
        source_x = source_x_list[0]

        # --------------------------------------------------
        # 2. 基準チャンネルの決定
        #    -> ref_distance, ref_height と一致する ch を探す。
        #    -> なければ震源に一番近い distance & ref_height の ch を採用
        # --------------------------------------------------
        # 各 sp ごとに ref_ch を決定（レイアウトが異なる場合に対応）
        ref_channels = [
            self._find_ref_channel(sp, ref_distance, ref_height, source_x)
            for sp in datalist
        ]
        if debug:
            print(f"Reference channel indices per file: {ref_channels}")

        # --------------------------------------------------
        # 3. 各データの time_delay を算出
        # --------------------------------------------------
        n_traces = len(datalist)
        if time_lags is not None and not use_cross_correlation:
            if len(time_lags) != n_traces:
                raise ValueError("time_lags length must equal len(datalist).")
        else:
            time_lags = [0] * n_traces

        # --------------------------------------------------
        # reverse_list の準備
        # --------------------------------------------------
        if reverse_list is None:
            reverse_list = [1] * n_traces
        elif len(reverse_list) != n_traces:
            raise ValueError("reverse_list length must equal len(datalist).")

        mother_sp = datalist[0]
        mother_ax, mother_analysis = mother_sp.getax_analysis(axis)
        n_samples = mother_ax.shape[1]
        mothersrc = mother_ax[ref_channels[0], :].astype(np.float64)
        mothersrc -= np.mean(mothersrc)

        if use_cross_correlation:
            for i, sp in enumerate(datalist):
                if i == 0:
                    continue
                child_ax, _ = sp.getax_analysis(axis)
                childsrc = child_ax[ref_channels[i], :].astype(np.float64)
                childsrc -= np.mean(childsrc)
                correlation = signal.correlate(childsrc, mothersrc, mode="full")
                time_lags[i] = int(np.argmax(correlation)) - (len(childsrc) - 1)
                if debug:
                    print(f"[{i}] time_delay = {time_lags[i]} [sample]")

        # --------------------------------------------------
        # 4. 各ファイルを時間合わせ → 位置ベースで蓄積
        #    distance は実チャンネル数分 (ax.shape[0]) だけ使用する。
        #    各ファイルの distance[0..n_ch-1] が実際の ch に対応。
        # --------------------------------------------------
        accum: dict[tuple, list] = {}  # (distance, height) -> list of traces

        for i, sp in enumerate(datalist):
            child_ax, _ = sp.getax_analysis(axis)
            n_ch = child_ax.shape[0]

            d_arr = np.array(sp.distance)[:n_ch]
            if hasattr(sp, "z_distance") and sp.z_distance is not None:
                h_arr = np.array(sp.z_distance)[:n_ch]
            else:
                h_arr = np.zeros(n_ch)

            td = time_lags[i]
            if td > 0:
                aligned = np.pad(child_ax, ((0, 0), (0, td)), mode="constant")[:, td:]
            elif td < 0:
                aligned = np.pad(child_ax, ((0, 0), (-td, 0)), mode="constant")[:, :td]
            else:
                aligned = child_ax.copy().astype(np.float64)

            aligned = aligned.astype(np.float64) * reverse_list[i]

            for ch_idx, (d, h) in enumerate(zip(d_arr, h_arr)):
                key = (float(d), float(h))
                accum.setdefault(key, []).append(aligned[ch_idx])

        # --------------------------------------------------
        # 5. 平均化 & distance 統合
        # --------------------------------------------------
        keys_sorted = sorted(accum.keys(), key=lambda x: (x[0], x[1]))
        new_distance = np.array([k[0] for k in keys_sorted], dtype=np.float32)
        new_height = np.array([k[1] for k in keys_sorted], dtype=np.float32)

        new_ax = np.zeros((len(keys_sorted), n_samples), dtype=np.float64)
        for pi, key in enumerate(keys_sorted):
            traces = accum[key]
            new_ax[pi] = np.mean(traces, axis=0)

        stacked_analysis = mother_analysis + f"_geom_align_stack_{n_traces}shots"

        # stacked_sp の生成
        stacked_sp = copy.deepcopy(mother_sp)
        stacked_sp.distance = new_distance
        stacked_sp.Num_sensor = len(new_distance)
        stacked_sp.sensor1_x = float(new_distance.min())
        if len(new_distance) > 1:
            stacked_sp.interval = float(np.median(np.diff(new_distance)))
        stacked_sp.z_distance = new_height
        stacked_sp.putax_analysis(axis, new_ax.astype(np.float32), stacked_analysis)

        if debug:
            print(
                f"Output: {len(new_distance)} channels, distance range "
                f"[{new_distance.min():.4f}, {new_distance.max():.4f}]"
            )

        return stacked_sp

    def _find_ref_channel(self, sp, ref_distance, ref_height, source_x):
        """
        align_stack_by_geometry() の参照チャネル決定ロジック。

        以前 align_stack_by_geometry() 内に入れ子定義されていた find_ref_channel を
        メソッドへ昇格。closure capture していた source_x を明示パラメータ化。
        use_height=True かつ完全一致が無いときのフォールバックでのみ source_x を参照する。
        """
        distances = np.array(sp.distance)
        use_height = (
            ref_height is not None
            and hasattr(sp, "z_distance")
            and sp.z_distance is not None
        )
        if use_height:
            heights = np.array(sp.z_distance)

        # 完全一致探索
        if use_height:
            idx = np.where(
                np.isclose(distances, ref_distance)
                & np.isclose(heights, ref_height)
            )[0]
        else:
            idx = np.where(np.isclose(distances, ref_distance))[0]
        if len(idx) > 0:
            return int(idx[0])

        # 一致チャンネルが無い場合
        if use_height:
            # まず height が一番 ref_height に近いものを絞り込み
            height_diff = np.abs(heights - ref_height)
            min_height = np.min(height_diff)
            candidate = np.where(height_diff == min_height)[0]
            distances_cand = distances[candidate]
            dist_diff = np.abs(distances_cand - source_x)
            best_idx_local = np.argmin(dist_diff)
            return int(candidate[best_idx_local])
        else:
            # distance だけで最近傍を返す
            dist_diff = np.abs(distances - ref_distance)
            return int(np.argmin(dist_diff))
