"""
HaraFormatNpzProcesser class for seismic wave analysis

This module provides the HaraFormatNpzProcesser class for processing and analyzing a grounp of seismic wave data files.
"""

#! /usr/bin/python3
import warnings

import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use("fast")

## Mixin import ##
from src.mixins.group.backscatter_distribution import backscatter_distribution
from src.mixins.group.cmp_gathering import CmpGathering
from src.mixins.group.dispersion import GroupDispersion
from src.mixins.group.fk_Migration import Fk_Migration
from src.mixins.group.geomet_align_stack import GeometryAlignedStacker
from src.mixins.group.kirchhoff_MG import Kirchhoff_MG
from src.mixins.group.nmo_correction import NMO_correction
from src.mixins.group.rayleigh_inversion import RayleighInversion
from src.processor.single_processor import SingleProcesser


class GroupProcesser(
    backscatter_distribution,
    CmpGathering,
    NMO_correction,
    Kirchhoff_MG,
    GeometryAlignedStacker,
    GroupDispersion,
    Fk_Migration,
    RayleighInversion,
):
    """
    複数ファイルの地震波データを対象としたグループ解析プロセッサ

    解析結果の描画を行う際は、各 SingleProcesser の描画メソッドを経由する。
    代表 kwargs は ``src/plotting/wrapper.py`` の docstring、
    描画 kwargs の完全一覧は
    ``src/plotting/backends/matplotlib_backend.py`` /
    ``src/plotting/backends/plotly_backend.py`` 冒頭を参照。
    """
    def __init__(self, datalist: list, name: str | None = None):
        """
        Build a group processor from multiple seismic survey records.

        Parameters
        ----------
        datalist : list
            Each element must be either a :class:`SingleProcesser`
            instance or a ``str`` path to a .npz file; paths are loaded
            into :class:`SingleProcesser` internally.  All records must
            share the same sampling frequency.
        name : str or None, optional
            Label for the survey line.  Stored as ``self.name``.

        Raises
        ------
        ValueError
            If two or more records in *datalist* have different sampling
            frequencies.
        TypeError
            If any element of *datalist* is neither a
            :class:`SingleProcesser` nor a ``str``.
        """
        fs = self._check_consistency_fs(datalist)
        datalist_sp = self._normalize_datalist(datalist)
        all_distance, counts = self._get_all_distance_and_counts(datalist)

        self.name = name
        self.datalist = datalist_sp
        self.fs = fs
        self.all_distance = all_distance
        self.count_distance = counts

        axis_bool = self._check_consistency_axis()
        self.axis_bool = axis_bool

    def _normalize_datalist(self, datalist):
        """
        datalist 内が SingleProcesser インスタンスならそのまま、
        文字列パスなら SingleProcesser にロードして揃えて返す。
        """
        normalized = []
        for item in datalist:
            if isinstance(item, SingleProcesser):
                # すでにロード済み
                normalized.append(item)
            elif isinstance(item, str):
                # npz パスとしてロード
                normalized.append(SingleProcesser(item))
            else:
                raise TypeError(
                    f"datalist の要素は SingleProcesser か str を期待しましたが、{type(item)} が来ました: {item!r}"
                )
        return normalized

    # datalist内のdistance要素をすべて取得したアレイを取り出す
    def _get_all_distance_and_counts(self, datalist):
        """
        datalist 内の各 data.distance に登場する値を
        - 重複なしの昇順配列 all_distance
        - 各値の出現回数 counts
        として返す。
        欠損で長さがバラバラでもOK。
        """

        # datalist 内の distance をすべて 1 次元に連結
        all_dist = np.concatenate(
            [np.asarray(data.distance).ravel() for data in datalist]
        )

        # ユニーク値とその出現回数を取得
        all_distance, counts = np.unique(all_dist, return_counts=True)

        return all_distance, counts

    def _check_consistency_fs(self, hr2c_list: list) -> None:
        """
        Ensure all Hr2c objects share sampling frequency and preprocessing.
        Raises ValueError if mismatches found.
        """
        base_fs = hr2c_list[0].fs

        for hr in hr2c_list[1:]:
            if hr.fs != base_fs:
                raise ValueError("Sampling frequency mismatch.")
            # if getattr(hr, f"analysis_{axis}") != base_pre:
            #     raise ValueError("Preprocessing mismatch.")
        return base_fs

    def _check_consistency_axis(self) -> dict[str, bool]:
        """
        datalist 内の各データが x, y, z 軸を持っているかをチェックする。

        Parameters
        ----------
        axis : str
            'x', 'y', 'z' のいずれかを想定（将来確認したい軸を増やす場合は呼び出し側で制御）。

        Returns
        -------
        ret : dict[str, bool]
            各軸ごとに、datalist 内の全要素がその属性を持っていれば True、
            どれか1つでも欠けていれば False。
        """
        axes = ["x", "y", "z"]
        ret: dict[str, bool] = {}

        for ax in axes:
            # datalist 内の data がすべて ax 属性を持っているか
            ret[ax] = all(hasattr(data, ax) for data in self.datalist)

        return ret
