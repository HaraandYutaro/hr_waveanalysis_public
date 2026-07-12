#! /usr/bin/python3
"""
HaraFormatNpzProcesser class for seismic wave analysis

This module provides the HaraFormatNpzProcesser class for processing and analyzing a single seismic wave data.
"""

#! /usr/bin/python3
import copy
import os
import warnings

import numpy as np
from matplotlib import pyplot as plt

warnings.filterwarnings("ignore")

plt.style.use("fast")

## Mixin import ##
from src.io.dataset_loader import FallbackMetadataWarning, load_dataset
from src.plotting.wrapper import PlotterWrapperMixin

# The module-level ``filterwarnings("ignore")`` above silences every
# UserWarning, including the fallback-metadata warnings emitted by
# load_dataset() on the sg2 path. Re-arm just our category so the user
# always hears about JSON / Excel / default-metadata transitions.
warnings.simplefilter("always", FallbackMetadataWarning)
from src.mixins.single.attenuation import Attenuation
from src.mixins.single.backscatteranalysis import BackscatterAnalysis
from src.mixins.single.cmp_params import CmpParams
from src.mixins.single.dispersion import Dispersion
from src.mixins.single.fft1d import FFT1D
from src.mixins.single.fft2d import FFT2D
from src.mixins.single.filter import Filter
from src.mixins.single.spectra import Spectra
from src.mixins.single.trace_editor import TraceEditor
from src.mixins.single.traveltime_tomography import TraveltimeTomography


class SingleProcesser(
    Attenuation,
    BackscatterAnalysis,
    CmpParams,
    Dispersion,
    FFT1D,
    FFT2D,
    Filter,
    Spectra,
    TraceEditor,
    TraveltimeTomography,
    PlotterWrapperMixin,
):
    """
    単一ファイルの地震波データを解析・描画するプロセッサ

    描画メソッド (seismogram, spectra_image, cmap 等) は
    ``PlotterWrapperMixin`` から継承しており、各メソッドの代表 kwargs は
    ``src/plotting/wrapper.py`` の docstring に記載されている。
    描画 kwargs の完全一覧は
    ``src/plotting/backends/matplotlib_backend.py`` /
    ``src/plotting/backends/plotly_backend.py`` 冒頭を参照
    """
    def __init__(self, file_path, backend="mpl", *, datasheet_path=None, obs_id=None):
        """
        Load a single seismic survey record from a ``.npz`` or ``.sg2`` file.

        Parameters
        ----------
        file_path : str
            Path to the input file. ``.npz`` is loaded directly. ``.sg2``
            is loaded together with metadata resolved in the order
            JSON sidecar -> Excel datasheet -> default placeholders.
        backend : str, default "mpl"
            Plotting backend identifier forwarded to the plotter wrapper.
            ``"mpl"`` selects Matplotlib.
        datasheet_path, obs_id : optional
            Used only on the ``.sg2`` path to enable the Excel fallback
            when the JSON sidecar is absent. Both must be supplied;
            ``obs_id`` is **never** inferred from the filename. If
            either is missing, the sg2 path skips Excel and falls
            through to default metadata.

        Notes
        -----
        ``unit`` encodes the physical quantity recorded on the axis
        arrays: ``"v"`` — particle velocity [m/s], ``"u"`` — displacement
        [m], ``"a"`` — acceleration [m/s²].

        ``analysis_x``, ``analysis_y``, and ``analysis_z`` are string
        attributes initialised to ``""`` that accumulate a history of
        processing steps applied to the corresponding axis array
        (e.g. ``"_lowpass_highpass"``).

        Two provenance attributes are always set:

        - ``metadata_source`` — one of ``"json"``, ``"excel"``,
          ``"fallback_default"``, or ``None`` (npz path).
        - ``metadata_is_fallback`` — ``True`` only when default
          placeholder values were used. Check this before running
          geometry-sensitive analyses on sg2 inputs.
        """
        file = load_dataset(
            file_path, datasheet_path=datasheet_path, obs_id=obs_id
        )
        self._input_name(file_path)
        self._input_axis(file)
        self._input_acquisition_parameters(file)
        self._input_conditions(file)
        self._input_optional_parameters(file)
        self.metadata_source = file.get("metadata_source")
        self.metadata_is_fallback = bool(file.get("metadata_is_fallback", False))
        self.backend = backend

    def _input_optional_parameters(self, file):
        if "x_distance" in file:
            self.x_distance = file["x_distance"]
        if "z_distance" in file:
            self.z_distance = file["z_distance"]

    def _input_name(self, file_path):
        self.name = os.path.splitext(os.path.basename(file_path))[0]

    def _input_conditions(self, file):
        if "shot" in file:
            self.shot = file["shot"]
        if "condition" in file:
            self.condition = str(file["condition"])

    def _input_acquisition_parameters(self, file):
        if "fs" in file:
            self.fs = file["fs"]
        if "source_x" in file:
            self.source_x = file["source_x"]
        if "sensor1_x" in file:
            self.sensor1_x = file["sensor1_x"]
        if "interval" in file:
            self.interval = file["interval"]
        if "Num_sensor" in file:
            self.Num_sensor = file["Num_sensor"]
        if "distance" in file:
            self.distance = file["distance"]
        if "unit" in file:
            self.unit = str(file["unit"])

    def _input_axis(self, file):
        if "x" in file:
            self.x = file["x"]
            self.analysis_x = ""
        if "y" in file:
            self.y = file["y"]
            self.analysis_y = ""
        if "z" in file:
            self.z = file["z"]
            self.analysis_z = ""

    # =================================================================
    # インスタンス複製用メソッド
    # =================================================================
    def _singleprocessor(self, sp):
        """
        SingleProcesserインスタンスの読み込み用内部メソッド。
        データの意図しない書き換え（副作用）を防ぐため、配列はコピーして引き継ぐ。
        """
        self.name = getattr(sp, "name", "Copied_Data")

        # 1. 軸データと解析履歴のコピー
        for axis in ["x", "y", "z", "ax1", "ax2"]:
            if hasattr(sp, axis):
                # NumPy配列の複製
                setattr(self, axis, getattr(sp, axis).copy())
                # 解析履歴文字列の複製
                setattr(self, f"analysis_{axis}", getattr(sp, f"analysis_{axis}", ""))

        # 2. 取得パラメータ・条件パラメータのコピー
        param_keys = [
            "fs",
            "interval",
            "source_x",
            "sensor1_x",
            "Num_sensor",
            "distance",
            "unit",
            "shot",
            "condition",
            "x_distance",
            "z_distance",
        ]
        for key in param_keys:
            if hasattr(sp, key):
                val = getattr(sp, key)
                # NumPy配列なら .copy()、それ以外（数値や文字列など）は deepcopy
                if isinstance(val, np.ndarray):
                    setattr(self, key, val.copy())
                else:
                    setattr(self, key, copy.deepcopy(val))

    # =================================================================
    # 直接引数(kwargs)用メソッド
    # =================================================================
    def _input_kwargs(self, **kwargs):
        """配列やパラメータを直接受け取るための内部メソッド"""
        self.name = kwargs.get("name", "Raw_Data")

        for axis in ["x", "y", "z"]:
            if axis in kwargs:
                setattr(self, axis, kwargs[axis])
                setattr(self, f"analysis_{axis}", kwargs.get(f"analysis_{axis}", ""))

        self.fs = kwargs.get("fs", 1.0)
        self.interval = kwargs.get("interval", 1.0)
        self.source_x = kwargs.get("source_x", None)
        self.sensor1_x = kwargs.get("sensor1_x", None)
        self.Num_sensor = kwargs.get("Num_sensor", 0)
        self.distance = kwargs.get("distance", None)
        self.unit = kwargs.get("unit", "v")
        self.shot = kwargs.get("shot", "")
        self.condition = kwargs.get("condition", "")

        if "x_distance" in kwargs:
            self.x_distance = kwargs["x_distance"]
        if "z_distance" in kwargs:
            self.z_distance = kwargs["z_distance"]

    def cat_information(self):
        """データの基本情報をまとめて表示"""
        info = {
            "名前": self.name,
            "サンプリング周波数 (Hz)": self.fs,
            "センサー数": self.Num_sensor,
            "センサー間隔 (m)": self.interval,
            "震源位置 (m)": self.source_x,
            "第1センサー位置 (m)": self.sensor1_x,
            "データ形状": self.y.shape,  # (96, 8192)
            "記録時間 (s)": self.y.shape[1] / self.fs,
            "震源最近傍のch": self.get_source_ch,
        }
        for key, value in info.items():
            print(f"{key}: {value}")
