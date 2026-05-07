## Wave analysis tools for seismic data processing

この波形処理プログラムは、原佑太郎の博士課程での研究成果をもとに、その波形の解析のためのコードを一般ユーザーに使ってもらえるよう、改変したものである。
探査側線は直線を基本とするが、起伏のある側線にも対応している

## License
This project is licensed under the GNU Lesser General Public License v2.1 (LGPL-2.1).
See the LICENSE file for details.

## Features

x,y,z,3成分における地震波データを受信機の位置や標高と紐づけることで、反射法探査やマイグレーション処理を数コマンドで迅速に行う。

## Parameters

.npz ファイルに必要なキー

1.x,y,zのどれかの波形データ.xyz方向の指定は以下の通り
-x:探査側線に平行な水平方向
-y:探査側線に垂直な水平方向
-z:鉛直方向

 *弾性波動論的には、x,z方向はP-SV波、y方向はSH波と呼ばれる
--- coordinate system ----------------
   y-axis
  /
 /  (sensor positions: o)
/___o__o__o__o__o__o__o____-> x-axis
|
|
|
v
z-axis(vertical)
--- coordinate system ----------------

2.サンプリング間隔
-fs サンプリング周波数(Hz)

3.受信システムの位置関係
-distance 受信機の探査軸を基準とした座標(m)
-interval 受信機の間隔(m)
-source_x 震源の探査軸を基準とした座標(m)
-Num_sensors 受信機数

*distanceの情報がない場合必要な変数
-sensor1_x 側線内の第一センサの座標(m), intervalと共に探査軸を作成する

*探査側線に起伏がある場合
-x_distance: 水平方向の座標(各センサの水平位置)
-z_distance: 鉛直方向の座標(各センサの標高)

*その他、探査側線の情報
-unit: センサーの次元("u"->変位,"v"->速度,"a"->加速度)
-name: 側線の名称
-shot: 震源条件についてのメモ
-condition: 探査側線条件についてのメモ

## Installation

### Using uv (recommended)

```bash
uv pip install -e .
```

### Using pip

```bash
pip install -r requirements.txt
```

requirements.txt contains:
- numpy, scipy, matplotlib, pandas, openpyxl, numba (core)
- dash, plotly (optional, uncomment in requirements.txt if needed)

## Project Structure

```
seismo-thru-any-ai/
├── src/
│   ├── converter/
│   │   └── seg2.py                      # SEG2Reader class
│   ├── processor/
│   │   ├── single_processor.py            # SingleProcesser (single file analysis)
│   │   ├── group_processor.py           # GroupProcesser (multi-file analysis)
│   │   └── plot_processor.py           # PlotProcesser (visualization)
│   ├── mixins/
│   │   ├── single/                      # Single-file analysis mixins
│   │   │   ├── attenuation.py
│   │   │   ├── backscatteranalysis.py
│   │   │   ├── cmp_params.py
│   │   │   ├── dispersion.py
│   │   │   ├── fft1d.py
│   │   │   ├── fft2d.py
│   │   │   ├── filter.py
│   │   │   ├── migration.py
│   │   │   ├── spectra.py
│   │   │   ├── trace_editor.py
│   │   │   └── traveltime_tomography.py
│   │   └── group/                       # Multi-file analysis mixins
│   │       ├── backscatter_distribution.py
│   │       ├── cmp_gathering.py
│   │       ├── dispersion.py
│   │       ├── fk_migration.py
│   │       ├── geomet_align_stack.py
│   │       ├── kirchhoff_MG.py
│   │       ├── nmo_correction.py
│   │       └── rayleigh_inversion.py
│   ├── inversion/
│   │   └── rayleigh/                    # Rayleigh-wave inversion
│   │       ├── forward/                 # Forward solvers
│   │       │   ├── base.py
│   │       │   ├── thomson_haskell.py
│   │       │   └── toy.py
│   │       ├── misfit/                   # Misfit functions
│   │       │   ├── base.py
│   │       │   └── weighted_l2.py
│   │       ├── engine/                 # Optimization engines
│   │       │   ├── base.py
│   │       │   └── damped_lsq.py
│   │       ├── model.py
│   │       ├── section.py
│   │       ├── init_model.py
│   │       ├── picking_qc.py
│   │       └── __init__.py
│   ├── plotting/                        # Plotting system
│   │   ├── __init__.py
│   │   ├── backend_base.py            # PlotterBase (backend ABC)
│   │   ├── wrapper.py                # PlotterWrapperMixin (user-facing API)
│   │   ├── results.py
│   │   └── backends/
│   │       ├── matplotlib_backend.py
│   │       └── plotly_backend.py
│   ├── plotters/                        # Dedicated plotter classes
│   │   ├── backscatter_plotter.py
│   │   ├── backscatter_distribution_plotter.py
│   │   ├── cmp_plotter.py
│   │   ├── dispersion_plotter.py
│   │   ├── fft1d_plotter.py
│   │   ├── fft_transfunc_plotter.py
│   │   ├── fk_plotter.py
│   │   ├── reflection_plotter.py
│   │   ├── seismogram_plotter.py
│   │   ├── spectra_plotter.py
│   │   └── traveltime_tomo_plotter.py
│   └── utils/
│       └── utils.py                     # Utility functions
├── examples/                          # Usage example scripts
│   ├── quickstart_single.py
│   ├── quickstart_group.py
│   ├── nmo_reflection_basic.py
│   ├── migration_kirchhoff.py
│   ├── backscatter_distribution.py
│   ├── stack_horizontal.py
│   ├── traveltime_tomo_uniform_model.py
│   ├── rayleigh_vs_inversion_basic.py
│   └── output/                        # Example output images
├── sample_data/                       # Example data files
│   ├── realdata/                    # Field data
│   ├── simudata/                  # Simulation data
│   └── hor_stack_before/           # Horizontal stacking data
├── output/                          # Output directory for results
├── pyproject.toml                   # uv/pip configuration
└── requirements.txt             # pip requirements (backup)
```
## Parameters 引数とデフォルトの設定

### 解析における引数
freq:list[float] 周波数(Hz). デフォルトは[1,200] ただし、attenuation_fitのfreq引数は"target_freq":floatである
c:list[float] 位相速度(m/s). デフォルトは[1,500]
t:list[float] 時刻(s).       時間範囲を示す
show:bool デフォルトTrue. 描画メソッドの表示。
save_name:str形式。デフォルトはNone 保存パスを設定すると、図を保存する
mode...str形式。Dispersion.dispersion_curve,BackscatterAnalysis.backscatter,_fft_transfunc_image_implで用いているmodeは「計算、解析」のモードなので、引数名はそのまま。
「fill, plot, scatter」といった描画モードを指しているものは、plot_modeとする。
title:str デフォルト:Noneに統一する。

### 描画における引数
各種プロットメソッドは `show`, `save_name`, `figsize`, `cmap`, `vmin`, `vmax`, `colorbar`, `xlabel`, `ylabel`, `title` などの表示オプションを **kwargs として受け取ります。
どの引数がどのバックエンドでどのように解釈されるかについては、`src/plotting/backends/matplotlib_backend.py` および `src/plotting/backends/plotly_backend.py` のファイル冒頭にある「描画オプション kwargs 一覧」を参照してください。

#### kwargs ナビゲーション

| 見たい情報 | 場所 |
|---|---|
| 基本的な使い方・メソッド一覧 | この README の Usage セクション |
| 各メソッドの入口と代表 kwargs | `src/plotting/wrapper.py` 内の各メソッド docstring |
| 描画 kwargs の完全一覧・backend 差異 | `src/plotting/backends/matplotlib_backend.py` / `plotly_backend.py` 冒頭 |

**kwargs 分類の目安**（代表例のみ・完全一覧は backend 冒頭を参照）:

- **signal 系** (`seismogram`, `cmap`, `backscatter_image` 等): `figsize`, `color`, `xlabel`, `ylabel`
- **imshow 系** (`spectra_image`, `reflection_image`, `fk_image`, `dispersion_image` 等): `cmap`, `interpolation`, `aspect`, `vmin`, `vmax`, `colorbar`, `figsize`
- **seismogram / CMP 系** (`seismogram`, `cmp_image`): `xmin`, `xmax`, `plot_mode`, `color`, `spacing`, `fill`, `agc`
- **FFT 系** (`fft1d_image`, `fft_transfunc_image`): `complex_mode`, `xlim`, `ylim`, `t_xlim`, `f_xlim`, `f_ylim`, `color`, `figsize`
- **attenuation 系** (`attenuation_fit`): `velocity`, `dB`


## Usage

```python
from src.processor.single_processor import SingleProcesser
from src.processor.group_processor import GroupProcesser
from src.processor.plot_processor import PlotProcesser
from src.converter.seg2 import SEG2Reader
import src.utils.utils as utils

# Read SEG2 files
reader = SEG2Reader('data.sg2')

# Process single wave data
hr = SingleProcesser('data.npz')

# Process multiple files as a group
import glob
datalist = [SingleProcesser(p) for p in glob.glob('data/*.npz')]
group = GroupProcesser(datalist)
```

### SingleProcesser

`SingleProcesser` inherits all analysis mixins:

| Mixin | Key methods |
|---|---|
| TraceEditor | `timepick`, `gain_comp`, `gain_geomet`, `gain_AGC`, `closs_corr`, `deconvolution_fft`, `envelope`, `integral`, `differential`, `remove`, `sort_distance`, `trace_amp_regularize`, `save` |
| Filter | `highpass`, `lowpass`, `bandpass`, `denoise_upgoing_wave` |
| FFT1D | `FFT`, `FFT_transfunc` |
| FFT2D | `fk` |
| Dispersion | `dispersion_curve`, `dispersion_curve_with_selected_chs` |
| Spectra | `spectra` |
| Attenuation | `attenuation_analysis` |
| Migration | `SH_stoltMig`, `SH_stoltMig_poststuck` |
| CmpParams | `get_all_CMP`, `get_all_CMPdist` |
| BackscatterAnalysis | `backscatter` |
| PlotterWrapperMixin | `seismogram`, `show`, `createcolormap_Amp`, `plot_seismogram` |

### GroupProcesser

`GroupProcesser` inherits all group analysis mixins:

| Mixin | Key methods |
|---|---|
| backscatter_distribution | `backscatter_distribution` |
| CmpGathering | `cmp_gathering` |
| NMO_correction | `NMO_correction` |
| Kirchhoff_MG | `Kirchhoff_migration` |
| GeometryAlignedStacker | `align_stack_by_geometry` |
| GroupDispersion | `dispersion_curve` |

See the `testcodes/` directory for runnable example scripts.
