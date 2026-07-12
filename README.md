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
```text
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
```

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

このプロジェクトは [`uv`](https://docs.astral.sh/uv/) による環境構築を推奨する。
`pyproject.toml` と `uv.lock` により、依存を固定した再現性のある `.venv` を作成できる。
動作確認環境: Python 3.11 / uv 0.11。`requires-python` は `>=3.8`。

### Using uv (recommended)

```bash
# 依存を .venv に同期 (uv.lock に基づく再現ビルド)
uv sync

# 可視化 (dash) の optional 依存も入れる場合
uv sync --extra viz

# 開発用 (pytest) まで含めてすべて入れる場合
uv sync --all-groups --all-extras
```

`uv sync` は `.venv/` を自動生成する。以降はコマンドを `uv run` 経由で実行する:

```bash
uv run python examples/quickstart_single.py
uv run pytest
```

環境が正しく構築できたかは検証スクリプトで確認できる:

```bash
uv run python verify_env.py
```

> 注: 本プロジェクトは `pyproject.toml` で `package = false` を指定しており、
> パッケージ自体はインストールせず依存のみを `.venv` に入れる。ソースは
> `src.` からの import と `pythonpath=["."]` で参照するため、
> `uv pip install -e .` は不要。

### Using pip

`uv` を使わない場合は仮想環境を作成し、requirements.txt から導入する:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

requirements.txt は `pyproject.toml` の依存を反映しており、以下を含む:
- numpy, scipy, matplotlib, pandas, openpyxl, numba, plotly, pyqt6, disba (core)
- dash (optional, 可視化。必要な場合 requirements.txt のコメントを外す)

## Project Structure

```
hr_waveanalysis_public/
├── src/
│   ├── converter/                       # 入力データの読み込み・変換レイヤ
│   │   ├── seg2.py                      # 後方互換の re-export (SEG2Reader)
│   │   ├── seg2_reader.py              # SEG2Reader class (SEG-2 バイナリ読み込み)
│   │   ├── datasheet_parser.py         # 調査野帳 Excel の解析
│   │   ├── json_sidecar.py            # 正規化メタデータの JSON sidecar I/O
│   │   └── npz_adapter.py             # npz 物理 I/O 境界
│   ├── io/
│   │   └── dataset_loader.py            # sg2 + JSON sidecar のデータセット読み込み
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
│   │   │   ├── spectra.py
│   │   │   ├── trace_editor.py
│   │   │   ├── traveltime_tomography.py   # SPM/Dijkstra 走時トモグラフィ
│   │   │   └── traveltime_fmm.py         # Fast Marching Method 走時ソルバ
│   │   └── group/                       # Multi-file analysis mixins
│   │       ├── backscatter_distribution.py
│   │       ├── cmp_gathering.py
│   │       ├── dispersion.py
│   │       ├── fk_Migration.py
│   │       ├── geomet_align_stack.py
│   │       ├── kirchhoff_MG.py
│   │       ├── nmo_correction.py
│   │       ├── rayleigh_inversion.py
│   │       └── love_inversion.py         # SH-Love wave インバージョン
│   ├── inversion/
│   │   ├── surface_wave/                # 波動種非依存の正準パッケージ
│   │   │   ├── forward/                 # Forward solvers (base)
│   │   │   ├── misfit/                   # Misfit functions
│   │   │   ├── engine/                 # Optimization engines
│   │   │   ├── model.py
│   │   │   ├── section.py
│   │   │   └── picking_qc.py
│   │   ├── rayleigh/                    # Rayleigh-wave inversion
│   │   │   ├── forward/                 # base / thomson_haskell / toy
│   │   │   ├── misfit/                   # base / determinant / nearest_neighbor / weighted_l2
│   │   │   ├── engine/                 # base / damped_lsq / lci / pso
│   │   │   ├── model.py
│   │   │   ├── section.py
│   │   │   ├── init_model.py
│   │   │   └── picking_qc.py
│   │   └── love/                        # Love-wave inversion
│   │       ├── forward/                 # thomson_haskell_love
│   │       ├── secular.py             # Love-wave 特性(secular)関数
│   │       └── init_model.py
│   ├── plotting/                        # Plotting system
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
│   │   ├── rayleigh_inversion_plotter.py
│   │   ├── love_inversion_plotter.py
│   │   ├── manual_pick_plotter.py
│   │   ├── reflection_plotter.py
│   │   ├── seismogram_plotter.py
│   │   ├── spectra_plotter.py
│   │   └── traveltime_tomo_plotter.py
│   └── utils/
│       └── utils.py                     # Utility functions
├── examples/                          # Usage example scripts
│   ├── quickstart_single.py
│   ├── quickstart_group.py
│   ├── quickstart_dat_sg2.py
│   ├── quickstart_dat_with_datasheet.py
│   ├── quickstart_backscatter_distribution.py
│   ├── quickstart_kirchhoff.py
│   ├── nmo_reflection_basic.py
│   ├── stack_horizontal.py
│   ├── rayleigh_inversion.py
│   ├── rayleigh_inversion_basic.py
│   └── output/                        # Example output images
├── sample_data/                       # Example data files
│   ├── datasheet/                   # 調査野帳 Excel の例
│   ├── npz/                         # npz 波形データ
│   │   ├── realdata/                # Field data
│   │   ├── simudata/               # Simulation data
│   │   └── hor_stack_before/       # Horizontal stacking data
│   └── sg2/                        # SEG-2 (.DAT) + JSON sidecar の例
├── tests/                           # pytest によるテスト
├── pyproject.toml                   # uv/pip configuration
├── requirements.txt             # pip requirements (backup)
└── uv.lock                      # uv lock file
```
## Parameters 引数とデフォルトの設定

### 解析における引数
`freq`:list[float] 周波数(Hz). デフォルトは[1,200] ただし、attenuation_fitのfreq引数は"target_freq":floatである
`c`:list[float] 位相速度(m/s). デフォルトは[1,500]
`t`:list[float] 時刻(s).       時間範囲を示す

### 描画における引数
各種プロットメソッドは `show`, `save_name`, `figsize`, `cmap`, `vmin`, `vmax`, `colorbar`, `xlabel`, `ylabel`, `title` などの表示オプションを **kwargs として受け取る。
どの引数がどのバックエンドでどのように解釈されるかについては、`src/plotting/backends/matplotlib_backend.py` および `src/plotting/backends/plotly_backend.py` のファイル冒頭にある「描画オプション kwargs 一覧」を参照すること。

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

See the `examples/` directory for runnable example scripts.
