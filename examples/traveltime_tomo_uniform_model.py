"""走時トモグラフィ例題スクリプト。

既存の SingleProcesser と traveltime_tomography() を用いて、
均一速度モデルから生成した数値シミュレーションデータで
SIRT 反復トモグラフィが収束することを確認する。

描画は plotting パイプライン経由で行う。
show=True を渡すことで、計算後に自動的に 2x2 描画が表示される。

テストデータ: sample_data/simudata/D0_6W2_0S0.npz

作成時のメモ（数値実験条件）:
- 均一速度モデルからの数値データを用いた example では、以下の設定で安定に収束を確認した:
    - 反復回数: max_iter = 20
    - 平滑化: smooth_sigma = 0.5
    - 収束判定: inversion_tol = 1e-5
    - スローネスのクリップ範囲: s_min = 1/8000 [s/m], s_max = 1/50 [s/m]
- 上記条件のもとで、RMS misfit は約 0.0030 → 0.0006 まで減少し、
  推定速度モデルは 77–180 m/s の範囲に収まり、数値的に安定した収束挙動を示した。
- ライブラリのデフォルト値は保存されており
    smooth_sigma = 1.0, inversion_tol = 1e-6,
    s_min = 1/6000 [s/m], s_max = 1/200 [s/m]
  である。上記は example 用の一例であり、問題設定に応じて調整するべきである。
- 収束挙動の診断のため、traveltime_tomography() の反復ループ内には
  初期 RMS、および各反復ごとの前後統計量・早期終了理由を stdout に出力する
  診断用 print() を追加している（API や返り値には影響しない）
"""

import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.single_processor import SingleProcesser

NPZ_PATH = "sample_data/simudata/D0_6W2_0S0.npz"
AXIS = "y"
BACKEND = "mpl"


def main():
    """走時トモグラフィの実行と結果の図示。"""
    # ------------------------------------------------------------------
    # 1) データ読み込み
    # ------------------------------------------------------------------
    hr = SingleProcesser(NPZ_PATH, BACKEND)
    hr.trace_amp_regularize(AXIS) # 振幅正規化（例題の数値データは振幅が大きく異なるため、トモグラフィの安定化のために正規化を行う）
    # ------------------------------------------------------------------
    # 2) トモグラフィ実行 (show=True でパイプライン経由の描画を自動表示)
    # ------------------------------------------------------------------
    result = hr.traveltime_tomography(
        AXIS,
        picking_mode="energy_threshold",
        initial_method="apparent_velocity",
        inversion_method="sirt",
        n_iter=20,
        smooth_sigma=0.5,
        inversion_tol=1e-5,
        vmin_init=20.0,
        vmax_init=200.0,
        s_min=1.0 / 8000.0,
        s_max=1.0 / 50.0,
        show=True,
    )


if __name__ == "__main__":
    main()