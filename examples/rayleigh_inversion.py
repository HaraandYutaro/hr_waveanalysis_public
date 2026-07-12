"""Quickstart: Rayleigh-wave Vs inversion in a single call.

このスクリプトは新しい一元化 API `GroupProcesser.rayleigh_inversion()` の
最小ユーザーコードを示す。多ステップ版 `rayleigh_vs_inversion_basic.py` と
等価な end-to-end ワークフロー (CMP 重合 → 分散曲線 → QC → 初期モデル生成 →
1D inversion × 全 CMP → Pseudo2D セクション化) を、内部で自動的に実行する。

実行方法:
    python examples/rayleigh_inversion_quickstart.py

前提:
    - disba (ThomsonHaskellSolver) がインストール済みであること
      pip install disba>=0.7.0
    - DATA_GLOB をユーザー環境の npz パスに差し替えること
"""

from __future__ import annotations

import glob
import os
import sys

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser


# TODO: ユーザー設定 — 実データの npz パスに差し替えること
DATA_GLOB = "testcodes/data/sinkhole_cav_ricker/*.npz"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/rayleigh_inversion_quickstart")


def main() -> None:
    # -- 入力ファイルの収集 --
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"[ERROR] データファイルが見つかりません: {DATA_GLOB}")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -- 前処理して SingleProcesser を束ねる --
    processors = []
    for path in paths:
        sp = SingleProcesser(path)
        sp.lowpass("z", fpass=100, fstop=400)
        # sp.remove("z", remove_chs=[0, 1, 2, 3])
        sp.trace_amp_regularize("z")
        processors.append(sp)

    group = GroupProcesser(processors)

    # -- 最小呼び出し: ほぼゼロ引数で全 CMP を解析 --
    # cmp_gathering / dispersion_curve / QC / 初期モデル生成 / 1D inversion /
    # Pseudo2DSectionBuilder が内部で順に走り、Pseudo2DVsSection を返す。
    section = group.rayleigh_inversion(
        save_name=os.path.join(OUTPUT_DIR, "vs_section.png"),
    )

    print(f"Done. CMPs analyzed: {len(section.cmpx)}")
    print(f"Vs section shape: {section.vsgrid.shape} (depth × CMP)")
    print(f"Saved: {OUTPUT_DIR}/vs_section.png")

    # -- オプション: 軸 / 層数 / 反復制御を変えた呼び出し例 --
    # section_detail = group.rayleigh_inversion(
    #     axis="z",
    #     n_layers=15,
    #     max_iter=100,
    #     rms_tol=0.3,
    #     save_name=os.path.join(OUTPUT_DIR, "vs_section_detailed.png"),
    # )

    # -- オプション: 単一 CMP のみ解析する例 (戻り値は RayleighInversionResult) --
    # result = group.rayleigh_inversion(target=float(section.cmpx[len(section.cmpx) // 2]))
    # print(f"Single CMP result: rms={result.rms:.3f}, converged={result.converged}")

    # -- オプション: LCI (横拘束同時逆解析) を使う例 --
    # lci_result = group.rayleigh_inversion(
    #     mode="lci", lambda_v=1.0, lambda_h=0.5,
    # )


if __name__ == "__main__":
    main()
