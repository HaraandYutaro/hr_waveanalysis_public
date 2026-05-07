"""Phase 1 診断用: タイミング計測版 (ロジック変更なし).

各ブロックの実行時間を計測し、ボトルネックを特定するための中間ファイル。
本番スクリプトは rayleigh_vs_inversion_basic.py を参照。

実行方法:
  python examples/rayleigh_vs_inversion_basic_timed.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.inversion.rayleigh import quality_control_dispersion_pick, DispersionPickQCResult
from src.inversion.rayleigh.model import PickedDispersionCurve, Pseudo2DVsSection
from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser


# =====================================================================
# Section 0: 定数
# =====================================================================

# TODO: ユーザー設定 — 実データの npz パスに差し替えること
DATA_GLOB = "sample_data/realdata/*.npz"
AXIS = "y"

CMP_SAVE_NAME = None

DISP_FREQ_RANGE = [1, 200]
DISP_VEL_RANGE = [1, 500]
DISP_DF = 0.5
DISP_DC = 1.0

N_LAYERS = 10
SENSITIVITY_CUTOFF = 0.1
ENGINE_MAX_ITER = 50
ENGINE_RMS_TOL = 0.5
ENGINE_LAMBDA0 = 1e-3

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/rayleigh_vs_inversion")


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    print("=" * 60)
    print("Rayleigh-wave Vs inversion: GroupProcesser workflow [TIMED]")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Section 1: GroupProcesser の構築と CMP 重合
    # -----------------------------------------------------------------
    import glob

    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"[ERROR] データファイルが見つかりません: {DATA_GLOB}")
        print("DATA_GLOB を実データパスに変更してください。")
        sys.exit(1)

    print(f"[Section 1] {len(paths)} files loaded for axis={AXIS!r}")

    t0 = time.perf_counter()
    processors = []
    for path in paths:
        sp = SingleProcesser(path)
        sp.lowpass(AXIS, fpass=100, fstop=400)
        sp.remove(AXIS, remove_chs=[0, 1, 2, 3])
        sp.trace_amp_regularize(AXIS)
        processors.append(sp)
    print(f"  [TIMING] SingleProcesser 構築ループ: {time.perf_counter()-t0:.3f} s")

    group = GroupProcesser(processors, AXIS)

    t0 = time.perf_counter()
    group.cmp_gathering(
        axis=AXIS,
        cross_corr=False,
        integrate=False,
        average=True,
        save_name=CMP_SAVE_NAME,
        show=False,
    )
    print(f"  [TIMING] cmp_gathering: {time.perf_counter()-t0:.3f} s")
    print(f"  CMP targets: {len(group.targets)} positions")
    if len(group.targets) > 0:
        print(f"  CMP range: {group.targets[0]:.1f} – {group.targets[-1]:.1f} m")

    # -----------------------------------------------------------------
    # Section 3: 全 CMP に対する分散曲線収集 (QC helper を使用)
    # -----------------------------------------------------------------
    print("\n[Section 3] 分散曲線の収集と QC 済み PickedDispersionCurve への変換")

    cmp_x_list: list[float] = []
    picks_list: list[PickedDispersionCurve] = []
    skipped: list[float] = []
    _disp_times: list[float] = []
    _qc_times: list[float] = []

    t_sec3 = time.perf_counter()
    for tgt in group.targets:
        t_disp = time.perf_counter()
        res, f_mesh, c_mesh, *_ = group.dispersion_curve(
            tgt,
            freq=DISP_FREQ_RANGE,
            c=DISP_VEL_RANGE,
            df=DISP_DF,
            dc=DISP_DC,
            show=False,
        )
        t_disp = time.perf_counter() - t_disp

        if res is None:
            warnings.warn(
                f"CMP {tgt:.1f} m: 分散曲線の計算に失敗 (trace不足)。スキップします。",
                stacklevel=2,
            )
            skipped.append(float(tgt))
            continue

        t_qc = time.perf_counter()
        qc: DispersionPickQCResult = quality_control_dispersion_pick(
            res, f_mesh, c_mesh,
            min_energy_ratio=0.05,
            continuity_rel_jump=0.15,
            max_secondary_peak_ratio=0.85,
            min_valid_points=8,
        )
        t_qc = time.perf_counter() - t_qc

        _disp_times.append(t_disp)
        _qc_times.append(t_qc)

        if qc.picked is None:
            valid_count = int(np.sum(qc.mask_valid))
            total_count = len(qc.mask_valid)
            warnings.warn(
                f"CMP {tgt:.1f} m: QC 後の有効点数不足 "
                f"({valid_count}/{total_count})。スキップします。",
                stacklevel=2,
            )
            skipped.append(float(tgt))
            continue

        valid_frac = float(np.mean(qc.mask_valid))
        median_qs = float(np.median(qc.quality_score[qc.mask_valid]))
        print(
            f"    CMP {tgt:.1f} m: dispersion_curve={t_disp:.3f}s, qc={t_qc:.4f}s"
            f"  [valid={valid_frac:.0%}, median_quality={median_qs:.2f},"
            f" points={qc.picked.n_points}]"
        )

        cmp_x_list.append(float(tgt))
        picks_list.append(qc.picked)

    print(f"  [TIMING] Section 3 全体: {time.perf_counter()-t_sec3:.3f} s")
    if _disp_times:
        print(
            f"  mean dispersion_curve time: {np.mean(_disp_times):.3f} s, "
            f"mean qc time: {np.mean(_qc_times):.4f} s"
        )

    if len(picks_list) == 0:
        print("[ERROR] 有効な分散曲線が 1 つも得られませんでした。")
        print("データの前処理パラメータ (周波数範囲等) を見直してください。")
        sys.exit(1)

    print(f"  有効 CMP: {len(picks_list)}/{group.targets}")
    if skipped:
        print(f"  スキップ CMP ({len(skipped)}): {skipped}")

    # -----------------------------------------------------------------
    # Section 4: rayleigh_vs_inversion_profile の呼び出し
    # -----------------------------------------------------------------
    print("\n[Section 4] rayleigh_vs_inversion_profile 実行")
    print(f"  forward=thomson_haskell, misfit=weighted_l2, engine=damped_lsq")
    print(f"  n_layers={N_LAYERS}, sensitivitycutoff={SENSITIVITY_CUTOFF}")
    print(f"  max_iter={ENGINE_MAX_ITER}, rms_tol={ENGINE_RMS_TOL}, lambda0={ENGINE_LAMBDA0}")

    t0 = time.perf_counter()
    try:
        section = group.rayleigh_vs_inversion_profile(
            cmp_x=np.array(cmp_x_list),
            picks=picks_list,
            forward="thomson_haskell",
            misfit="weighted_l2",
            engine="damped_lsq",
            n_layers=N_LAYERS,
            sensitivitycutoff=SENSITIVITY_CUTOFF,
            max_iter=ENGINE_MAX_ITER,
            rms_tol=ENGINE_RMS_TOL,
            lambda0=ENGINE_LAMBDA0,
        )
    except ImportError as e:
        print(f"[ERROR] disba が必要です: pip install disba>=0.7.0\n{e}")
        sys.exit(1)
    print(f"  [TIMING] rayleigh_vs_inversion_profile: {time.perf_counter()-t0:.3f} s")

    # -----------------------------------------------------------------
    # Section 5: 結果の表示と保存
    # -----------------------------------------------------------------
    print("\n[Section 5] 結果の表示と保存")

    print(f"\n  {'CMP x [m]':>12s} | {'converged':>10s} | {'rms [m/s]':>12s} | {'n_iter':>7s}")
    print("  " + "-" * 50)
    for i, r in enumerate(section.percmpresults):
        print(
            f"  {section.cmpx[i]:12.1f} | "
            f"{'True' if r.converged else 'False':>10s} | "
            f"{r.rms:12.4f} | "
            f"{r.n_iter:7d}"
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, "vs_section.csv")
    with open(csv_path, "w") as f:
        header = "depth_m," + ",".join(f"cmp_{x:.1f}" for x in section.cmpx)
        f.write(header + "\n")
        for zi in range(section.depthz.size):
            row_parts = [f"{section.depthz[zi]:.4f}"]
            for ki in range(section.cmpx.size):
                val = section.vsgrid[zi, ki]
                row_parts.append(f"{val:.2f}" if np.isfinite(val) else "NaN")
            f.write(",".join(row_parts) + "\n")
    print(f"  CSV saved: {csv_path}")

    print(f"\n  Section info:")
    print(f"    n_layers: {section.metadata.get('n_layers', '?')}")
    print(f"    is_halfspace: {section.metadata.get('is_halfspace', '?')}")
    print(f"    maxdepthfactor: {section.metadata.get('maxdepthfactor', '?')}")
    print(f"    sensitivitycutoff: {section.metadata.get('sensitivitycutoff', '?')}")
    print(f"    confidence_normalization: {section.metadata.get('confidence_normalization', '?')}")

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        X, Z = np.meshgrid(section.cmpx, section.depthz)
        vs_display = section.vsgrid.copy()
        pcm = ax.pcolormesh(X, Z, vs_display, shading="auto", cmap="jet_r")
        ax.invert_yaxis()
        ax.set_xlabel("CMP position [m]")
        ax.set_ylabel("Depth [m]")
        ax.set_title("Pseudo-2D Vs section")
        plt.colorbar(pcm, ax=ax, label="Vs [m/s]")

        plot_path = os.path.join(OUTPUT_DIR, "vs_section.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"  Plot saved: {plot_path}")
        plt.close(fig)
    except ImportError:
        print("  matplotlib not available; skipping plot")

    print("\nDone.")


if __name__ == "__main__":
    main()
