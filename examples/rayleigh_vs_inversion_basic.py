"""Step ?-5 example: GroupProcesser workflow → Rayleigh-wave Vs inversion.

このスクリプトは GroupProcesser で CMP 重合 → 分散曲線 → Rayleigh-wave Vs inversion
の end-to-end ワークフローを示す。

Section 構成:
  0: imports と定数
  1: GroupProcesser の構築と CMP 重合 (既存ワークフローの再掲)
  3: 全 CMP に対する分散曲線の収集 (QC helper を使用)
  4: rayleigh_vs_inversion_profile の呼び出し
  4b: 分散曲線フィギュアの保存 (--save-dispersion 指定時)
  5: 結果の表示と保存

実行方法:
  python examples/rayleigh_vs_inversion_basic.py
  python examples/rayleigh_vs_inversion_basic.py --save-dispersion
  python examples/rayleigh_vs_inversion_basic.py --dry-run --save-dispersion --dispersion-dir dispersion_curves

注意:
  - disba (ThomsonHaskellSolver) が必要。
    未インストールの場合は pip install disba>=0.7.0 で導入すること。
  - 実データパスはユーザー環境に合わせて変更すること (# TODO 箇所)。
  - USE_PARALLEL=True で ProcessPoolExecutor による並列 CMP 逆解析が有効になる。
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np

# プロジェクトルートを sys.path に追加 (examples/ からの相対パス)
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.inversion.rayleigh import quality_control_dispersion_pick, DispersionPickQCResult
from src.inversion.rayleigh.init_model import build_default_init_model
from src.inversion.rayleigh.model import PickedDispersionCurve, Pseudo2DVsSection
from src.inversion.rayleigh.section import Pseudo2DSectionBuilder
from src.processor.group_processor import GroupProcesser
from src.processor.plot_processor import PlotProcesser
from src.processor.single_processor import SingleProcesser


# =====================================================================
# Section 0: 定数 (ユーザー設定箇所は # TODO で示す)
# =====================================================================

# TODO: ユーザー設定 — 実データの npz パスに差し替えること
DATA_GLOB = "sample_data/realdata/*.npz"
AXIS = "y"

# CMP 重合パラメータ (既存コードの慣習に従う)
CMP_SAVE_NAME = None  # None の場合保存しない

# 分散曲線パラメータ
DISP_FREQ_RANGE = [1, 200]  # [fmin, fmax] Hz
DISP_VEL_RANGE = [1, 500]   # [cmin, cmax] m/s
DISP_DF = 0.5               # 周波数刻み Hz
DISP_DC = 1.0               # 速度刻み m/s

# Vs inversion パラメータ (Step ?-4 での検証済みパラメータを踏襲)
N_LAYERS = 10
SENSITIVITY_CUTOFF = 0.1
ENGINE_MAX_ITER = 50
ENGINE_RMS_TOL = 0.5
ENGINE_LAMBDA0 = 1e-3

# TODO: True に変更すると並列化が有効 (ProcessPoolExecutor, 要テスト)
# pickle 不可のオブジェクトが含まれる場合は False のままにすること。
USE_PARALLEL = False

# 有効探査深度係数: Z_max = lambda_max * DEPTH_FRACTION
# 0.5 → lambda/2（Rix & Leipski 1991 標準）、0.33 → lambda/3（Tokimatsu 1995 保守的）
DEPTH_FRACTION = 0.5

# QC後ピッキング点を線形リサンプリングする際の目標点数
# 高周波帯でQCにより疎になった picked.f/c を均等周波数グリッドに再配置する
N_FREQ_RESAMPLE = 50

# 出力ディレクトリ
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output/rayleigh_vs_inversion")  # 例: examples/output/rayleigh_vs_inversion

# 分散曲線保存ディレクトリ (--save-dispersion / --dispersion-dir 未指定時のデフォルト)
DISP_CURVES_DIR = os.path.join(os.path.dirname(__file__), "output/rayleigh_vs_inversion/dispersion_curves")


# =====================================================================
# Module-level worker (ProcessPoolExecutor 用)
# =====================================================================
# このクラスは import 可能な位置 (モジュールトップレベル) に置く必要がある。
# フォワードソルバー (ThomsonHaskellSolver / disba) を subprocess 内で再構成し、
# pickle 不可能なオブジェクトを worker プロセスへ送信しないようにする。

def _build_init_model_from_qc(
    qc,
    depth_template: "LayeredEarthModel",
    n_layers: int,
    vp_vs_ratio: float = float(np.sqrt(3.0)),
    rho: float = 2000.0,
    vs_bounds_factor: tuple = (0.5, 2.0),
    vs_min: float = 50.0,
    vs_max: float = 2000.0,
) -> "LayeredEarthModel":
    """QC 後の有効 c_filtered から初期 Vs モデルを構築する。

    Rayleigh 波基本モード近似: Vs ≈ c_phase / 0.92

    層ごとの Vs 推定:
      - valid な c を周波数昇順にソート（低周波→深部、高周波→浅部）
      - n_layers への線形補間で各層 Vs を割り当て

    Pseudo2DSectionBuilder の要件に合わせ、層厚 h は depth_template と共通の値を使用する。
    Vs / Vp / vs_bounds のみ CMP ごとに推定する。
    """
    from src.inversion.rayleigh.model import LayeredEarthModel

    _RAYLEIGH_VS_RATIO = 0.92
    c_valid = np.asarray(qc.picked.c, dtype=float)
    f_valid = np.asarray(qc.picked.f, dtype=float)

    # 周波数昇順ソート（低周波→深部の Vs、高周波→浅部の Vs）
    sort_idx = np.argsort(f_valid)
    c_sorted = c_valid[sort_idx]  # c_sorted[0]→深部, c_sorted[-1]→浅部

    vs_sorted = c_sorted / _RAYLEIGH_VS_RATIO

    # n_layers への線形補間
    # dst_x[0]=最浅部（高周波側）, dst_x[-1]=最深部（低周波側）なので vs_sorted を逆順に
    src_x = np.linspace(0.0, 1.0, len(vs_sorted))
    dst_x = np.linspace(0.0, 1.0, n_layers)
    vs_interp = np.interp(dst_x, src_x, vs_sorted[::-1])  # 浅部→深部の順

    vs_arr = np.clip(vs_interp, vs_min, vs_max)
    vp_arr = vp_vs_ratio * vs_arr
    rho_arr = np.full(n_layers, float(rho))
    lo_f, hi_f = vs_bounds_factor
    vs_lo = lo_f * vs_arr
    vs_hi = hi_f * vs_arr

    return LayeredEarthModel(
        vs=vs_arr,
        vp=vp_arr,
        rho=rho_arr,
        h=depth_template.h.copy(),  # 全 CMP で共通の層厚を使用
        vs_bounds=(vs_lo, vs_hi),
    )


def _invert_single_cmp(args: tuple):
    """並列 CMP 逆解析ワーカー。フォワードソルバーを subprocess 内で再構成する。

    args: (x, pick, init_model, params)
      x          : float — CMP 位置 [m]
      pick       : PickedDispersionCurve
      init_model : LayeredEarthModel — 各 CMP の独立コピー
      params     : dict — {"forward", "misfit", "engine", "engine_opts"}
    """
    x, pick, init_model, params = args
    from src.mixins.group.rayleigh_inversion import (
        _resolve_forward,
        _resolve_misfit,
        _resolve_engine_cls,
    )
    fwd_obj = _resolve_forward(params["forward"])
    mis_obj = _resolve_misfit(params["misfit"])
    _, engine_cls = _resolve_engine_cls(params["engine"])
    eng = engine_cls(forward=fwd_obj, misfit=mis_obj, **params["engine_opts"])
    result = eng.run_single(pick, init_model)
    result.metadata.update({
        "cmp_target": x,
        "forward_name": params["forward"],
        "misfit_name": params["misfit"],
        "engine_name": params["engine"],
    })
    return result


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Rayleigh-wave Vs inversion: GroupProcesser workflow"
    )
    parser.add_argument(
        "--save-dispersion",
        action="store_true",
        help=(
            "分散曲線フィギュアを --dispersion-dir に保存する。"
            "各 CMP ごとに imshow + 観測ピーク + 理論曲線を描画する。"
        ),
    )
    parser.add_argument(
        "--dispersion-dir",
        default=DISP_CURVES_DIR,
        metavar="PATH",
        help="分散曲線フィギュアの保存先ディレクトリ (既定: dispersion_curves/)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="各フィギュアを保存後に plt.show() で表示する。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "実データを読み込まず、合成分散曲線データで可視化パイプラインをテストする。"
            "GroupProcesser は構築しない。"
        ),
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Rayleigh-wave Vs inversion: GroupProcesser workflow")
    print("=" * 60)
    if args.dry_run:
        print("[DRY-RUN] モード: 合成データで可視化パイプラインをテストします。")

    # -----------------------------------------------------------------
    # Section 1: GroupProcesser の構築と CMP 重合
    # (既存コードの再掲。新機能ではない)
    # -----------------------------------------------------------------
    if not args.dry_run:
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
        print(f"  [TIMING] SingleProcesser 構築: {time.perf_counter()-t0:.3f} s")

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
    else:
        group = None
        print("[Section 1] DRY-RUN: GroupProcesser をスキップします。")

    # -----------------------------------------------------------------
    # Section 3: 全 CMP に対する分散曲線収集 (QC helper を使用)
    # -----------------------------------------------------------------
    print("\n[Section 3] 分散曲線の収集と QC 済み PickedDispersionCurve への変換")

    cmp_x_list: list[float] = []
    picks_list: list[PickedDispersionCurve] = []
    # rayleigh_dispersion_fit_image 用: QC 通過 CMP のみ
    qc_results: list[DispersionPickQCResult] = []
    f_meshes: list[np.ndarray] = []
    c_meshes: list[np.ndarray] = []
    res_maps: list[np.ndarray] = []
    # 描画診断用リスト (QC 失敗 CMP も含む / inversion には渡さない)
    disp_cmp_x_list: list[float] = []
    disp_res_list: list[np.ndarray] = []
    disp_f_mesh_list: list[np.ndarray] = []
    disp_c_mesh_list: list[np.ndarray] = []
    disp_qc_list: list[DispersionPickQCResult] = []
    skipped: list[float] = []
    _disp_times: list[float] = []
    _qc_times: list[float] = []

    t_sec3 = time.perf_counter()

    if not args.dry_run:
        _n_targets = len(group.targets)
        for i_tgt, tgt in enumerate(group.targets):
            t_disp = time.perf_counter()
            res, f_mesh, c_mesh, dx_cmp, n_sensors_cmp = group.dispersion_curve(
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

            # dx_cmp / n_sensors_cmp が無効（早期リターン由来の nan / 0）の場合は
            # 波数制約をスキップして後方互換モードで QC を実行する
            _dx_arg = dx_cmp if (not np.isnan(dx_cmp) and dx_cmp > 0) else None
            _nr_arg = int(n_sensors_cmp) if (n_sensors_cmp is not None and n_sensors_cmp > 0) else None

            t_qc = time.perf_counter()
            qc: DispersionPickQCResult = quality_control_dispersion_pick(
                res, f_mesh, c_mesh,
                dx=_dx_arg,
                n_receivers=_nr_arg,
                min_energy_ratio=0.05,
                continuity_rel_jump=0.15,
                max_secondary_peak_ratio=0.85,
                min_valid_points=8,
            )
            t_qc = time.perf_counter() - t_qc

            _disp_times.append(t_disp)
            _qc_times.append(t_qc)

            # res が取得できた時点で診断用リストに追加
            # (QC の結果によらず、分散曲線が存在する CMP すべて)
            disp_cmp_x_list.append(float(tgt))
            disp_res_list.append(res)
            disp_f_mesh_list.append(f_mesh)
            disp_c_mesh_list.append(c_mesh)
            disp_qc_list.append(qc)

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

            # QC サマリ (オプション)
            valid_frac = float(np.mean(qc.mask_valid))
            median_qs = float(np.median(qc.quality_score[qc.mask_valid]))
            n_below = int(np.sum(qc.flags["below_kmax_limit"]))
            n_above = int(np.sum(qc.flags["above_kmin_limit"]))
            print(
                f"  CMP {tgt:.1f} m: valid={valid_frac:.0%}, "
                f"median quality={median_qs:.2f}, "
                f"points={qc.picked.n_points}, "
                f"disp={t_disp:.3f}s, qc={t_qc:.4f}s"
            )
            print(
                f"    k-limit excluded: below_kmax={n_below}, above_kmin={n_above}"
            )
            if (i_tgt + 1) % 5 == 0 or (i_tgt + 1) == _n_targets:
                print(f"  [進捗] Section 3: {i_tgt + 1}/{_n_targets} CMP 処理済み")

            # 修正 1: QC後残存点を均等周波数グリッドへリサンプリング
            # QC フィルタ後の picked.f は高周波帯で疎になる可能性があるため
            # [f_min, f_max] を N_FREQ_RESAMPLE 点の線形グリッドに再配置する。
            _f_raw = np.asarray(qc.picked.f, dtype=float)
            _c_raw = np.asarray(qc.picked.c, dtype=float)
            _n_resample = min(N_FREQ_RESAMPLE, _f_raw.size)
            _f_rs = np.linspace(_f_raw[0], _f_raw[-1], _n_resample)
            _c_rs = np.interp(_f_rs, _f_raw, _c_raw)
            _cstd_rs = None
            if qc.picked.c_std is not None:
                _cstd_rs = np.interp(_f_rs, _f_raw, qc.picked.c_std)
            qc.picked = PickedDispersionCurve(
                f=_f_rs, c=_c_rs, c_std=_cstd_rs, mode=qc.picked.mode
            )

            cmp_x_list.append(float(tgt))
            picks_list.append(qc.picked)
            qc_results.append(qc)
            f_meshes.append(f_mesh)
            c_meshes.append(c_mesh)
            res_maps.append(res)

        n_total_cmp = len(group.targets)

    else:
        # --dry-run: 合成分散曲線を 3 CMP 分生成してパイプラインをテストする
        print("[Section 3] DRY-RUN: 合成分散曲線データを生成します (3 CMP)")
        _n_dry = 3
        for i in range(_n_dry):
            cmp_x = float(i * 10)
            # 1D 配列を使用 (quality_control_dispersion_pick は 1D を要求)
            f_vec = np.linspace(5.0, 50.0, 60)
            c_vec = np.linspace(100.0, 600.0, 80)
            # 理論分散曲線: c_true(f) = 300 - 2*(f-5) [m/s]
            c_true = 300.0 - 2.0 * (f_vec - 5.0)
            # Gaussian ridge で energy map を構築 (shape: n_f × n_c)
            sigma_c = 15.0
            res = np.zeros((len(f_vec), len(c_vec)), dtype=float)
            for fi in range(len(f_vec)):
                res[fi, :] = np.exp(-0.5 * ((c_vec - c_true[fi]) / sigma_c) ** 2)
            res += 0.05 * np.random.default_rng(seed=i).random(res.shape)

            qc: DispersionPickQCResult = quality_control_dispersion_pick(
                res, f_vec, c_vec,
                min_energy_ratio=0.05,
                continuity_rel_jump=0.15,
                max_secondary_peak_ratio=0.85,
                min_valid_points=8,
            )

            disp_cmp_x_list.append(cmp_x)
            disp_res_list.append(res)
            disp_f_mesh_list.append(f_vec)
            disp_c_mesh_list.append(c_vec)
            disp_qc_list.append(qc)

            if qc.picked is None:
                warnings.warn(
                    f"[DRY-RUN] CMP {cmp_x:.1f} m: QC に失敗しました。スキップします。",
                    stacklevel=2,
                )
                skipped.append(cmp_x)
                continue

            print(
                f"  [DRY-RUN] CMP {cmp_x:.1f} m: 合成データ生成 OK, "
                f"points={qc.picked.n_points}"
            )
            cmp_x_list.append(cmp_x)
            picks_list.append(qc.picked)
            qc_results.append(qc)
            f_meshes.append(f_vec)
            c_meshes.append(c_vec)
            res_maps.append(res)

        n_total_cmp = _n_dry

    print(f"  [TIMING] Section 3 全体: {time.perf_counter()-t_sec3:.3f} s")
    if _disp_times:
        _mean_disp = float(np.mean(_disp_times))
        _mean_qc = float(np.mean(_qc_times))
        print(
            f"  mean dispersion_curve time: {_mean_disp:.3f} s, "
            f"mean qc time: {_mean_qc:.4f} s"
        )
    else:
        _mean_disp = 0.0

    if len(picks_list) == 0:
        print("[ERROR] 有効な分散曲線が 1 つも得られませんでした。")
        print("データの前処理パラメータ (周波数範囲等) を見直してください。")
        sys.exit(1)

    print(f"  有効 CMP: {len(picks_list)}/{n_total_cmp}")
    if skipped:
        print(f"  スキップ CMP ({len(skipped)}): {skipped}")

    # -----------------------------------------------------------------
    # Section 4: 1D Vs inversion (per-CMP 明示ループ)
    # -----------------------------------------------------------------
    # sensitivitycutoff は engine opt ではなく Pseudo2DSectionBuilder への引数。

    K = len(picks_list)
    # 逆解析時間は分散計算より大幅に長い傾向があるため 5.0 s/CMP をデフォルト推定値とする。
    _default_per_cmp_sec = 5.0
    estimated_s = _default_per_cmp_sec * K

    print("\n[Section 4] rayleigh_vs_inversion_profile 実行")
    print(f"  forward=thomson_haskell, misfit=weighted_l2, engine=damped_lsq")
    print(f"  n_layers={N_LAYERS}, sensitivitycutoff={SENSITIVITY_CUTOFF}")
    print(f"  max_iter={ENGINE_MAX_ITER}, rms_tol={ENGINE_RMS_TOL}, lambda0={ENGINE_LAMBDA0}")
    print(f"[Section 4] 解析開始: {K} CMP × 最大{ENGINE_MAX_ITER}反復")
    print(f"  推定時間: ~{estimated_s:.0f}s (初回のみ、以降はキャッシュ可能)")

    # 共通の層厚構造を代表 CMP（中央）から構築する
    # Pseudo2DSectionBuilder は全 CMP で同一の h 配列を要求するため。
    _rep_idx = len(picks_list) // 2
    _depth_template = build_default_init_model(picks_list[_rep_idx], n_layers=N_LAYERS)

    # 各 CMP の分散曲線から Vs のみ CMP ごとに推定し、h は共通テンプレートを流用する。
    # Rayleigh-Vs 比 0.92 を用いて c_phase → Vs に変換。
    initmodels = [
        _build_init_model_from_qc(qc, depth_template=_depth_template, n_layers=N_LAYERS)
        for qc in qc_results
    ]
    _vs_medians = [float(np.median(m.vs)) for m in initmodels]
    print(f"  初期モデル Vs 中央値: min={min(_vs_medians):.1f}, "
          f"max={max(_vs_medians):.1f}, mean={np.mean(_vs_medians):.1f} m/s")

    _inversion_results: list = [None] * K

    # Section 4 の共通パラメータ (並列・DRY-RUN 共用)
    _inv_params = {
        "forward": "thomson_haskell",
        "misfit": "weighted_l2",
        "engine": "damped_lsq",
        "engine_opts": {
            "max_iter": ENGINE_MAX_ITER,
            "rms_tol": ENGINE_RMS_TOL,
            "lambda0": ENGINE_LAMBDA0,
        },
    }

    try:
        t_sec4 = time.perf_counter()

        if USE_PARALLEL:
            from concurrent.futures import ProcessPoolExecutor, as_completed
            import multiprocessing

            N_WORKERS = min(K, max(1, multiprocessing.cpu_count() - 1))
            print(f"  並列: {N_WORKERS} workers (USE_PARALLEL=True)")

            with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
                futures_map = {
                    executor.submit(_invert_single_cmp, (x, pick, init_m, _inv_params)): i
                    for i, (x, pick, init_m) in enumerate(
                        zip(cmp_x_list, picks_list, initmodels)
                    )
                }
                done_count = 0
                for fut in as_completed(futures_map):
                    idx = futures_map[fut]
                    _inversion_results[idx] = fut.result()
                    done_count += 1
                    if done_count % 5 == 0 or done_count == K:
                        print(f"  [進捗] Section 4 (並列): {done_count}/{K} CMP 完了")

        elif args.dry_run:
            # --dry-run: group オブジェクトなしで _invert_single_cmp を直接呼び出す
            print(f"  [DRY-RUN] _invert_single_cmp を直接呼び出します")
            _results_dry = []
            for i_seq, (x, pick, init_m) in enumerate(
                zip(cmp_x_list, picks_list, initmodels)
            ):
                r = _invert_single_cmp((x, pick, init_m, _inv_params))
                _results_dry.append(r)
                if (i_seq + 1) % 5 == 0 or (i_seq + 1) == K:
                    rms_disp = r.rms if np.isfinite(r.rms) else float("nan")
                    print(
                        f"  [進捗] Section 4 (DRY-RUN): "
                        f"{i_seq + 1}/{K} CMP 完了, rms={rms_disp:.2f}"
                    )
            _inversion_results = _results_dry

        else:
            print(f"  逐次実行 (USE_PARALLEL=False; 並列化は USE_PARALLEL=True で有効)")
            _results_seq = []
            for i_seq, (x, pick, init_m) in enumerate(zip(cmp_x_list, picks_list, initmodels)):
                r = group.rayleigh_vs_inversion_1d(
                    target=x,
                    picked=pick,
                    init_model=init_m,
                    forward="thomson_haskell",
                    misfit="weighted_l2",
                    engine="damped_lsq",
                    max_iter=ENGINE_MAX_ITER,
                    rms_tol=ENGINE_RMS_TOL,
                    lambda0=ENGINE_LAMBDA0,
                )
                _results_seq.append(r)
                if (i_seq + 1) % 5 == 0 or (i_seq + 1) == K:
                    rms_disp = r.rms if np.isfinite(r.rms) else float("nan")
                    print(f"  [進捗] Section 4 (逐次): {i_seq + 1}/{K} CMP 完了, rms={rms_disp:.2f}")
            _inversion_results = _results_seq

        print(f"  [TIMING] Section 4 全体: {time.perf_counter()-t_sec4:.3f} s")

        section = Pseudo2DSectionBuilder.build(
            cmpx=np.array(cmp_x_list),
            results=_inversion_results,
            sensitivitycutoff=SENSITIVITY_CUTOFF,
            picked_curves=picks_list,
            depth_fraction=DEPTH_FRACTION,
        )

    except ImportError as e:
        print(f"[ERROR] disba が必要です: pip install disba>=0.7.0\n{e}")
        sys.exit(1)
    args.save_dispersion=True
    # -----------------------------------------------------------------
    # Section 4b: 分散曲線フィギュアの保存 (--save-dispersion 指定時)
    # -----------------------------------------------------------------
    if args.save_dispersion:
        print("\n[Section 4b] 分散曲線フィギュアの保存")
        os.makedirs(args.dispersion_dir, exist_ok=True)
        result_by_cmpx = {
            float(section.cmpx[i]): section.percmpresults[i]
            for i in range(len(section.cmpx))
        }
        for cmp_x, qc, f_mesh, c_mesh, res in zip(
            cmp_x_list, qc_results, f_meshes, c_meshes, res_maps
        ):
            r = result_by_cmpx.get(float(cmp_x))
            theory_c = None
            theory_f = None
            if r is not None and bool(r.converged) and qc.picked is not None:
                theory_c = np.asarray(r.predicted_c, dtype=float)
                theory_f = np.asarray(qc.picked.f, dtype=float)
            save_path = os.path.join(args.dispersion_dir, f"disp_cmp_{cmp_x:.1f}m.png")
            group.rayleigh_dispersion_fit_image(
                res=res,
                f_mesh=f_mesh,
                c_mesh=c_mesh,
                qc_result=qc,
                theory_c=theory_c,
                theory_f=theory_f,
                target=f"{cmp_x:.1f} m",
                show=args.show,
                save_name=save_path,
            )
            print(f"  Saved dispersion figure: {save_path}")

    # -----------------------------------------------------------------
    # Section 5: 結果の表示と保存
    # -----------------------------------------------------------------
    print("\n[Section 5] 結果の表示と保存")

    # 各 CMP の収束状況を表形式で出力
    print(f"\n  {'CMP x [m]':>12s} | {'converged':>10s} | {'rms [m/s]':>12s} | {'n_iter':>7s}")
    print("  " + "-" * 50)
    for i, r in enumerate(section.percmpresults):
        print(
            f"  {section.cmpx[i]:12.1f} | "
            f"{'True' if r.converged else 'False':>10s} | "
            f"{r.rms:12.4f} | "
            f"{r.n_iter:7d}"
        )

    # vsgrid, depthz, cmpx を CSV に保存
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

    # z_max_per_cmp を CSV に保存
    import csv as _csv
    os.makedirs(DISP_CURVES_DIR, exist_ok=True)
    zmax_csv_path = os.path.join(DISP_CURVES_DIR, "zmax_per_cmp.csv")
    _z_max_arr = section.metadata.get("z_max_per_cmp", np.full(len(section.cmpx), np.nan))
    _lm_arr = section.metadata.get("lambda_max_per_cmp", np.full(len(section.cmpx), np.nan))
    with open(zmax_csv_path, "w", newline="") as _f:
        _writer = _csv.writer(_f)
        _writer.writerow(["cmp_x", "z_max_m", "lambda_max_m"])
        for _x, _zm, _lm in zip(section.cmpx, _z_max_arr, _lm_arr):
            _writer.writerow([
                f"{_x:.2f}",
                f"{_zm:.2f}" if np.isfinite(_zm) else "NaN",
                f"{_lm:.2f}" if np.isfinite(_lm) else "NaN",
            ])
    print(f"  z_max CSV saved: {zmax_csv_path}")

    # Vs section metadata
    print(f"\n  Section info:")
    print(f"    n_layers: {section.metadata.get('n_layers', '?')}")
    print(f"    is_halfspace: {section.metadata.get('is_halfspace', '?')}")
    print(f"    maxdepthfactor: {section.metadata.get('maxdepthfactor', '?')}")
    print(f"    sensitivitycutoff: {section.metadata.get('sensitivitycutoff', '?')}")
    print(f"    confidence_normalization: {section.metadata.get('confidence_normalization', '?')}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plot_path = os.path.join(OUTPUT_DIR, "vs_section.png")
    plotter = PlotProcesser(backend="mpl")
    plotter.vs_section(
        section=section,
        show=False,
        save_name=plot_path,
    )
    print(f"  Plot saved: {plot_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
