"""
Reflection cross-section Mixin.

Based on Liu et al. (2005), "Accurate elevation and normal moveout corrections..."
- Eq. (9): simultaneous Elevation + NMO time shift.
- Eq. (11): velocity spectrum (semblance over trial velocities).

Public API:
    Reflection.reflection(...) — CMP gather + Elev/NMO + 反射断面の出力
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import interpolate
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **_kw):
        return iterable


from src.mixins.group.cmp_gathering import merge_close_targets
from src.plotting.wrapper import PlotterWrapperMixin


# =============================================================================
#  Dataclasses for stacking velocity results
# =============================================================================
@dataclass
class StackingVelocityResult:
    """
    Stacking velocity analysis results from semblance-based velocity scanning.

    These are stacking velocities (often called NMO velocities) determined by
    semblance analysis. Under the horizontal-layer (hyperbolic NMO) assumption,
    they correspond to RMS velocities. They are NOT depth velocities or interval
    velocities.

    Attributes
    ----------
    stacking_velocity_picks : np.ndarray, shape (n_cmp, n_gates)
        Best velocity pick per CMP per time gate [m/s].
    stacking_velocity_model : np.ndarray, shape (n_cmp, n_samples)
        Time-dependent velocity model v(t) used for NMO correction [m/s].
        Obtained by interpolating stacking_velocity_picks to every sample.
    stacking_velocity_time : np.ndarray, shape (n_samples,)
        Time axis corresponding to stacking_velocity_model [s].
    gate_time_centers : np.ndarray, shape (n_gates,)
        Center time of each analysis gate [s].
    cmp_positions : np.ndarray, shape (n_cmp,)
        CMP horizontal positions [m].
    targets_sorted : np.ndarray, shape (n_cmp,)
        Sorted CMP target keys.
    """

    stacking_velocity_picks: np.ndarray
    stacking_velocity_model: np.ndarray
    stacking_velocity_time: np.ndarray
    gate_time_centers: np.ndarray
    cmp_positions: np.ndarray
    targets_sorted: np.ndarray


@dataclass
class NMOFullResult:
    """
    Container for NMO correction results with optional stacking velocity info.

    Attributes
    ----------
    stack_horiz : np.ndarray
        Reflection cross-section (masked above surface).
        In "constant" depth_conversion_mode, shape is (n_cmp, n_samples_time)
        and rows correspond to the original time axis.
        In "stacking_velocity" mode, shape is (n_cmp, n_depth_samples)
        and rows have been resampled onto a common depth grid.
    cmp_pos : np.ndarray
        CMP horizontal positions [m].
    elev_axis : np.ndarray
        Elevation axis [m].
        In "constant" mode, this is surf_ref - (t * depth_vel / 2).
        In "stacking_velocity" mode, this is surf_ref - common_depth_axis.
    targets_sorted : np.ndarray
        Sorted CMP target keys.
    stacking_velocity : StackingVelocityResult | None
        Per-CMP depth curves z_i(t), shape (n_cmp, n_samples_time).
        Only set when depth_conversion_mode="stacking_velocity".
        Each row gives the approximate depth z_i(t_j) for CMP i,
        computed as z_i(t) ≈ 0.5 * ∫₀ᵗ v_stack,i(τ) dτ.
        None in "constant" mode.
    common_depth_axis : np.ndarray | None
        Common depth grid, shape (n_depth_samples,).
        Only set when depth_conversion_mode="stacking_velocity".
        The depth axis to which stack_horiz has been resampled.
        None in "constant" mode.
    """

    stack_horiz: np.ndarray
    cmp_pos: np.ndarray
    elev_axis: np.ndarray
    targets_sorted: np.ndarray
    stacking_velocity: StackingVelocityResult | None = None
    depth_per_cmp: np.ndarray | None = None
    common_depth_axis: np.ndarray | None = None


# =============================================================================
#  Elevation + NMO 補正本体 (Liu et al. 2005, Eq. 9)
# =============================================================================
class ElevNMO:
    """
    Δt(v) = sqrt((x/v)**2 + (t + (Δh_s + Δh_r)/v)**2) - t  (Eq. 9)
    """

    def __init__(self, data, offsets, hs, hr, dt):
        self.data = np.asarray(data, dtype=float)
        self.offsets = np.asarray(offsets, dtype=float)
        self.hs = np.asarray(hs, dtype=float)
        self.hr = np.asarray(hr, dtype=float)
        self.hm = (self.hs + self.hr) / 2.0
        self.dt = dt
        self.t = np.arange(self.data.shape[1]) * dt

    def _time_shifts(self, v_t):
        dhs = (self.hs - self.hm)[:, None]
        dhr = (self.hr - self.hm)[:, None]
        t0 = self.t[None, :]
        x = 2 * self.offsets[:, None]
        return np.sqrt((x / v_t) ** 2 + (t0 + (dhs + dhr) / v_t) ** 2) - t0

    def apply(self, v_t):
        shifts = self._time_shifts(v_t)
        corrected = np.zeros_like(self.data)
        for i, (trace, dt_shift) in enumerate(zip(self.data, shifts)):
            corrected[i, :] = np.interp(
                self.t + dt_shift, self.t, trace, left=0.0, right=0.0
            )
        return corrected


# =============================================================================
#  Velocity spectrum (Liu et al. 2005, Eq. 11)
# =============================================================================
def velocity_spectrum(cmp_data, offsets, hs, hr, dt, v_samples, gate_len, gate_step):
    n_tr, n_samp = cmp_data.shape
    K = max(int(gate_len / dt), 1)
    M = max(int(gate_step / dt), 1)
    centers = np.arange(K // 2, n_samp - K // 2, M)
    spec = np.zeros((len(v_samples), len(centers)))

    enmo = ElevNMO(cmp_data, offsets, hs, hr, dt)

    for iv, v in enumerate(v_samples):
        v_t = np.full(n_samp, v)
        corr = enmo.apply(v_t)
        for ig, c0 in enumerate(centers):
            win = slice(c0 - K // 2, c0 + K // 2)
            mean_trace = corr[:, win].mean(axis=0)
            spec[iv, ig] = np.sum(mean_trace**2)

    t_centers = centers * dt
    return spec, t_centers


def est_velocity(
    v_min,
    v_max,
    n_vel,
    target,
    data,
    offsets,
    hs,
    hr,
    dt,
    gate_len,
    gate_step,
    show=False,
):
    v_samples = np.linspace(v_min, v_max, n_vel)
    spec, t_centers = velocity_spectrum(
        data, offsets, hs, hr, dt, v_samples, gate_len, gate_step
    )
    best_vs = v_samples[spec.argmax(axis=0)]
    if show:
        print(f"CMP {target}: picked velocities (m/s) at gates: {best_vs}")
    return best_vs, t_centers


def build_vt(t_centers, best_vs, dt, n_samples):
    v_interp = interp1d(
        t_centers,
        best_vs,
        bounds_error=False,
        fill_value=(best_vs[0], best_vs[-1]),
    )
    return v_interp(np.arange(n_samples) * dt)


def apply_enmo(data, offsets, hs, hr, dt, v_t):
    enmo = ElevNMO(data, offsets, hs, hr, dt)
    corr = enmo.apply(v_t)
    return corr.mean(axis=0)


# =============================================================================
#  CMP loop helpers
# =============================================================================
def _bring_target_data(cmp, offsets_dict, src_h, rec_h, target):
    data = cmp[target]
    offsets = offsets_dict[target]
    hs = np.asarray(src_h[target], dtype=float)
    hr = np.asarray(rec_h[target], dtype=float)
    hm = (hs + hr) / 2.0
    return data, offsets, hs, hr, hm


def remove_single_trace_gathers(
    cmp, offsets_dict, src_h, rec_h, targets, tar_heights, criterion=1
):
    """Drop CMP gathers that have <= criterion traces. Returns filtered tuple."""
    targets = np.asarray(targets)
    tar_heights = np.asarray(tar_heights)

    keep_mask = np.array(
        [cmp.get(t) is not None and cmp[t].shape[0] > criterion for t in targets],
        dtype=bool,
    )

    for t, keep in zip(targets, keep_mask):
        if not keep:
            for d in (cmp, offsets_dict, src_h, rec_h):
                d.pop(t, None)

    return cmp, offsets_dict, src_h, rec_h, targets[keep_mask], tar_heights[keep_mask]


def main_loop(
    cmp,
    offsets_dict,
    src_h,
    rec_h,
    targets,
    dt,
    v_min,
    v_max,
    n_vel,
    gate_len,
    gate_step,
    show_est_vel,
    return_velocity=False,
):
    stacks, mids, pos = [], [], []
    if return_velocity:
        all_picks, all_vt, all_tc = [], [], []
    for tgt in tqdm(targets, desc="Velocity analysis"):
        data, offsets, hs, hr, hm = _bring_target_data(
            cmp, offsets_dict, src_h, rec_h, tgt
        )
        best_vs, t_centers = est_velocity(
            v_min,
            v_max,
            n_vel,
            tgt,
            data,
            offsets,
            hs,
            hr,
            dt,
            gate_len,
            gate_step,
            show=show_est_vel,
        )
        v_t = build_vt(t_centers, best_vs, dt, data.shape[1])
        stack = apply_enmo(data, offsets, hs, hr, dt, v_t)
        stacks.append(stack)
        mids.append(hm.mean())
        pos.append(tgt)
        if return_velocity:
            all_picks.append(best_vs)
            all_vt.append(v_t)
            all_tc.append(t_centers)
    if return_velocity:
        return (
            stacks,
            mids,
            pos,
            np.array(all_picks),
            np.array(all_vt),
            np.array(all_tc),
        )
    return stacks, mids, pos


# =============================================================================
#  Depth conversion helpers
# =============================================================================

def depth_from_constant_velocity(depth_vel, t_axis):
    """
    Compute a depth axis from constant velocity.

    z(t) = t * depth_vel / 2

    For a two-way travel time axis t, this gives a linear mapping to depth.
    With depth_vel=2.0, the vertical axis is approximately equal to
    one-way travel time in seconds, which is convenient for initial
    visual inspection.

    Parameters
    ----------
    depth_vel : float
        Constant time-to-depth conversion velocity [m/s].
    t_axis : np.ndarray, shape (n_samples,)
        Two-way travel time axis [s].

    Returns
    -------
    depth : np.ndarray, shape (n_samples,)
        Depth axis [m].
    """
    return t_axis * depth_vel / 2.0


def depth_from_stacking_velocity(stacking_velocity_model, stacking_velocity_time):
    """
    Compute per-CMP depth curves via direct integration of stacking velocity.

    z_i(t) = 0.5 * cumulative_integral(v_stack_i, tau, 0..t)

    **Physical approximation warning**: This treats stacking (NMO) velocity
    as if it were interval velocity and integrates directly. Under the
    horizontal-layer (hyperbolic NMO) assumption, stacking velocity
    approximates RMS velocity, and integrating it directly is a rough
    approximation. For a more rigorous depth conversion, the RMS velocity
    should first be converted to interval velocity via Dix differentiation,
    then integrated. That path is reserved for a future
    "dix_interval_velocity" depth_conversion_mode.

    Parameters
    ----------
    stacking_velocity_model : np.ndarray, shape (n_cmp, n_samples)
        Stacking (NMO) velocity model v(t) [m/s] for each CMP.
    stacking_velocity_time : np.ndarray, shape (n_samples,)
        Two-way travel time axis [s] for the velocity model.

    Returns
    -------
    depth_per_cmp : np.ndarray, shape (n_cmp, n_samples)
        Depth z_i(t_j) [m] for each CMP i at each time sample j.
    """
    n_cmp, n_samples = stacking_velocity_model.shape
    depth_per_cmp = np.zeros_like(stacking_velocity_model)
    for i in range(n_cmp):
        depth_per_cmp[i] = 0.5 * cumulative_trapezoid(
            stacking_velocity_model[i], stacking_velocity_time, initial=0
        )
    return depth_per_cmp


def resample_to_common_depth(stack_horiz_time, depth_per_cmp, t_axis,
                              n_depth_samples=None):
    """
    Resample a time-domain stack section onto a common depth grid.

    Each CMP trace has its own depth curve z_i(t), so different traces
    sample different depths at the same time. This function interpolates
    all traces onto a uniform depth grid so the result can be displayed
    as a 2D image (CMP position vs. depth).

    Out-of-range depths (deeper than a given CMP's maximum depth, or
    shallower than zero) are filled with 0.0, which is consistent with
    the surface-statics masking convention used elsewhere.

    Parameters
    ----------
    stack_horiz_time : np.ndarray, shape (n_cmp, n_samples_time)
        Time-domain stack section (after surface statics correction).
    depth_per_cmp : np.ndarray, shape (n_cmp, n_samples_time)
        Per-CMP depth curves z_i(t) from depth_from_stacking_velocity().
    t_axis : np.ndarray, shape (n_samples_time,)
        Two-way travel time axis [s] (not used directly, but defines the
        time-domain shape of the input).
    n_depth_samples : int | None
        Number of points in the common depth grid.
        If None, defaults to n_samples_time (preserving data density).

    Returns
    -------
    resampled : np.ndarray, shape (n_cmp, n_depth_samples)
        Stack section resampled onto the common depth grid.
    common_depth_axis : np.ndarray, shape (n_depth_samples,)
        Common depth grid [m], starting at 0.
    """
    n_cmp, n_samples_time = stack_horiz_time.shape
    if n_depth_samples is None:
        n_depth_samples = n_samples_time

    z_max = depth_per_cmp.max()
    if z_max <= 0:
        z_max = 1.0
    common_depth_axis = np.linspace(0.0, z_max, n_depth_samples)

    resampled = np.zeros((n_cmp, n_depth_samples))
    for i in range(n_cmp):
        resampled[i] = np.interp(
            common_depth_axis,
            depth_per_cmp[i],
            stack_horiz_time[i],
            left=0.0,
            right=0.0,
        )

    return resampled, common_depth_axis
def sort_cmp_position(pos, stacks, mids, targets, tar_heights):
    cmp_pos = np.array(pos)
    sort_idx = np.argsort(cmp_pos)
    cmp_pos = cmp_pos[sort_idx]
    stack_mat = np.stack(stacks)[sort_idx]
    mid_elev = np.array(mids)[sort_idx]
    targets = np.array(targets)[sort_idx]
    tar_heights = np.array(tar_heights)[sort_idx]
    return cmp_pos, stack_mat, mid_elev, targets, tar_heights


def shift_to_surface(stack_mat, mid_elev, depth_vel, dt, *,
                     depth_conversion_mode="constant",
                     stacking_velocity_model=None,
                     stacking_velocity_time=None,
                     n_depth_samples=None):
    """
    Apply surface-statics shift (Step A) and time-to-depth conversion (Step B).

    Step A (surface statics) uses constant ``depth_vel`` for all CMPs,
    regardless of ``depth_conversion_mode``.  A per-CMP velocity could
    theoretically improve the statics correction, but the interpretation
    is less straightforward; this is deferred to a future extension.

    Step B (depth-axis generation) branches on ``depth_conversion_mode``:

    * ``"constant"``: linear mapping  z = t * depth_vel / 2.
    * ``"stacking_velocity"``: per-CMP depth curves via direct integration
      of stacking velocity, then resampling onto a common depth grid.

    Parameters
    ----------
    stack_mat : np.ndarray, shape (n_cmp, n_samples)
        Stacked section in two-way travel time.
    mid_elev : np.ndarray, shape (n_cmp,)
        Mean elevation of source + receiver for each CMP.
    depth_vel : float
        Constant velocity [m/s] used for surface statics (Step A)
        and for time-to-depth conversion when mode="constant".
    dt : float
        Time sampling interval [s].
    depth_conversion_mode : {"constant", "stacking_velocity"}
        "constant":  constant-velocity depth conversion (default, backward-compatible).
        "stacking_velocity": per-CMP depth curves from stacking velocity model.
    stacking_velocity_model : np.ndarray | None
        Required when mode="stacking_velocity".  Shape (n_cmp, n_samples).
        Stacking (NMO) velocity v(t) [m/s].
    stacking_velocity_time : np.ndarray | None
        Required when mode="stacking_velocity".  Shape (n_samples,).
        Two-way travel time axis [s] for the velocity model.
    n_depth_samples : int | None
        Number of points in the common depth grid (stacking_velocity mode).
        If None, defaults to n_samples (preserving data density).

    Returns
    -------
    stack_horiz : np.ndarray
        Surface-shifted (and depth-resampled if stacking_velocity mode) stack.
    surf_ref : float
        Surface reference elevation.
    elev_axis : np.ndarray
        Elevation axis (= surf_ref − depth axis).
    depth_per_cmp : np.ndarray | None
        Per-CMP depth curves z_i(t).  None in "constant" mode.
    common_depth_axis : np.ndarray | None
        Common depth grid.  None in "constant" mode.
    """
    # Step A: Surface statics correction (constant depth_vel for all CMPs).
    # NOTE: A per-CMP stacking velocity could be used here instead of
    # depth_vel, but the interpretation and stability implications are
    # less straightforward.  This extension is deferred for now.
    surf_ref = mid_elev.max()
    shift_tw = 2 * (surf_ref - mid_elev) / depth_vel
    shift_samp = (shift_tw / dt).astype(int)
    stack_horiz = np.zeros_like(stack_mat)
    for i, tr in enumerate(stack_mat):
        stack_horiz[i] = np.interp(
            np.arange(tr.size) + shift_samp[i],
            np.arange(tr.size),
            tr,
            left=0.0,
            right=0.0,
        )

    t = np.arange(stack_mat.shape[1]) * dt

    # Step B: Depth-axis generation (mode-dependent).
    if depth_conversion_mode == "constant":
        depth = depth_from_constant_velocity(depth_vel, t)
        elev_axis = surf_ref - depth
        depth_per_cmp = None
        common_depth_axis = None
    elif depth_conversion_mode == "stacking_velocity":
        if stacking_velocity_model is None or stacking_velocity_time is None:
            raise ValueError(
                "stacking_velocity_model and stacking_velocity_time are required "
                "when depth_conversion_mode='stacking_velocity'"
            )
        depth_per_cmp = depth_from_stacking_velocity(
            stacking_velocity_model, stacking_velocity_time
        )
        stack_horiz, common_depth_axis = resample_to_common_depth(
            stack_horiz, depth_per_cmp, t,
            n_depth_samples=n_depth_samples,
        )
        elev_axis = surf_ref - common_depth_axis
    else:
        raise ValueError(
            f"Unknown depth_conversion_mode: {depth_conversion_mode!r}. "
            "Supported: 'constant', 'stacking_velocity'"
        )

    return stack_horiz, surf_ref, elev_axis, depth_per_cmp, common_depth_axis


def surface_spline(cmp_pos, elev, n_points):
    if len(cmp_pos) < 3:
        spl = interpolate.interp1d(
            cmp_pos, elev, kind="linear", fill_value="extrapolate"
        )
    else:
        spl = interpolate.interp1d(
            cmp_pos, elev, kind="quadratic", fill_value="extrapolate"
        )
    dense_x = np.linspace(cmp_pos.min(), cmp_pos.max(), n_points)
    return dense_x, spl(dense_x)


def zero_upper_surface(cmp_pos, stack_horiz, elev_axis, dense_x, hm_dense, pad=np.nan):
    hm_img = np.interp(cmp_pos, dense_x, hm_dense)[:, None]
    mask_above = elev_axis[None, :] > hm_img
    out = stack_horiz.copy()
    out[mask_above] = pad
    return out


# =============================================================================
#  Reflection Mixin
# =============================================================================
class NMO_correction(PlotterWrapperMixin):
    """
    Reflection cross-section を導出する Mixin。
    GroupProcesser に CmpGathering と一緒に多重継承させて gp.reflection(...) で利用する。
    """

    def NMO_correction(
        self,
        axis: str,
        v_min: float = 2.0,
        v_max: float = 200.0,
        n_vel: int = 99,
        gate_len: float = 0.02,
        gate_step: float = 0.01,
        depth_vel: float = 56.0,
        criterion: int = 1,
        round_targets: int = 4,
        show_est_vel: bool = False,
        return_stacking_velocity: bool = False,
        depth_conversion_mode: str = "constant",
        n_depth_samples: int | None = None,
        plot_x_as_distance: bool = True,
        title: str = "",
        xmin: float | None = None,
        xmax: float | None = None,
        ymin: float | None = None,
        ymax: float | None = None,
        num_ticks: int = 8,
        save_name: str | None = None,
        show: bool = True,
        **kw,
    ):
        """
        反射断面を出力する

        return stack_horiz:array(反射断面), cmp_pos(cmpの位置), elev_axis(cmp点の高度), targets_sorted

        手順:
            1. cmp_gathering() で CMP 集約
            2. CMP 標高補間
            3. トレース数の少ない CMP を除外
            4. 各 CMP で速度推定 + Elev+NMO 補正 + スタック
            5. 位置でソート → 地表基準にシフト
            6. 表面スプライン → 上部マスク
            7. 反射断面の描画 / 保存

        Parameters
        ----------
        axis : 'x' | 'y' | 'z'
        v_min, v_max, n_vel : 速度推定用の試行速度範囲とサンプル数
        gate_len, gate_step : 速度推定窓 [s]
        depth_vel : 時刻→深度変換速度 [m/s]
            depth_conversion_mode="constant" のとき time→depth 変換に使われる。
            depth_conversion_mode="stacking_velocity" のときは surface statics
            (Step A) にのみ使われる。depth_vel=2.0 とすると縦軸が時間 (s) に
            近似されるため、解析の第一段階で便利。
        criterion : この本数以下のトレースしか持たない CMP は除外
        round_targets : int, default 4
            CMP ターゲット位置を丸める小数点以下桁数。float 精度で分裂した
            近接 CMP を同一視して統合する（丸めたキーが衝突するトレースを結合）。
            既定 4 桁は get_all_CMP の丸めと一致するため実質 no-op（後方互換）。
            桁を下げると近接 CMP が統合され、各 CMP の重合数が増える。
            None で統合を無効化。
        show_est_vel : True で推定速度を print
        return_stacking_velocity : True で stacking velocity 解析結果を NMOFullResult で返す
        depth_conversion_mode : {"constant", "stacking_velocity"}
            "constant": constant-velocity depth conversion (default, backward-compatible).
                Uses depth_vel for both surface statics (Step A) and time→depth
                conversion (Step B).  The vertical axis is z = t * depth_vel / 2.
            "stacking_velocity": per-CMP depth curves from stacking velocity model.
                Surface statics (Step A) still uses constant depth_vel.
                Depth axis (Step B) uses per-CMP numerical integration of the
                stacking velocity model:
                    z_i(t) ≈ 0.5 * ∫₀ᵗ v_stack,i(τ) dτ
                This is an approximation — see docstrings of
                depth_from_stacking_velocity() for caveats.
                Requires return_stacking_velocity=True or will be forced internally.
            Future extension: "dix_interval_velocity" will convert stacking
            velocity to interval velocity via Dix differentiation before
            integrating to depth.
        n_depth_samples : int | None
            Number of points in the common depth grid (stacking_velocity mode).
            If None (default), uses the same number of points as the time axis.
        plot_x_as_distance : True で X 軸を距離 [m]、False で CMP position
        title : 図タイトル
        xmin, xmax, ymin, ymax : プロット範囲
        num_ticks : X 軸 tick 数
        save_name : 出力ファイルパス。None で保存しない
        show : True で plt.show()
        **kw : backend 固有の描画引数（matplotlib 系）。
            代表例: cmap / figsize / aspect / interpolation。
            未指定時は figsize=(14, 8) / cmap="gray" / aspect="auto" を採用する。
            描画時に backend (_reflection_image_impl) が **kw から必要分を取り出す。
            完全一覧と backend ごとの差異は
            src/plotting/backends/matplotlib_backend.py 冒頭を参照。

        Returns
        -------
        When return_stacking_velocity=False (default):
            tuple of (stack_horiz, cmp_pos, elev_axis, targets_sorted)
        When return_stacking_velocity=True:
            NMOFullResult with stacking_velocity attribute containing
            StackingVelocityResult (picks, model, time axis, gate centers),
            and optional depth_per_cmp / common_depth_axis attributes
            (set when depth_conversion_mode="stacking_velocity").
        """
        (
            stack_horiz,
            cmp_pos,
            elev_axis,
            targets_sorted,
            dense_x,
            height_dense,
            stacking_vel,
            depth_per_cmp,
            common_depth_axis,
        ) = self._NMO_correction_core(
            axis,
            v_min,
            v_max,
            n_vel,
            gate_len,
            gate_step,
            depth_vel,
            criterion,
            show_est_vel,
            round_targets=round_targets,
            return_stacking_velocity=return_stacking_velocity,
            depth_conversion_mode=depth_conversion_mode,
            n_depth_samples=n_depth_samples,
        )

        # 8) 反射断面の描画 / 保存 / 表示 — PlotterWrapperMixin に委譲
        # NMO_correction 既定の matplotlib 系描画設定を kw に注入（未指定時のみ）。
        # 現行 public 挙動（figsize=(14,8) / cmap="gray" / aspect="auto"）を維持しつつ、
        # backend 固有の描画引数は **kw 経由で backend まで素通しする。
        kw.setdefault("figsize", (14, 8))
        kw.setdefault("cmap", "gray")
        kw.setdefault("aspect", "auto")

        self.reflection_image(
            stack_horiz,
            cmp_pos,
            elev_axis,
            targets_sorted,
            plot_x_as_distance=plot_x_as_distance,
            title=title,
            num_ticks=num_ticks,
            dense_x=dense_x,
            hm_dense=height_dense,
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            save_name=save_name,
            show=show,
            **kw,
        )

        if return_stacking_velocity:
            return NMOFullResult(
                stack_horiz=stack_horiz,
                cmp_pos=cmp_pos,
                elev_axis=elev_axis,
                targets_sorted=targets_sorted,
                stacking_velocity=stacking_vel,
                depth_per_cmp=depth_per_cmp,
                common_depth_axis=common_depth_axis,
            )
        return stack_horiz, cmp_pos, elev_axis, targets_sorted

    def _NMO_correction_core(
        self,
        axis,
        v_min,
        v_max,
        n_vel,
        gate_len,
        gate_step,
        depth_vel,
        criterion,
        show_est_vel,
        round_targets: int = 4,
        return_stacking_velocity=False,
        depth_conversion_mode="constant",
        n_depth_samples=None,
    ):
        """
        NMO_correction() の純粋計算部 (描画なし)。

        Section 1: CMP gathering (self.cmp_gathering を呼び self の属性を更新)
        Section 2: 各 CMP 点の標高補間
        Section 3: トレース数の少ない CMP を除外
        Section 4: 速度推定 + Elev+NMO + スタック (進捗 print 含む)
        Section 5: 位置でソート
        Section 6: 地表基準にシフト + depth conversion (mode-dependent)
        Section 7: 表面スプライン + 上部マスク

        Parameters
        ----------
        round_targets : int | None, default 4
            Number of decimal places to round CMP target positions to before
            velocity analysis.  Near-duplicate targets that only differ by
            floating-point noise are merged (their traces concatenated) into a
            single higher-fold CMP.  Default 4 matches get_all_CMP rounding, so
            it is effectively a no-op; None disables merging.
        depth_conversion_mode : {"constant", "stacking_velocity"}
            "constant": uses depth_vel for both statics and depth axis.
            "stacking_velocity": uses stacking velocity for depth axis
                (per-CMP integration), but still uses depth_vel for statics.
                Forces return_stacking_velocity=True internally if needed.
        n_depth_samples : int | None
            Number of points in the common depth grid (stacking_velocity mode).
            If None, uses the same number of points as the time axis.

        Returns
        -------
        stack_horiz     : np.ndarray, 反射断面
        cmp_pos         : np.ndarray, CMP 水平位置 [m]
        elev_axis       : np.ndarray, 標高軸
        targets_sorted  : np.ndarray, ソート済み CMP target キー
        dense_x         : np.ndarray, surface spline の補間 X
        height_dense    : np.ndarray, surface spline の補間高さ
        stacking_vel    : StackingVelocityResult | None
            Stacking velocity analysis results (only when return_stacking_velocity=True
            or depth_conversion_mode="stacking_velocity").
        depth_per_cmp   : np.ndarray | None
            Per-CMP depth curves z_i(t). None unless mode="stacking_velocity".
        common_depth_axis : np.ndarray | None
            Common depth grid. None unless mode="stacking_velocity".
        """
        # 1) CMP gathering — CmpGathering mixin に委譲
        self.cmp_gathering(axis=axis, average=False, show=False)

        # 1b) 近接ターゲットの統合 — float 精度で分裂した CMP を round_targets 桁で
        #     丸めて同一視し、トレースを結合する。remove_single_trace_gathers より
        #     前に行うことで、単独トレース bin が結合されて criterion を通過しうる。
        (
            self.cmp,
            self.offsets,
            self.src_h,
            self.rec_h,
            self.targets,
        ) = merge_close_targets(
            self.cmp,
            self.offsets,
            self.src_h,
            self.rec_h,
            self.targets,
            decimals=round_targets,
        )

        cmp = self.cmp
        offsets_dict = self.offsets
        src_h = self.src_h
        rec_h = self.rec_h
        targets = np.asarray(self.targets)
        dt = 1.0 / self.fs

        # 2) 各 CMP 点の標高補間
        if hasattr(self.datalist[0], "z_distance"):
            tar_heights = self.get_all_target_height()
        else:
            tar_heights = np.zeros_like(targets, dtype=float)

        # 3) トレース数の少ない CMP を除外
        cmp, offsets_dict, src_h, rec_h, targets, tar_heights = (
            remove_single_trace_gathers(
                cmp,
                offsets_dict,
                src_h,
                rec_h,
                targets,
                tar_heights,
                criterion=criterion,
            )
        )

        if len(targets) == 0:
            raise ValueError(
                f"criterion={criterion} を超えるトレースを持つ CMP が存在しません。criterion引数の整数値を下げてください。"
            )

        # 4) 速度推定 + Elev+NMO + スタック
        # When depth_conversion_mode="stacking_velocity", we need the velocity
        # model even if the user did not request return_stacking_velocity.
        need_velocity = return_stacking_velocity or (depth_conversion_mode == "stacking_velocity")

        ml_result = main_loop(
            cmp,
            offsets_dict,
            src_h,
            rec_h,
            targets,
            dt,
            v_min,
            v_max,
            n_vel,
            gate_len,
            gate_step,
            show_est_vel,
            return_velocity=need_velocity,
        )
        if need_velocity:
            stacks, mids, pos, all_picks, all_vt, all_tc = ml_result
        else:
            stacks, mids, pos = ml_result
        print(f"Processing complete. Total CMP gathers: {len(targets)}")

        # 5) 位置でソート
        cmp_pos, stack_mat, mid_elev, targets_sorted, tar_heights = sort_cmp_position(
            pos, stacks, mids, targets, tar_heights
        )

        # ソート順序を速度配列にも適用
        if need_velocity:
            sort_idx = np.argsort(pos)
            all_picks = all_picks[sort_idx]
            all_vt = all_vt[sort_idx]

        # Save original time sample count (before depth-conversion resampling).
        # StackingVelocityResult.stacking_velocity_time must always use the
        # original time axis, regardless of depth conversion mode.
        n_samples_time = stack_mat.shape[1]

        # 6) 地表基準にシフト + depth conversion
        if depth_conversion_mode == "stacking_velocity":
            sv_model = all_vt  # shape (n_cmp, n_samples_time), already sorted
            sv_time = np.arange(n_samples_time) * dt
            stack_horiz, surf_ref, elev_axis, depth_per_cmp, common_depth_axis = shift_to_surface(
                stack_mat, mid_elev, depth_vel, dt,
                depth_conversion_mode="stacking_velocity",
                stacking_velocity_model=sv_model,
                stacking_velocity_time=sv_time,
                n_depth_samples=n_depth_samples,
            )
        else:
            stack_horiz, surf_ref, elev_axis, depth_per_cmp, common_depth_axis = shift_to_surface(
                stack_mat, mid_elev, depth_vel, dt,
                depth_conversion_mode="constant",
            )

        # 7) 表面スプライン + 上部マスク
        dense_x, height_dense = surface_spline(cmp_pos, tar_heights, len(cmp_pos))
        stack_horiz = zero_upper_surface(
            cmp_pos, stack_horiz, elev_axis, dense_x, height_dense
        )

        # 7b) migration 等の後続処理向けに地表情報を attribute として公開
        self._last_nmo_surface = {
            "surf_ref": float(surf_ref),
            "dense_x": dense_x,
            "height_dense": height_dense,
        }

        if need_velocity:
            t_time = np.arange(n_samples_time) * dt
            stacking_vel = StackingVelocityResult(
                stacking_velocity_picks=all_picks,
                stacking_velocity_model=all_vt,
                stacking_velocity_time=t_time,
                gate_time_centers=all_tc[0],
                cmp_positions=cmp_pos,
                targets_sorted=targets_sorted,
            )
        else:
            stacking_vel = None

        return stack_horiz, cmp_pos, elev_axis, targets_sorted, dense_x, height_dense, stacking_vel, depth_per_cmp, common_depth_axis
