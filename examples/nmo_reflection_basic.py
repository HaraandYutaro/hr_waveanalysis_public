"""Basic NMO reflection analysis on CMP gathers.

This example demonstrates a standard reflection seismology workflow:
multiple shot gathers from one survey line are preprocessed, grouped into
CMP gathers, and then NMO (Normal Moveout) corrected and stacked to produce
a reflection cross-section.  The Y-axis is used here (SH-wave / Love-wave
component), but the same workflow applies to Z (vertical) or X (inline)
by changing AXIS.

The velocity model used is a simple constant-velocity scan (v_min to v_max
with n_vel samples), not an inversion result.  For production use, replace
it with a more accurate velocity model derived from velocity analysis.

Main steps: load → preprocess → CMP gather → NMO correction + stack → plot.
After NMO correction, stacking velocity picks and the interpolated velocity
model v(t) are also visualized and saved.

NOTE: The extracted velocities are stacking velocities (also called NMO
velocities), which correspond to RMS velocities under the horizontal-layer
(hyperbolic NMO) assumption.  They are NOT depth velocities or interval
velocities.
"""

import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser

# -- Input / output paths --
DATA_GLOB = "sample_data/realdata/*.npz"
SAVE_DIR = "examples/output/nmo_reflection"
SAVE_PATH = os.path.join(SAVE_DIR, "reflection.png")
SAVE_VEL_PICKS_PATH = os.path.join(SAVE_DIR, "stacking_velocity_picks.png")
SAVE_VEL_MODEL_PATH = os.path.join(SAVE_DIR, "stacking_velocity_model.png")

# -- Axis selection --
AXIS = "y"  # SH-wave component (use "z" for vertical P-wave reflection)

# -- Velocity analysis parameters --
V_MIN = 2.0         # minimum trial velocity [m/s]
V_MAX = 200.0       # maximum trial velocity [m/s]
N_VEL = 99          # number of trial velocities
GATE_LEN = 0.02     # velocity-estimation window length [s]
GATE_STEP = 0.01    # velocity-estimation window step [s]
DEPTH_VEL = 56.0    # time-to-depth conversion velocity [m/s]

# -- Filter parameters for preprocessing --
LOWPASS_FPASS = 200  # lowpass passband [Hz]
LOWPASS_FSTOP = 800  # lowpass stopband [Hz]
HIGHPASS_FPASS = 10  # highpass passband [Hz]
HIGHPASS_FSTOP = 2   # highpass stopband [Hz]


def plot_stacking_velocity_picks(vel, save_path, show=True):
    """
    Plot stacking velocity picks (semblance max) as a pcolormesh.

    These are stacking velocities (= NMO velocities) determined by
    semblance analysis.  Under the horizontal-layer (hyperbolic NMO)
    assumption, they correspond to RMS velocities.  They are NOT depth
    velocities or interval velocities.

    Parameters
    ----------
    vel : StackingVelocityResult
        Must contain stacking_velocity_picks, gate_time_centers, cmp_positions.
    save_path : str
        File path to save the figure.
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        vel.cmp_positions,
        vel.gate_time_centers,
        vel.stacking_velocity_picks.T,
        shading="auto",
        cmap="viridis",
    )
    ax.invert_yaxis()
    ax.set_xlabel("CMP Position [m]")
    ax.set_ylabel("Time [s]")
    ax.set_title("Stacking Velocity Picks (semblance max per gate)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Velocity [m/s]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_stacking_velocity_model(vel, save_path, show=True):
    """
    Plot the time-dependent stacking velocity model v(t) as a pcolormesh.

    This is the interpolated velocity model actually used for NMO correction.
    It is a stacking velocity (= NMO velocity) model, NOT a depth velocity
    model.  Under the horizontal-layer assumption, it corresponds to the
    RMS velocity as a function of two-way travel time.

    Parameters
    ----------
    vel : StackingVelocityResult
        Must contain stacking_velocity_model, stacking_velocity_time,
        cmp_positions.
    save_path : str
        File path to save the figure.
    show : bool
        Whether to call plt.show().
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(
        vel.cmp_positions,
        vel.stacking_velocity_time,
        vel.stacking_velocity_model.T,
        shading="auto",
        cmap="viridis",
    )
    ax.invert_yaxis()
    ax.set_xlabel("CMP Position [m]")
    ax.set_ylabel("Time [s]")
    ax.set_title("NMO Velocity Model v(t) (interpolated)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Velocity [m/s]")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150)
    print(f"Saved: {save_path}")
    if show:
        plt.show()
    plt.close(fig)


def main():
    # 1. Collect input files
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No files found matching {DATA_GLOB}")
        return
    print(f"Found {len(paths)} files for axis={AXIS!r}")

    # 2. Preprocess each shot gather
    single_list = []
    for path in paths:
        sp = SingleProcesser(path)
        sp.sort_distance()
        sp.highpass(AXIS, fpass=HIGHPASS_FPASS, fstop=HIGHPASS_FSTOP)
        sp.lowpass(AXIS, fpass=LOWPASS_FPASS, fstop=LOWPASS_FSTOP)
        single_list.append(sp)

    # 3. Build GroupProcesser and run NMO correction with velocity extraction
    group = GroupProcesser(single_list, AXIS)

    os.makedirs(SAVE_DIR, exist_ok=True)

    result = group.NMO_correction(
        axis=AXIS,
        v_min=V_MIN,
        v_max=V_MAX,
        n_vel=N_VEL,
        gate_len=GATE_LEN,
        gate_step=GATE_STEP,
        depth_vel=DEPTH_VEL,
        show_est_vel=False,
        return_stacking_velocity=True,
        save_name=SAVE_PATH,
        show=True,
    )

    # 4. Inspect NMO results (access via NMOFullResult attributes)
    stack_horiz = result.stack_horiz
    cmp_pos = result.cmp_pos
    elev_axis = result.elev_axis
    targets_sorted = result.targets_sorted

    print(f"Reflection section shape : {stack_horiz.shape}")
    print(f"CMP positions             : {cmp_pos.shape[0]} points")
    print(f"Elevation axis range      : {elev_axis.min():.2f} — {elev_axis.max():.2f}")

    # 5. Inspect and visualize stacking velocity results
    vel = result.stacking_velocity
    if vel is None:
        raise RuntimeError(
            "stacking_velocity is None despite return_stacking_velocity=True. "
            "This should not happen; check the NMO_correction implementation."
        )

    print(f"Stacking velocity picks shape : {vel.stacking_velocity_picks.shape}")
    print(f"Stacking velocity model shape : {vel.stacking_velocity_model.shape}")
    print(f"Gate time centers             : {vel.gate_time_centers.shape[0]} gates")
    print(f"Velocity time axis            : {vel.stacking_velocity_time.shape[0]} samples")

    # 5a. Stacking velocity picks (semblance max per gate)
    plot_stacking_velocity_picks(vel, SAVE_VEL_PICKS_PATH, show=True)

    # 5b. NMO velocity model v(t) (interpolated)
    plot_stacking_velocity_model(vel, SAVE_VEL_MODEL_PATH, show=True)


if __name__ == "__main__":
    main()