"""2D SH-wave Kirchhoff depth migration on a seismic survey line.

This example demonstrates a Kirchhoff depth migration workflow: multiple shot
gathers are preprocessed, grouped into CMP gathers, NMO-corrected and stacked,
then migrated to produce a depth image of the subsurface.  The Y-axis (SH-wave)
is used, but the same workflow applies to other components by changing AXIS.

The velocity model (VS_MODEL) is a simple constant-velocity example for
educational purposes, not an inversion result.  For production use, replace
it with a velocity model derived from velocity analysis.

Main steps: load → preprocess → CMP + NMO → Kirchhoff migration → plot.
"""

import glob
import os
import sys

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser

# -- Input / output paths --
DATA_GLOB = "testcodes/data/sinkhole_cav_ricker/*.npz"
SAVE_PATH = "examples/output/kirchhoff_migration/kirchhoff.png"

# -- Axis selection --
AXIS = "y"  # SH-wave component (use "z" for vertical P-wave)

# -- NMO correction parameters --
V_MIN = 2.0          # minimum trial velocity [m/s]
V_MAX = 400.0        # maximum trial velocity [m/s]
N_VEL = 99           # number of trial velocities
GATE_LEN = 0.02      # velocity-estimation window length [s]
GATE_STEP = 0.01     # velocity-estimation window step [s]
DEPTH_VEL = 200.0     # time-to-depth conversion velocity [m/s]

# -- Kirchhoff migration parameters --
VS_MODEL = np.array([200.0, 200.0])  # 1D SH-wave velocity model [m/s] (constant)
Z_MAX = 10.0          # maximum migration depth [m]
DZ = 0.05             # depth sampling interval [m]
APERTURE_HALF = None  # None = use all CMPs (no aperture limit)
AMPLITUDE_WEIGHT = True  # apply geometrical spreading & obliquity correction

# -- Filter parameters for preprocessing --
LOWPASS_FPASS = 200   # lowpass passband [Hz]
LOWPASS_FSTOP = 800   # lowpass stopband [Hz]
HIGHPASS_FPASS = 10   # highpass passband [Hz]
HIGHPASS_FSTOP = 2    # highpass stopband [Hz]


def main():
    # -- collect input files --
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No files found matching {DATA_GLOB}")
        return
    print(f"Found {len(paths)} files for axis={AXIS!r}")

    # -- preprocess each shot gather --
    single_list = []
    for path in paths:
        sp = SingleProcesser(path)
        sp.sort_distance()
        sp.highpass(AXIS, fpass=HIGHPASS_FPASS, fstop=HIGHPASS_FSTOP)
        sp.lowpass(AXIS, fpass=LOWPASS_FPASS, fstop=LOWPASS_FSTOP)
        single_list.append(sp)

    # -- Kirchhoff migration over the group --
    group = GroupProcesser(single_list, AXIS)

    image, z_img, cmp_pos, targets_sorted = group.Kirchhoff_migration(
        axis=AXIS,
        v_min=V_MIN,
        v_max=V_MAX,
        n_vel=N_VEL,
        gate_len=GATE_LEN,
        gate_step=GATE_STEP,
        depth_vel=DEPTH_VEL,
        vs_model=VS_MODEL,
        z_max=Z_MAX,
        dz=DZ,
        aperture_half=APERTURE_HALF,
        amplitude_weight=AMPLITUDE_WEIGHT,
        save_name=SAVE_PATH,
        show=True,
        show_nmo=False,
    )

    # -- inspect results --
    print(f"Migration complete. image shape: {image.shape}")
    print(f"Depth range: 0 — {z_img[-1]:.2f} m")
    print(f"CMP positions: {cmp_pos.shape[0]} points")


if __name__ == "__main__":
    main()