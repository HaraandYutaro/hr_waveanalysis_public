"""Quickstart: group processing of multiple seismic records.

Demonstrates a one-command multi-file workflow using GroupProcesser and
SingleProcesser.  By specifying a single axis (e.g. "y" for SH / Love waves),
you can preprocess multiple shot records, build CMP gathers, and visualize
dispersion characteristics -- all with a small number of user-visible
variables and sensible defaults.

Design philosophy
----------------
- One-command multi-file processing: build a list of SingleProcesser instances
  from a glob of .npz files, then hand them to GroupProcesser with an axis.
- Axis-based workflow: switch between "x", "y", "z" by changing AXIS; no code
  duplication needed for each component.
- Sensible defaults that can be overridden at the top of the file.
- Minimal, linear flow that is easy to read for beginners.

What is visualized
------------------
- CMP (Common Mid Point) gathers built from the input shots.
- Dispersion curves for Love waves (Y-direction, SH-wave) at each CMP target,
  showing the relationship between frequency and phase velocity.
"""

import glob
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser

DATA_GLOB = "sample_data/npz/realdata/*.npz"
AXIS = "y"
CMP_SAVE_NAME = None
DISP_SAVE_NAME = None


def main():
    # -- collect input files --
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No files found matching {DATA_GLOB}")
        return
    print(f"Found {len(paths)} files for axis={AXIS!r}")

    # -- build SingleProcesser list with preprocessing --
    processors = []
    for path in paths:
        sp = SingleProcesser(path)
        sp.lowpass(AXIS, fpass=100, fstop=400)
        sp.remove(AXIS, remove_chs=[0, 1, 2, 3])
        sp.trace_amp_regularize(AXIS)
        processors.append(sp)

    # -- build CMP gathers --
    group = GroupProcesser(processors, AXIS)
    group.cmp_gathering(
        axis=AXIS,
        cross_corr=False,
        integrate=False,
        average=True,
        save_name=CMP_SAVE_NAME,
        show=False,
    )

    # -- dispersion curve per CMP target --
    for tgt in group.targets:
        res, f_mesh, c_mesh, *_ = group.dispersion_curve(tgt, show=False, save_name=DISP_SAVE_NAME, title = f"{tgt:.1f} m: Dispersion curve (axis={AXIS})")


if __name__ == "__main__":
    main()