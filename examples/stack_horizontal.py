"""Horizontal stacking of repeated shot gathers for S/N improvement.

This example demonstrates geometry-aligned horizontal stacking: multiple shot
or receiver gathers recorded at the same line are time-aligned using
cross-correlation at a reference distance, then averaged to produce a
single stacked trace gather with improved signal-to-noise ratio.  The Y-axis
(SH-wave component) is used here, but the same workflow applies to other
components by changing AXIS.

Horizontal stacking is useful when repeated shots at the same location allow
random noise to cancel out while coherent signals add constructively.

Main steps: load → align by geometry → stack → normalize → visualize.
"""

import glob
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser

# -- Input / output paths --
DATA_GLOB = "sample_data/hor_stack_before/*.npz"

# -- Axis and alignment settings --
AXIS = "y"              # SH-wave component
REF_DISTANCE = -0.01    # reference distance for geometry alignment [m]
BACKEND = "mpl"
STACK_NAME = "line_horiz_stack"


def main():
    # 1. Collect input files
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No files found matching {DATA_GLOB}")
        return
    print(f"Found {len(paths)} files for axis={AXIS!r}")

    # 2. Build SingleProcesser list
    single_list = []
    for path in paths:
        sp = SingleProcesser(path, backend=BACKEND)
        single_list.append(sp)

    # 3. Align and stack by geometry
    group = GroupProcesser(single_list, name=STACK_NAME)
    stacked = group.align_stack_by_geometry(
        single_list,
        AXIS,
        ref_distance=REF_DISTANCE,
    )

    # 4. Normalize and visualize
    stacked.trace_amp_regularize(AXIS)
    stacked.seismogram(AXIS)


if __name__ == "__main__":
    main()