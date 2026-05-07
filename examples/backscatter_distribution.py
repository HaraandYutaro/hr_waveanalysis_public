"""Backscatter amplitude distribution of SH waves along a survey line.

This example uses multiple shot gathers from the same survey line to compute
and plot the averaged backscatter (reflection) amplitude distribution in the
Y-direction, which corresponds to SH-waves (Love waves).  An FFT-based
sliding-window method (backscatter mode="fft") is applied to each shot, and
results are averaged across all shots at each distance.

Main steps: load → preprocess (lowpass + highpass) → group → analyze
(backscatter_distribution) → inspect results.
"""

import glob
import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser

DATA_GLOB = "sample_data/realdata/*.npz"
AXIS = "y"
SAVE_PATH = "examples/output/backscatter_distribution/distribution.png"
BACKEND = "mpl"


def main():
    # 1. Collect input files
    paths = sorted(glob.glob(DATA_GLOB))
    if not paths:
        print(f"No files found matching {DATA_GLOB}")
        return
    print(f"Found {len(paths)} files for axis={AXIS!r}")

    # 2. Build SingleProcesser list with preprocessing
    single_list = []
    for path in paths:
        sp = SingleProcesser(path, BACKEND)
        sp.gain_AGC(AXIS, batchsize=128, clip=10)
        sp.lowpass(AXIS, fpass=50, fstop=200)
        sp.highpass(AXIS, fpass=10, fstop=2)
        
        single_list.append(sp)

    # 3. Create GroupProcesser and run backscatter distribution
    group = GroupProcesser(single_list, AXIS)
    avg, all_distance = group.backscatter_distribution(
        axis=AXIS,
        N=6,
        mode="fft",
        show=False,
        save_name=SAVE_PATH,
    )

    # 4. Inspect results
    print(f"Averaged amplitude distribution shape: {avg.shape}")
    print(f"All-distance array shape:                {all_distance.shape}")
    print(f"Non-zero entries: {(avg > 0).sum()} / {avg.size}")


if __name__ == "__main__":
    main()