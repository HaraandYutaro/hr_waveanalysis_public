"""Quickstart: single-file seismic waveform analysis.

Demonstrates the standard analysis flow for one shot gather (NPZ file):
distance sort → amplitude normalization → AGC gain → lowpass filter →
seismogram / cmap / spectra visualization.  The Y-axis is used throughout,
corresponding to the SH-wave (Love-wave) component.

Design philosophy
----------------
- Collect user-configurable parameters at the top of the file.
- Use sensible defaults so the script runs out-of-the-box.
- Call only public APIs on SingleProcesser (inherited via PlotterWrapperMixin).
- Keep the pipeline short and linear; beginners can copy, modify, and extend.
"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.single_processor import SingleProcesser

NPZ_PATH = "sample_data/npz/simudata/D0_6W2_0S0.npz"
AXIS = "y"
SAVE_DIR = None
BACKEND = "mpl"


def main():
    # -- load data --
    hr = SingleProcesser(NPZ_PATH, BACKEND)

    # -- preprocessing --
    hr.sort_distance()
    hr.trace_amp_regularize(AXIS)
    hr.gain_AGC(AXIS, batchsize=128, clip=10)

    # -- filtering --
    hr.lowpass(AXIS, fpass=200, fstop=800)

    # -- seismogram plot --
    hr.seismogram(AXIS, show=True)

    # -- cmap (time-distance colormap) --
    hr.cmap(hr, AXIS, show=True)

    # -- spectra plot --
    hr.spectra(axis=AXIS, show=True)

    # -- optional: backscatter amplitude analysis (FFT mode) --
    # Uncomment the lines below to compute and plot reflection amplitude
    # using a sliding-window f-k spectrum method.
    ind, amp = hr.backscatter(
        AXIS,
        mode="fft",
        N=6,
        show=True,
        show_fk=False,
    )
    print(f"backscatter done: {len(ind)} windows")


if __name__ == "__main__":
    main()