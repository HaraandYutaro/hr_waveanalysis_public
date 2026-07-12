"""Quickstart: load a .DAT file (SEG-2 format) with fallback metadata.

Demonstrates that load_dataset accepts .DAT as a SEG-2 candidate.
When no JSON sidecar or Excel datasheet is supplied, SingleProcesser
falls back to default metadata (all channels on the y-axis).
FallbackMetadataWarning is intentionally left visible on stderr.


"""

import os
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.processor.single_processor import SingleProcesser

DAT_PATH = "sample_data/sg2/GEO_1021.DAT"
AXIS = "y"


def main():
    # -- load data (.DAT, SEG-2 with fallback metadata) --
    sp = SingleProcesser(DAT_PATH)

    # -- inspect metadata and loaded components --
    print(f"metadata_source     : {sp.metadata_source}")
    print(f"metadata_is_fallback: {sp.metadata_is_fallback}")
    print(f"Num_sensor          : {sp.Num_sensor}")

    for name in ("x", "y", "z"):
        arr = getattr(sp, name, None)
        shape = arr.shape if arr is not None else None
        print(f"  {name} shape: {shape}")

    # -- preprocessing --
    sp.trace_amp_regularize(AXIS)

    # -- seismogram plot --
    sp.seismogram(AXIS, show=True)


if __name__ == "__main__":
    main()
