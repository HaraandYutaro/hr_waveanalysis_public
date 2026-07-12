"""Quickstart: .DAT (SEG-2) + Excel datasheet → auto-created JSON sidecar.

End-to-end example for the datasheet-paired loading path:

- ``testcodes/data/amenomiya20260629/dats/GEO_2532.DAT`` is interpreted
  as SEG-2 (.DAT and .sg2 share the SEG-2 format).
- ``testcodes/data/amenomiya20260629/datasheet/datasheet_amenomiya.xlsx``
  supplies normalized metadata, looked up by ``obs_id="2532"`` (the
  digits of the .DAT filename — row 3 in Sheet1).
- On first run, no JSON sidecar exists next to the .DAT, so the loader
  auto-creates ``.../dats/GEO_2532.json`` from the datasheet.
  Subsequent runs read the existing JSON directly.
- This is a **2-component** survey: 24 sensors record both a Y and a Z
  component, so the datasheet declares ``センサー数=24`` with channel
  ranges ``Y方向のch="1-24"`` and ``Z方向のch="25-48"`` (48 channels
  total). The .DAT logger appends 2 trigger channels after the 48 real
  channels, so the SEG-2 header reports 50 channels. The loader slices
  the real-sensor block into ``sp.y`` (channels 1-24) and ``sp.z``
  (channels 25-48); the 2 trigger channels are silently dropped.

Immutability:
- The raw ``.DAT`` file is opened read-only by SEG2Reader and is never
  modified.
- The Excel datasheet is opened with ``read_only=True`` and is never
  modified.
- Only the JSON sidecar at ``GEO_2532.json`` is written, and only when
  it does not already exist.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.processor.single_processor import SingleProcesser

OBS_ID = "1021"
AXIS = "y"
BACKEND = "mpl"

DAT_PATH = Path(str(PROJECT_ROOT) + f'/sample_data/sg2/GEO_{OBS_ID}.DAT')
DATASHEET_PATH = Path(str(PROJECT_ROOT) + f'/sample_data/datasheet/datasheet_example.xlsx')


def main():
    # -- load data (.DAT + datasheet, auto-creates JSON sidecar) --
    sp = SingleProcesser(
        str(DAT_PATH),
        BACKEND,
        datasheet_path=str(DATASHEET_PATH),
        obs_id=OBS_ID,
    )
    json_sidecar = DAT_PATH.with_suffix(".json")

    # -- inspect metadata and loaded components --
    print(f"input file          : {DAT_PATH}")
    print(f"datasheet           : {DATASHEET_PATH}")
    print(f"obs_id              : {OBS_ID}")
    print(f"json sidecar exists : {json_sidecar.exists()} ({json_sidecar})")
    print(f"metadata_source     : {sp.metadata_source}")
    print(f"metadata_is_fallback: {sp.metadata_is_fallback}")
    print(f"Num_sensor          : {sp.Num_sensor}")

    for name in ("x", "y", "z"):
        arr = getattr(sp, name, None)
        shape = arr.shape if arr is not None else None
        print(f"  {name} shape: {shape}")

    # -- preprocessing --
    sp.sort_distance()

    # -- seismogram plot --
    sp.seismogram(AXIS, show=True, agc=True)


if __name__ == "__main__":
    main()
