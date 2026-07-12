#! /usr/bin/python3
"""
Dataset loading orchestration for the sg2 + JSON sidecar input route.

This module is the first entry of the **dataset loading layer**
introduced by the SEG-2 / JSON sidecar input refactor (see
``private/docs/ai_refactor.md`` §3 and §4). Its sole responsibility
in this slice is to combine:

- :class:`~src.converter.seg2_reader.SEG2Reader` — pure SEG-2 binary
  reader (kept intentionally unaware of metadata)
- :func:`~src.converter.json_sidecar.resolve_sidecar_for_sg2` —
  same-directory JSON sidecar resolver (kept intentionally unaware
  of SEG-2 binary I/O)

into a single explicit orchestration entry point so callers do not
have to wire the two halves manually.

This module:

- never modifies the ``.sg2`` file (``SEG2Reader`` only opens it
  read-only with ``open(..., "rb")``)
- never modifies the Excel datasheet
- never modifies the JSON sidecar
- never reads, writes, or touches ``.npz`` files
- never imports or modifies ``SingleProcesser`` / ``GroupProcesser`` /
  plotting code

Public entry points (this slice):

- :func:`load_from_sg2_json`  — load an sg2 waveform together with
  its same-directory JSON sidecar; raise :class:`FileNotFoundError`
  if either is missing.
- :class:`Sg2WithSidecar`     — minimal :class:`typing.NamedTuple`
  return container.

Out of scope for this slice (deferred to later slices):

- ``load_from_npz(...)``
- Excel auto-generation fallback when the sidecar is missing
  (``json_sidecar.ensure_sidecar_from_excel`` remains a placeholder)
- dataclass / dataset-object normalization beyond a minimal
  ``NamedTuple``
- any bridge to ``SingleProcesser.__init__``
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np

from src.converter.datasheet_parser import (
    NormalizedMetadata,
    get_record,
    load_datasheet,
)
from src.converter.json_sidecar import (
    NumSensorMismatchError,
    ensure_sidecar_from_excel,
    get_sidecar_json_path,
    load_json_sidecar,
    resolve_sidecar_for_sg2,
)
from src.converter.npz_adapter import load_npz
from src.converter.seg2_reader import SEG2Reader


class FallbackMetadataWarning(UserWarning):
    """Emitted by :func:`load_dataset` when an ``.sg2`` input falls back
    from JSON sidecar to Excel datasheet to default metadata.

    Subclassing :class:`UserWarning` lets callers (including
    :mod:`src.processor.single_processor`, which globally silences
    ``UserWarning``) re-enable just this category via
    ``warnings.simplefilter("always", FallbackMetadataWarning)``.
    """


class Sg2WithSidecar(NamedTuple):
    """Minimal container returned by :func:`load_from_sg2_json`.

    Attributes
    ----------
    reader
        A :class:`SEG2Reader` opened on the raw ``.sg2`` file. The
        underlying file is opened read-only and is never modified.
    metadata
        The :class:`NormalizedMetadata` loaded from the
        same-directory JSON sidecar.
    sg2_path
        The resolved path of the ``.sg2`` file.
    json_path
        The resolved path of the JSON sidecar (always
        ``sg2_path.with_suffix(".json")``).
    """

    reader: SEG2Reader
    metadata: NormalizedMetadata
    sg2_path: Path
    json_path: Path


def load_from_sg2_json(sg2_path: Union[str, Path]) -> Sg2WithSidecar:
    """Load an ``.sg2`` waveform together with its JSON sidecar.

    The orchestration order is deliberately:

    1. Verify ``sg2_path`` exists on disk.
    2. Resolve and load the same-directory JSON sidecar via
       :func:`~src.converter.json_sidecar.resolve_sidecar_for_sg2`
       with ``missing_ok=False``. If the sidecar is missing, a
       :class:`FileNotFoundError` is raised **before** the binary
       file is opened.
    3. Only then construct :class:`SEG2Reader` on ``sg2_path``
       (read-only open).

    This fail-fast order ensures we never spend SEG-2 binary I/O on
    records that are missing their normalized metadata.

    Parameters
    ----------
    sg2_path
        Path of the raw ``.sg2`` file.

    Returns
    -------
    Sg2WithSidecar
        Named tuple containing the open :class:`SEG2Reader`, the
        :class:`NormalizedMetadata` loaded from the sidecar, and the
        resolved ``sg2_path`` / ``json_path``.

    Raises
    ------
    FileNotFoundError
        If ``sg2_path`` does not exist, or the same-directory JSON
        sidecar does not exist.
    ValueError
        Propagated from
        :func:`~src.converter.json_sidecar.resolve_sidecar_for_sg2`
        when the sidecar contents fail schema validation.

    Notes
    -----
    - This function never writes to ``sg2_path``; the file is opened
      read-only by :class:`SEG2Reader`.
    - This function never writes to ``json_path``.
    - Excel auto-generation when the sidecar is absent is **not**
      implemented in this slice. Callers that need that behavior
      must first invoke
      :func:`~src.converter.json_sidecar.refresh_sidecar_from_datasheet`
      or :func:`~src.converter.json_sidecar.save_sidecar`.
    """
    sg2_path = Path(sg2_path)
    if not sg2_path.exists():
        raise FileNotFoundError(
            f".sg2 file not found: {sg2_path}"
        )

    # Fail-fast on missing sidecar BEFORE touching the binary file.
    metadata = resolve_sidecar_for_sg2(sg2_path, missing_ok=False)
    json_path = get_sidecar_json_path(sg2_path)

    # SEG2Reader opens the file read-only; the .sg2 is never modified.
    reader = SEG2Reader(str(sg2_path))

    return Sg2WithSidecar(
        reader=reader,
        metadata=metadata,
        sg2_path=sg2_path,
        json_path=json_path,
    )


def load_dataset(
    file_path: Union[str, Path],
    *,
    datasheet_path: Union[str, Path, None] = None,
    obs_id: Union[int, str, None] = None,
) -> Dict[str, Any]:
    """Unified loader returning an npz-shaped ``dict`` for SingleProcesser.

    Dispatch is by file suffix:

    - ``.npz`` -> delegates to :func:`load_npz` unchanged.
    - ``.sg2`` -> resolves metadata via the json -> Excel -> default
      fallback chain and assembles a dict whose keys match the npz
      schema consumed by ``SingleProcesser._input_*`` (``y``, ``fs``,
    - ``.dat`` / ``.DAT`` -> same SEG-2 code path as ``.sg2``; any
      SEG2Reader parse error is wrapped with a descriptive message.
      ``Num_sensor``, ``interval``, ``source_x``, ``sensor1_x``,
      ``distance``, ``x_distance``, ``z_distance``, ``unit``, ``shot``,
      ``condition``), plus the provenance keys ``metadata_source`` and
      ``metadata_is_fallback``.

    Parameters
    ----------
    file_path
        Path of the input file. Suffix is matched case-insensitively.
    datasheet_path, obs_id
        Optional Excel fallback inputs. Both must be supplied for the
        Excel branch to be attempted; ``obs_id`` is **never** inferred
        from the filename. If either is missing, the loader skips
        Excel and falls through to default metadata when the JSON
        sidecar is absent.

    Returns
    -------
    dict[str, Any]
        Mapping consumable by ``SingleProcesser._input_*``.

    Raises
    ------
    FileNotFoundError
        If ``file_path`` does not exist.
    ValueError
        If the extension is unsupported, or if a JSON sidecar exists
        but is structurally invalid (corrupted sidecars are NOT
        silently demoted to the Excel branch).
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")

    suffix = p.suffix.lower()
    if suffix == ".npz":
        return load_npz(p)
    if suffix == ".sg2":
        return _load_sg2_as_dict(
            p, datasheet_path=datasheet_path, obs_id=obs_id
        )
    if suffix == ".dat":
        try:
            return _load_sg2_as_dict(
                p, datasheet_path=datasheet_path, obs_id=obs_id
            )
        except Exception as exc:
            raise ValueError(
                f"File {p} has .DAT extension but could not be"
                f" parsed as SEG-2: {exc}"
            ) from exc
    raise ValueError(
        f"Unsupported input extension {p.suffix!r} for {p}; "
        f"expected '.npz', '.sg2', or '.dat'."
    )


def _load_sg2_as_dict(
    sg2_path: Path,
    *,
    datasheet_path: Union[str, Path, None],
    obs_id: Union[int, str, None],
) -> Dict[str, Any]:
    """Read raw ``.sg2`` + resolve metadata via json -> Excel -> default.

    The raw ``.sg2`` file is opened read-only via :class:`SEG2Reader`
    and is never modified. The Excel workbook is opened read-only via
    :func:`load_datasheet` and is never modified.
    """
    # SEG2Reader opens the file with open(..., "rb") only.
    reader = SEG2Reader(str(sg2_path))
    waveform = np.asarray(reader.get_all_numpy_array()).T  # (n_ch, n_samples)
    fs = reader.get_frequency()
    n_sensor = reader.get_max_ch()

    # 1) JSON sidecar (same directory, same stem).
    json_path = get_sidecar_json_path(sg2_path)

    if json_path.exists():
        if obs_id is not None:
            warnings.warn(
                f"JSON sidecar already exists for '{sg2_path}' "
                f"(at '{json_path}'); ignoring obs_id={obs_id!r}. "
                f"Use refresh_sidecar_from_datasheet(...) to regenerate.",
                FallbackMetadataWarning,
                stacklevel=3,
            )
    elif datasheet_path is not None and obs_id is not None:
        # Auto-create the sidecar from the Excel datasheet before
        # falling through to the JSON-load branch. The SEG-2 header's
        # max_ch is passed so a mismatched pairing fails BEFORE any
        # file is written.
        try:
            ensure_sidecar_from_excel(
                sg2_path,
                datasheet_path=datasheet_path,
                obs_id=obs_id,
                expected_num_sensor=n_sensor,
            )
        except (FileNotFoundError, ValueError, FileExistsError) as exc:
            warnings.warn(
                f"JSON auto-creation from Excel failed for '{sg2_path}' "
                f"(datasheet='{datasheet_path}', obs_id={obs_id!r}): "
                f"{exc}. Falling back to in-memory Excel metadata.",
                FallbackMetadataWarning,
                stacklevel=3,
            )

    if json_path.exists():
        # A structurally invalid sidecar must NOT silently demote to
        # Excel/default: re-raise so the user fixes the JSON instead of
        # losing real metadata to a typo.
        md = load_json_sidecar(json_path)
        _check_num_sensor_match(md.num_sensor, n_sensor, sg2_path, json_path)
        return _metadata_to_dict(md, waveform, fs, source="json")

    warnings.warn(
        f"Sidecar JSON not found for '{sg2_path}' "
        f"(looked at '{json_path}'); trying Excel fallback.",
        FallbackMetadataWarning,
        stacklevel=3,
    )

    # 2) Excel datasheet. Per design, obs_id is never auto-inferred
    # from the filename; both inputs must be passed explicitly.
    if datasheet_path is not None and obs_id is not None:
        try:
            index = load_datasheet(datasheet_path)
            md = get_record(index, obs_id)
            _check_num_sensor_match(
                md.num_sensor, n_sensor, sg2_path, datasheet_path
            )
            return _metadata_to_dict(md, waveform, fs, source="excel")
        except (FileNotFoundError, ValueError) as exc:
            warnings.warn(
                f"Excel fallback failed for '{sg2_path}' "
                f"(datasheet='{datasheet_path}', obs_id={obs_id!r}): "
                f"{exc}",
                FallbackMetadataWarning,
                stacklevel=3,
            )

    # 3) Default metadata.
    warnings.warn(
        f"Proceeding with default metadata for '{sg2_path}'. "
        f"fs ({fs}) and Num_sensor ({n_sensor}) come from the sg2 "
        f"file; source_x, distance, sensor1_x, x_distance, z_distance "
        f"are placeholder values. Check metadata_is_fallback before "
        f"running geometry-sensitive analyses.",
        FallbackMetadataWarning,
        stacklevel=3,
    )
    return _default_dict(waveform, fs, n_sensor)


def _check_num_sensor_match(
    md_num_sensor: int,
    sg2_max_ch: int,
    sg2_path: Path,
    source: Union[Path, str],
) -> None:
    """Raise :class:`NumSensorMismatchError` if the SEG-2 file has fewer
    channels than the datasheet/JSON declares.

    Acquisition data loggers commonly append trigger / auxiliary
    channels after the real sensor channels (e.g. GEO_1021.DAT exposes
    98 channels = 96 sensors + 2 trigger channels). The chs_range
    fields in the datasheet (``X方向のch`` / ``Y方向のch`` / ``Z方向のch``)
    pick out the real-sensor indices, so ``sg2_max_ch > md_num_sensor``
    is expected and accepted silently. The strict failure case is
    ``sg2_max_ch < md_num_sensor``: the SEG-2 file cannot supply the
    channels the datasheet promises, which indicates the wrong
    ``obs_id`` was paired with the file.
    """
    if md_num_sensor > sg2_max_ch:
        raise NumSensorMismatchError(
            f"num_sensor mismatch for '{sg2_path}': SEG-2 header reports "
            f"only {sg2_max_ch} channels but '{source}' declares "
            f"num_sensor={md_num_sensor}. The SEG-2 file does not "
            f"contain enough channels to satisfy the datasheet; fix the "
            f"datasheet/sidecar or pair the SEG-2 file with the correct "
            f"obs_id."
        )


def _parse_chs_range(
    spec: Any, total_channels: int
) -> Optional[Tuple[int, ...]]:
    """Parse a channel-range spec into 0-based, ordered, unique indices.

    Accepted spec forms (1-based on input to match the spreadsheet UX):

    - ``None`` or empty string -> returns ``None`` (axis absent).
    - ASCII range string such as ``"1-96"`` or ``"1-48,73-96"``. Only
      ``,`` and ``-`` are allowed as separators; whitespace around
      segments is tolerated.
    - ``list`` or ``tuple`` of integers (1-based).
    - A single integer (1-based).

    Anything else (including unicode dashes, decimals, or arbitrary
    objects) raises :class:`ValueError`.

    ``total_channels`` is the upper bound of the valid channel index
    range. For multi-component surveys it is ``num_sensor *
    n_components`` (e.g. ``Z方向のch="25-48"`` is valid when
    ``total_channels`` is 48), so it is **not** the per-component
    ``num_sensor``.

    Validation:
    - segments with ``lo > hi`` -> ``ValueError``
    - duplicates across segments -> ``ValueError``
    - indices outside ``[1, total_channels]`` -> ``ValueError``

    Returns a ``tuple[int, ...]`` of 0-based indices in the order the
    spec declared them, or ``None`` when the spec is empty / absent.
    """
    if spec is None:
        return None

    out: List[int]
    if isinstance(spec, str):
        s = spec.strip()
        if not s:
            return None
        out = []
        for seg in s.split(","):
            seg = seg.strip()
            if not seg:
                continue
            if "-" in seg:
                a, _, b = seg.partition("-")
                try:
                    lo, hi = int(a.strip()), int(b.strip())
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported chs_range segment {seg!r} in "
                        f"{spec!r}: {exc}"
                    ) from exc
                if lo > hi:
                    raise ValueError(
                        f"Invalid chs_range segment {seg!r} in "
                        f"{spec!r}: low ({lo}) > high ({hi})."
                    )
                out.extend(range(lo, hi + 1))
            else:
                try:
                    out.append(int(seg))
                except ValueError as exc:
                    raise ValueError(
                        f"Unsupported chs_range segment {seg!r} in "
                        f"{spec!r}: {exc}"
                    ) from exc
    elif isinstance(spec, bool):
        # bool is a subclass of int; reject so 'True' does not silently
        # become channel 1.
        raise ValueError(
            f"Unsupported chs_range spec {spec!r}: bool is not allowed."
        )
    elif isinstance(spec, int):
        out = [spec]
    elif isinstance(spec, (list, tuple)):
        out = []
        for v in spec:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(
                    f"Unsupported chs_range element {v!r} in {spec!r}: "
                    f"expected int."
                )
            out.append(v)
    else:
        raise ValueError(
            f"Unsupported chs_range spec type {type(spec).__name__}: "
            f"{spec!r}."
        )

    if not out:
        return None
    if len(set(out)) != len(out):
        raise ValueError(
            f"Duplicate channel indices in chs_range {spec!r}: {out}."
        )
    for v in out:
        if v < 1 or v > total_channels:
            raise ValueError(
                f"Channel index {v} out of range [1, {total_channels}] "
                f"in chs_range {spec!r}."
            )
    return tuple(v - 1 for v in out)


def _split_waveform(
    waveform: "np.ndarray",
    x_idx: Optional[Tuple[int, ...]],
    y_idx: Optional[Tuple[int, ...]],
    z_idx: Optional[Tuple[int, ...]],
) -> Tuple[Optional["np.ndarray"], Optional["np.ndarray"], Optional["np.ndarray"]]:
    """Slice ``waveform`` (shape ``(n_ch, n_samples)``) into x / y / z.

    Each returned array is a freshly-allocated copy so callers cannot
    accidentally mutate the SEG2Reader-owned underlying buffer. ``None``
    indices map to ``None`` outputs.
    """

    def _take(idx: Optional[Tuple[int, ...]]) -> Optional["np.ndarray"]:
        if idx is None:
            return None
        return waveform[list(idx), :].copy()

    return _take(x_idx), _take(y_idx), _take(z_idx)


def _metadata_to_dict(
    md: NormalizedMetadata,
    waveform: "np.ndarray",
    fs: int,
    *,
    source: str,
) -> Dict[str, Any]:
    """Pack ``NormalizedMetadata`` + waveform into the npz-shaped dict.

    The waveform is split into ``x`` / ``y`` / ``z`` per the
    ``{x,y,z}_chs_range`` fields of ``md``. Axes whose chs_range is
    absent or empty are **not** added to the returned dict so the
    existing ``SingleProcesser._input_axis`` (``if "x" in file:``) gate
    skips them without modification.

    The chs_range fields and the per-channel arrays (``distance`` /
    ``x_distance`` / ``z_distance``) are indexed in **total-channel**
    space (``num_sensor * n_components``). The per-channel arrays are
    sliced down to the **primary axis** (``y`` if present, else ``x``,
    else ``z``) so that ``distance`` / ``x_distance`` / ``z_distance``
    line up 1:1 with the primary waveform array. For single-component
    sheets ``total_channels == num_sensor`` and the slice is an
    identity reindex, so behavior is unchanged.

    ``Num_sensor`` is set to the **per-axis** channel count of the
    primary axis (``y`` if present, else ``x``, else ``z``, else the
    total channel count as a last-resort fallback). Per-axis
    semantics match how downstream filter / analysis code addresses
    each axis with its own array.
    """
    # Per-channel arrays are indexed in total-channel space
    # (num_sensor * n_components). Their length is the authoritative
    # total-channel count used to bound the chs_range parsing and to
    # slice the waveform. Falls back to num_sensor for records without
    # per-channel arrays (single-component legacy shape).
    z_full = np.asarray(md.z_distance, dtype=float)
    x_full = np.asarray(md.x_distance, dtype=float)
    dist_full = (
        np.asarray(md.distance, dtype=float)
        if md.distance is not None
        else None
    )
    total_channels = (
        int(z_full.shape[0]) if z_full.shape[0] > 0 else int(md.num_sensor)
    )

    xi = _parse_chs_range(md.x_chs_range, total_channels)
    yi = _parse_chs_range(md.y_chs_range, total_channels)
    zi = _parse_chs_range(md.z_chs_range, total_channels)

    if total_channels > waveform.shape[0]:
        raise NumSensorMismatchError(
            f"Datasheet declares {total_channels} total channels "
            f"(num_sensor={md.num_sensor}) but the SEG-2 file provides "
            f"only {waveform.shape[0]} channels; cannot slice the "
            f"requested channel ranges."
        )

    x_arr, y_arr, z_arr = _split_waveform(waveform, xi, yi, zi)

    # Primary axis drives the per-axis channel count and the slicing of
    # the total-channel per-channel arrays.
    if y_arr is not None:
        per_axis_n = int(y_arr.shape[0])
        primary_idx = yi
    elif x_arr is not None:
        per_axis_n = int(x_arr.shape[0])
        primary_idx = xi
    elif z_arr is not None:
        per_axis_n = int(z_arr.shape[0])
        primary_idx = zi
    else:
        per_axis_n = int(total_channels)
        primary_idx = None

    def _to_primary_axis(arr: "np.ndarray") -> "np.ndarray":
        # Reduce a total-channel array to the primary axis so it aligns
        # 1:1 with the primary waveform array. Single-component records
        # (primary_idx spanning the whole array) reindex to an identical
        # array, preserving legacy behavior.
        if primary_idx is None or arr.shape[0] == 0:
            return arr
        return arr[list(primary_idx)]

    # ``md.excdistance`` is the scalar source-to-sensor1 offset and is
    # preserved separately as ``source_x`` / ``sensor1_x``. When the
    # workbook's optional 'distance' sheet supplied per-channel values
    # for this obs_id, they win over the derivation (sliced to the
    # primary axis); otherwise the canonical sensor1_x + arange*interval
    # construction is used unchanged.
    if dist_full is not None:
        distance = _to_primary_axis(dist_full)
    else:
        distance = (
            float(md.sensor1_x)
            + np.arange(per_axis_n, dtype=float) * float(md.interval)
        )
    out: Dict[str, Any] = {
        "fs": fs,
        "interval": md.interval,
        "source_x": md.source_x,
        "sensor1_x": md.sensor1_x,
        "Num_sensor": per_axis_n,
        "distance": distance,
        "x_distance": _to_primary_axis(x_full),
        "z_distance": _to_primary_axis(z_full),
        "unit": "v",
        "shot": md.shot_method if md.shot_method is not None else "",
        "condition": md.condition if md.condition is not None else "",
        "metadata_source": source,
        "metadata_is_fallback": False,
    }
    if x_arr is not None:
        out["x"] = x_arr
    if y_arr is not None:
        out["y"] = y_arr
    if z_arr is not None:
        out["z"] = z_arr
    return out


def _default_dict(
    waveform: "np.ndarray", fs: int, n_sensor: int
) -> Dict[str, Any]:
    """Default-metadata dict used when both JSON and Excel are unavailable."""
    interval = 1.0
    return {
        "y": waveform,
        "fs": fs,
        "interval": interval,
        "source_x": 0.0,
        "sensor1_x": 0.0,
        "Num_sensor": n_sensor,
        "distance": np.arange(n_sensor, dtype=float) * interval,
        "x_distance": np.arange(n_sensor, dtype=float) * interval,
        "z_distance": np.zeros(n_sensor, dtype=float),
        "unit": "v",
        "shot": "",
        "condition": "fallback_default",
        "metadata_source": "fallback_default",
        "metadata_is_fallback": True,
    }


__all__ = [
    "FallbackMetadataWarning",
    "Sg2WithSidecar",
    "load_dataset",
    "load_from_sg2_json",
]
