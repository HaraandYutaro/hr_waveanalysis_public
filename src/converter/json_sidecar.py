#! /usr/bin/python3
"""
JSON sidecar I/O for normalized survey metadata.

This module is the dedicated **JSON sidecar layer** introduced as
Slice 3 of the SEG-2 / JSON sidecar input refactor (see
``private/docs/ai_refactor.md`` §3 and §8). Its sole responsibility is
to turn a :class:`~src.converter.datasheet_parser.NormalizedMetadata`
into a human-readable JSON file on disk and back, with explicit schema
versioning, atomic writes, and timestamped backups before overwrite.

This module:

- never modifies the Excel workbook (only reads via
  :func:`~src.converter.datasheet_parser.load_datasheet`)
- never reads or writes ``.sg2`` files
- never touches NPZ files
- never imports or modifies ``SingleProcesser`` or any plotting code

Public entry points:

- :func:`save_sidecar`                  — write a sidecar JSON file
- :func:`load_sidecar`                  — read a sidecar JSON file back
  into a :class:`NormalizedMetadata`
- :func:`refresh_sidecar_from_datasheet` — re-derive the sidecar JSON
  from the canonical Excel datasheet for one observation id
- :func:`get_sidecar_json_path`         — compute the canonical
  JSON-sidecar path next to a given ``.sg2`` file (pure path operation;
  the ``.sg2`` file itself is never opened)
- :func:`load_json_sidecar`             — thin alias of
  :func:`load_sidecar` provided as the canonical entry point for
  sg2-side callers
- :func:`resolve_sidecar_for_sg2`       — locate and load the JSON
  sidecar adjacent to an ``.sg2`` file, with explicit missing-file
  branch (``FileNotFoundError`` or ``UserWarning + None``)
- :func:`ensure_sidecar_from_excel`     — create the JSON sidecar from
  the Excel datasheet on first load when it does not yet exist
  (idempotent; no-op if the sidecar is already present)

JSON layout (schema version ``"1.0"``):

.. code-block:: json

    {
      "_schema_version": "1.0",
      "_datasheet_path": "sample_data/datasheet/datasheet_example.xlsx",
      "_obs_id": "891",
      "metadata": {
        "obs_id": 891,
        "shot_method": "...",
        "condition": "...",
        "excdistance": -0.075,
        "source_x": -0.075,
        "sensor1_x": 0.0,
        "interval": 0.15,
        "num_sensor": 96,
        "x_chs_range": null,
        "y_chs_range": "1-96",
        "z_chs_range": null,
        "x_reverse": null,
        "y_reverse": null,
        "z_reverse": null,
        "z_distance": [24.5308, 24.5477, ...],
        "x_distance": [0.0, 0.149, ...]
      }
    }

Field names inside ``"metadata"`` mirror the attribute names of
:class:`NormalizedMetadata` exactly. ``z_distance`` and ``x_distance``
are JSON arrays on disk and are restored to ``tuple[float, ...]`` on
load so the round trip yields a value-equal :class:`NormalizedMetadata`.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import warnings
from dataclasses import fields
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Union

from src.converter.datasheet_parser import (
    NormalizedMetadata,
    get_record,
    load_datasheet,
)


# ---------------------------------------------------------------------------
# Schema constants (frozen for this phase).
# ---------------------------------------------------------------------------

class NumSensorMismatchError(Exception):
    """Raised when the datasheet's ``num_sensor`` disagrees with the
    SEG-2 header's ``max_ch``.

    Intentionally **not** a :class:`ValueError` subclass: the dataset
    loader catches ``ValueError`` to demote recoverable parse failures
    to ``FallbackMetadataWarning``, but a num_sensor mismatch is a
    configuration error that must surface to the caller instead of
    being silently demoted to default metadata.
    """


_SCHEMA_VERSION_SUPPORTED: str = "1.0"

_KEY_SCHEMA_VERSION: str = "_schema_version"
_KEY_DATASHEET_PATH: str = "_datasheet_path"
_KEY_OBS_ID: str = "_obs_id"
_KEY_METADATA: str = "metadata"

_REQUIRED_TOP_LEVEL_KEYS = (
    _KEY_SCHEMA_VERSION,
    _KEY_DATASHEET_PATH,
    _KEY_OBS_ID,
    _KEY_METADATA,
)

# Fields on NormalizedMetadata that need list <-> tuple conversion.
_TUPLE_FIELDS = ("z_distance", "x_distance")

# Optional tuple fields: same list<->tuple conversion as ``_TUPLE_FIELDS``
# but the JSON key may be missing (older sidecars) or ``null``. Schema
# stays at ``"1.0"`` because these are strictly additive — older readers
# that don't know the key fall back to their existing derivation, and
# newer readers tolerate sidecars that pre-date this field.
_OPTIONAL_TUPLE_FIELDS = ("distance",)

# Fields that must round-trip as float.
_FLOAT_FIELDS = ("source_x", "sensor1_x", "interval")

# Fields that must round-trip as int.
_INT_FIELDS = ("num_sensor",)


# ---------------------------------------------------------------------------
# Public API.
# ---------------------------------------------------------------------------


def save_sidecar(
    path: Union[str, Path],
    metadata: NormalizedMetadata,
    *,
    datasheet_path: Union[str, Path, None] = None,
    schema_version: str = _SCHEMA_VERSION_SUPPORTED,
    overwrite: bool = False,
    backup: bool = True,
) -> None:
    """Write ``metadata`` to ``path`` as a JSON sidecar.

    Parameters
    ----------
    path
        Destination JSON file. Parent directory is created if missing.
    metadata
        The :class:`NormalizedMetadata` to serialize.
    datasheet_path
        Optional source Excel path recorded in the JSON top-level
        ``_datasheet_path`` field for auditability. Stored as ``str``.
    schema_version
        Top-level ``_schema_version`` value. Currently must equal
        ``"1.0"``; any other value is rejected.
    overwrite
        If ``False`` (default) and ``path`` already exists, raise
        :class:`FileExistsError`. If ``True``, the existing file is
        replaced atomically.
    backup
        When ``overwrite=True`` and ``path`` exists, copy the existing
        file to ``<path>.YYYYMMDDHHMMSS.bak`` (with a numeric suffix
        on same-second collision) before replacing it.

    Notes
    -----
    The new content is first written to a sibling temporary file in
    the same directory, then renamed via :func:`os.replace`, so a
    crash mid-write cannot corrupt an existing sidecar.
    """
    if not isinstance(metadata, NormalizedMetadata):
        raise TypeError(
            f"metadata must be a NormalizedMetadata instance, got "
            f"{type(metadata).__name__}."
        )
    if schema_version != _SCHEMA_VERSION_SUPPORTED:
        raise ValueError(
            f"schema_version must be {_SCHEMA_VERSION_SUPPORTED!r} for "
            f"this writer, got {schema_version!r}."
        )

    path = Path(path)

    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Sidecar already exists: {path}. Pass overwrite=True to "
            f"replace it."
        )

    parent = path.parent
    if str(parent) and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)

    if path.exists() and overwrite and backup:
        _backup_existing(path)

    payload = {
        _KEY_SCHEMA_VERSION: schema_version,
        _KEY_DATASHEET_PATH: (
            str(datasheet_path) if datasheet_path is not None else None
        ),
        _KEY_OBS_ID: str(metadata.obs_id),
        _KEY_METADATA: _metadata_to_dict(metadata),
    }

    _atomic_write_json(path, payload)


def load_sidecar(path: Union[str, Path]) -> NormalizedMetadata:
    """Read a sidecar JSON file and return the :class:`NormalizedMetadata`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file is not valid JSON, the top-level structure is
        wrong, the schema version is not supported, or any required
        ``metadata`` field is missing or has the wrong type.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sidecar not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        try:
            payload = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Sidecar '{path}' is not valid JSON ({exc})."
            ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            f"Sidecar '{path}' must contain a JSON object at the top "
            f"level, got {type(payload).__name__}."
        )

    missing_top = [k for k in _REQUIRED_TOP_LEVEL_KEYS if k not in payload]
    if missing_top:
        raise ValueError(
            f"Sidecar '{path}' is missing required top-level key(s): "
            f"{missing_top}."
        )

    schema_version = payload[_KEY_SCHEMA_VERSION]
    if schema_version != _SCHEMA_VERSION_SUPPORTED:
        raise ValueError(
            f"Sidecar '{path}' has unsupported schema version "
            f"{schema_version!r}; this loader only supports "
            f"{_SCHEMA_VERSION_SUPPORTED!r}."
        )

    md = payload[_KEY_METADATA]
    if not isinstance(md, dict):
        raise ValueError(
            f"Sidecar '{path}' top-level '{_KEY_METADATA}' must be a "
            f"JSON object, got {type(md).__name__}."
        )

    return _dict_to_metadata(md, source_for_errors=path)


def refresh_sidecar_from_datasheet(
    json_path: Union[str, Path],
    datasheet_path: Union[str, Path],
    obs_id: Union[int, str],
) -> None:
    """Re-derive the sidecar JSON from the canonical Excel datasheet.

    Reads ``datasheet_path`` via :func:`load_datasheet`, looks up
    ``obs_id`` via :func:`get_record`, and writes the result to
    ``json_path`` with ``overwrite=True, backup=True`` so the previous
    sidecar (if any) is preserved as a timestamped backup.

    Neither the Excel workbook nor the existing JSON content is
    mutated in place; the JSON is regenerated and atomically swapped
    in.

    Raises
    ------
    ValueError
        If the observation id is not present in the datasheet, or any
        downstream validation in :func:`get_record` fails. The
        original error is re-raised with sidecar-context attached.
    """
    json_path = Path(json_path)
    datasheet_path = Path(datasheet_path)

    index = load_datasheet(datasheet_path)
    try:
        metadata = get_record(index, obs_id)
    except ValueError as exc:
        raise ValueError(
            f"Cannot refresh sidecar '{json_path}' from datasheet "
            f"'{datasheet_path}' for observation id {obs_id!r}: {exc}"
        ) from exc

    save_sidecar(
        json_path,
        metadata,
        datasheet_path=datasheet_path,
        overwrite=True,
        backup=True,
    )


def get_sidecar_json_path(sg2_path: Union[str, Path]) -> Path:
    """Return the canonical JSON sidecar path next to an ``.sg2`` file.

    The sidecar lives in the same directory as the ``.sg2`` file, with
    the same stem and a ``.json`` suffix. This helper is a **pure path
    operation**: the ``.sg2`` file itself is never opened, read, or
    modified.

    Parameters
    ----------
    sg2_path
        Path of the raw ``.sg2`` file.

    Returns
    -------
    pathlib.Path
        ``sg2_path`` with its suffix replaced by ``.json``.

    Examples
    --------
    >>> get_sidecar_json_path("/data/record001.sg2")
    PosixPath('/data/record001.json')
    """
    return Path(sg2_path).with_suffix(".json")


def load_json_sidecar(json_path: Union[str, Path]) -> NormalizedMetadata:
    """Minimal sidecar JSON loader for sg2-side callers.

    Thin wrapper around :func:`load_sidecar` provided as the canonical
    entry point invoked from sg2-loading code paths. The return value
    and the validation semantics are identical to :func:`load_sidecar`:

    - returns a :class:`NormalizedMetadata`
    - raises :class:`FileNotFoundError` if ``json_path`` does not exist
    - raises :class:`ValueError` for schema / structure violations

    Parameters
    ----------
    json_path
        Path of the JSON sidecar file to read.
    """
    return load_sidecar(json_path)


def resolve_sidecar_for_sg2(
    sg2_path: Union[str, Path],
    *,
    missing_ok: bool = False,
) -> Union[NormalizedMetadata, None]:
    """Locate and load the JSON sidecar adjacent to an ``.sg2`` file.

    The sidecar path is derived purely from ``sg2_path`` via
    :func:`get_sidecar_json_path` — the ``.sg2`` file is **never**
    opened, read, or written by this helper. Callers that also need
    the waveform must invoke :class:`~src.converter.seg2_reader.SEG2Reader`
    separately; this function only resolves the metadata sidecar.

    Parameters
    ----------
    sg2_path
        Path of the raw ``.sg2`` file. Only its parent directory and
        stem are consulted to compute the sidecar location.
    missing_ok
        If ``False`` (default), raise :class:`FileNotFoundError` when
        the sidecar does not exist. If ``True``, emit a
        :class:`UserWarning` and return ``None`` so callers can fall
        back to alternate metadata sources (e.g. a future Excel hook).

    Returns
    -------
    NormalizedMetadata or None
        The loaded metadata, or ``None`` when ``missing_ok=True`` and
        the sidecar is absent.

    Raises
    ------
    FileNotFoundError
        When the sidecar does not exist and ``missing_ok=False``.
    ValueError
        Propagated from :func:`load_json_sidecar` on schema violations.
    """
    json_path = get_sidecar_json_path(sg2_path)
    if json_path.exists():
        return load_json_sidecar(json_path)
    if missing_ok:
        warnings.warn(
            f"JSON sidecar not found next to '{sg2_path}' "
            f"(expected at '{json_path}'); proceeding without normalized "
            f"metadata. Generate it via refresh_sidecar_from_datasheet(...) "
            f"or save_sidecar(...).",
            UserWarning,
            stacklevel=2,
        )
        return None
    raise FileNotFoundError(
        f"JSON sidecar not found next to '{sg2_path}' "
        f"(expected at '{json_path}'). Pass missing_ok=True to fall back "
        f"to None, or generate the sidecar via "
        f"refresh_sidecar_from_datasheet(...) / save_sidecar(...)."
    )


def ensure_sidecar_from_excel(
    sg2_path: Union[str, Path],
    *,
    datasheet_path: Union[str, Path, None] = None,
    obs_id: Union[int, str, None] = None,
    expected_num_sensor: Union[int, None] = None,
) -> Path:
    """Create the JSON sidecar from the Excel datasheet on first load.

    The ``.sg2`` file is never opened, read, or written here; only its
    parent directory and stem are consulted (via
    :func:`get_sidecar_json_path`) to compute where the new sidecar
    should live. The Excel workbook is opened read-only.

    Behavior:

    1. If the sidecar already exists at the canonical path, return it
       immediately without touching the file (idempotent re-entry).
    2. Otherwise, ``datasheet_path`` and ``obs_id`` must both be
       supplied; load the datasheet, look up the record, optionally
       cross-check ``expected_num_sensor`` against the datasheet's
       ``センサー数``, then atomically write the sidecar with
       ``overwrite=False`` so a concurrent writer cannot be clobbered.

    Parameters
    ----------
    sg2_path
        Path of the raw ``.sg2`` file. Used only to compute the
        sidecar path.
    datasheet_path
        Path of the Excel datasheet to derive the sidecar from.
        Required when the sidecar does not yet exist.
    obs_id
        Observation id within the datasheet. Required when the
        sidecar does not yet exist.
    expected_num_sensor
        Optional channel count from the SEG-2 header. When supplied
        and not equal to the datasheet's ``num_sensor``, raises
        :class:`ValueError` **before** writing, so a bad pairing
        cannot leave a corrupt sidecar on disk.

    Returns
    -------
    pathlib.Path
        The canonical sidecar path (existing or newly written).

    Raises
    ------
    ValueError
        If ``datasheet_path`` / ``obs_id`` is missing when the sidecar
        does not exist, if the datasheet record is missing or invalid,
        or if ``expected_num_sensor`` disagrees with the datasheet.
    FileNotFoundError
        If ``datasheet_path`` does not exist on disk.
    """
    json_path = get_sidecar_json_path(sg2_path)
    if json_path.exists():
        return json_path

    if datasheet_path is None or obs_id is None:
        raise ValueError(
            "ensure_sidecar_from_excel requires both datasheet_path and "
            "obs_id when the JSON sidecar does not yet exist "
            f"(expected at '{json_path}')."
        )

    index = load_datasheet(datasheet_path)
    metadata = get_record(index, obs_id)

    # The per-channel arrays are indexed in total-channel space
    # (num_sensor * n_components), so the SEG-2 file must supply at
    # least that many channels. Compare the total-channel count — not
    # the per-component num_sensor — against the SEG-2 header.
    total_channels = (
        len(metadata.z_distance)
        if len(metadata.z_distance) > 0
        else metadata.num_sensor
    )
    if (
        expected_num_sensor is not None
        and total_channels > expected_num_sensor
    ):
        # SEG-2 max_ch > total channels is allowed (extra trigger / aux
        # channels are dropped by the chs_range slicing later). Only
        # refuse when the SEG-2 cannot supply enough channels.
        raise NumSensorMismatchError(
            f"channel-count mismatch for '{sg2_path}': SEG-2 header "
            f"reports only {expected_num_sensor} channels but datasheet "
            f"'{datasheet_path}' obs_id={obs_id!r} declares "
            f"{total_channels} total channels "
            f"(num_sensor={metadata.num_sensor}). Refusing to write a "
            f"sidecar that promises more channels than the SEG-2 file "
            f"contains."
        )

    save_sidecar(
        json_path,
        metadata,
        datasheet_path=datasheet_path,
        overwrite=False,
        backup=False,
    )
    return json_path


# ---------------------------------------------------------------------------
# Internal helpers.
# ---------------------------------------------------------------------------


def _metadata_to_dict(metadata: NormalizedMetadata) -> Mapping[str, Any]:
    """Convert a :class:`NormalizedMetadata` to a JSON-ready dict.

    Tuple-typed fields are converted to ``list`` so the JSON output is
    a plain JSON array. All other fields are passed through unchanged
    (``json.dump`` handles ``int``, ``float``, ``str``, ``None``).
    """
    out: dict = {}
    for f in fields(metadata):
        v = getattr(metadata, f.name)
        if f.name in _TUPLE_FIELDS:
            out[f.name] = list(v)
        elif f.name in _OPTIONAL_TUPLE_FIELDS:
            out[f.name] = list(v) if v is not None else None
        else:
            out[f.name] = v
    return out


def _dict_to_metadata(
    md: Mapping[str, Any], *, source_for_errors: Path
) -> NormalizedMetadata:
    """Construct a :class:`NormalizedMetadata` from a parsed JSON dict.

    Missing fields, wrong container types for ``z_distance`` /
    ``x_distance``, and non-numeric values for the float / int fields
    all raise :class:`ValueError` with sidecar path and field name in
    the message.
    """
    expected_field_names = [
        f.name
        for f in fields(NormalizedMetadata)
        if f.name not in _OPTIONAL_TUPLE_FIELDS
    ]
    missing = [n for n in expected_field_names if n not in md]
    if missing:
        raise ValueError(
            f"Sidecar '{source_for_errors}' '{_KEY_METADATA}' is "
            f"missing required field(s): {missing}."
        )

    kwargs: dict = {}
    for f in fields(NormalizedMetadata):
        if f.name in _OPTIONAL_TUPLE_FIELDS:
            v = md.get(f.name)
            if v is None:
                kwargs[f.name] = None
                continue
            if not isinstance(v, list):
                raise ValueError(
                    f"Sidecar '{source_for_errors}' "
                    f"'{_KEY_METADATA}.{f.name}' must be a JSON array "
                    f"of numbers or null, got {type(v).__name__}."
                )
            try:
                kwargs[f.name] = tuple(float(x) for x in v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sidecar '{source_for_errors}' "
                    f"'{_KEY_METADATA}.{f.name}' contains non-numeric "
                    f"value(s) ({exc})."
                ) from exc
            continue
        v = md[f.name]
        if f.name in _TUPLE_FIELDS:
            if not isinstance(v, list):
                raise ValueError(
                    f"Sidecar '{source_for_errors}' "
                    f"'{_KEY_METADATA}.{f.name}' must be a JSON array "
                    f"of numbers, got {type(v).__name__}."
                )
            try:
                v = tuple(float(x) for x in v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sidecar '{source_for_errors}' "
                    f"'{_KEY_METADATA}.{f.name}' contains non-numeric "
                    f"value(s) ({exc})."
                ) from exc
        elif f.name in _FLOAT_FIELDS:
            try:
                v = float(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sidecar '{source_for_errors}' "
                    f"'{_KEY_METADATA}.{f.name}' must be numeric, got "
                    f"{v!r} ({exc})."
                ) from exc
        elif f.name in _INT_FIELDS:
            try:
                v = int(v)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Sidecar '{source_for_errors}' "
                    f"'{_KEY_METADATA}.{f.name}' must be an integer, "
                    f"got {v!r} ({exc})."
                ) from exc
        kwargs[f.name] = v

    return NormalizedMetadata(**kwargs)


def _backup_existing(path: Path) -> Path:
    """Copy ``path`` to ``<path>.YYYYMMDDHHMMSS.bak`` before overwrite.

    On same-second collisions, append a numeric tiebreaker so we never
    overwrite an existing backup.
    """
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    backup_path = Path(str(path) + f".{timestamp}.bak")
    n = 1
    while backup_path.exists():
        backup_path = Path(str(path) + f".{timestamp}.{n}.bak")
        n += 1
    shutil.copy2(path, backup_path)
    return backup_path


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` as pretty-printed UTF-8 JSON atomically.

    The content is first written to a sibling temporary file in the
    same directory and then renamed via :func:`os.replace`. On any
    error, the temporary file is removed.
    """
    parent = path.parent
    target_dir = str(parent) if str(parent) else "."

    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=target_dir
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                payload, f, ensure_ascii=False, indent=2, sort_keys=False
            )
            f.write("\n")
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise


__all__ = [
    "NumSensorMismatchError",
    "save_sidecar",
    "load_sidecar",
    "refresh_sidecar_from_datasheet",
    "get_sidecar_json_path",
    "load_json_sidecar",
    "resolve_sidecar_for_sg2",
    "ensure_sidecar_from_excel",
]
