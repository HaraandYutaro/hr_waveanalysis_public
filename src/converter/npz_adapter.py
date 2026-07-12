#! /usr/bin/python3
"""
Minimal npz read adapter for the SEG-2 / JSON sidecar input refactor.

This module is the dedicated **npz physical I/O boundary** introduced as
Slice A of the SEG-2 / JSON sidecar input refactor (see
``private/docs/ai_refactor.md`` §3 and the "SEG-2 / JSON sidecar input
refactor plan" section). Its sole responsibility is to encapsulate the
``np.load(..., allow_pickle=True)`` call and the associated file-handle
context management behind a single read entry point.

Scope boundaries for this slice (intentionally narrow):

- read-only API; no write helpers are introduced here yet
- no schema knowledge (required vs. optional keys is not enforced)
- no value normalization: zero-dimensional ndarrays are returned as-is so
  that existing callers (``SingleProcesser._input_*``) can continue to
  apply their own ``str(...)`` casts and ``"<key>" in file`` checks
  without behavioural drift
- the returned ``dict[str, Any]`` preserves the npz key set verbatim;
  callers must continue to gate optional keys with membership tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Union

import numpy as np


def load_npz(path: Union[str, Path]) -> Dict[str, Any]:
    """Load an ``.npz`` file and return its contents as a plain ``dict``.

    The npz handle is opened with ``allow_pickle=True`` and is closed
    before this function returns. Every key listed in ``NpzFile.files``
    is read once and stored in the returned mapping verbatim; no value
    coercion (scalar unwrapping, string decoding, etc.) is performed.

    Parameters
    ----------
    path
        Filesystem path of the ``.npz`` file to read.

    Returns
    -------
    dict[str, Any]
        Mapping of every npz key to its raw value. Zero-dimensional
        ndarrays are preserved as ndarrays so downstream callers can
        continue to apply their own coercion (e.g. ``str(file["unit"])``)
        and membership checks (``"<key>" in file``) without observable
        behavioural change.
    """
    with np.load(path, allow_pickle=True) as file:
        return {key: file[key] for key in file.files}
