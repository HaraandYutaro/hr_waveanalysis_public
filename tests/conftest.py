"""
conftest.py — project-level pytest configuration.

The production code (src/plotting/backends/matplotlib_backend.py) calls
matplotlib.use("qtagg") at import time, which fails in headless environments
without Qt.  We set the Agg backend first and then replace matplotlib.use with
a no-op so subsequent calls from production code cannot switch away from it.
"""

import os
import pathlib

os.environ["MPLBACKEND"] = "Agg"

import matplotlib  # noqa: E402 (must come after env var is set)

matplotlib.use("Agg")
matplotlib.use = lambda *a, **kw: None  # freeze backend for all subsequent imports

import pytest  # noqa: E402

REALDATA_DIR = pathlib.Path(__file__).parent.parent / "sample_data" / "realdata"


@pytest.fixture
def sample_npz_path():
    """Path to a single real .npz file."""
    p = REALDATA_DIR / "776.npz"
    if not p.exists():
        pytest.skip(f"sample data not found: {p}")
    return str(p)


@pytest.fixture
def sample_npz_paths():
    """Paths to three real .npz files that share the same sampling frequency."""
    names = ["776.npz", "778.npz", "780.npz"]
    paths = [REALDATA_DIR / n for n in names]
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"sample data not found: {missing}")
    return [str(p) for p in paths]
