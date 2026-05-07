import numpy as np
import pytest

from src.processor.group_processor import GroupProcesser
from src.processor.single_processor import SingleProcesser


# ---------------------------------------------------------------------------
# SingleProcesser
# ---------------------------------------------------------------------------


def test_single_init_attributes(sample_npz_path):
    hr = SingleProcesser(sample_npz_path, backend="mpl")

    assert hasattr(hr, "fs"), "fs not set after init"
    assert float(hr.fs) > 0, "fs must be positive"

    assert hasattr(hr, "Num_sensor"), "Num_sensor not set after init"
    assert hasattr(hr, "distance"), "distance not set after init"
    assert len(hr.distance) == int(hr.Num_sensor)


def test_single_axis_y_loaded(sample_npz_path):
    hr = SingleProcesser(sample_npz_path, backend="mpl")

    assert hasattr(hr, "y"), "y axis not loaded"
    assert isinstance(hr.y, np.ndarray)
    assert hr.y.ndim == 2, "y should be a 2-D (n_traces, n_samples) array"
    assert hr.y.shape[0] == int(hr.Num_sensor)


def test_single_name_derived_from_path(sample_npz_path):
    hr = SingleProcesser(sample_npz_path, backend="mpl")

    assert isinstance(hr.name, str)
    assert len(hr.name) > 0


# ---------------------------------------------------------------------------
# GroupProcesser
# ---------------------------------------------------------------------------


def test_group_init(sample_npz_paths):
    sps = [SingleProcesser(p, backend="mpl") for p in sample_npz_paths]
    grp = GroupProcesser(sps, name="test_line")

    assert grp.fs == sps[0].fs
    assert len(grp.datalist) == len(sps)
    assert grp.name == "test_line"
    assert isinstance(grp.axis_bool, dict)
    assert "y" in grp.axis_bool


def test_group_cmp_gathering(sample_npz_paths):
    sps = [SingleProcesser(p, backend="mpl") for p in sample_npz_paths]
    grp = GroupProcesser(sps, name="test_line")

    grp.cmp_gathering(
        axis="y",
        closs_corr=False,
        integrate=False,
        average=True,
        show=False,
        save_name=None,
    )

    assert hasattr(grp, "cmp"), "cmp dict not set after cmp_gathering"
    assert hasattr(grp, "targets"), "targets not set after cmp_gathering"
    assert len(grp.targets) > 0, "no CMP targets found"
    assert grp.cmp is not None

    first_target = grp.targets[0]
    waveform = grp.cmp[first_target]
    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 2, "CMP waveform should be (n_traces, n_samples)"
