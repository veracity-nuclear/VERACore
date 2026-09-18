"""H5 reader tests against small files written to tmp_path.

The ROM path (RomH5DatasetSource, and open_vera_file_data_source when the file
has a ROM group) is not covered: rom.py was not available.
"""

import h5py
import numpy as np
import pytest

from vera_core.data import model
from vera_core.data.dtypes import VeraDtype
from vera_core.data.readers.h5 import (
    H5DatasetSource,
    _get_units,
    _search_for_datasets,
    open_vera_file_data_source,
)
from vera_core.data.readers.mock import DictDatasetSource, build_core, core_arrays, state_arrays

# STATE_2 is unpadded so lexicographic order (0001, 0010, 2) differs from
# numeric order (1, 2, 10).
EXPOSURES = {"STATE_0001": 0.0, "STATE_2": 10.0, "STATE_0010": 20.0}


@pytest.fixture(autouse=True)
def no_veraout(monkeypatch):
    """Keep pyvera out of these tests: the calculator is never needed here."""

    def unavailable(filename):
        raise RuntimeError("VERAout disabled in tests")

    monkeypatch.setattr(model, "VERAout", unavailable)


@pytest.fixture
def h5_path(tmp_path):
    core = build_core()
    path = tmp_path / "run.h5"
    with h5py.File(path, "w") as f:
        for name, data in core_arrays().items():
            f[f"CORE/{name}"] = data
        for group, exposure in EXPOSURES.items():
            for name, data in state_arrays(core, exposure).items():
                f[f"{group}/{name}"] = data
        s1 = f["STATE_0001"]
        s1["pin_powers"].attrs["physical_units"] = b"W"
        s1["axial_powers"].attrs["units"] = np.array([b"cm"])
        s1["scalar0d"] = 2.5
        s1["NODAL_XS/AXIALMESH"] = np.arange(3.0)
        f["NOT_STATE"] = np.zeros(1)
    return path


@pytest.fixture
def h5file(h5_path):
    with h5py.File(h5_path, "r") as f:
        yield f


@pytest.fixture
def state1(h5file):
    return H5DatasetSource(h5file, "STATE_0001", build_core().shape_to_dtype)


# ---------------------------------------------------------------- helpers


def test_units_from_physical_units_bytes(h5file):
    assert _get_units(h5file["STATE_0001/pin_powers"]) == "W"


def test_units_from_units_array(h5file):
    assert _get_units(h5file["STATE_0001/axial_powers"]) == "cm"


def test_units_default(h5file):
    assert _get_units(h5file["STATE_0001/keff"]) == "Unitless"


def test_search_finds_nested_and_normalizes_scalars(h5file):
    found = _search_for_datasets(h5file["STATE_0001"])
    assert found["NODAL_XS/AXIALMESH"] == (3,)
    assert found["scalar0d"] == (1,)
    assert found["pin_powers"] == (3, 3, 4, 9)


# ---------------------------------------------------------------- H5DatasetSource


def test_non_group_dir_rejected(h5file):
    with pytest.raises(ValueError, match="not a group"):
        H5DatasetSource(h5file, "NOT_STATE")


def test_provenance_is_filename(state1, h5_path):
    assert state1.provenance == str(h5_path)


def test_names_include_nested(state1):
    assert {"pin_powers", "keff", "scalar0d", "NODAL_XS/AXIALMESH"} <= set(state1.names())


def test_shape_lookup(state1):
    assert state1.shape("pin_powers") == (3, 3, 4, 9)
    assert state1.shape("scalar0d") == (1,)
    assert state1.shape("nope") is None


def test_shape_falls_back_to_root_path(h5file):
    core_src = H5DatasetSource(h5file, "CORE")
    assert core_src.shape("STATE_0001/pin_powers") == (3, 3, 4, 9)


def test_load_types_by_shape_and_reads_units(state1):
    ds = state1.load("pin_powers")
    assert (ds.dataset_type, ds.name, ds.physical_units) == (VeraDtype.PIN, "pin_powers", "W")


def test_load_scalar_becomes_length_one(state1):
    ds = state1.load("scalar0d")
    assert ds.shape == (1,)
    assert ds[0] == 2.5
    assert ds.dataset_type == VeraDtype.SCALAR


def test_load_missing_or_group_returns_none(state1):
    assert state1.load("nope") is None
    assert state1.load("NODAL_XS") is None


def test_load_without_dtype_map_is_unknown(h5file):
    assert H5DatasetSource(h5file, "STATE_0001").load("pin_powers").dataset_type == (
        VeraDtype.UNKNOWN
    )


def test_sample(state1):
    assert state1.sample("scalar0d", ()) == 2.5
    np.testing.assert_array_equal(state1.sample("exposure", ()), [0.0])
    assert state1.sample("pin_powers", (0, 0, 0, 0)) == state1.load("pin_powers")[0, 0, 0, 0]
    assert state1.sample("nope", ()) is None


def test_default_units_match_dict_source(state1):
    dict_units = DictDatasetSource({"keff": [1.0]}).load("keff").physical_units
    assert state1.load("keff").physical_units == dict_units


# ---------------------------------------------------------------- open_vera_file_data_source


@pytest.fixture
def opened(h5_path):
    src = open_vera_file_data_source(str(h5_path))
    yield src
    src.close()


def test_open_sorts_states_numerically(opened):
    assert opened.time_axes()["exposure"] == [0.0, 10.0, 20.0]


def test_open_builds_core_and_metadata(opened, h5_path):
    assert opened.core.core_shape == (3, 3, 4, 9)
    assert opened.provenance == str(h5_path)
    assert opened.vera_calculator is None


def test_open_ignores_non_state_groups(opened):
    assert len(opened.states) == 3


def test_open_respects_active_state(h5_path):
    src = open_vera_file_data_source(str(h5_path), active_state_idx=2)
    try:
        assert src.active_state_index == 2
    finally:
        src.close()


def test_open_passes_core_overrides(h5_path):
    src = open_vera_file_data_source(str(h5_path), core_overrides={"npin": 5})
    try:
        assert src.core.core_shape == (5, 5, 4, 9)
    finally:
        src.close()


def test_open_reads_state_data(opened):
    assert opened.get_dataset_units("pin_powers") == "W"
    assert opened.get_dataset_dtype("pin_powers") == VeraDtype.PIN


def test_close_releases_file(h5_path):
    src = open_vera_file_data_source(str(h5_path))
    handle = src.core.source._f
    src.close()
    assert not handle.id.valid


@pytest.mark.skip(reason="rom.py not available; ROM branch untested.")
def test_open_rom_file():
    pass
