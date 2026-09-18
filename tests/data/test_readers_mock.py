import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.readers.mock import (
    DictDatasetSource,
    build_core,
    build_source,
    core_arrays,
    core_map,
    state_arrays,
)

# ---------------------------------------------------------------- DictDatasetSource


@pytest.fixture
def dsrc():
    return DictDatasetSource(
        {"vec": [1.0, 2.0], "scalar": 3.0},
        provenance="here",
        dataset_dtypes={(2,): VeraDtype.AXIAL, (): VeraDtype.SCALAR},
        units={"vec": "W"},
    )


def test_names_and_provenance(dsrc):
    assert dsrc.names() == ["vec", "scalar"]
    assert dsrc.provenance == "here"


def test_scalar_shape_reported_as_length_one(dsrc):
    assert dsrc.shape("scalar") == (1,)
    assert dsrc.shape("vec") == (2,)


def test_missing_name(dsrc):
    assert dsrc.shape("nope") is None
    assert dsrc.load("nope") is None
    assert not dsrc.has("nope")


def test_load_wraps_and_types(dsrc):
    vec = dsrc.load("vec")
    assert isinstance(vec, VeraDataset)
    assert (vec.dataset_type, vec.name, vec.physical_units) == (VeraDtype.AXIAL, "vec", "W")


def test_load_scalar_becomes_length_one_and_types_by_raw_shape(dsrc):
    s = dsrc.load("scalar")
    assert s.shape == (1,)
    assert s.dataset_type == VeraDtype.SCALAR
    assert s.physical_units == "Unitless"


def test_load_without_dtype_map_is_unknown():
    assert DictDatasetSource({"x": [1.0]}).load("x").dataset_type == VeraDtype.UNKNOWN


def test_load_count_tracks_reads(dsrc):
    dsrc.load("vec")
    dsrc.load("vec")
    dsrc.load("nope")
    assert dsrc.load_count == {"vec": 2}


# ---------------------------------------------------------------- core_map


def test_full_core_map_numbers_every_cell():
    cm = core_map(5, 1)
    np.testing.assert_array_equal(np.sort(cm.ravel()), np.arange(1, 26))


@pytest.mark.parametrize("n", [3, 5, 7])
def test_quarter_core_map_is_mirror_symmetric(n):
    cm = core_map(n, 4)
    np.testing.assert_array_equal(cm, cm[::-1, :])
    np.testing.assert_array_equal(cm, cm[:, ::-1])


def test_quarter_core_map_quadrant_is_sequential():
    cm = core_map(5, 4)
    np.testing.assert_array_equal(cm[2:, 2:], np.arange(1, 10).reshape(3, 3))
    assert len(np.unique(cm)) < cm.size


# ---------------------------------------------------------------- core_arrays


def test_core_arrays_defaults():
    a = core_arrays()
    assert set(a) == {"core_map", "core_sym", "axial_mesh", "npin", "pin_volumes"}
    assert a["pin_volumes"].shape == (3, 3, 4, 9)
    assert (a["pin_volumes"][0, 0] == 0).all()
    assert a["pin_volumes"].sum() == (9 - 1) * 4 * 9


@pytest.mark.parametrize(
    "kwargs, absent",
    [
        ({"core_sym": None}, "core_sym"),
        ({"with_axial_mesh": False}, "axial_mesh"),
        ({"npin": None}, "npin"),
        ({"npin": None}, "pin_volumes"),
        ({"with_pin_volumes": False}, "pin_volumes"),
    ],
)
def test_core_arrays_omissions(kwargs, absent):
    assert absent not in core_arrays(**kwargs)


@pytest.mark.parametrize(
    "kwargs, present",
    [
        ({"aspect_ratio": 2.0}, {"aspect_ratio"}),
        ({"apitch": 21.5}, {"apitch"}),
        ({"with_comp_core": True}, {"computational_core_map", "computational_axial_mesh"}),
        ({"with_detectors": True}, {"detector_map", "detector_axial_mesh"}),
        ({"labels": True}, {"xlabel", "ylabel"}),
    ],
)
def test_core_arrays_additions(kwargs, present):
    assert present <= set(core_arrays(**kwargs))


def test_core_arrays_labels_are_bytes():
    a = core_arrays(labels=True)
    assert list(a["xlabel"]) == [b"A", b"B", b"C", b"D", b"E"]
    assert list(a["ylabel"]) == [b"1", b"2", b"3", b"4", b"5"]


def test_map_sym_decouples_layout_from_declaration():
    a = core_arrays(core_sym=None, map_sym=4)
    assert "core_sym" not in a
    np.testing.assert_array_equal(a["core_map"], core_map(5, 4))


# ---------------------------------------------------------------- state_arrays / build


def test_state_arrays_match_core_shape():
    core = build_core()
    a = state_arrays(core, 10.0)
    assert a["pin_powers"].shape == core.core_shape
    assert a["pin_radial_powers"].shape == (3, 3, 9)
    assert a["assembly_powers"].shape == (1, 4, 9)
    assert a["axial_powers"].shape == (4,)


def test_state_arrays_deterministic_per_exposure():
    core = build_core()
    np.testing.assert_array_equal(
        state_arrays(core, 10.0)["pin_powers"], state_arrays(core, 10.0)["pin_powers"]
    )
    assert not np.array_equal(
        state_arrays(core, 10.0)["pin_powers"], state_arrays(core, 20.0)["pin_powers"]
    )


def test_state_arrays_scale():
    core = build_core()
    np.testing.assert_allclose(
        state_arrays(core, 10.0, scale=2.0)["pin_powers"],
        2.0 * state_arrays(core, 10.0)["pin_powers"],
    )


def test_build_source_state_count_and_exposures():
    assert len(build_source(4).states) == 4
    src = build_source(exposures=[1.0, 7.0])
    assert src.time_axes()["exposure"] == [1.0, 7.0]


def test_build_source_passes_core_kwargs():
    assert build_source(core_kwargs={"nax": 6}).core.nax == 6


def test_build_source_state_caching_flag():
    src = build_source(state_caching=False)
    assert "pin_powers" not in src.active_state.source.load_count
