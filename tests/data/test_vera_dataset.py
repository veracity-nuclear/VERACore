import copy
import pickle

import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDim, VeraDtype

T = VeraDtype
META = ("dataset_type", "name", "physical_units")


def meta(x):
    return tuple(getattr(x, attr) for attr in META)


@pytest.fixture
def ds():
    return VeraDataset(np.arange(6.0).reshape(2, 3), T.RADIAL_NODE, "flux", "W")


# ---------------------------------------------------------------- construction


def test_defaults():
    assert meta(VeraDataset([1.0])) == (T.UNKNOWN, None, "Unitless")


def test_keyword_construction():
    d = VeraDataset([1.0], dataset_type=T.AXIAL, name="n", physical_units="cm")
    assert meta(d) == (T.AXIAL, "n", "cm")


@pytest.mark.parametrize(
    "data, shape",
    [(3.0, ()), ([3.0], (1,)), ([[1, 2], [3, 4]], (2, 2)), (np.zeros((2, 3, 4)), (2, 3, 4))],
)
def test_shape_follows_input(data, shape):
    assert VeraDataset(data).shape == shape


@pytest.mark.parametrize("np_dtype", [np.int64, np.float32, np.float64, np.bool_])
def test_numeric_dtype_preserved(np_dtype):
    assert VeraDataset(np.zeros(2, dtype=np_dtype)).dtype == np_dtype


def test_is_ndarray_subclass(ds):
    assert isinstance(ds, np.ndarray)


def test_wraps_ndarray_without_copying():
    """The constructor is a view. Callers that must not alias the source
    copy first (VeraDataSource.get_dataset does)."""
    src = np.zeros(3)
    wrapped = VeraDataset(src)
    src[0] = 9.0
    assert wrapped[0] == 9.0


def test_rewrapping_replaces_metadata(ds):
    assert meta(VeraDataset(ds, T.NODAL)) == (T.NODAL, None, "Unitless")


def test_shape_is_not_checked_against_dtype():
    assert VeraDataset(np.zeros(5), T.PIN).dataset_type == T.PIN


# ---------------------------------------------------------------- delegation


@pytest.mark.parametrize("dtype", list(VeraDtype), ids=str)
def test_predicates_delegate(dtype):
    d = VeraDataset([0.0], dtype)
    assert d.is_computational() == dtype.is_computational()
    assert d.is_assembly() == dtype.is_assembly()


# ---------------------------------------------------------------- metadata propagation


@pytest.mark.parametrize(
    "op",
    [
        lambda d: d[0],
        lambda d: d[:, 1:],
        lambda d: d[[0, 1]],
        lambda d: d[d > 2],
        lambda d: d * 2,
        lambda d: 2 + d,
        lambda d: -d,
        lambda d: d**2,
        lambda d: np.sqrt(d),
        lambda d: np.ones((2, 3)) + d,
        lambda d: d.copy(),
        lambda d: copy.deepcopy(d),
        lambda d: d.astype(np.float32),
        lambda d: d.reshape(-1),
        lambda d: d.T,
        lambda d: d.transpose(1, 0),
        lambda d: d.mean(axis=0),
        lambda d: d.sum(),
        lambda d: d == d,
    ],
    ids=[
        "row", "slice", "fancy", "mask", "mul", "radd", "neg", "pow", "ufunc", "ndarray_add",
        "copy", "deepcopy", "astype", "reshape", "T", "transpose", "mean_axis", "sum", "eq",
    ],
)
def test_metadata_survives(ds, op):
    out = op(ds)
    assert isinstance(out, VeraDataset)
    assert meta(out) == meta(ds)


def test_element_access_returns_plain_scalar(ds):
    assert not isinstance(ds[0, 0], VeraDataset)
    assert ds[0, 0] == 0.0


@pytest.mark.parametrize(
    "op",
    [
        lambda d: np.asarray(d),
        lambda d: d.view(np.ndarray),
        lambda d: np.where(d > 2, d, np.nan),
        lambda d: np.stack([d, d]),
        lambda d: np.concatenate([d, d]),
    ],
    ids=["asarray", "view", "where", "stack", "concatenate"],
)
def test_metadata_dropped_by(ds, op):
    assert type(op(ds)) is np.ndarray


def test_asanyarray_keeps_subclass(ds):
    assert np.asanyarray(ds) is ds


def test_left_operand_metadata_wins(ds):
    """Documents a hazard: combining datasets keeps the left operand's
    metadata without checking dtype or units."""
    other = VeraDataset(np.ones((2, 3)), T.NODAL, "other", "cm")
    assert meta(ds + other) == meta(ds)
    assert meta(other + ds) == meta(other)


def test_reduction_keeps_dtype_of_unreduced_layout(ds):
    """Documents a hazard: dataset_type describes the source layout, not the
    result. A full reduction is still tagged RADIAL_NODE."""
    assert ds.sum().shape == ()
    assert ds.sum().dataset_type == T.RADIAL_NODE


def test_derived_arrays_do_not_share_metadata_objects(ds):
    out = ds * 1
    out.name = "renamed"
    assert ds.name == "flux"


# ---------------------------------------------------------------- select


def pin_ds():
    return VeraDataset(np.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5), T.PIN, "p", "W")


@pytest.mark.parametrize(
    "indices, index",
    [
        ({VeraDim.AXIAL: 1}, np.s_[:, :, 1, :]),
        ({VeraDim.ASSEMBLY: -1}, np.s_[:, :, :, -1]),
        ({VeraDim.PIN_Y: 1, VeraDim.PIN_X: 2}, np.s_[1, 2, :, :]),
        ({VeraDim.AXIAL: 3, VeraDim.ASSEMBLY: 4}, np.s_[:, :, 3, 4]),
        ({VeraDim.PIN_Y: 0, VeraDim.PIN_X: 0, VeraDim.AXIAL: 0, VeraDim.ASSEMBLY: 0}, np.s_[0, 0, 0, 0]),
    ],
)
def test_select_on_pin(indices, index):
    d = pin_ds()
    np.testing.assert_array_equal(d.select(indices), np.asarray(d)[index])


def test_select_keeps_metadata():
    assert meta(pin_ds().select({VeraDim.AXIAL: 0})) == (T.PIN, "p", "W")


def test_select_full_selection_is_zero_dimensional():
    out = pin_ds().select(
        {VeraDim.PIN_Y: 1, VeraDim.PIN_X: 2, VeraDim.AXIAL: 3, VeraDim.ASSEMBLY: 4}
    )
    assert out.shape == ()
    assert out == np.asarray(pin_ds())[1, 2, 3, 4]


@pytest.mark.parametrize("empty", [None, {}])
def test_select_nothing_on_pin_is_identity(empty):
    d = pin_ds()
    np.testing.assert_array_equal(d.select(empty), d)


@pytest.mark.parametrize(
    "dtype, shape, index",
    [
        (T.ASSEMBLY, (2, 4, 5), np.s_[0]),
        (T.COMP_ASSY, (2, 4, 5), np.s_[0]),
        (T.ASSY_ENERGY, (3, 2, 4, 5), np.s_[:, 0]),
        (T.ASSY_SURFACE, (6, 3, 2, 4, 5), np.s_[:, :, 0]),
    ],
    ids=str,
)
def test_select_nothing_still_collapses_fixed_axis(dtype, shape, index):
    data = np.arange(np.prod(shape)).reshape(shape)
    np.testing.assert_array_equal(VeraDataset(data, dtype).select(), data[index])


def test_select_ignores_dims_absent_from_dtype():
    d = VeraDataset(np.arange(4.0), T.AXIAL)
    out = d.select({VeraDim.GROUP: 0, VeraDim.NODE: 1, VeraDim.PIN_Y: 0})
    np.testing.assert_array_equal(out, d)


@pytest.mark.parametrize("dim", [VeraDim.PIN_Y, VeraDim.PIN_X])
def test_select_one_pin_axis_rejected(dim):
    with pytest.raises(ValueError, match="together"):
        pin_ds().select({dim: 0})


def test_select_out_of_range_raises_index_error():
    with pytest.raises(IndexError):
        pin_ds().select({VeraDim.AXIAL: 4})


def test_select_on_scalar_and_unknown_is_identity():
    for dtype in (T.SCALAR, T.UNKNOWN):
        d = VeraDataset(np.arange(3.0), dtype)
        np.testing.assert_array_equal(d.select({VeraDim.AXIAL: 0}), d)
