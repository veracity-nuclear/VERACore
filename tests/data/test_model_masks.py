"""nan_out_reflected and nan_out_non_fuel, checked against the original
dtype-by-dtype implementations for every dtype those handled."""

import numpy as np
import pytest

from vera_core.data.dtypes import NUM_NODES, VeraDataset, VeraDtype
from vera_core.data.model import nan_out_non_fuel, nan_out_reflected

T = VeraDtype
CM = np.arange(1, 10).reshape(3, 3)  # the mock's reduced quarter-core map
NPIN, NAX, NASS, NG = 3, 4, 9, 2


def old_nan_out_reflected(cm, core_sym, array):
    """Verbatim copy of the original, as the reference."""
    ax, ay = cm.shape
    dtype = array.dataset_type
    has_reflected_pins = dtype in (T.PIN, T.CHANNEL, T.RADIAL)
    if has_reflected_pins and core_sym == 4:
        hpy = array.shape[0] // 2
        hpx = array.shape[1] // 2
        match array.dataset_type:
            case T.PIN | T.CHANNEL:
                array[:hpy, :, :, :ax] = np.nan
                array[:, :hpx, :, cm[:, 0] - 1] = np.nan
            case T.RADIAL:
                array[:hpy, :, :ax] = np.nan
                array[:, :hpx, cm[:, 0] - 1] = np.nan
    elif (dtype == T.COMP_NODAL or dtype == T.NODAL) and core_sym == 4:
        array[: int(NUM_NODES / 2), :, :ax] = np.nan
        array[0, :, cm[:, 0] - 1] = np.nan
        array[2, :, cm[:, 0] - 1] = np.nan
    elif dtype == T.COMP_NODAL_ENERGY and core_sym == 4:
        array[:, : int(NUM_NODES / 2), :, :ax] = np.nan
        array[:, 0, :, cm[:, 0] - 1] = np.nan
        array[:, 2, :, cm[:, 0] - 1] = np.nan
    elif dtype == T.COMP_NODAL_SURFACE and core_sym == 4:
        array[:, :, : int(NUM_NODES / 2), :, :ax] = np.nan
        array[:, :, 0, :, cm[:, 0] - 1] = np.nan
        array[:, :, 2, :, cm[:, 0] - 1] = np.nan
    return array


SHAPES = {
    T.PIN: (NPIN, NPIN, NAX, NASS),
    T.COMP_PIN: (NPIN, NPIN, NAX, NASS),
    T.CHANNEL: (NPIN + 1, NPIN + 1, NAX, NASS),
    T.RADIAL: (NPIN, NPIN, NASS),
    T.CHANNEL_RADIAL: (NPIN + 1, NPIN + 1, NASS),
    T.NODAL: (NUM_NODES, NAX, NASS),
    T.COMP_NODAL: (NUM_NODES, NAX, NASS),
    T.RADIAL_NODE: (NUM_NODES, NASS),
    T.NODAL_ENERGY: (NG, NUM_NODES, NAX, NASS),
    T.COMP_NODAL_ENERGY: (NG, NUM_NODES, NAX, NASS),
    T.NODAL_SURFACE: (6, NG, NUM_NODES, NAX, NASS),
    T.COMP_NODAL_SURFACE: (6, NG, NUM_NODES, NAX, NASS),
    T.ASSEMBLY: (1, NAX, NASS),
    T.ASSY_ENERGY: (NG, 1, NAX, NASS),
    T.ASSY_SURFACE: (6, NG, 1, NAX, NASS),
    T.RADIAL_ASSEMBLY: (NASS,),
    T.AXIAL: (NAX,),
    T.POINT_DETECTOR: (NAX, NASS),
}
OLD_HANDLED = {
    T.PIN,
    T.CHANNEL,
    T.RADIAL,
    T.NODAL,
    T.COMP_NODAL,
    T.COMP_NODAL_ENERGY,
    T.COMP_NODAL_SURFACE,
}
NEWLY_HANDLED = {T.COMP_PIN, T.CHANNEL_RADIAL, T.RADIAL_NODE, T.NODAL_ENERGY, T.NODAL_SURFACE}
UNTOUCHED = set(SHAPES) - OLD_HANDLED - NEWLY_HANDLED


def ones(dtype):
    return VeraDataset(np.ones(SHAPES[dtype]), dtype)


@pytest.mark.parametrize("dtype", sorted(OLD_HANDLED, key=str), ids=str)
def test_matches_original_where_original_applied(dtype):
    np.testing.assert_array_equal(
        nan_out_reflected(CM, 4, ones(dtype)), old_nan_out_reflected(CM, 4, ones(dtype))
    )


@pytest.mark.parametrize("dtype", sorted(UNTOUCHED, key=str), ids=str)
def test_dtypes_without_lattice_untouched(dtype):
    assert np.isfinite(nan_out_reflected(CM, 4, ones(dtype))).all()


# Each newly handled dtype must match the masking of its sibling layout.
SIBLING = {
    T.COMP_PIN: T.PIN,
    T.CHANNEL_RADIAL: T.RADIAL,
    T.RADIAL_NODE: T.NODAL,
    T.NODAL_ENERGY: T.COMP_NODAL_ENERGY,
    T.NODAL_SURFACE: T.COMP_NODAL_SURFACE,
}


@pytest.mark.parametrize("dtype", sorted(NEWLY_HANDLED, key=str), ids=str)
def test_newly_handled_match_sibling(dtype):
    sibling = SIBLING[dtype]
    got = np.isnan(nan_out_reflected(CM, 4, ones(dtype)))
    ref = np.isnan(old_nan_out_reflected(CM, 4, ones(sibling)))
    if dtype == T.CHANNEL_RADIAL:  # channel lattice is one wider than pins
        ref = np.isnan(old_nan_out_reflected(CM, 4, VeraDataset(np.ones(SHAPES[dtype]), T.RADIAL)))
    if dtype == T.RADIAL_NODE:  # NODAL without the axial axis
        ref = ref[:, 0, :]
    assert got.any()
    np.testing.assert_array_equal(got, ref)


def test_full_core_untouched():
    assert np.isfinite(nan_out_reflected(CM, 1, ones(T.PIN))).all()


def test_first_row_uses_map_ids_not_positions():
    """The original assumed ids 1..n_cols fill row 0; ids are read from cm now."""
    cm = np.array([[3, 1, 2], [4, 5, 6], [7, 8, 9]])
    out = nan_out_reflected(cm, 4, ones(T.RADIAL))
    assert np.isnan(out[0, :, [0, 1, 2]]).all()
    assert np.isfinite(out[0, 1:, 4]).all()


# ---------------------------------------------------------------- nan_out_non_fuel


def volumes():
    v = np.ones((NPIN, NPIN, NAX, NASS))
    v[0, 0] = 0.0  # guide tube, every level
    v[2, 2, 1] = 0.0  # non-fuel at one level only
    return v


def test_pin_masks_exactly_zero_volume():
    ds = VeraDataset(np.arange(np.prod(SHAPES[T.PIN]), dtype=float).reshape(SHAPES[T.PIN]), T.PIN)
    out = nan_out_non_fuel(ds, volumes())
    np.testing.assert_array_equal(np.isnan(out), volumes() == 0)
    assert np.isfinite(ds).all()  # input untouched
    assert (out.dataset_type, out.name) == (ds.dataset_type, ds.name)


def test_radial_masks_pins_non_fuel_at_every_level():
    out = nan_out_non_fuel(ones(T.RADIAL), volumes())
    assert np.isnan(out[0, 0]).all()
    assert np.isfinite(out[2, 2]).all()
    assert np.isnan(out).sum() == NASS


@pytest.mark.parametrize("dtype", [T.COMP_PIN, T.CHANNEL, T.NODAL, T.ASSEMBLY], ids=str)
def test_other_dtypes_returned_as_is(dtype):
    ds = ones(dtype)
    assert nan_out_non_fuel(ds, volumes()) is ds


def test_shape_mismatch_returned_as_is():
    ds = VeraDataset(np.ones((NPIN, NPIN, NAX + 1, NASS)), T.PIN)
    assert nan_out_non_fuel(ds, volumes()) is ds


def test_no_volumes_returned_as_is():
    ds = ones(T.PIN)
    assert nan_out_non_fuel(ds, None) is ds
