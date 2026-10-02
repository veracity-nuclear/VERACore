"""dtypes behavior not pinned elsewhere: build_core_dtypes branch conditions,
point_indices on zero-axis and split-only dtypes, recipe dicts, and
VeraDataset corners."""

import numpy as np
import pytest

from vera_core.data.dtypes import (
    VeraDataset,
    VeraDim,
    VeraDtype,
    build_core_dtypes,
    derive_recipe,
    point_indices,
)

T = VeraDtype

# ---------------------------------------------------------------- build_core_dtypes


def test_comp_dtypes_skipped_when_comp_grid_equals_base_grid():
    d = build_core_dtypes(3, 3, nax=4, nass=9, comp_nax=4, comp_nass=9)
    assert not any(t.is_computational() for t in d.values())


def test_comp_dtypes_added_when_only_one_comp_size_differs():
    d = build_core_dtypes(3, 3, nax=4, nass=9, comp_nax=4, comp_nass=12)
    assert d[(3, 3, 4, 12)] == T.COMP_PIN


def test_comp_pin_key_contains_none_without_pin_counts():
    """Documents a hazard: COMP_PIN is keyed on (npiny, npinx, ...) even when
    the pin counts are None, so the map holds a key no real shape can match."""
    d = build_core_dtypes(nax=4, nass=9, comp_nax=6, comp_nass=12)
    assert d[(None, None, 6, 12)] == T.COMP_PIN


def test_detector_skipped_when_ndet_and_ndax_both_equal_nass():
    """Documents a hazard: the guard compares ndax with ndet, not with nax, so
    a detector grid with ndet == ndax == nass gets no detector dtype."""
    d = build_core_dtypes(nax=4, nass=9, ndet=9, ndax=9)
    assert (9, 9) not in d


def test_detector_mapped_when_ndet_equals_nass_but_ndax_differs():
    d = build_core_dtypes(nax=4, nass=9, ndet=9, ndax=7)
    assert d[(7, 9)] == T.POINT_DETECTOR


def test_no_base_dtypes_without_nass():
    assert build_core_dtypes(npiny=3, npinx=3, nax=4) == {}


def test_non_square_pin_lattice():
    d = build_core_dtypes(npiny=2, npinx=3, nax=4, nass=9)
    assert d[(2, 3, 4, 9)] == T.PIN
    assert d[(3, 4, 4, 9)] == T.CHANNEL


# ---------------------------------------------------------------- point_indices


@pytest.mark.parametrize("shape, expected", [((), [()]), ((1,), [(0,)]), ((1, 1), [(0, 0)])])
def test_zero_axis_dtype_single_value(shape, expected):
    assert point_indices(T.SCALAR, shape, {}) == expected


def test_split_only_selection_enumerates_every_element():
    assert point_indices(T.AXIAL, (3,), {}, split=(VeraDim.AXIAL,)) == [(0,), (1,), (2,)]


def test_split_over_two_dims_is_row_major():
    sel = {VeraDim.AXIAL: 1, VeraDim.ASSEMBLY: 2}
    points = point_indices(
        T.ASSY_SURFACE, (6, 2, 1, 5, 9), sel, split=(VeraDim.SURFACE, VeraDim.GROUP)
    )
    assert len(points) == 12
    assert points[:3] == [(0, 0, 0, 1, 2), (0, 1, 0, 1, 2), (1, 0, 0, 1, 2)]


def test_split_dims_absent_from_dtype_are_ignored():
    sel = {VeraDim.AXIAL: 2}
    assert point_indices(T.AXIAL, (5,), sel) == [(2,)]


def test_selection_keys_absent_from_dtype_are_ignored():
    sel = {VeraDim.AXIAL: 2, VeraDim.NODE: 3, VeraDim.SURFACE: 1}
    assert point_indices(T.AXIAL, (5,), sel) == [(2,)]


def test_indices_are_plain_ints():
    [point] = point_indices(T.AXIAL, (5,), {VeraDim.AXIAL: np.int64(2)})
    assert type(point[0]) is int


def test_split_dim_also_in_selection_splits_anyway():
    """split wins over selection: the selected value is overwritten."""
    points = point_indices(T.AXIAL, (3,), {VeraDim.AXIAL: 1}, split=(VeraDim.AXIAL,))
    assert points == [(0,), (1,), (2,)]


# ---------------------------------------------------------------- recipes


def test_derive_recipe_is_exact():
    """Pins the stored key names. They differ from the argument names
    (use_factors -> use_factor, exclude_non_fuel_rods -> exclude_fuel_rods);
    saved sessions depend on these keys."""
    r = derive_recipe("a", "pin_powers", "avg", "Average", "AXIAL", True, False)
    assert r == {
        "kind": "derive",
        "src_id": "a",
        "source_array": "pin_powers",
        "name": "avg",
        "method": "Average",
        "axes": "AXIAL",
        "use_factor": True,
        "exclude_fuel_rods": False,
    }


def test_recipes_are_fresh_dicts():
    a = derive_recipe("a", "x", "n", "Average", "CORE", True, False)
    b = derive_recipe("a", "x", "n", "Average", "CORE", True, False)
    assert a == b and a is not b


# ---------------------------------------------------------------- VeraDtype


@pytest.mark.parametrize("dtype", [d for d in T if not d.has_pin_level_dim()], ids=str)
def test_make_slice_ignores_pin_pair_on_dtype_without_pins(dtype):
    assert dtype.make_slice(pin_idxs=(1,)) == dtype.make_slice()


# ---------------------------------------------------------------- VeraDataset


def test_select_does_not_copy():
    ds = VeraDataset(np.zeros((4, 5)), T.POINT_DETECTOR)
    view = ds.select({VeraDim.ASSEMBLY: 1})
    view[0] = 7.0
    assert ds[0, 1] == 7.0


def test_select_accepts_numpy_integer_indices():
    ds = VeraDataset(np.arange(5.0), T.AXIAL)
    assert ds.select({VeraDim.AXIAL: np.int64(3)}) == 3.0


def test_select_reads_fixed_axis_at_zero_even_when_longer():
    ds = VeraDataset(np.arange(2 * 3 * 4).reshape(2, 3, 4), T.ASSEMBLY)
    np.testing.assert_array_equal(ds.select(), np.asarray(ds)[0])


def test_arrange_pad_adds_absent_dim():
    ds = VeraDataset(np.arange(5.0), T.AXIAL)
    [out] = ds.arrange(order=(VeraDim.NODE, VeraDim.AXIAL), pad=(VeraDim.NODE,))
    assert out.shape == (1, 5)


def test_arrange_pad_multiple_absent_dims_in_order_position():
    ds = VeraDataset(np.arange(5.0), T.AXIAL)
    order = (VeraDim.GROUP, VeraDim.AXIAL, VeraDim.NODE)
    [out] = ds.arrange(order=order, pad=(VeraDim.NODE, VeraDim.GROUP))
    assert out.shape == (1, 5, 1)
