import numpy as np
import pytest

from vera_core.data.dtypes import (
    _INFO,
    MAX_NUM_GROUPS,
    MIN_NUM_GROUPS,
    NUM_DF,
    NUM_NODES,
    VeraDataset,
    VeraDim,
    VeraDtype,
    _Info,
    build_core_dtypes,
    derive_recipe,
    diff_recipe,
    make_slice,
)

ALL = slice(None)


# ---------------------------------------------------------------- _Info


def test_info_ndim_counts_every_axis_including_fixed():
    assert _Info(fixed_idxs={0: 0}, axial_idx=1, assembly_id_idx=2).ndim == 3


def test_info_with_no_axes_has_ndim_zero():
    assert _Info().ndim == 0


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"pin_idxs": (0,)}, ValueError),
        ({"axial_idx": -1}, ValueError),
        ({"axial_idx": True}, TypeError),
        ({"axial_idx": 1.0}, TypeError),
        ({"axial_idx": 0, "assembly_id_idx": 0}, ValueError),  # duplicate
        ({"axial_idx": 0, "assembly_id_idx": 2}, ValueError),  # gap
        ({"fixed_idxs": {0: 1.5}}, TypeError),
        ({"fixed_idxs": {0: True}}, TypeError),
    ],
)
def test_info_rejects_invalid_layouts(kwargs, error):
    with pytest.raises(error):
        _Info(**kwargs)


def test_every_dtype_has_info():
    assert set(_INFO) == set(VeraDtype)


@pytest.mark.parametrize("dtype", list(VeraDtype))
def test_dim_axes_and_fixed_axes_cover_ndim(dtype):
    axes = set(dtype.dim_axes.values()) | set(dtype._info.fixed_idxs)
    assert axes == set(range(dtype.ndim))


# ---------------------------------------------------------------- VeraDtype


def test_core_is_alias_for_scalar():
    assert VeraDtype.CORE is VeraDtype.SCALAR
    assert str(VeraDtype.CORE) == "SCALAR"


def test_str_and_title_use_member_name():
    assert str(VeraDtype.PIN) == VeraDtype.PIN.str == VeraDtype.PIN.title == "PIN"


def test_pin_axis_accessors():
    pin = VeraDtype.PIN
    assert pin.pin_dim_idxs == (0, 1)
    assert pin.axial_dim_idx == 2
    assert pin.assembly_id_dim_idx == 3
    assert pin.ndim == 4


@pytest.mark.parametrize(
    "accessor",
    [
        "axial_dim_idx",
        "energy_group_dim_idx",
        "pin_dim_idxs",
        "node_dim_idx",
        "assembly_id_dim_idx",
        "surface_dim_idx",
    ],
)
def test_axis_accessors_raise_when_axis_absent(accessor):
    with pytest.raises(ValueError):
        getattr(VeraDtype.SCALAR, accessor)


@pytest.mark.parametrize(
    "dtype, predicate",
    [
        (VeraDtype.PIN, "has_fuel_pins"),
        (VeraDtype.COMP_PIN, "is_computational"),
        (VeraDtype.NODAL, "is_nodal"),
        (VeraDtype.ASSEMBLY, "is_assembly"),
        (VeraDtype.CHANNEL, "is_channel"),
        (VeraDtype.POINT_DETECTOR, "is_detector"),
        (VeraDtype.COMP_NODAL_SURFACE, "has_surface_dim"),
        (VeraDtype.COMP_ASSY_ENERGY, "has_energy_group_dim"),
    ],
)
def test_flag_predicates(dtype, predicate):
    assert getattr(dtype, predicate)()
    assert not getattr(VeraDtype.SCALAR, predicate)()


def test_dim_axes_for_pin():
    assert VeraDtype.PIN.dim_axes == {
        VeraDim.PIN_Y: 0,
        VeraDim.PIN_X: 1,
        VeraDim.AXIAL: 2,
        VeraDim.ASSEMBLY: 3,
    }


# ---------------------------------------------------------------- make_slice


def test_make_slice_without_selectors_is_all_slices():
    assert make_slice(VeraDtype.PIN) == (ALL, ALL, ALL, ALL)


def test_make_slice_sets_requested_axes():
    assert make_slice(VeraDtype.PIN, axial_idx=2, pin_idxs=(0, 1)) == (0, 1, 2, ALL)


def test_make_slice_applies_fixed_axes():
    assert make_slice(VeraDtype.ASSEMBLY, assembly_id=5) == (0, ALL, 5)


def test_make_slice_ignores_selectors_for_absent_axes():
    assert make_slice(VeraDtype.AXIAL, group_idx=1, assembly_id=3) == (ALL,)


def test_method_matches_function():
    kwargs = {"axial_idx": 1, "node_idx": 2}
    assert VeraDtype.NODAL.make_slice(**kwargs) == make_slice(VeraDtype.NODAL, **kwargs)


# ---------------------------------------------------------------- VeraDataset


@pytest.fixture
def pin_ds():
    return VeraDataset(np.arange(2 * 3 * 4 * 5.0).reshape(2, 3, 4, 5), VeraDtype.PIN, "p", "W")


def test_dataset_defaults():
    ds = VeraDataset([1.0])
    assert (ds.dataset_type, ds.name, ds.physical_units) == (VeraDtype.UNKNOWN, None, "Unitless")


def test_metadata_survives_slicing_and_arithmetic(pin_ds):
    for derived in (pin_ds[0], pin_ds * 2, pin_ds + pin_ds):
        assert isinstance(derived, VeraDataset)
        assert derived.dataset_type == VeraDtype.PIN
        assert derived.physical_units == "W"
        assert derived.name == "p"


def test_predicates_delegate_to_dtype():
    assert VeraDataset([1.0], VeraDtype.COMP_ASSY).is_computational()
    assert VeraDataset([1.0], VeraDtype.COMP_ASSY).is_assembly()
    assert not VeraDataset([1.0], VeraDtype.PIN).is_assembly()


def test_select_picks_semantic_axes(pin_ds):
    out = pin_ds.select({VeraDim.AXIAL: 1, VeraDim.ASSEMBLY: 2})
    np.testing.assert_array_equal(out, pin_ds[:, :, 1, 2])


def test_select_ignores_absent_dims(pin_ds):
    out = pin_ds.select({VeraDim.GROUP: 0})
    np.testing.assert_array_equal(out, pin_ds)


def test_select_with_none_returns_all(pin_ds):
    np.testing.assert_array_equal(pin_ds.select(None), pin_ds)


def test_select_rejects_single_pin_axis(pin_ds):
    with pytest.raises(ValueError, match="together"):
        pin_ds.select({VeraDim.PIN_Y: 0})


def test_arrange_reorders_axes(pin_ds):
    order = (VeraDim.ASSEMBLY, VeraDim.AXIAL, VeraDim.PIN_Y, VeraDim.PIN_X)
    [out] = pin_ds.arrange(order=order)
    np.testing.assert_array_equal(out, pin_ds.transpose(3, 2, 0, 1))


def test_arrange_selects_then_orders(pin_ds):
    [out] = pin_ds.arrange(order=(VeraDim.PIN_X, VeraDim.PIN_Y), axial=1, assembly=4)
    np.testing.assert_array_equal(out, pin_ds[:, :, 1, 4].T)


def test_arrange_splits_into_one_array_per_index(pin_ds):
    parts = pin_ds.arrange(
        order=(VeraDim.PIN_Y, VeraDim.PIN_X), split=(VeraDim.AXIAL,), assembly=0
    )
    assert len(parts) == 4
    for z, part in enumerate(parts):
        np.testing.assert_array_equal(part, pin_ds[:, :, z, 0])


def test_arrange_split_over_two_dims_is_row_major(pin_ds):
    parts = pin_ds.arrange(
        order=(VeraDim.PIN_Y, VeraDim.PIN_X), split=(VeraDim.AXIAL, VeraDim.ASSEMBLY)
    )
    assert len(parts) == 4 * 5
    np.testing.assert_array_equal(parts[6], pin_ds[:, :, 1, 1])


def test_arrange_ignores_dims_absent_from_dtype():
    ds = VeraDataset(np.arange(4.0), VeraDtype.AXIAL)
    [out] = ds.arrange(order=(VeraDim.AXIAL, VeraDim.GROUP), split=(VeraDim.NODE,), group=0)
    np.testing.assert_array_equal(out, ds)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"order": (VeraDim.AXIAL, VeraDim.AXIAL)}, "order"),
        ({"order": (), "split": (VeraDim.AXIAL, VeraDim.AXIAL)}, "split"),
        ({"order": (), "require": (VeraDim.AXIAL, VeraDim.AXIAL)}, "require"),
        ({"order": (VeraDim.AXIAL,), "split": (VeraDim.AXIAL,)}, "both"),
        ({"order": (), "require": (VeraDim.GROUP,)}, "missing required"),
        ({"order": (VeraDim.AXIAL,)}, "unhandled"),
    ],
)
def test_arrange_validation(pin_ds, kwargs, match):
    with pytest.raises(ValueError, match=match):
        pin_ds.arrange(**kwargs)


# ---------------------------------------------------------------- build_core_dtypes


def test_build_core_dtypes_empty_without_dimensions():
    assert build_core_dtypes() == {}


def test_build_core_dtypes_base_and_pin_shapes():
    # Dimensions chosen so no two dtypes share a shape (see collision test).
    d = build_core_dtypes(npiny=17, npinx=17, nax=6, nass=9)
    assert d[(1, 6, 9)] == VeraDtype.ASSEMBLY
    assert d[(6,)] == VeraDtype.AXIAL
    assert d[(9,)] == VeraDtype.RADIAL_ASSEMBLY
    assert d[(NUM_NODES, 6, 9)] == VeraDtype.NODAL
    assert d[(NUM_NODES, 9)] == VeraDtype.RADIAL_NODE
    assert d[(1,)] == d[()] == VeraDtype.SCALAR
    assert d[(17, 17, 6, 9)] == VeraDtype.PIN
    assert d[(18, 18, 6, 9)] == VeraDtype.CHANNEL
    assert d[(17, 17, 9)] == VeraDtype.RADIAL
    assert d[(18, 18, 9)] == VeraDtype.CHANNEL_RADIAL


def test_shape_collision_is_resolved_silently_by_insertion_order():
    """Documents a hazard, not a contract. With npin + 1 == NUM_NODES == nax,
    CHANNEL_RADIAL (npin+1, npin+1, nass) and NODAL (NUM_NODES, nax, nass)
    share a shape and the later entry wins without warning. The mock's
    defaults (npin=3, nax=4) hit this."""
    d = build_core_dtypes(npiny=3, npinx=3, nax=NUM_NODES, nass=9)
    assert d[(NUM_NODES, NUM_NODES, 9)] == VeraDtype.CHANNEL_RADIAL


def test_build_core_dtypes_pin_shapes_need_pins():
    d = build_core_dtypes(nax=4, nass=9)
    assert VeraDtype.PIN not in d.values()


def test_build_core_dtypes_group_range_is_inclusive():
    d = build_core_dtypes(npiny=3, npinx=3, nax=4, nass=9, comp_nax=6, comp_nass=12)
    assert d[(3, 3, 6, 12)] == VeraDtype.COMP_PIN
    assert d[(MIN_NUM_GROUPS, 1, 6, 12)] == VeraDtype.COMP_ASSY_ENERGY
    assert d[(MAX_NUM_GROUPS, NUM_NODES, 6, 12)] == VeraDtype.COMP_NODAL_ENERGY
    assert d[(NUM_DF, MAX_NUM_GROUPS, 1, 6, 12)] == VeraDtype.COMP_ASSY_SURFACE
    assert (MIN_NUM_GROUPS - 1, 1, 6, 12) not in d
    assert (MAX_NUM_GROUPS + 1, 1, 6, 12) not in d


def test_build_core_dtypes_detectors():
    d = build_core_dtypes(nax=4, nass=9, ndet=5, ndax=7)
    assert d[(7, 5)] == VeraDtype.POINT_DETECTOR
    assert d[(5,)] == VeraDtype.RADIAL_POINT_DETECTOR
    d = build_core_dtypes(nax=4, nass=9, ndet=5, ndax=7, continous_det=1)
    assert d[(7, 5)] == VeraDtype.CONTINOUS_DETECTOR


def test_radial_point_detector_skipped_when_ndet_equals_nass():
    d = build_core_dtypes(nax=4, nass=9, ndet=9, ndax=4)
    assert d[(9,)] == VeraDtype.RADIAL_ASSEMBLY


# ---------------------------------------------------------------- recipes


def test_derive_recipe_keys():
    r = derive_recipe("a", "pin_powers", "avg", "Average", "AXIAL", True, False)
    assert r["kind"] == "derive"
    assert (r["src_id"], r["source_array"], r["name"], r["axes"]) == (
        "a",
        "pin_powers",
        "avg",
        "AXIAL",
    )


def test_diff_recipe_keys():
    r = diff_recipe("a", "x", "b", "y", "d", 1, 2.0, 3.0, "W")
    assert r == {
        "kind": "diff",
        "ref_src_id": "a",
        "ref_array": "x",
        "comp_src_id": "b",
        "comp_array": "y",
        "name": "d",
        "interp_degree": 1,
        "ref_scale": 2.0,
        "comp_scale": 3.0,
        "units": "W",
    }
