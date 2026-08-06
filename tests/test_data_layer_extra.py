"""Deeper tests for the data layer.

test_data_layer.py covers the happy paths of each class. This module covers the
vocabulary (VeraDtype / VeraDataset), cross-cutting invariants, parametrized
sweeps over dtype and symmetry, and the operations that combine two sources.
"""

import itertools

import numpy as np
import pytest

from vera_core.data.dtypes import (
    NUM_NODES,
    DerivationMethod,
    Surface,
    VeraAxes,
    VeraDataset,
    VeraDtype,
    build_core_dtypes,
    derive_recipe,
    diff_recipe,
)
from vera_core.data.model import DatasetStore, nan_out_reflected
from vera_core.data.readers.mock import DictDatasetSource, build_core, build_source, state_arrays
from vera_core.data.registry import VeraDataRegistry
from vera_core.data.thresholds import apply_thresholds

ALL_DTYPES = [d for d in VeraDtype if d is not VeraDtype.UNKNOWN]


# --------------------------------------------------------------------------
# VeraDtype vocabulary
# --------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", list(VeraDtype))
def test_every_dtype_has_an_info_entry(dtype):
    """A dtype without an _INFO row raises KeyError the first time it's asked."""
    assert dtype._info is not None


@pytest.mark.parametrize("dtype", list(VeraDtype))
def test_predicates_return_bools(dtype):
    for pred in (
        dtype.is_computational,
        dtype.is_nodal,
        dtype.is_assembly,
        dtype.is_surface,
        dtype.is_channel,
        dtype.is_detector,
        dtype.has_fuel_pins,
        dtype.has_axial_dim,
    ):
        assert isinstance(pred(), bool)


@pytest.mark.parametrize("dtype", list(VeraDtype))
def test_axial_dim_idx_agrees_with_has_axial_dim(dtype):
    if dtype.has_axial_dim():
        assert dtype.axial_dim_idx >= 0
    else:
        with pytest.raises(ValueError):
            _ = dtype.axial_dim_idx


@pytest.mark.parametrize("dtype", list(VeraDtype))
def test_title_is_uppercase_name(dtype):
    assert dtype.title == dtype.name.upper() == str(dtype).upper()


def test_core_is_an_alias_for_scalar():
    assert VeraDtype.CORE is VeraDtype.SCALAR


def test_computational_dtypes_are_named_comp():
    for dtype in VeraDtype:
        if dtype.is_computational():
            assert dtype.name.startswith("COMP_")


def test_surface_enum_covers_six_faces():
    assert len(Surface) == 6
    assert Surface.WEST.str == "WEST"


def test_derivation_methods_are_strings():
    assert DerivationMethod.AVERAGE == "Average"


def test_vera_axes_names_are_unique():
    assert len({a.name for a in VeraAxes}) == len(VeraAxes)


# --------------------------------------------------------------------------
# build_core_dtypes
# --------------------------------------------------------------------------


def test_shape_map_is_empty_without_dimensions():
    assert build_core_dtypes() == {}


def test_shape_map_classifies_pin_and_channel():
    m = build_core_dtypes(npiny=17, npinx=17, nax=49, nass=193)
    assert m[(17, 17, 49, 193)] is VeraDtype.PIN
    assert m[(18, 18, 49, 193)] is VeraDtype.CHANNEL
    assert m[(17, 17, 193)] is VeraDtype.RADIAL


def test_shape_map_includes_both_scalar_spellings():
    m = build_core_dtypes(nax=49, nass=193)
    assert m[()] is VeraDtype.SCALAR
    assert m[(1,)] is VeraDtype.SCALAR


def test_shape_map_adds_computational_entries():
    m = build_core_dtypes(nax=49, nass=193, comp_nax=25, comp_nass=193)
    assert m[(NUM_NODES, 25, 193)] is VeraDtype.COMP_NODAL
    assert m[(1, 25, 193)] is VeraDtype.COMP_ASSY


def test_detector_entries_skipped_when_shapes_collide_with_assemblies():
    """ndet == nass would make detector shapes indistinguishable, so they're omitted."""
    m = build_core_dtypes(nax=49, nass=193, ndet=193, ndax=49)
    assert (193,) not in m or m[(193,)] is VeraDtype.RADIAL_ASSEMBLY


def test_shape_map_values_are_all_dtypes():
    m = build_core_dtypes(npiny=3, npinx=3, nax=4, nass=9, comp_nax=4, comp_nass=9)
    assert all(isinstance(v, VeraDtype) for v in m.values())


# --------------------------------------------------------------------------
# VeraDataset
# --------------------------------------------------------------------------


def test_dataset_defaults_to_unknown_and_unitless():
    ds = VeraDataset(np.zeros(3))
    assert ds.dataset_type is VeraDtype.UNKNOWN
    assert ds.physical_units == "unitless"


def test_dataset_is_an_ndarray():
    ds = VeraDataset(np.arange(4.0), VeraDtype.AXIAL, "W")
    assert isinstance(ds, np.ndarray)
    assert ds.sum() == 6.0


def test_metadata_survives_arithmetic():
    ds = VeraDataset(np.ones(3), VeraDtype.PIN, "W")
    assert (ds * 2).dataset_type is VeraDtype.PIN


def test_metadata_survives_copy():
    ds = VeraDataset(np.ones(3), VeraDtype.PIN, "W")
    copied = ds.copy()
    assert copied.dataset_type is VeraDtype.PIN
    assert copied.physical_units == "W"


def test_dataset_type_is_not_updated_by_slicing():
    """Known sharp edge: the tag rides along even when the shape stops matching.

    Callers must capture dataset_type before transforming, not read it after.
    """
    ds = VeraDataset(np.zeros((3, 3, 4, 9)), VeraDtype.PIN, "W")
    sliced = ds[:, :, 0]
    assert sliced.shape == (3, 3, 9)
    assert sliced.dataset_type is VeraDtype.PIN  # documents current behavior


def test_dataset_convenience_predicates():
    assert VeraDataset(np.zeros(1), VeraDtype.COMP_NODAL).is_computational()
    assert VeraDataset(np.zeros(1), VeraDtype.ASSEMBLY).is_assembly()


# --------------------------------------------------------------------------
# Parametrized sweeps -- geometry across configurations
# --------------------------------------------------------------------------


@pytest.mark.parametrize("core_sym", [1, 4])
@pytest.mark.parametrize("nass_side", [3, 5, 7])
def test_core_builds_across_symmetry_and_size(core_sym, nass_side):
    core = build_core(nass_side=nass_side, core_sym=core_sym)
    expected = nass_side if core_sym == 1 else nass_side // 2 + 1
    assert core.reduced_core_map.shape == (expected, expected)
    assert core.nass > 0
    assert len(core.reduced_core_map_row_labels) == expected


@pytest.mark.parametrize("nax", [1, 2, 10, 49])
def test_axial_mesh_derivatives_scale_with_nax(nax):
    core = build_core(nax=nax)
    assert len(core.axial_mesh) == nax + 1
    assert len(core.axial_mesh_means) == nax
    assert len(core.axial_mesh_pixels) == nax


@pytest.mark.parametrize("npin", [1, 3, 17])
def test_lattice_size_round_trips(npin):
    core = build_core(npin=npin)
    assert core.core_shape[0] == core.core_shape[1] == npin


@pytest.mark.parametrize(
    "flags",
    list(itertools.product([False, True], repeat=3)),
    ids=lambda f: f"comp{int(f[0])}_det{int(f[1])}_lbl{int(f[2])}",
)
def test_optional_core_features_are_independent(flags):
    comp, det, labels = flags
    core = build_core(with_comp_core=comp, with_detectors=det, labels=labels)
    assert core.has_comp_core() is comp
    assert (core.ndet is not None) is det
    assert core.reduced_core_map is not None


@pytest.mark.parametrize("n_states", [1, 2, 5])
def test_source_builds_for_any_state_count(n_states):
    src = build_source(n_states)
    assert len(src.states) == n_states
    assert src.time_axes()["state_count"] == list(range(n_states))
    for idx in range(n_states):
        assert src.get_dataset("pin_powers", state_idx=idx).shape == src.core.core_shape


@pytest.mark.parametrize(
    "name", ["pin_powers", "pin_radial_powers", "assembly_powers", "axial_powers", "keff"]
)
def test_every_state_dataset_is_reachable(name):
    src = build_source(2)
    assert name in src.states[0]
    assert src.get_dataset(name) is not None
    assert src.get_dataset_dtype(name) is not VeraDtype.UNKNOWN


@pytest.mark.parametrize("mask", [True, False])
@pytest.mark.parametrize("core_sym", [1, 4])
def test_get_dataset_shape_is_independent_of_masking(mask, core_sym):
    src = build_source(1, core_kwargs={"core_sym": core_sym})
    arr = src.get_dataset("pin_powers", mask_reflected=mask)
    assert arr.shape == src.core.core_shape


# --------------------------------------------------------------------------
# Invariants
# --------------------------------------------------------------------------


def test_core_shape_matches_pin_dataset_shape():
    src = build_source(1)
    assert src.get_dataset("pin_powers").shape == src.core.core_shape


def test_reduced_map_ids_are_within_assembly_count():
    core = build_core()
    ids = core.reduced_core_map[core.reduced_core_map > 0]
    assert ids.max() <= core.nass


def test_every_assembly_id_resolves_to_a_position():
    core = build_core()
    for idx in np.unique(core.reduced_core_map[core.reduced_core_map > 0]) - 1:
        i, j = core.reduced_core_map_ij(int(idx))
        assert core.reduced_core_map[j, i] == idx + 1


def test_gross_axial_mesh_covers_the_core_mesh():
    core = build_core()
    assert set(np.round(core.axial_mesh_means, 6)) <= set(np.round(core.gross_axial_mesh, 6))


def test_axial_mesh_means_are_monotonic():
    core = build_core(nax=10)
    assert np.all(np.diff(core.axial_mesh_means) > 0)


def test_masking_only_adds_nans():
    src = build_source(1, core_kwargs={"core_sym": 4})
    raw = src.get_dataset("pin_powers", mask_reflected=False)
    masked = src.get_dataset("pin_powers", mask_reflected=True)
    kept = ~np.isnan(masked)
    assert np.allclose(masked[kept], raw[kept])


def test_masking_is_idempotent():
    src = build_source(1, core_kwargs={"core_sym": 4})
    once = src.get_dataset("pin_powers")
    twice = nan_out_reflected(src.core.reduced_core_map, src.core.core_sym, once.copy())
    assert np.array_equal(np.isnan(once), np.isnan(twice))


def test_reading_a_state_does_not_disturb_its_neighbours():
    src = build_source(3)
    before = [src.get_dataset("pin_powers", state_idx=i).copy() for i in range(3)]
    src.get_dataset("pin_powers", state_idx=1)
    after = [src.get_dataset("pin_powers", state_idx=i) for i in range(3)]
    assert all(np.array_equal(b, a, equal_nan=True) for b, a in zip(before, after, strict=False))


def test_state_caching_does_not_change_values():
    cached = build_source(2, state_caching=True).get_dataset("pin_powers", state_idx=1)
    lazy = build_source(2, state_caching=False).get_dataset("pin_powers", state_idx=1)
    assert np.allclose(cached, lazy, equal_nan=True)


# --------------------------------------------------------------------------
# Diffs
# --------------------------------------------------------------------------


def test_diff_of_a_source_against_itself_is_zero():
    a, b = build_source(2), build_source(2)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "zero")
    assert np.allclose(np.nan_to_num(a.get_dataset("zero")), 0.0)


def test_diff_applies_scales():
    a, b = build_source(1), build_source(1)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "scaled", ref_scale=3.0, comp_scale=1.0)
    raw = a.get_dataset("pin_powers", mask_reflected=False)
    diff = a.get_dataset("scaled", mask_reflected=False)
    assert np.allclose(diff, raw * 2.0)


def test_diff_sets_units():
    a, b = build_source(1), build_source(1)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d", units="W")
    assert a.get_dataset_units("d") == "W"


def test_diff_appears_in_state_listing():
    a, b = build_source(1), build_source(1)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert "d" in a.states[0].full_core_keys


def test_diff_covers_only_overlapping_states():
    a, b = build_source(3), build_source(1)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert "d" in a.states[0]
    assert "d" not in a.states[2]


def test_diff_survives_cache_cycling():
    """Diffs are pinned, so evicting the file-backed cache must not drop them."""
    a, b = build_source(1), build_source(1)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    a.states[0].cache_all()
    a.states[0].uncache_all()
    assert a.get_dataset("d") is not None


def test_diff_between_mismatched_dtypes_is_skipped():
    a, b = build_source(1), build_source(1)
    with pytest.raises(ValueError):
        a.add_new_diff_dataset("pin_powers", b, "axial_powers", "d")


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "op,value,kept",
    [
        (">", 0.5, [False, False, True]),
        (">=", 0.5, [False, True, True]),
        ("<", 0.5, [True, False, False]),
        ("<=", 0.5, [True, True, False]),
        ("==", 0.5, [False, True, False]),
        ("!=", 0.5, [True, False, True]),
    ],
)
def test_every_threshold_operator(op, value, kept):
    arr = VeraDataset(np.array([0.0, 0.5, 1.0]), VeraDtype.PIN)
    out = apply_thresholds(arr, [{"op": op, "value": value}])
    assert list(~np.isnan(out)) == kept


def test_thresholds_do_not_mutate_the_input():
    arr = VeraDataset(np.array([0.1, 0.9]), VeraDtype.PIN)
    original = arr.copy()
    apply_thresholds(arr, [{"op": ">", "value": 0.5}])
    assert np.array_equal(arr, original)


def test_thresholds_preserve_shape():
    arr = VeraDataset(np.zeros((3, 4, 5)), VeraDtype.PIN)
    assert apply_thresholds(arr, [{"op": ">", "value": 1.0}]).shape == (3, 4, 5)


def test_existing_nans_stay_nan():
    arr = VeraDataset(np.array([np.nan, 1.0]), VeraDtype.PIN)
    out = apply_thresholds(arr, [{"op": ">", "value": 0.0}])
    assert np.isnan(out[0])


def test_impossible_threshold_nans_everything():
    arr = VeraDataset(np.array([0.1, 0.9]), VeraDtype.PIN)
    out = apply_thresholds(arr, [{"op": ">", "value": 1.0}, {"op": "<", "value": 0.0}])
    assert np.isnan(out).all()


# --------------------------------------------------------------------------
# Recipes
# --------------------------------------------------------------------------


def test_derive_recipe_round_trips_through_the_registry():
    reg = VeraDataRegistry()
    reg.add_src(build_source(1), "a")
    recipe = derive_recipe("a", "pin_powers", "avg", "Average", "CORE", True, False)
    assert recipe["kind"] == "derive"
    with pytest.raises(RuntimeError):  # no calculator on an in-memory source
        reg.apply_recipe(recipe)


def test_diff_recipe_applies_through_the_registry():
    reg = VeraDataRegistry()
    reg.add_src(build_source(2), "a")
    reg.add_src(build_source(2), "b")
    reg.apply_recipe(
        diff_recipe("a", "pin_powers", "b", "pin_powers", "d", 1, 1.0, 1.0, "unitless")
    )
    assert "d" in reg.get("a").states[0]


def test_unknown_recipe_kind_raises():
    reg = VeraDataRegistry()
    with pytest.raises(ValueError, match="Unknown recipe kind"):
        reg.apply_recipe({"kind": "nonsense"})


# --------------------------------------------------------------------------
# Registry, further
# --------------------------------------------------------------------------


def test_registry_contains_operator():
    reg = VeraDataRegistry()
    reg.add_src(build_source(1), "a")
    assert "a" in reg and "b" not in reg


def test_global_axial_mesh_unions_across_sources():
    reg = VeraDataRegistry()
    reg.add_src(build_source(1, core_kwargs={"nax": 4}), "a")
    before = len(reg.global_axial_mesh)
    reg.add_src(build_source(1, core_kwargs={"nax": 7}), "b")
    assert len(reg.global_axial_mesh) >= before


def test_get_ds_dtype_for_unknown_source():
    reg = VeraDataRegistry()
    assert reg.get_ds_dtype("nope", "pin_powers") is VeraDtype.UNKNOWN


def test_time_axis_value_reads_the_longest_source():
    reg = VeraDataRegistry()
    reg.add_src(build_source(exposures=[0.0, 5.0, 9.0]), "a")
    assert reg.time_axis_value("exposure", 1) == pytest.approx(5.0)
    assert reg.time_axis_value("state_count", 2) == 2.0


def test_removing_the_last_source_clears_the_default():
    reg = VeraDataRegistry()
    reg.add_src(build_source(1), "a")
    reg.remove_src("a")
    assert reg.default_src_id is None
    assert not reg.has_src()


def test_src_ids_reflect_additions_and_removals():
    reg = VeraDataRegistry()
    for name in ("a", "b", "c"):
        reg.add_src(build_source(1), name)
    reg.remove_src("b")
    assert set(reg.src_ids()) == {"a", "c"}


# --------------------------------------------------------------------------
# Store edge cases
# --------------------------------------------------------------------------


def test_store_over_an_empty_source():
    store = DatasetStore(DictDatasetSource({}))
    assert store.get("anything") is None
    assert "anything" not in store
    store.cache_all()
    store.uncache_all()


def test_pin_accepts_a_name_the_source_does_not_have():
    store = DatasetStore(DictDatasetSource({}))
    ds = VeraDataset(np.ones(2), VeraDtype.AXIAL)
    store.pin("derived", ds)
    assert store["derived"] is ds


def test_zero_valued_scalar_is_not_treated_as_missing():
    """Guards against truthiness checks on arrays: 0.0 is a value, not absence."""
    store = DatasetStore(DictDatasetSource({"zero": np.array([0.0])}))
    assert store.get("zero") is not None
    assert "zero" in store


def test_empty_array_is_retrievable():
    store = DatasetStore(DictDatasetSource({"empty": np.zeros(0)}))
    assert store.get("empty") is not None
    assert store.get("empty").size == 0


def test_state_arrays_helper_matches_core_shape():
    core = build_core()
    arrays = state_arrays(core, 0.0)
    assert arrays["pin_powers"].shape == core.core_shape
