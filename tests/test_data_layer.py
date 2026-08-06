"""Tests for the data layer.

Everything here runs against in-memory sources (readers/mock.py), so no HDF5
file is needed and cases the sample files don't cover -- full-core symmetry,
missing core properties, absent optional datasets -- are reachable.
"""

import pathlib

import numpy as np
import pytest

from vera_core.data.dtypes import VeraAxes, VeraDataset, VeraDtype
from vera_core.data.model import CorePropMissing, DatasetStore, VeraOutCore, nan_out_reflected
from vera_core.data.readers.mock import (
    DictDatasetSource,
    build_core,
    build_source,
    core_arrays,
)
from vera_core.data.registry import VeraDataRegistry
from vera_core.data.thresholds import apply_thresholds

# --------------------------------------------------------------------------
# DatasetSource contract
# --------------------------------------------------------------------------


def test_source_reports_names_and_shapes():
    src = DictDatasetSource({"a": np.zeros((2, 3)), "b": np.array([1.0])})
    assert set(src.names()) == {"a", "b"}
    assert src.shape("a") == (2, 3)
    assert src.has("a")


def test_source_returns_none_for_unknown_name():
    src = DictDatasetSource({"a": np.zeros(2)})
    assert src.shape("missing") is None
    assert src.load("missing") is None
    assert not src.has("missing")


def test_scalars_report_shape_1():
    """Matches H5DatasetSource: a 0-d dataset is exposed as shape (1,)."""
    src = DictDatasetSource({"keff": np.array(1.05)})
    assert src.shape("keff") == (1,)
    assert src.load("keff").shape == (1,)


def test_load_applies_dtype_from_shape_map():
    src = DictDatasetSource({"x": np.zeros((3, 4))}, dataset_dtypes={(3, 4): VeraDtype.RADIAL})
    assert src.load("x").dataset_type is VeraDtype.RADIAL


def test_load_defaults_to_unknown_dtype():
    src = DictDatasetSource({"x": np.zeros((3, 4))})
    assert src.load("x").dataset_type is VeraDtype.UNKNOWN


# --------------------------------------------------------------------------
# DatasetStore
# --------------------------------------------------------------------------


@pytest.fixture
def store():
    return DatasetStore(DictDatasetSource({"a": np.zeros((2, 2)), "b": np.ones(3)}))


def test_store_get_and_getitem_agree(store):
    assert np.array_equal(store.get("a"), store["a"])


def test_store_getitem_raises_on_missing(store):
    with pytest.raises(KeyError):
        store["nope"]


def test_store_get_returns_fallback_on_missing(store):
    sentinel = VeraDataset(np.array([9.0]))
    assert store.get("nope") is None
    assert store.get("nope", sentinel) is sentinel


def test_store_contains_matches_get(store):
    """`in` and `get` must not disagree -- callers do `if x in s: s.get(x)`."""
    for name in ("a", "b", "nope"):
        assert (name in store) == (store.get(name) is not None)


def test_uncached_get_rereads_source(store):
    store.get("a")
    store.get("a")
    assert store._source.load_count["a"] == 2


def test_cached_get_reads_once(store):
    store.cache_dataset("a")
    before = store._source.load_count["a"]
    store.get("a")
    store.get("a")
    assert store._source.load_count["a"] == before


def test_uncache_reverts_to_lazy(store):
    store.cache_dataset("a")
    store.uncache_dataset("a")
    store.get("a")
    assert store._source.load_count["a"] == 2


def test_cache_all_then_uncache_all(store):
    store.cache_all()
    assert len(store._cache) == 2
    store.uncache_all()
    assert store._cache == {}


def test_cache_dataset_rejects_unknown_name(store):
    with pytest.raises(KeyError):
        store.cache_dataset("nope")


def test_pinned_dataset_survives_uncache_all(store):
    pinned = VeraDataset(np.array([7.0]), VeraDtype.SCALAR)
    store.pin("derived", pinned)
    store.cache_all()
    store.uncache_all()
    assert store.get("derived") is pinned
    assert "derived" in store


def test_pinned_takes_precedence_over_source(store):
    override = VeraDataset(np.full((2, 2), 5.0))
    store.pin("a", override)
    assert np.all(store.get("a") == 5.0)


def test_store_exposes_provenance():
    store = DatasetStore(DictDatasetSource({"a": np.zeros(1)}, provenance="here"))
    assert store.provenance == "here"


# --------------------------------------------------------------------------
# VeraOutCore -- geometry
# --------------------------------------------------------------------------


def test_quarter_core_reduces_map():
    core = build_core(nass_side=5, core_sym=4)
    assert core.core_sym == 4
    assert core.reduced_core_map.shape == (3, 3)
    assert core.reduced_core_map_start_index == 2


def test_full_core_keeps_whole_map():
    core = build_core(nass_side=5, core_sym=1)
    assert core.reduced_core_map.shape == (5, 5)
    assert core.reduced_core_map_start_index == 0


def test_symmetry_inferred_when_absent():
    """Repeated assembly ids mean a mirrored (quarter) map."""
    assert build_core(core_sym=None, map_sym=4).core_sym == 4
    assert build_core(core_sym=None, map_sym=1).core_sym == 1


def test_core_shape_from_pin_volumes():
    core = build_core(nass_side=5, core_sym=4, nax=6, npin=3)
    npy, npx, nax, nass = core.core_shape
    assert (npy, npx, nax) == (3, 3, 6)


def test_missing_core_map_raises():
    arrays = core_arrays()
    del arrays["core_map"]
    with pytest.raises(RuntimeError, match="core_map"):
        VeraOutCore(DictDatasetSource(arrays))


def test_missing_lattice_raises_core_prop_missing():
    with pytest.raises(CorePropMissing) as exc:
        build_core(npin=None, with_axial_mesh=False, with_pin_volumes=False)
    assert "npin" in exc.value.missing
    assert "nax" in exc.value.missing
    assert "nass" in exc.value.inferred


def test_overrides_supply_missing_lattice():
    core = build_core(
        npin=None,
        with_axial_mesh=False,
        with_pin_volumes=False,
        overrides={"npin": 17, "nax": 6},
    )
    assert core.npy == core.npx == 17
    assert core.nax == 6
    assert core.has_axial_mesh()


def test_overrides_outrank_discovered_values():
    core = build_core(npin=3, overrides={"npin": 21})
    assert core.npy == 21


def test_axial_mesh_fallback_when_absent():
    core = build_core(with_axial_mesh=False)
    assert core.has_axial_mesh()
    assert len(core.axial_mesh) > 1


def test_axial_mesh_means_are_midpoints():
    core = build_core(nax=4)
    expected = (core.axial_mesh[:-1] + core.axial_mesh[1:]) / 2
    assert np.allclose(core.axial_mesh_means, expected)


def test_axial_mesh_pixels_are_positive_ints():
    core = build_core(nax=4)
    assert core.axial_mesh_pixels.dtype.kind == "i"
    assert (core.axial_mesh_pixels > 0).all()


def test_non_fuel_locs_found_from_pin_volumes():
    core = build_core()
    assert core.non_fuel_locs is not None
    assert len(core.non_fuel_locs[0]) > 0


def test_non_fuel_locs_none_without_pin_volumes():
    core = build_core(with_pin_volumes=False)
    assert core.non_fuel_locs is None


def test_aspect_ratio_defaults_to_one():
    assert float(build_core().aspect_ratio) == 1.0


def test_aspect_ratio_read_from_source():
    assert float(build_core(aspect_ratio=0.5).aspect_ratio) == 0.5


def test_pin_pitch_from_apitch():
    core = build_core(npin=3, apitch=6.0)
    assert core.pin_pitch == pytest.approx(2.0)


def test_pin_pitch_default_without_apitch():
    assert build_core().pin_pitch > 0


def test_labels_default_to_letters_and_numbers():
    core = build_core(nass_side=5, core_sym=4, labels=False)
    assert len(core.reduced_core_map_column_labels) == 3
    assert len(core.reduced_core_map_row_labels) == 3


def test_labels_read_from_source_are_decoded():
    core = build_core(nass_side=5, core_sym=4, labels=True)
    assert all(isinstance(x, str) for x in core.reduced_core_map_column_labels)


def test_computational_core_absent_by_default():
    assert not build_core().has_comp_core()


def test_computational_core_present_when_supplied():
    core = build_core(with_comp_core=True)
    assert core.has_comp_core()
    assert core.comp_nax is not None


def test_detectors_absent_by_default():
    assert build_core().ndet is None


def test_detectors_present_when_supplied():
    core = build_core(with_detectors=True)
    assert core.ndet > 0
    assert core.det_axial_mesh_means is not None


def test_core_dtypes_classifies_by_shape():
    core = build_core()
    assert core.core_dtypes(core.core_shape) is VeraDtype.PIN
    assert core.core_dtypes((999, 999)) is VeraDtype.UNKNOWN


def test_assembly_label_round_trips():
    core = build_core()
    idx = int(core.reduced_core_map[core.reduced_core_map > 0][0]) - 1
    i, j = core.reduced_core_map_ij(idx)
    assert core.reduced_core_map_assembly(i, j) == idx
    assert "-" in core.reduced_core_map_label(idx)


def test_row_and_col_indices_include_the_assembly():
    core = build_core()
    idx = int(core.reduced_core_map[core.reduced_core_map > 0][0]) - 1
    assert idx in core.row_assembly_indices(idx)
    assert idx in core.col_assembly_indices(idx)


# --------------------------------------------------------------------------
# Masking
# --------------------------------------------------------------------------


def test_reflected_region_masked_on_quarter_core():
    core = build_core(core_sym=4)
    arr = VeraDataset(np.ones(core.core_shape), VeraDtype.PIN)
    out = nan_out_reflected(core.reduced_core_map, core.core_sym, arr)
    assert np.isnan(out).any()


def test_no_masking_on_full_core():
    core = build_core(core_sym=1)
    arr = VeraDataset(np.ones(core.core_shape), VeraDtype.PIN)
    out = nan_out_reflected(core.reduced_core_map, core.core_sym, arr)
    assert not np.isnan(out).any()


# --------------------------------------------------------------------------
# VeraOutState
# --------------------------------------------------------------------------


def test_state_categorizes_datasets_by_shape():
    src = build_source(1)
    state = src.states[0]
    assert "pin_powers" in state.categorized_ds_names[VeraDtype.PIN]
    assert "keff" in state.categorized_ds_names[VeraDtype.SCALAR]


def test_full_core_keys_are_flat_and_nonempty():
    state = build_source(1).states[0]
    keys = state.full_core_keys
    assert "pin_powers" in keys
    assert len(keys) == len(set(keys))


def test_state_holds_core_reference():
    src = build_source(1)
    assert src.states[0].core is src.core


# --------------------------------------------------------------------------
# VeraDataSource
# --------------------------------------------------------------------------


def test_states_are_ordered_and_counted():
    src = build_source(4)
    assert len(src.states) == 4


def test_active_state_defaults_to_first():
    assert build_source(3).active_state_index == 0


def test_active_state_index_clamps():
    src = build_source(3)
    src.active_state_index = 99
    assert src.active_state_index == 2
    src.active_state_index = -5
    assert src.active_state_index == 0


def test_get_dataset_returns_expected_shape():
    src = build_source(1)
    assert src.get_dataset("pin_powers").shape == src.core.core_shape


def test_get_dataset_unknown_name_raises():
    with pytest.raises(RuntimeError, match="Could not find"):
        build_source(1).get_dataset("not_a_dataset")


def test_get_dataset_reads_requested_state():
    src = build_source(3)
    a = src.get_dataset("pin_powers", state_idx=0)
    b = src.get_dataset("pin_powers", state_idx=2)
    assert not np.allclose(np.nansum(a), np.nansum(b))


def test_get_dataset_state_idx_out_of_range_raises():
    src = build_source(2)
    with pytest.raises(IndexError):
        src.get_dataset("pin_powers", state_idx=5)
    with pytest.raises(IndexError):
        src.get_dataset("pin_powers", state_idx=-1)


def test_get_dataset_does_not_change_active_state():
    src = build_source(3)
    src.get_dataset("pin_powers", state_idx=2)
    assert src.active_state_index == 0


def test_get_dataset_returns_owned_copy():
    """Callers may write to the result without corrupting anything cached."""
    src = build_source(1)
    a = src.get_dataset("pin_powers")
    a[:] = np.nan
    b = src.get_dataset("pin_powers")
    assert not np.isnan(b).all()


def test_mask_reflected_can_be_disabled():
    src = build_source(1, core_kwargs={"core_sym": 4})
    masked = src.get_dataset("pin_powers", mask_reflected=True)
    raw = src.get_dataset("pin_powers", mask_reflected=False)
    assert np.isnan(masked).sum() > np.isnan(raw).sum()


def test_core_arrays_resolve_against_the_core():
    src = build_source(1)
    assert src.get_dataset("pin_volumes").shape == src.core.core_shape


def test_dataset_dtype_and_units_and_shape():
    src = build_source(1)
    assert src.get_dataset_dtype("pin_powers") is VeraDtype.PIN
    assert src.get_dataset_units("pin_powers") == "W"
    assert src.get_dataset_shape("pin_powers") == src.core.core_shape


def test_dtype_of_unknown_name_is_unknown():
    src = build_source(1)
    assert src.get_dataset_dtype("nope") is VeraDtype.UNKNOWN
    assert src.get_dataset_shape("nope") == ()


def test_time_axes_include_exposure_and_state_count():
    src = build_source(3)
    axes = src.time_axes()
    assert axes["state_count"] == [0, 1, 2]
    assert axes["exposure"] == [0.0, 10.0, 20.0]


def test_non_monotonic_time_axis_is_dropped():
    src = build_source(exposures=[0.0, 30.0, 10.0])
    assert "exposure" not in src.time_axes()


def test_default_datasets_prefers_pin_powers():
    src = build_source(1)
    assert src.default_datasets()[VeraDtype.PIN.title] == "pin_powers"


def test_derivation_raises_without_calculator():
    """Derivation needs pyvera, which needs a file; in-memory sources can't."""
    src = build_source(1)
    assert src.vera_calculator is None
    with pytest.raises(RuntimeError, match="derivation"):
        src.add_new_derived_dataset("pin_powers", "avg", None, VeraAxes.CORE)
    assert "avg" not in src.states[0]


def test_diff_works_without_calculator():
    """Diffs use only numpy and the core mesh, so they work for any source."""
    a, b = build_source(2), build_source(2)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "delta")
    assert "delta" in a.states[0]
    assert a.get_dataset("delta").shape == a.core.core_shape


def test_diff_with_no_overlap_raises():
    a, b = build_source(2), build_source(2)
    with pytest.raises(ValueError, match="No overlapping"):
        a.add_new_diff_dataset("not_there", b, "not_there", "delta")


def test_close_is_safe_without_a_callback():
    build_source(1).close()


# --------------------------------------------------------------------------
# Thresholds
# --------------------------------------------------------------------------


def test_threshold_nans_out_values_below_cutoff():
    arr = VeraDataset(np.array([0.1, 0.5, 0.9]), VeraDtype.PIN)
    out = apply_thresholds(arr, [{"op": ">", "value": 0.4}])
    assert np.isnan(out[0])
    assert out[2] == pytest.approx(0.9)


def test_thresholds_combine_as_and():
    arr = VeraDataset(np.array([0.1, 0.5, 0.9]), VeraDtype.PIN)
    out = apply_thresholds(arr, [{"op": ">", "value": 0.2}, {"op": "<", "value": 0.8}])
    assert np.isnan(out[0]) and np.isnan(out[2])
    assert out[1] == pytest.approx(0.5)


def test_empty_thresholds_keep_everything():
    arr = VeraDataset(np.array([0.1, 0.5]), VeraDtype.PIN)
    assert np.allclose(apply_thresholds(arr, []), arr)


def test_thresholds_preserve_dtype():
    arr = VeraDataset(np.array([1.0]), VeraDtype.RADIAL)
    assert apply_thresholds(arr, []).dataset_type is VeraDtype.RADIAL


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


@pytest.fixture
def registry():
    reg = VeraDataRegistry()
    reg.add_src(build_source(3), "a")
    return reg


def test_empty_registry_has_no_default(registry=None):
    reg = VeraDataRegistry()
    assert reg.default_src is None
    assert not reg.has_src()
    assert reg.max_state == 0


def test_first_source_becomes_default(registry):
    assert registry.default_src_id == "a"
    assert registry.has_src()


def test_duplicate_src_id_rejected(registry):
    with pytest.raises(ValueError):
        registry.add_src(build_source(1), "a")


def test_max_state_spans_all_sources(registry):
    registry.add_src(build_source(5), "b")
    assert registry.max_state == 4


def test_get_unknown_src_returns_none(registry):
    assert registry.get("nope") is None
    assert registry.get(None) is None


def test_remove_source_reassigns_default(registry):
    registry.add_src(build_source(1), "b")
    registry.remove_src("a")
    assert registry.default_src_id == "b"
    assert "a" not in registry


def test_remove_unknown_src_raises(registry):
    with pytest.raises(ValueError):
        registry.remove_src("nope")


def test_clear_empties_registry(registry):
    registry.clear()
    assert not registry.has_src()
    assert list(registry.src_ids()) == []


def test_all_sources_maps_id_to_provenance(registry):
    assert registry.all_sources() == {"a": "<memory>"}


def test_change_active_state_applies_to_all(registry):
    registry.add_src(build_source(3), "b")
    registry.change_all_active_state(2)
    assert all(registry.get(i).active_state_index == 2 for i in ("a", "b"))


def test_change_active_state_unknown_src_raises(registry):
    with pytest.raises(ValueError):
        registry.change_active_state("nope", 1)


def test_shared_time_axes_are_intersected(registry):
    registry.add_src(build_source(2), "b")
    shared = registry.shared_time_axes()
    assert "state_count" in shared


def test_axial_index_round_trips(registry):
    core = registry.get("a").core
    z = core.axial_mesh_means[1]
    assert (
        registry.global_axial_idx_to_src_idx("a", VeraDtype.PIN, registry.get_axial_index(z)) == 1
    )


def test_time_axis_value_falls_back_to_index():
    reg = VeraDataRegistry()
    assert reg.time_axis_value("exposure", 3) == 3.0


# --------------------------------------------------------------------------
# Layering
# --------------------------------------------------------------------------


DATA_ROOT = pathlib.Path(__file__).resolve().parents[1] / "vera_core" / "data"

# Directories exempt from the layering rules below.
#   readers    -- the storage boundary; knowing the format is their job
#   vera_tools -- vendored third-party (pyvera), not ours to restructure
EXEMPT_DIRS = {"readers", "vera_tools"}


def _modules_mentioning(token: str) -> list[str]:
    """Data-layer modules containing `token`, excluding exempt directories."""
    return sorted(
        path.relative_to(DATA_ROOT).as_posix()
        for path in DATA_ROOT.rglob("*.py")
        if not EXEMPT_DIRS.intersection(path.relative_to(DATA_ROOT).parts[:-1])
        and token in path.read_text()
    )


def test_no_h5py_outside_readers():
    """Only the readers may know the storage format."""
    assert _modules_mentioning("h5py") == []


def test_no_trame_in_data_layer():
    """The data layer must import without the UI framework installed."""
    assert _modules_mentioning("trame") == []
