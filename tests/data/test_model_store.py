import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.model import DatasetStore, VeraOutState
from vera_core.data.readers.mock import DictDatasetSource, build_core, state_arrays


@pytest.fixture
def backend():
    return DictDatasetSource({"a": np.arange(3.0), "b": np.array(5.0)})


@pytest.fixture
def store(backend):
    return DatasetStore(backend)


# ---------------------------------------------------------------- DatasetStore


def test_get_reads_lazily_each_time(store, backend):
    store.get("a")
    store.get("a")
    assert backend.load_count["a"] == 2


def test_cached_dataset_is_read_once(store, backend):
    store.cache_dataset("a")
    store.get("a")
    store.get("a")
    assert backend.load_count["a"] == 1


def test_uncache_reverts_to_lazy(store, backend):
    store.cache_dataset("a")
    store.uncache_dataset("a")
    store.get("a")
    assert backend.load_count["a"] == 2


def test_uncache_unknown_name_is_noop(store):
    store.uncache_dataset("missing")


def test_cache_all_and_uncache_all(store, backend):
    store.cache_all()
    store.get("a"), store.get("b")
    store.uncache_all()
    store.get("a")
    assert backend.load_count == {"a": 2, "b": 1}


def test_cache_dataset_rejects_name_outside_manifest(store):
    with pytest.raises(KeyError):
        store.cache_dataset("missing")


def test_pinned_dataset_wins_over_source(store, backend):
    pinned = VeraDataset([9.0])
    store.pin("a", pinned)
    assert store.get("a") is pinned
    assert "a" not in backend.load_count


def test_pinned_name_is_contained(store):
    store.pin("extra", VeraDataset([1.0]))
    assert "extra" in store


def test_get_missing_returns_fallback(store):
    fallback = VeraDataset([0.0])
    assert store.get("missing") is None
    assert store.get("missing", fallback) is fallback


def test_getitem_missing_raises_key_error(store):
    with pytest.raises(KeyError):
        store["missing"]


def test_getitem_returns_dataset(store):
    np.testing.assert_array_equal(store["a"], [0.0, 1.0, 2.0])


def test_scalar_reported_and_loaded_as_length_one(store):
    assert store.shape("b") == (1,)
    assert store["b"].shape == (1,)


def test_shape_missing_is_none(store):
    assert store.shape("missing") is None


def test_provenance_and_source_pass_through(store, backend):
    assert store.provenance == "<memory>"
    assert store.source is backend


# ---------------------------------------------------------------- VeraOutState


@pytest.fixture
def core():
    return build_core()


@pytest.fixture
def state(core):
    arrays = state_arrays(core, 10.0) | {"odd_shape": np.zeros((7, 7, 7))}
    return VeraOutState(DictDatasetSource(arrays, dataset_dtypes=core.shape_to_dtype), 0, core)


def test_state_categorizes_by_shape(state):
    cats = state.categorized_ds_names
    assert cats[VeraDtype.PIN] == {"pin_powers"}
    assert cats[VeraDtype.RADIAL] == {"pin_radial_powers"}
    assert cats[VeraDtype.ASSEMBLY] == {"assembly_powers"}
    assert cats[VeraDtype.AXIAL] == {"axial_powers"}
    assert cats[VeraDtype.SCALAR] == {"exposure", "keff"}


def test_unknown_shapes_are_not_categorized(state):
    assert "odd_shape" not in state.full_core_keys
    assert not state.categorized_ds_names[VeraDtype.UNKNOWN]


def test_scalar_datasets_sorted(state):
    assert state.scalar_datasets == ["exposure", "keff"]


def test_grouped_keys_follow_enum_order_and_skip_empty(state):
    assert [group for group, _ in state.grouped_full_core_keys] == [
        "PIN",
        "ASSEMBLY",
        "AXIAL",
        "RADIAL",
        "SCALAR",
    ]


def test_full_core_keys_flattens_groups(state):
    assert state.full_core_keys == [
        "pin_powers",
        "assembly_powers",
        "axial_powers",
        "pin_radial_powers",
        "exposure",
        "keff",
    ]


def test_state_exposes_core(state, core):
    assert state.core is core


@pytest.mark.parametrize("method, registry", [
    ("add_diff_dataset", "diff_datasets"),
    ("add_derived_dataset", "derived_datasets"),
])
def test_added_dataset_is_pinned_tracked_and_categorized(state, method, registry):
    ds = VeraDataset(np.zeros(4), VeraDtype.AXIAL)
    getattr(state, method)("new", ds)
    assert state.get("new") is ds
    assert getattr(state, registry) == {"new": VeraDtype.AXIAL}
    assert "new" in state.categorized_ds_names[VeraDtype.AXIAL]


@pytest.mark.parametrize("method", ["add_diff_dataset", "add_derived_dataset"])
def test_added_unknown_dataset_is_pinned_but_not_categorized(state, method):
    getattr(state, method)("new", VeraDataset([1.0]))
    assert "new" in state
    assert "new" not in state.full_core_keys
