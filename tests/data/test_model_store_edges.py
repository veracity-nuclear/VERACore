"""DatasetStore manifest and precedence rules, and side effects of the
masking helpers."""

import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.model import DatasetStore, _make_ji_safe, nan_out_non_fuel, nan_out_reflected

from .source_support import TypedSource

T = VeraDtype


@pytest.fixture
def src():
    return TypedSource({"a": [1.0]})


@pytest.fixture
def store(src):
    return DatasetStore(src)


# ---------------------------------------------------------------- manifest snapshot


def test_name_added_to_source_later_still_resolves(src, store):
    src.arrays["late"] = np.array([2.0])
    assert "late" in store
    np.testing.assert_array_equal(store.get("late"), [2.0])


def test_name_added_to_source_later_cannot_be_cached(src, store):
    src.arrays["late"] = np.array([2.0])
    with pytest.raises(KeyError):
        store.cache_dataset("late")


def test_cache_all_ignores_names_added_later(src, store):
    src.arrays["late"] = np.array([2.0])
    store.cache_all()
    store.get("late")
    store.get("late")
    assert src.loads["late"] == 2


# ---------------------------------------------------------------- precedence


def test_pin_wins_over_cache(store):
    store.cache_dataset("a")
    pinned = VeraDataset([9.0])
    store.pin("a", pinned)
    assert store.get("a") is pinned


def test_uncache_does_not_remove_pin(store):
    pinned = VeraDataset([9.0])
    store.pin("a", pinned)
    store.uncache_all()
    assert store.get("a") is pinned


def test_pinned_shape_is_the_pinned_arrays_shape(store):
    store.pin("a", VeraDataset(np.zeros((2, 3))))
    assert store.shape("a") == (2, 3)


def test_pin_replaces_previous_pin(store):
    store.pin("x", VeraDataset([1.0]))
    second = VeraDataset([2.0])
    store.pin("x", second)
    assert store.get("x") is second


def test_cached_dataset_is_the_same_object_each_get(store):
    store.cache_dataset("a")
    assert store.get("a") is store.get("a")


def test_lazy_reads_return_independent_objects(store):
    assert store.get("a") is not store.get("a")


# ---------------------------------------------------------------- load returning None


class _NullLoad(TypedSource):
    def load(self, name):
        return None


def test_listed_name_whose_load_fails_returns_fallback():
    store = DatasetStore(_NullLoad({"a": [1.0]}))
    fallback = VeraDataset([0.0])
    assert store.get("a", fallback) is fallback


def test_listed_name_whose_load_fails_is_not_cached():
    store = DatasetStore(_NullLoad({"a": [1.0]}))
    store.cache_dataset("a")
    assert store.get("a") is None


# ---------------------------------------------------------------- mask helpers


def test_nan_out_reflected_mutates_its_input():
    """Documents a contract callers rely on: it masks in place, so
    VeraDataSource.get_dataset copies before calling it."""
    ds = VeraDataset(np.ones((3, 3, 9)), T.RADIAL)
    out = nan_out_reflected(np.arange(1, 10).reshape(3, 3), 4, ds)
    assert out is ds
    assert np.isnan(ds).any()


def test_nan_out_reflected_on_integer_data_raises():
    """NaN cannot be stored in an integer array; callers must pass floats."""
    ds = VeraDataset(np.ones((3, 3, 9), dtype=int), T.RADIAL)
    with pytest.raises(ValueError):
        nan_out_reflected(np.arange(1, 10).reshape(3, 3), 4, ds)


def test_nan_out_reflected_ignores_ids_absent_from_data():
    """Map ids beyond the assembly axis are skipped rather than raising."""
    ds = VeraDataset(np.ones((3, 3, 2)), T.RADIAL)
    out = nan_out_reflected(np.arange(1, 10).reshape(3, 3), 4, ds)
    assert np.isnan(out[0, :, 0]).all()
    assert np.isnan(out[0, :, 1]).all()


def test_nan_out_non_fuel_promotes_integers_and_keeps_units():
    ds = VeraDataset(np.ones((3, 3, 2, 9), dtype=int), T.PIN, "n", "W")
    vols = np.ones((3, 3, 2, 9))
    vols[0, 0] = 0
    out = nan_out_non_fuel(ds, vols)
    assert out.dtype.kind == "f"
    assert out.physical_units == "W"
    assert np.isnan(out).sum() == 2 * 9


def test_nan_out_non_fuel_radial_needs_zero_volume_at_every_level():
    vols = np.ones((3, 3, 2, 9))
    vols[1, 1, 0] = 0  # zero at one level only
    out = nan_out_non_fuel(VeraDataset(np.ones((3, 3, 9)), T.RADIAL), vols)
    assert np.isfinite(out).all()


@pytest.mark.parametrize(
    "j, i, expected", [(-4, 99, (0, 4)), (1, 2, (1, 2)), (10, -1, (2, 0))]
)
def test_make_ji_safe_clips_to_array(j, i, expected):
    assert tuple(int(v) for v in _make_ji_safe(j, i, np.zeros((3, 5)))) == expected
