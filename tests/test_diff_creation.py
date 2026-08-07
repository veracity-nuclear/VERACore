"""Tests for VeraDataSource.add_new_diff_dataset.

Its own module because the function has four guards, three subtraction paths
(scalar, matching mesh, interpolated mesh) and an atomicity contract -- enough
surface that folding it into the general suite would bury it.
"""

import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.model import VeraOutState
from vera_core.data.readers.mock import DictDatasetSource, build_source

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def set_scalar(source, name, values):
    """Pin a known scalar onto each state so diffs have predictable arithmetic."""
    for state, value in zip(source.states, values, strict=False):
        state.pin(name, VeraDataset(np.array([value]), VeraDtype.SCALAR, "unitless"))


def drop_dataset(source, state_idx, name):
    """Rebuild one state without `name`, to simulate heterogeneous files."""
    state = source.states[state_idx]
    arrays = {n: np.asarray(state.get(n)) for n in state.full_core_keys if n != name}
    source.states[state_idx] = VeraOutState(
        DictDatasetSource(arrays, dataset_dtypes=source.core.shape_to_dtype),
        state_idx,
        source.core,
    )


@pytest.fixture
def pair():
    """Two independent sources over identical geometry."""
    kw = {"nass_side": 5, "npin": 3, "nax": 4}
    return build_source(3, core_kwargs=kw), build_source(3, core_kwargs=kw)


# --------------------------------------------------------------------------
# Guards
# --------------------------------------------------------------------------


def test_diffing_a_dataset_against_itself_is_rejected():
    """Always zero, so it is a mistake rather than a result."""
    src = build_source(1)
    with pytest.raises(ValueError, match="against itself"):
        src.add_new_diff_dataset("keff", src, "keff", "d")


def test_two_datasets_in_one_source_may_be_diffed(pair):
    """Self-diffing is only rejected when both names match.

    Different names in one source reach the dtype check and are skipped there,
    not blocked by the guard -- so the error is about compatibility, not identity.
    """
    a, _ = pair
    with pytest.raises(ValueError, match="No overlapping"):
        a.add_new_diff_dataset("pin_powers", a, "pin_radial_powers", "d")


def test_duplicate_diff_name_is_rejected(pair):
    a, b = pair
    a.add_new_diff_dataset("keff", b, "keff", "KEFF_DIFF")
    with pytest.raises(ValueError, match="already exists"):
        a.add_new_diff_dataset("keff", b, "keff", "KEFF_DIFF")


def test_duplicate_check_sees_file_backed_names(pair):
    """A diff may not shadow a real dataset."""
    a, b = pair
    with pytest.raises(ValueError, match="already exists"):
        a.add_new_diff_dataset("keff", b, "keff", "pin_powers")


def test_unknown_dataset_name_reports_no_overlap(pair):
    a, b = pair
    with pytest.raises(ValueError, match="No overlapping"):
        a.add_new_diff_dataset("not_a_dataset", b, "not_a_dataset", "d")


def test_shape_mismatch_outside_axial_is_rejected():
    """Same dtype but different core sizes: a broadcast error made legible."""
    a = build_source(1, core_kwargs={"nass_side": 5, "npin": 3})
    b = build_source(1, core_kwargs={"nass_side": 7, "npin": 3})
    with pytest.raises(ValueError, match="differ outside the axial dimension"):
        a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")


def test_mismatched_dtypes_are_skipped(pair):
    a, b = pair
    with pytest.raises(ValueError, match="No overlapping"):
        a.add_new_diff_dataset("pin_powers", b, "axial_powers", "d")


# --------------------------------------------------------------------------
# Scalars -- no axial dimension, so the mesh must not be consulted
# --------------------------------------------------------------------------


def test_scalar_diff_subtracts(pair):
    a, b = pair
    set_scalar(a, "keff", [1.0, 1.0, 1.0])
    set_scalar(b, "keff", [0.25, 0.5, 0.75])
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert [float(s.get("d")[0]) for s in a.states] == pytest.approx([0.75, 0.5, 0.25])


def test_scalar_diff_ignores_a_differing_axial_mesh():
    """A scalar has no axial extent, so mesh disagreement is irrelevant."""
    a = build_source(1, core_kwargs={"nax": 4})
    b = build_source(1, core_kwargs={"nax": 9})
    set_scalar(a, "keff", [1.0])
    set_scalar(b, "keff", [0.25])
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert float(a.states[0].get("d")[0]) == pytest.approx(0.75)


def test_identical_scalars_diff_to_zero(pair):
    """Zero is a legitimate answer, not a failure."""
    a, b = pair
    set_scalar(a, "keff", [1.0, 1.0, 1.0])
    set_scalar(b, "keff", [1.0, 1.0, 1.0])
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert all(float(s.get("d")[0]) == 0.0 for s in a.states)


# --------------------------------------------------------------------------
# Arrays -- matching mesh
# --------------------------------------------------------------------------


def test_array_diff_is_elementwise(pair):
    a, b = pair
    ref = a.get_dataset("pin_powers", mask_reflected=False)
    comp = b.get_dataset("pin_powers", mask_reflected=False)
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert np.allclose(a.get_dataset("d", mask_reflected=False), ref - comp, equal_nan=True)


def test_diff_preserves_shape_and_dtype(pair):
    a, b = pair
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert a.get_dataset_shape("d") == a.core.core_shape
    assert a.get_dataset_dtype("d") is VeraDtype.PIN


def test_scales_are_applied_to_each_side(pair):
    a, b = pair
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d", ref_scale=3.0, comp_scale=1.0)
    ref = a.get_dataset("pin_powers", mask_reflected=False)
    comp = b.get_dataset("pin_powers", mask_reflected=False)
    assert np.allclose(a.get_dataset("d", mask_reflected=False), 3.0 * ref - comp, equal_nan=True)


def test_units_are_taken_from_the_argument(pair):
    a, b = pair
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d", units="dW")
    assert a.get_dataset_units("d") == "dW"


def test_diff_is_categorized_and_listed(pair):
    a, b = pair
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert "d" in a.states[0].full_core_keys
    assert "d" in a.states[0].categorized_ds_names[VeraDtype.PIN]
    assert "d" in a.states[0].diff_datasets


def test_diff_lands_on_every_overlapping_state(pair):
    a, b = pair
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert all("d" in s for s in a.states)


def test_diff_is_pinned_and_survives_cache_cycling(pair):
    a, b = pair
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    for state in a.states:
        state.cache_all()
        state.uncache_all()
    assert a.get_dataset("d") is not None


def test_a_diff_can_itself_be_diffed(pair):
    a, b = pair
    set_scalar(a, "keff", [4.0, 4.0, 4.0])
    set_scalar(b, "keff", [1.0, 1.0, 1.0])
    a.add_new_diff_dataset("keff", b, "keff", "d1")  # 3.0
    a.add_new_diff_dataset("d1", b, "keff", "d2")  # 3.0 - 1.0
    assert float(a.states[0].get("d2")[0]) == pytest.approx(2.0)


# --------------------------------------------------------------------------
# Arrays -- interpolated mesh
# --------------------------------------------------------------------------


def test_differing_axial_meshes_interpolate_onto_the_reference():
    a = build_source(1, core_kwargs={"nass_side": 5, "npin": 3, "nax": 4})
    b = build_source(1, core_kwargs={"nass_side": 5, "npin": 3, "nax": 7})
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert a.get_dataset("d").shape == a.core.core_shape  # reference geometry wins


def test_interpolated_diff_of_identical_fields_is_near_zero():
    """A constant field interpolates exactly, whatever the meshes."""
    kw = {"nass_side": 5, "npin": 3}
    a = build_source(1, core_kwargs={**kw, "nax": 4})
    b = build_source(1, core_kwargs={**kw, "nax": 7})
    for src in (a, b):
        npy, npx, nax, nass = src.core.core_shape
        src.states[0].pin("flat", VeraDataset(np.ones((npy, npx, nax, nass)), VeraDtype.PIN, "W"))
    a.add_new_diff_dataset("flat", b, "flat", "d")
    out = a.get_dataset("d", mask_reflected=False)
    assert np.allclose(out[~np.isnan(out)], 0.0, atol=1e-9)


def test_interpolation_order_is_accepted():
    kw = {"nass_side": 5, "npin": 3}
    a = build_source(1, core_kwargs={**kw, "nax": 6})
    b = build_source(1, core_kwargs={**kw, "nax": 9})
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d", interpolation_order=3)
    assert a.get_dataset("d").shape == a.core.core_shape


# --------------------------------------------------------------------------
# Heterogeneous and mismatched state counts
# --------------------------------------------------------------------------


def test_comparison_with_fewer_states_covers_the_overlap_only():
    kw = {"nass_side": 5, "npin": 3}
    a = build_source(3, core_kwargs=kw)
    b = build_source(1, core_kwargs=kw)
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert "d" in a.states[0]
    assert "d" not in a.states[2]


def test_no_overlap_at_all_raises():
    """Guards the `break` path: it must not skip the produced == 0 check."""
    kw = {"nass_side": 5, "npin": 3}
    a = build_source(2, core_kwargs=kw)
    b = build_source(2, core_kwargs=kw)
    b._states = []
    with pytest.raises(ValueError, match="No overlapping"):
        a.add_new_diff_dataset("keff", b, "keff", "d")


def test_states_missing_the_dataset_are_skipped(pair):
    a, b = pair
    drop_dataset(b, 1, "keff")
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert "d" in a.states[0] and "d" not in a.states[1] and "d" in a.states[2]


# --------------------------------------------------------------------------
# Atomicity
# --------------------------------------------------------------------------


def test_a_failed_diff_leaves_no_partial_result():
    """A mismatch found on a later state must not leave earlier ones written."""
    kw = {"nass_side": 5, "npin": 3}
    a = build_source(2, core_kwargs=kw)
    b = build_source(2, core_kwargs=kw)

    # give b's second state a differently sized field under the same name
    npy, npx, nax, nass = a.core.core_shape
    b.states[1].pin(
        "pin_powers",
        VeraDataset(np.ones((npy, npx, nax, nass + 1)), VeraDtype.PIN, "W"),
    )
    with pytest.raises(ValueError, match="differ outside the axial dimension"):
        a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert all("d" not in s for s in a.states)


def test_a_rejected_diff_leaves_the_name_free(pair):
    a, b = pair
    with pytest.raises(ValueError):
        a.add_new_diff_dataset("nope", b, "nope", "d")
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert "d" in a.states[0]


def test_source_is_unchanged_when_the_guard_rejects(pair):
    a, b = pair
    before = sorted(a.states[0].full_core_keys)
    with pytest.raises(ValueError):
        a.add_new_diff_dataset("keff", a, "keff", "d")
    assert sorted(a.states[0].full_core_keys) == before


# --------------------------------------------------------------------------
# The comparison source must not be touched
# --------------------------------------------------------------------------


def test_comparison_source_does_not_receive_the_diff(pair):
    a, b = pair
    a.add_new_diff_dataset("keff", b, "keff", "d")
    assert all("d" not in s for s in b.states)


def test_comparison_data_is_not_mutated(pair):
    a, b = pair
    before = b.get_dataset("pin_powers").copy()
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d")
    assert np.array_equal(b.get_dataset("pin_powers"), before, equal_nan=True)


def test_reference_data_is_not_mutated(pair):
    a, b = pair
    before = a.get_dataset("pin_powers").copy()
    a.add_new_diff_dataset("pin_powers", b, "pin_powers", "d", ref_scale=5.0)
    assert np.array_equal(a.get_dataset("pin_powers"), before, equal_nan=True)


def test_diffs_in_both_directions_are_negatives(pair):
    a, b = pair
    set_scalar(a, "keff", [1.0, 1.0, 1.0])
    set_scalar(b, "keff", [0.25, 0.25, 0.25])
    a.add_new_diff_dataset("keff", b, "keff", "fwd")
    b.add_new_diff_dataset("keff", a, "keff", "rev")
    assert float(a.states[0].get("fwd")[0]) == pytest.approx(-float(b.states[0].get("rev")[0]))


def test_repeated_diffs_between_the_same_pair_are_consistent(pair):
    """Guards against accumulated state: the Nth call must equal the first."""
    a, b = pair
    set_scalar(a, "keff", [2.0, 2.0, 2.0])
    set_scalar(b, "keff", [0.5, 0.5, 0.5])
    results = []
    for i in range(4):
        a.add_new_diff_dataset("keff", b, "keff", f"d{i}")
        results.append(float(a.states[0].get(f"d{i}")[0]))
    assert results == pytest.approx([1.5] * 4)
