import numpy as np
import pytest
from conftest import FakeSource

from vera_core.app.core import VeraAxes, VeraDtype
from vera_core.app.core.vera_data_registry import VeraDataRegistry


def test_registry_add_lookup_default_and_global_mesh():
    registry = VeraDataRegistry()
    first = FakeSource(path="first.h5", mesh=(10, 30), state_count=2)
    second = FakeSource(path="second.h5", mesh=(20, 30, 40), state_count=4)

    registry.add_src(first, "first")
    registry.add_src(second, "second")

    assert registry.default_src is first
    assert registry.default_src_id == "first"
    assert registry.get("second") is second
    assert registry.get(None) is None
    assert list(registry.src_ids()) == ["first", "second"]
    np.testing.assert_array_equal(registry.global_axial_mesh, [10, 20, 30, 40])
    assert registry.max_state == 3
    assert registry.all_sources() == {"first": "first.h5", "second": "second.h5"}


def test_registry_rejects_duplicate_ids():
    registry = VeraDataRegistry()
    registry.add_src(FakeSource(path="a.h5"), "same")
    with pytest.raises(ValueError, match="already exists"):
        registry.add_src(FakeSource(path="b.h5"), "same")


def test_registry_changes_state_on_one_or_all_sources():
    registry = VeraDataRegistry()
    first = FakeSource(path="a.h5")
    second = FakeSource(path="b.h5")
    registry.add_src(first, "a")
    registry.add_src(second, "b")

    registry.change_active_state("a", 2)
    assert first.active_state_index == 2
    assert second.active_state_index == 0

    registry.change_all_active_state(5)
    assert first.active_state_index == 5
    assert second.active_state_index == 5

    with pytest.raises(ValueError, match="Could not find"):
        registry.change_active_state("missing", 1)


def test_registry_remove_and_clear_close_sources():
    registry = VeraDataRegistry()
    first = FakeSource(path="a.h5", mesh=(10, 20))
    second = FakeSource(path="b.h5", mesh=(20, 30))
    registry.add_src(first, "a")
    registry.add_src(second, "b")

    registry.remove_src("a")
    assert first.closed
    assert registry.default_src_id == "b"
    np.testing.assert_array_equal(registry.global_axial_mesh, [20, 30])

    registry.clear()
    assert second.closed
    assert not registry.has_src()
    assert registry.default_src is None


def test_registry_shared_time_axes_and_values():
    registry = VeraDataRegistry()
    registry.add_src(
        FakeSource(
            path="a.h5",
            state_count=2,
            time_axes={"state_count": [0, 1], "exposure": [0.0, 10.0]},
        ),
        "a",
    )
    registry.add_src(
        FakeSource(
            path="b.h5",
            state_count=3,
            time_axes={"state_count": [0, 1, 2], "exposure": [0.0, 5.0, 15.0]},
        ),
        "b",
    )

    assert registry.shared_time_axes() == ["exposure", "state_count"]
    assert registry.time_axis_value("exposure", 2) == 15.0
    assert registry.time_axis_value("state_count", 2) == 2.0
    assert registry.time_axis_value("missing", 2) == 2.0


def test_registry_axial_index_conversion():
    registry = VeraDataRegistry()
    registry.add_src(FakeSource(path="a.h5", mesh=(10, 30)), "a")
    registry.add_src(FakeSource(path="b.h5", mesh=(20, 30, 40)), "b")

    assert registry.src_axial_idx_to_global_idx("a", VeraDtype.PIN, 1) == 2
    assert registry.global_axial_idx_to_src_idx("a", VeraDtype.PIN, 1) == 1
    assert registry.global_axial_idx_to_src_idx("missing", VeraDtype.PIN, 1) == 0


def test_registry_apply_recipe_dispatches_to_source():
    registry = VeraDataRegistry()
    ref = FakeSource(path="a.h5")
    comp = FakeSource(path="b.h5")
    registry.add_src(ref, "ref")
    registry.add_src(comp, "comp")

    registry.apply_recipe(
        {
            "kind": "derive",
            "src_id": "ref",
            "source_array": "pin_powers",
            "name": "avg_power",
            "method": "Average",
            "axes": VeraAxes.AXIAL.name,
        }
    )
    assert ref.derived_calls[0]["new_dataset_name"] == "avg_power"
    assert ref.derived_calls[0]["axes"] is VeraAxes.AXIAL

    registry.apply_recipe(
        {
            "kind": "diff",
            "ref_src_id": "ref",
            "ref_array": "a",
            "comp_src_id": "comp",
            "comp_array": "b",
            "name": "delta",
            "interp_degree": 1,
            "ref_scale": 1.0,
            "comp_scale": 2.0,
            "units": "W",
        }
    )
    assert ref.diff_calls[0][1] is comp

    with pytest.raises(ValueError, match="Unknown recipe kind"):
        registry.apply_recipe({"kind": "invalid"})
