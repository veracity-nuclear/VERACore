"""VeraDataRegistry behavior not covered by test_registry.py.

Source "a": 5 levels, means 10, 30, 50, 70, 90.
Source "b": 8 levels, means 6.25, 18.75, ..., 93.75.
"""

import numpy as np
import pytest

from vera_core.data.dtypes import VeraDtype, derive_recipe, diff_recipe
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry

from .slice_support import quarter_map
from .source_support import make_core, make_source

T = VeraDtype


def derive_all(name="avg"):
    return derive_recipe(None, "pin_powers", name, "Average", "CORE", True, False) | {
        "all_sources": True
    }


def keff_diff(comp_array="keff"):
    return diff_recipe("a", "keff", "b", comp_array, "d", 1, 1.0, 1.0, "W")


@pytest.fixture
def reg():
    r = VeraDataRegistry()
    r.add_src(make_source(), "a")
    r.add_src(make_source(core=make_core(nax=8)), "b")
    return r


def global_idx(reg, height):
    return int(np.flatnonzero(reg.global_axial_mesh == height)[0])


# ---------------------------------------------------------------- axial mapping


def test_global_height_between_levels_maps_to_next_level_up(reg):
    """Documents a hazard: searchsorted picks the next level up, not the
    nearest. 18.75 maps to 30 even though 10 is closer."""
    assert reg.global_axial_idx_to_src_idx("a", T.PIN, global_idx(reg, 18.75)) == 1


def test_global_height_below_first_level_maps_to_first(reg):
    assert reg.global_axial_idx_to_src_idx("a", T.PIN, global_idx(reg, 6.25)) == 0


def test_get_axial_index_of_absent_height_is_insertion_point(reg):
    assert reg.get_axial_index(11.0) == global_idx(reg, 18.75)


def test_comp_dtype_maps_through_comp_mesh(reg):
    core = make_core(
        computational_core_map=quarter_map(), computational_axial_mesh=np.linspace(0, 100, 4)
    )
    reg.add_src(make_source(core=core), "c")
    g = reg.src_axial_idx_to_global_idx("c", T.COMP_PIN, 0)
    assert reg.global_axial_mesh[g] == pytest.approx(16.6667)
    assert reg.global_axial_idx_to_src_idx("c", T.COMP_PIN, g) == 0


def test_global_mesh_is_sorted_and_unique(reg):
    mesh = reg.global_axial_mesh
    assert (np.diff(mesh) > 0).all()


# ---------------------------------------------------------------- recipe bookkeeping


def test_failed_diff_is_not_recorded(reg):
    with pytest.raises(ValueError):
        reg.apply_recipe(keff_diff(comp_array="axial_powers"))
    assert reg._recipes == []


def test_failed_all_sources_derivation_is_partial_and_still_auto_applied(reg):
    """Documents a hazard: sources before the failing one keep the dataset,
    the recipe is not recorded for replay, yet it stays registered for
    auto-derivation on every future source."""
    reg.get("b").vera_calculator = None
    with pytest.raises(RuntimeError):
        reg.apply_recipe(derive_all())
    assert "avg" in reg.get("a").states[0]
    assert "avg" not in reg.get("b").states[0]
    assert reg._recipes == []
    reg.add_src(make_source(), "c")
    assert "avg" in reg.get("c").states[0]


def test_all_sources_derivation_skips_sources_without_array(reg):
    for state in reg.get("b").states:
        del state.source.arrays["pin_powers"]
        state.uncache_all()  # the active state was cached on construction
    reg.apply_recipe(derive_all())
    assert reg.get_ds_dtype("a", "avg") == T.SCALAR
    assert reg.get_ds_dtype("b", "avg") == T.UNKNOWN


def test_removing_source_keeps_all_sources_auto_derivation(reg):
    reg.apply_recipe(derive_all())
    reg.remove_src("a")
    reg.add_src(make_source(), "c")
    assert reg.get_ds_dtype("c", "avg") == T.SCALAR


def test_replacing_comparison_source_leaves_diff_stale(reg):
    """Documents a hazard: only recipes whose ref_src_id is the replaced
    source are replayed, so the diff on "a" still reflects the old "b"."""
    reg.apply_recipe(keff_diff())
    before = reg.get("a").states[0].get("d").copy()
    new_b = make_source(core=make_core(nax=8))
    for state in new_b.states:
        state.source.arrays["keff"] = state.source.arrays["keff"] + 5.0
    reg.replace_src("b", new_b)
    np.testing.assert_array_equal(reg.get("a").states[0].get("d"), before)


def test_replayed_recipes_keep_their_order(reg):
    reg.apply_recipe(derive_recipe("a", "pin_powers", "first", "Average", "CORE", True, False))
    reg.apply_recipe(derive_recipe("a", "first", "second", "Average", "CORE", True, False))
    reg.replace_src("a", make_source())
    assert "second" in reg.get("a").states[0]


# ---------------------------------------------------------------- states and time


def test_max_state_ignores_sources_without_states(reg):
    reg.add_src(VeraDataSource(core=make_core(), states=[], provenance="x"), "e")
    assert reg.max_state == 2


def test_max_state_of_only_empty_sources_is_zero():
    reg = VeraDataRegistry()
    reg.add_src(VeraDataSource(core=make_core(), states=[], provenance="x"), "e")
    assert reg.max_state == 0


def test_negative_state_clamps_to_zero_everywhere(reg):
    reg.change_all_active_state(-3)
    assert [reg.get(i).active_state_index for i in ("a", "b")] == [0, 0]


def test_time_axis_value_ties_resolve_to_first_added(reg):
    reg.add_src(make_source(), "c")
    reg.get("c").time_axes()["exposure"][1] = 999.0
    assert reg.time_axis_value("exposure", 1) == 10.0


def test_shared_time_axes_drops_axis_missing_from_one_source(reg):
    reg.get("b").time_axes().pop("exposure")
    assert reg.shared_time_axes() == ["state_count"]


def test_time_axis_value_with_missing_sample_falls_back_to_index():
    src = make_source()
    del src.states[1].source.arrays["exposure"]
    reg = VeraDataRegistry()
    reg.add_src(VeraDataSource(core=src.core, states=src.states, provenance="x"), "a")
    assert reg.time_axis_value("exposure", 1) == 1.0


# ---------------------------------------------------------------- defaults


def test_default_moves_to_next_source_and_stays_on_add(reg):
    reg.remove_src("a")
    reg.add_src(make_source(), "z")
    assert reg.default_src_id == "b"


def test_first_source_mesh_is_copied_not_aliased():
    reg = VeraDataRegistry()
    src = make_source()
    reg.add_src(src, "a")
    reg.global_axial_mesh[0] = -1.0
    assert src.core.gross_axial_mesh[0] == 10.0
