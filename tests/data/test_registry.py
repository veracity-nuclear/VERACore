import numpy as np
import pytest

from vera_core.data.dtypes import VeraDtype, derive_recipe, diff_recipe
from vera_core.data.registry import VeraDataRegistry, _recipe_sources

from .conftest import make_source

# Source "a": 4 axial levels, means [12.5, 37.5, 62.5, 87.5].
# Source "b": 8 axial levels, means [6.25, 18.75, ..., 93.75].


def derive(src_id="a", name="avg", axes="CORE", source_array="pin_powers"):
    return derive_recipe(src_id, source_array, name, "Average", axes, True, False)


def derive_all(name="avg"):
    return derive(src_id=None, name=name) | {"all_sources": True}


def diff(ref="a", comp="b", name="d", array="keff"):
    return diff_recipe(ref, array, comp, array, name, 1, 1.0, 1.0, "W")


@pytest.fixture
def reg():
    return VeraDataRegistry()


@pytest.fixture
def two(reg):
    reg.add_src(make_source(3), "a")
    reg.add_src(make_source(5, core_kwargs={"nax": 8}), "b")
    return reg


# ---------------------------------------------------------------- _recipe_sources


def test_recipe_sources_collects_ids():
    assert _recipe_sources(diff()) == {"a", "b"}
    assert _recipe_sources(derive()) == {"a"}


def test_recipe_sources_empty_for_all_sources():
    assert _recipe_sources(derive_all()) == set()


# ---------------------------------------------------------------- empty registry


def test_empty_registry(reg):
    assert reg.default_src is None
    assert not reg.has_src()
    assert reg.max_state == 0
    assert list(reg.src_ids()) == []
    assert reg.get(None) is None
    assert reg.get("a") is None
    assert "a" not in reg
    assert reg.all_sources() == {}


def test_time_axis_value_on_empty_registry(reg):
    assert reg.time_axis_value("exposure", 3) == 3.0


def test_shared_time_axes_on_empty_registry_is_empty_list(reg):
    assert reg.shared_time_axes() == []


# ---------------------------------------------------------------- add / lookup


def test_first_source_becomes_default(two):
    assert two.default_src_id == "a"
    assert two.default_src is two.get("a")
    assert two.has_src()


def test_add_sets_source_name(two):
    assert two.get("b").name == "b"


def test_duplicate_id_rejected(two):
    with pytest.raises(ValueError, match="already exists"):
        two.add_src(make_source(), "a")


def test_contains_and_ids(two):
    assert "a" in two and "b" in two
    assert list(two.src_ids()) == ["a", "b"]


def test_all_sources_maps_to_provenance(two):
    assert two.all_sources() == {"a": "<memory>", "b": "<memory>"}


def test_global_mesh_is_union(two):
    expected = np.union1d(two.get("a").core.gross_axial_mesh, two.get("b").core.gross_axial_mesh)
    np.testing.assert_array_equal(two.global_axial_mesh, expected)
    assert len(two.global_axial_mesh) == 12


def test_global_mesh_does_not_alias_core_mesh(reg):
    src = make_source()
    reg.add_src(src, "a")
    assert reg.global_axial_mesh is not src.core.gross_axial_mesh


def test_max_state_uses_longest_source(two):
    assert two.max_state == 4


def test_get_ds_dtype(two):
    assert two.get_ds_dtype("a", "pin_powers") == VeraDtype.PIN
    assert two.get_ds_dtype("a", "nope") == VeraDtype.UNKNOWN
    assert two.get_ds_dtype("zz", "pin_powers") == VeraDtype.UNKNOWN


# ---------------------------------------------------------------- active state


def test_change_active_state_single(two):
    two.change_active_state("b", 3)
    assert two.get("b").active_state_index == 3
    assert two.get("a").active_state_index == 0


def test_change_active_state_unknown_rejected(two):
    with pytest.raises(ValueError):
        two.change_active_state("zz", 0)


def test_change_all_active_state_clamps_per_source(two):
    two.change_all_active_state(4)
    assert two.get("a").active_state_index == 2
    assert two.get("b").active_state_index == 4


# ---------------------------------------------------------------- time axes


def test_shared_time_axes_is_sorted_intersection(two):
    assert two.shared_time_axes() == ["exposure", "state_count"]


def test_time_axis_value_reads_longest_source(two):
    assert two.time_axis_value("exposure", 4) == 40.0


def test_time_axis_value_falls_back_to_index(two):
    assert two.time_axis_value("state_count", 2) == 2.0
    assert two.time_axis_value("exposure", 99) == 99.0
    assert two.time_axis_value("missing_axis", 1) == 1.0


# ---------------------------------------------------------------- axial index mapping


def test_get_axial_index_exact_match(two):
    assert two.get_axial_index(12.5) == int(np.flatnonzero(two.global_axial_mesh == 12.5)[0])


def test_src_to_global_to_src_round_trip(two):
    for src_id, nax in (("a", 4), ("b", 8)):
        for i in range(nax):
            g = two.src_axial_idx_to_global_idx(src_id, VeraDtype.PIN, i)
            assert two.global_axial_idx_to_src_idx(src_id, VeraDtype.PIN, g) == i


def test_src_to_global_maps_to_same_height(two):
    g = two.src_axial_idx_to_global_idx("b", VeraDtype.PIN, 0)
    assert two.global_axial_mesh[g] == 6.25


def test_global_to_src_clamps_past_end(two):
    last = len(two.global_axial_mesh) - 1
    assert two.global_axial_idx_to_src_idx("a", VeraDtype.PIN, last) == 3


def test_src_to_global_unknown_source_rejected(two):
    with pytest.raises(ValueError):
        two.src_axial_idx_to_global_idx("zz", VeraDtype.PIN, 0)


def test_global_to_src_unknown_source_rejected(two):
    with pytest.raises(ValueError):
        two.global_axial_idx_to_src_idx("zz", VeraDtype.PIN, 0)


# ---------------------------------------------------------------- remove / clear


def test_remove_reassigns_default_and_closes(two):
    src = two.get("a")
    two.remove_src("a")
    assert "a" not in two
    assert two.default_src_id == "b"
    assert src.vera_calculator.closed


def test_remove_recomposes_mesh(two):
    two.remove_src("b")
    np.testing.assert_array_equal(two.global_axial_mesh, [12.5, 37.5, 62.5, 87.5])


def test_remove_last_source_empties_registry(two):
    two.remove_src("a")
    two.remove_src("b")
    assert two.default_src_id is None
    assert len(two.global_axial_mesh) == 0


def test_remove_unknown_rejected(two):
    with pytest.raises(ValueError):
        two.remove_src("zz")


def test_remove_drops_recipes_that_reference_source(two):
    two.apply_recipe(diff())
    two.apply_recipe(derive("a"))
    two.remove_src("b")
    assert [r["kind"] for r in two._recipes] == ["derive"]


def test_clear_closes_and_resets(two):
    srcs = [two.get("a"), two.get("b")]
    two.clear()
    assert all(s.vera_calculator.closed for s in srcs)
    assert not two.has_src()
    assert list(two.src_ids()) == []

def test_clear_resets_global_mesh(two):
    two.clear()
    assert len(two.global_axial_mesh) == 0


def test_clear_stops_auto_derivation(two):
    two.apply_recipe(derive_all())
    two.clear()
    two.add_src(make_source(), "c")
    assert "avg" not in two.get("c").states[0]


# ---------------------------------------------------------------- apply_recipe


def test_derive_recipe_on_one_source(two):
    two.apply_recipe(derive("a", axes="AXIAL"))
    assert two.get_ds_dtype("a", "avg") == VeraDtype.AXIAL
    assert two.get_ds_dtype("b", "avg") == VeraDtype.UNKNOWN


def test_derive_recipe_unknown_source_rejected(two):
    with pytest.raises(ValueError, match="not found in registry"):
        two.apply_recipe(derive("zz"))


def test_derive_recipe_missing_array_rejected(two):
    with pytest.raises(ValueError, match="not found in src"):
        two.apply_recipe(derive("a", source_array="missing"))


def test_derive_all_sources(two):
    two.apply_recipe(derive_all())
    assert all(two.get_ds_dtype(i, "avg") == VeraDtype.SCALAR for i in ("a", "b"))


def test_derive_all_sources_auto_applies_to_new_sources(two):
    two.apply_recipe(derive_all())
    two.add_src(make_source(), "c")
    assert two.get_ds_dtype("c", "avg") == VeraDtype.SCALAR


def test_auto_derivation_failure_does_not_block_add(two, capsys):
    two.apply_recipe(derive_all())
    two.add_src(make_source(calculator=False), "c")
    assert "c" in two
    assert "Internal warning" in capsys.readouterr().out


def test_diff_recipe_adds_to_reference_source(two):
    two.apply_recipe(diff())
    assert two.get_ds_dtype("a", "d") == VeraDtype.SCALAR
    assert two.get_ds_dtype("b", "d") == VeraDtype.UNKNOWN


def test_diff_recipe_unknown_source_rejected(two):
    with pytest.raises(ValueError):
        two.apply_recipe(diff(ref="zz"))


def test_unknown_recipe_kind_rejected(two):
    with pytest.raises(ValueError, match="Unknown recipe kind"):
        two.apply_recipe({"kind": "bogus"})


# ---------------------------------------------------------------- replace_src


def test_replace_swaps_closes_and_names(two):
    old, new = two.get("a"), make_source()
    two.replace_src("a", new)
    assert two.get("a") is new
    assert new.name == "a"
    assert old.vera_calculator.closed


def test_replace_unknown_rejected(two):
    with pytest.raises(ValueError):
        two.replace_src("zz", make_source())


def test_replace_recomposes_mesh(two):
    two.replace_src("b", make_source())
    np.testing.assert_array_equal(two.global_axial_mesh, [12.5, 37.5, 62.5, 87.5])


@pytest.mark.parametrize("recipe", [derive("a"), derive_all(), diff()], ids=["one", "all", "diff"])
def test_replace_replays_recipes_for_that_source(two, recipe):
    two.apply_recipe(recipe)
    two.replace_src("a", make_source())
    assert recipe["name"] in two.get("a").states[0]


def test_replace_does_not_replay_other_sources_recipes(two):
    two.apply_recipe(derive("b"))
    two.replace_src("a", make_source())
    assert "avg" not in two.get("a").states[0]


def test_replay_failure_is_silent(two):
    two.apply_recipe(derive("a"))
    two.replace_src("a", make_source(calculator=False))
    assert "avg" not in two.get("a").states[0]
