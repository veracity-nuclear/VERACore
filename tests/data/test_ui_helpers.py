"""vera_core.app.ui.helpers with a dict-backed stand-in for trame's State."""

import numpy as np
import pytest

from vera_core.app.ui import helpers as h
from vera_core.data.registry import VeraDataRegistry

from .source_support import make_source
from .ui_support import FakeState

SEL = dict(
    selected_time=1,
    selected_i=1,
    selected_j=2,
    selected_layer=3,
    selected_surface=0,
    selected_assembly_ij={"i": 1, "j": 1},
)


@pytest.fixture
def reg():
    r = VeraDataRegistry()
    r.add_src(make_source(), "a")
    return r


@pytest.fixture
def state():
    return FakeState(
        SEL,
        selected_src_id_1="a",
        selected_array_1="pin_powers",
        locked_1={},
        view_loading_1=False,
        thresholds={},
        multi_selected_1=[],
        grid_view_1={"name": "core"},
    )


# ---------------------------------------------------------------- labels and layout


@pytest.mark.parametrize(
    "kwargs, label",
    [
        ({}, "PIN POWERS | a"),
        ({"group": 2}, "PIN POWERS / Group 2 | a"),
        ({"source_identifier": False}, "PIN POWERS"),
    ],
)
def test_format_label(kwargs, label):
    assert h.format_label("a", "pin_powers", **kwargs) == label


@pytest.mark.parametrize(
    "layout, y",
    [([], 0), ([{"y": 0, "h": 8}], 8), ([{"y": 10, "h": 2}, {"y": 0, "h": 5}], 12), ([{}], 1)],
)
def test_next_y_is_bottom_of_lowest_item(layout, y):
    assert h.get_next_y_from_layout(layout) == y


# ---------------------------------------------------------------- lock and time


@pytest.mark.parametrize(
    "locked, loading, expected",
    [({}, False, False), ({"a": 1}, False, True), (True, False, True), ({"a": 1}, True, False)],
)
def test_is_view_locked(state, locked, loading, expected):
    state.update(locked_1=locked, view_loading_1=loading)
    assert h.is_view_locked(state, 1) is expected


def test_is_view_locked_without_loading_key(state):
    del state["view_loading_1"]
    state["locked_1"] = {"a": 1}
    assert h.is_view_locked(state, 1)


def test_non_active_view_when_names_differ_or_locked(state):
    assert h.is_non_active_view(state, 1, {"name": "table"})
    assert not h.is_non_active_view(state, 1, {"name": "core"})
    state["locked_1"] = {"a": 1}
    assert h.is_non_active_view(state, 1, {"name": "core"})


@pytest.mark.parametrize("locked, expected", [({}, 1), ({"selected_time": 4}, 4)])
def test_get_time(state, locked, expected):
    """Only unambiguous cases. A lock without a frozen time (True, or a dict
    missing selected_time) currently gives None, which crashes get_safe_idxs;
    see test_safe_idxs_with_boolean_lock. Not pinned here so either fix
    location (get_time or get_safe_idxs) stays open."""
    state["locked_1"] = locked
    assert h.get_time(state, 1) == expected


# ---------------------------------------------------------------- get_safe_idxs


def test_safe_idxs_unlocked(state, reg):
    assert h.get_safe_idxs(1, state, reg) == (2, 1, 3, 4, "a", "pin_powers", 1, 0)


def test_safe_idxs_clip_pins_and_time(state, reg):
    state.update(selected_i=99, selected_j=-3, selected_time=50)
    j, i, _, _, _, _, time, _ = h.get_safe_idxs(1, state, reg)
    assert (j, i, time) == (0, 2, 2)


def test_safe_idxs_read_frozen_selection_when_locked(state, reg):
    state["locked_1"] = dict(SEL, selected_i=0)
    state["selected_i"] = 2
    assert h.get_safe_idxs(1, state, reg)[1] == 0


def test_safe_idxs_overrides_take_precedence(state, reg):
    out = h.get_safe_idxs(1, state, reg, sel_src_id="a", sel_dataset_name="keff")
    assert out[5] == "keff"


@pytest.mark.parametrize("field, value", [("selected_src_id_1", "zz"), ("selected_array_1", "nope")])
def test_safe_idxs_none_for_unknown_source_or_dataset(state, reg, field, value):
    state[field] = value
    assert h.get_safe_idxs(1, state, reg) is None


def test_safe_idxs_with_boolean_lock(state, reg):
    """ViewSession.locked is typed dict | bool and the lock icon treats True
    as locked, so a True value can arrive from a saved session."""
    state["locked_1"] = True
    assert h.get_safe_idxs(1, state, reg) is not None


def test_safe_idxs_clip_out_of_range_layer(state, reg):
    """A locked view can hold a layer from before a source was removed."""
    state["selected_layer"] = 99
    assert h.get_safe_idxs(1, state, reg)[2] == 4


# ---------------------------------------------------------------- set_info


def test_set_info_fills_label(state, reg):
    h.set_info(1, state, reg)
    assert state["label_info_1"] == {
        "Exposure": 10.0,
        "Assembly": "B-4",
        "Layer": 70.0,
        "Pin_x": 1,
        "Pin_y": 2,
    }


def test_set_info_skips_locked_view(state, reg):
    state["locked_1"] = dict(SEL)
    h.set_info(1, state, reg)
    assert "label_info_1" not in state


def test_set_info_without_exposure(state, reg):
    for s in reg.get("a").states:
        del s.source.arrays["exposure"]
        s.uncache_all()
    h.set_info(1, state, reg)
    assert state["label_info_1"]["Exposure"] == "not recorded"


@pytest.mark.parametrize(
    "mesh, layer, expected",
    [([10.0, 30.0], 1, 30.0), ([[0.0, 10.0], [20.0, 30.0]], 1, 25.0), ([1.0], 5, None), ([1.0], -1, None)],
)
def test_layer_elevation(mesh, layer, expected):
    assert h._layer_elevation(mesh, layer) == expected


def test_layer_elevation_rounds_to_two_places():
    assert h._layer_elevation([1.23456], 0) == 1.23


# ---------------------------------------------------------------- small helpers


@pytest.mark.parametrize("j, i, node", [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3), (5, 5, 3)])
def test_convert_ji_to_node(j, i, node):
    assert h.convert_ji_to_node(j, i) == node


def test_requires_src_skips_call_on_empty_registry(reg):
    calls = []
    wrapped = h.requires_src(lambda registry: calls.append(1) or "ran")
    assert wrapped(VeraDataRegistry()) is None
    assert wrapped(reg) == "ran"
    assert calls == [1]


@pytest.mark.parametrize(
    "names, expected",
    [
        ({}, None),
        ({"ASSEMBLY": "x", "PIN": "pin_powers"}, "pin_powers"),
        ({"ASSEMBLY": "x", "PIN": "p2"}, "p2"),
        ({"ASSEMBLY": "x", "AXIAL": "y"}, "x"),
    ],
)
def test_default_dataset_name(names, expected):
    assert h.default_dataset_name(names) == expected


def test_get_thresholds_combines_source_and_global_keys(state):
    state["thresholds"] = {
        "PIN POWERS | a": [{"op": ">", "value": 1}],
        "PIN POWERS": [{"op": "<", "value": 2}],
        "PIN POWERS | b": [{"op": "==", "value": 3}],
    }
    assert h.get_thresholds(state, 1) == [{"op": ">", "value": 1}, {"op": "<", "value": 2}]


@pytest.mark.parametrize(
    "group, expected", [(None, [1, 2, 3]), (0, [1, 2, 3]), (1, [1]), (3, [3]), (9, [3])]
)
def test_pick_group_is_one_based_and_clamped(group, expected):
    assert h.pick_group([1, 2, 3], group) == expected


def test_pick_group_on_empty_list():
    assert h.pick_group([], 2) == []


# ---------------------------------------------------------------- tokens


TRIPLES = [("a", "pin_powers", None), ("b", "flux", 2), ("c", "x", 0)]


def test_tokens_round_trip():
    assert h.decode_tokens(h.encode_tokens(TRIPLES)) == TRIPLES


def test_decode_drops_tokens_without_separator():
    assert h.decode_tokens(["legacy", f"a{h.MULTI_SEP}b"]) == [("a", "b", None)]


def test_decode_treats_trailing_separator_as_all_groups():
    assert h.decode_tokens([f"a{h.MULTI_SEP}b{h.MULTI_SEP}"]) == [("a", "b", None)]


def test_get_multi_selected_src_reads_view_key(state):
    state["multi_selected_1"] = h.encode_tokens(TRIPLES[:1])
    assert h.get_multi_selected_src(state, 1) == TRIPLES[:1]


def test_group_indices_are_python_ints():
    [(_, _, group)] = h.decode_tokens(h.encode_tokens([("a", "b", np.int64(2))]))
    assert type(group) is int
