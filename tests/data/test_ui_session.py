"""vera_core.app.ui.session: what build_session captures and filters, and
that save_session output loads back into the dataclasses."""

import json
from dataclasses import asdict

import pytest

from vera_core.app.ui.session import (
    SESSION_VERSION,
    ViewSession,
    build_session,
    recipe_sources,
    save_session,
)
from vera_core.data.dtypes import derive_recipe, diff_recipe
from vera_core.data.registry import VeraDataRegistry, _recipe_sources

from .source_support import make_source
from .ui_support import FakeState

VIEW_IDS = ["1", "2"]
DERIVE_A = derive_recipe("a", "pin_powers", "avg", "Average", "CORE", True, False)
DERIVE_ALL = derive_recipe(None, "pin_powers", "all", "Average", "CORE", True, False) | {
    "all_sources": True
}
DIFF_AB = diff_recipe("a", "keff", "b", "keff", "d", 1, 1.0, 1.0, "W")
DIFF_AS = diff_recipe("a", "keff", "s", "keff", "ds", 1, 1.0, 1.0, "W")


def view_state(vid, **overrides):
    keys = {
        "grid_view": {"name": "core"},
        "selected_src_id": "a",
        "selected_array": "pin_powers",
        "selected_label": "PIN POWERS | a",
        "selected_group": None,
        "multi_selected": ["a\x1fpin_powers"],
        "multi_label": "1 selected",
        "locked": {},
        "crop_enabled": False,
        "crop_mode": "keep",
        "crop_x": [0.0, 1.0],
        "crop_y": [0.0, 1.0],
        "crop_z": [0.0, 1.0],
    } | overrides
    return {f"{k}_{vid}": v for k, v in keys.items()}


@pytest.fixture
def reg():
    r = VeraDataRegistry()
    for src_id in ("a", "b", "s"):
        r.add_src(make_source(1), src_id)
    return r


@pytest.fixture
def state():
    return FakeState(
        **view_state("1", assembly_decimals=4, camera={"zoom": 2}),
        **view_state("2"),
        grid_layout=[{"i": "1", "x": 0, "y": 0, "w": 6, "h": 8}],
        ports_opened={"s": 9000},
        file_overrides={},
        recipes=[DERIVE_A, DIFF_AB, DIFF_AS],
        selected_time=0,
        dark_mode=False,
        unrelated_key=1,
    )


@pytest.fixture
def session(state, reg):
    return build_session(state, reg, VIEW_IDS)


# ---------------------------------------------------------------- build_session


def test_version_and_default_source(session):
    assert session.version == SESSION_VERSION
    assert session.default_src_id == "a"


def test_stream_sources_are_not_saved_as_files(session):
    assert session.file_paths == {"a": "<test>", "b": "<test>"}
    assert session.stream_ports == {"s": 9000}


def test_recipes_depending_on_stream_sources_are_dropped(session):
    assert [r["name"] for r in session.recipes] == ["avg", "d"]


def test_globals_include_only_session_keys_present(session):
    assert session.globals == {"selected_time": 0, "dark_mode": False}


def test_view_layout_matches_grid_entry_or_none(session):
    layouts = {v.view_id: v.layout for v in session.views}
    assert layouts == {"1": {"i": "1", "x": 0, "y": 0, "w": 6, "h": 8}, "2": None}


def test_optional_view_fields_default_when_absent(session):
    v1, v2 = session.views
    assert (v1.assembly_decimals, v1.camera) == (4, {"zoom": 2})
    assert (v2.assembly_decimals, v2.camera) == (2, None)


def test_list_fields_are_copied_not_aliased(state, session):
    state["multi_selected_1"].append("changed")
    state["crop_x_1"][0] = 0.5
    assert session.views[0].multi_selected == ["a\x1fpin_powers"]
    assert session.views[0].crop_x == [0.0, 1.0]


def test_missing_recipes_key_gives_empty_list(state, reg):
    del state["recipes"]
    assert build_session(state, reg, VIEW_IDS).recipes == []


# ---------------------------------------------------------------- all-sources recipes


@pytest.mark.xfail(strict=True, reason="recipe_sources returns {None} for all-sources derivations")
def test_all_sources_recipe_is_saved(state, reg):
    state["recipes"] = [DERIVE_ALL]
    assert build_session(state, reg, VIEW_IDS).recipes == [DERIVE_ALL]


@pytest.mark.xfail(strict=True, reason="session and registry disagree on all-sources recipes")
def test_recipe_sources_agrees_with_registry():
    for recipe in (DERIVE_A, DIFF_AB, DERIVE_ALL):
        assert recipe_sources(recipe) == _recipe_sources(recipe)


def test_recipe_sources_of_unknown_kind_is_empty():
    assert recipe_sources({"kind": "other"}) == set()


# ---------------------------------------------------------------- serialization


def test_saved_file_loads_back_into_dataclasses(state, reg, session, tmp_path):
    path = tmp_path / "session.json"
    save_session(state, reg, VIEW_IDS, str(path))
    raw = json.loads(path.read_text())
    assert [ViewSession(**v) for v in raw["views"]] == session.views
    assert raw["recipes"] == session.recipes
    assert raw["file_paths"] == session.file_paths


def test_saved_file_matches_asdict(state, reg, session, tmp_path):
    path = tmp_path / "session.json"
    save_session(state, reg, VIEW_IDS, str(path))
    assert json.loads(path.read_text()) == json.loads(json.dumps(asdict(session)))


def test_pre_crop_view_entry_still_loads():
    old = {
        "view_id": "1",
        "option": {"name": "core"},
        "layout": None,
        "selected_src_id": "a",
        "selected_array": "pin_powers",
        "selected_label": "",
        "multi_selected": [],
        "multi_label": "",
        "locked": False,
    }
    view = ViewSession(**old)
    assert (view.crop_enabled, view.crop_x, view.camera, view.selected_group) == (
        False,
        [0.0, 1.0],
        None,
        None,
    )


def test_crop_defaults_are_not_shared_between_instances():
    fields = dict.fromkeys(
        ["option", "layout", "selected_src_id", "selected_array", "selected_label", "multi_label", "locked"]
    )
    a = ViewSession(view_id=1, multi_selected=[], **fields)
    b = ViewSession(view_id=2, multi_selected=[], **fields)
    a.crop_x[0] = 0.5
    assert b.crop_x == [0.0, 1.0]
