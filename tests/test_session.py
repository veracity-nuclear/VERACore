import json

from conftest import FakeSource, FakeState
from vera_core.app.core.session import (
    SESSION_VERSION,
    ViewSession,
    build_session,
    derive_recipe,
    diff_recipe,
    recipe_sources,
    save_session,
)
from vera_core.app.core.vera_data_registry import VeraDataRegistry


def test_view_session_mutable_defaults_are_not_shared() -> None:
    required = {
        "view_id": 1,
        "option": {},
        "layout": None,
        "selected_src_id": "src",
        "selected_array": "array",
        "selected_label": "label",
        "multi_selected": [],
        "multi_label": "none",
        "locked": False,
    }

    first = ViewSession(**required)
    second = ViewSession(**required)

    first.crop_x.append(2.0)

    assert second.crop_x == [0.0, 1.0]


def test_recipe_builders_and_source_dependencies() -> None:
    derive = derive_recipe(
        "a",
        "power",
        "avg",
        "Average",
        "AXIAL",
        True,
        False,
    )
    diff = diff_recipe(
        "a",
        "power",
        "b",
        "power",
        "delta",
        1,
        1.0,
        1.0,
        "W",
    )

    assert derive["kind"] == "derive"
    assert recipe_sources(derive) == {"a"}

    assert diff["kind"] == "diff"
    assert recipe_sources(diff) == {"a", "b"}

    assert recipe_sources({"kind": "unknown"}) == set()


def _state_for_one_view() -> FakeState:
    return FakeState(
        {
            "grid_layout": [
                {
                    "i": "1",
                    "x": 0,
                    "y": 0,
                    "w": 2,
                    "h": 3,
                }
            ],
            "grid_view_1": {"name": "core"},
            "selected_src_id_1": "src-a",
            "selected_array_1": "pin_powers",
            "selected_label_1": "PIN POWERS | src-a",
            "multi_selected_1": ["src-a\x1fpin_powers"],
            "multi_label_1": "1 selected",
            "locked_1": True,
            "assembly_decimals_1": 3,
            "crop_enabled_1": True,
            "crop_mode_1": "remove",
            "crop_x_1": [0.1, 0.9],
            "crop_y_1": [0.2, 0.8],
            "crop_z_1": [0.3, 0.7],
            "camera_1": {"position": [1, 2, 3]},
            "selected_layer": 2,
            "dark_mode": True,
            "file_overrides": {
                "a.h5": {
                    "nax": 3,
                }
            },
            "recipes": [
                {
                    "kind": "derive",
                    "src_id": "src-a",
                },
                {
                    "kind": "derive",
                    "src_id": "missing",
                },
            ],
        }
    )


def test_build_session_captures_state_and_filters_invalid_recipes() -> None:
    registry = VeraDataRegistry()
    registry.add_src(FakeSource(path="a.h5"), "src-a")

    state = _state_for_one_view()
    session = build_session(state, registry, ["1"])

    assert session.version == SESSION_VERSION
    assert session.file_paths == {"src-a": "a.h5"}
    assert session.default_src_id == "src-a"
    assert session.globals == {
        "selected_layer": 2,
        "dark_mode": True,
    }
    assert session.file_overrides == {
        "a.h5": {
            "nax": 3,
        }
    }

    assert len(session.views) == 1

    view = session.views[0]

    assert view.layout is not None
    assert view.layout["h"] == 3
    assert view.crop_enabled is True
    assert view.crop_mode == "remove"
    assert view.crop_x == [0.1, 0.9]
    assert view.camera == {"position": [1, 2, 3]}

    assert session.recipes == [
        {
            "kind": "derive",
            "src_id": "src-a",
        }
    ]


def test_save_session_writes_json(tmp_path) -> None:
    registry = VeraDataRegistry()
    registry.add_src(FakeSource(path="a.h5"), "src-a")

    state = _state_for_one_view()
    output = tmp_path / "session.json"

    save_session(state, registry, ["1"], str(output))

    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["version"] == SESSION_VERSION
    assert payload["file_paths"] == {"src-a": "a.h5"}
    assert payload["file_overrides"] == {
        "a.h5": {
            "nax": 3,
        }
    }
    assert payload["views"][0]["selected_array"] == "pin_powers"
