import numpy as np
import pytest

from vera_core.app.ui.helpers import (
    _layer_elevation,
    array_range,
    convert_ji_to_node,
    default_dataset_name,
    format_label,
    get_next_y_from_layout,
)


def test_format_label():
    assert format_label("source-a", "pin_powers") == "PIN POWERS | source-a"


@pytest.mark.parametrize(
    ("array", "expected"),
    [
        (np.array([1.0, 4.0, np.nan]), (1.0, 4.0)),
        (np.array([np.nan, np.inf]), (0.0, 1.0)),
    ],
)
def test_array_range(array, expected):
    assert array_range(array) == expected


def test_array_range_expands_constant_values():
    lo, hi = array_range(np.array([5.0, 5.0]))
    assert lo == 5.0
    assert hi > lo


def test_get_next_y_from_layout_uses_lowest_free_row():
    layout = [{"y": 0, "h": 4}, {"y": 7, "h": 3}, {"y": 2}]
    assert get_next_y_from_layout(layout) == 10


@pytest.mark.parametrize(
    ("mesh", "layer", "expected"),
    [
        ([10.123, 20.456], 1, 20.46),
        ([[0.0, 2.0], [4.0, 8.0]], 1, 6.0),
        ([1.0], -1, None),
        ([1.0], 2, None),
    ],
)
def test_layer_elevation(mesh, layer, expected):
    assert _layer_elevation(mesh, layer) == expected


@pytest.mark.parametrize(
    ("j", "i", "expected"),
    [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 1, 3), (10, 10, 3)],
)
def test_convert_ji_to_node(j, i, expected):
    assert convert_ji_to_node(j, i) == expected


def test_default_dataset_name_priority():
    assert default_dataset_name({}) is None
    assert default_dataset_name({"ASSEMBLY": "power", "PIN": "pin_powers"}) == "pin_powers"
    assert default_dataset_name({"PIN": "temperature", "ASSEMBLY": "power"}) == "temperature"
    assert default_dataset_name({"ASSEMBLY": "power"}) == "power"
