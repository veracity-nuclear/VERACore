"""ViewStyle is now the whole of the non-data rendering contract, so its
validation is what stops a bad choice reaching matplotlib."""

from dataclasses import replace

import pytest

from vera_core.data.analysis.color import DEFAULT_CMAP, ColorScope
from vera_core.data.renders.styles import DARK, LIGHT, ViewStyle


def test_defaults_are_the_documented_ones():
    style = ViewStyle()
    assert style.cmap == DEFAULT_CMAP
    assert style.color_scope is ColorScope.SLICE_ALL
    assert style.show_caption is True
    assert style.panel_columns is None
    assert style.theme is DARK


def test_color_scope_accepts_a_plain_string():
    """A caller writing a style by hand should not have to import the enum."""
    assert ViewStyle(color_scope="dataset_group").color_scope is ColorScope.DATASET_GROUP


def test_unknown_color_scope_fails_at_construction():
    with pytest.raises(ValueError):
        ViewStyle(color_scope="not_a_scope")


@pytest.mark.parametrize("width", [0.0, -1.0])
def test_non_positive_panel_width_fails(width):
    with pytest.raises(ValueError):
        ViewStyle(panel_width=width)


def test_panel_columns_below_one_fails():
    with pytest.raises(ValueError):
        ViewStyle(panel_columns=0)


def test_replace_revalidates():
    """dataclasses.replace goes through __init__, so a bad value cannot be
    slipped into a style that was valid when it was built."""
    style = ViewStyle()
    assert replace(style, cmap="magma").cmap == "magma"
    with pytest.raises(ValueError):
        replace(style, panel_columns=-2)


def test_style_is_frozen():
    with pytest.raises(Exception):
        ViewStyle().cmap = "magma"


def test_theme_with_background_leaves_the_original_alone():
    swapped = LIGHT.with_background("#abcdef")
    assert swapped.background == "#abcdef"
    assert LIGHT.background == "#ffffff"
    assert swapped.foreground == LIGHT.foreground
