"""CoreSlice.cropped and the rows/cols selection, on the mock source."""

from dataclasses import replace

import numpy as np
import pytest

from vera_core.data.analysis.core_slice import CoreSlice
from vera_core.data.renders.core_view import CoreView
from vera_core.data.renders.styles import ViewStyle


@pytest.fixture
def core_slice(mock_source):
    return CoreSlice.create_core_slice(mock_source, "pin_powers", 0)


# -- cropped ---------------------------------------------------------------


def test_the_mock_core_is_the_shape_the_tests_assume(core_slice):
    assert core_slice.grid_shape == (5, 5)


def test_cropped_narrows_the_grid_and_the_labels(core_slice):
    sub = core_slice.cropped(slice(1, 4), slice(1, 4))
    assert sub.grid_shape == (3, 3)
    assert np.array_equal(sub.core_map, core_slice.core_map[1:4, 1:4])
    if core_slice.x_labels:
        assert list(sub.x_labels) == list(core_slice.x_labels[1:4])
        assert list(sub.y_labels) == list(core_slice.y_labels[1:4])


def test_cropped_survives_validation(core_slice):
    """replace() runs __post_init__, so a label list left at the old length
    would raise here rather than at draw time."""
    assert core_slice.cropped(slice(1, 4), slice(1, 4)).validate() == []


def test_a_full_window_is_a_no_op_on_shape(core_slice):
    sub = core_slice.cropped(slice(None), slice(None))
    assert sub.grid_shape == core_slice.grid_shape


def test_cropped_nans_the_assemblies_it_drops(core_slice):
    """Otherwise a dropped assembly still sets the colorbar range, even
    though to_grid never draws it."""
    sub = core_slice.cropped(slice(0, 1), slice(0, 1))
    kept = {int(sub.core_map[0, 0]) - 1}
    values = sub.data_groups[0]
    drawn = {
        index
        for index in range(len(values))
        if np.isfinite(np.asarray(values[index], dtype=float)).any()
    }
    assert drawn == kept


def test_cropped_does_not_touch_the_original(core_slice):
    before = np.asarray(core_slice.data_groups[0], dtype=float).copy()
    core_slice.cropped(slice(0, 1), slice(0, 1))
    after = np.asarray(core_slice.data_groups[0], dtype=float)
    assert np.array_equal(before, after, equal_nan=True)


def test_to_grid_of_a_cropped_slice_is_the_window(core_slice):
    side = core_slice.assembly_side
    grid = core_slice.cropped(slice(1, 4), slice(1, 4)).to_grid(0)
    assert grid.shape == (3 * side, 3 * side)


# -- the selection ---------------------------------------------------------


def test_rows_and_cols_crop_through_build_slice(mock_source):
    view = CoreView(mock_source)
    assert view.select("pin_powers", z=0).slice().grid_shape == (5, 5)
    assert view.select("pin_powers", z=0, rows=slice(1, 4)).slice().grid_shape == (3, 5)
    assert view.select("pin_powers", z=0, cols=slice(1, 4)).slice().grid_shape == (5, 3)
    assert view.select(
        "pin_powers", z=0, rows=slice(1, 4), cols=slice(2, 5)
    ).slice().grid_shape == (3, 3)


def test_a_window_survives_replace(mock_source):
    """A sweep rebuilds every frame through select, so a collage of a
    cropped selection must stay cropped."""
    selection = CoreView(mock_source).select(
        "pin_powers", z=0, rows=slice(1, 4), cols=slice(1, 4)
    )
    assert selection.replace(state=1).slice().grid_shape == (3, 3)


def test_the_window_shows_in_the_label(mock_source):
    label = CoreView(mock_source).select("pin_powers", z=0, rows=slice(1, 4)).label()
    assert "rows=" in label and "cols" not in label


# -- rendering -------------------------------------------------------------


def test_the_figure_uses_the_style_colormap(mock_source, quiet_style):
    selection = CoreView(mock_source).select("pin_powers", z=0)
    figure = selection.figure(style=replace(quiet_style, cmap="magma"))
    images = [image for ax in figure.axes for image in ax.images]
    assert images and all(image.cmap.name == "magma" for image in images)


def test_the_cropped_figure_is_sized_for_the_window(mock_source, quiet_style):
    """Canvas reads grid_shape, so the crop has to reach the panel aspect and
    not just the pixels drawn inside it."""
    style = replace(quiet_style, panel_width=4.0)
    view = CoreView(mock_source)
    full = view.select("pin_powers", z=0).figure(style=style)
    narrow = view.select("pin_powers", z=0, cols=slice(0, 2)).figure(style=style)
    assert narrow.get_figheight() > full.get_figheight()


def test_caption_is_drawn_when_the_style_asks_for_it(mock_source):
    """Exercises slice_.info.caption(); the other render tests stay off it."""
    figure = (
        CoreView(mock_source)
        .select("pin_powers", z=0)
        .figure(style=ViewStyle(show_caption=True, show_values=False))
    )
    assert figure._supxlabel is not None


def test_an_unknown_array_names_the_selection(mock_source):
    with pytest.raises(Exception) as caught:
        CoreView(mock_source).select("not_a_dataset", z=0).slice()
    assert "not_a_dataset" in str(caught.value)
