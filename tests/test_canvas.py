"""Layout arithmetic and the correction pass. Nothing here writes a file:
Canvas.figure() is the finished figure, and the assertions read it."""

import numpy as np
import pytest

from vera_core.data.analysis.color import ColorSpec
from vera_core.data.renders import draw
from vera_core.data.renders.canvas import (
    Canvas,
    block_layout,
    map_aspect,
    panel_grid,
    square_columns,
)
from vera_core.data.renders.styles import ViewStyle

SPEC = ColorSpec(vmin=0.0, vmax=1.0, cmap="viridis")


def draw_into(panel, style):
    return draw.cells(panel.ax, np.arange(4.0).reshape(2, 2), SPEC, style)


# -- layout arithmetic -----------------------------------------------------


@pytest.mark.parametrize(
    "n_panels,expected",
    [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2)), (5, (3, 2))],
)
def test_panel_grid_default_is_two_wide(n_panels, expected):
    assert panel_grid(n_panels) == expected


@pytest.mark.parametrize(
    "n_panels,columns,expected",
    [(4, 4, (1, 4)), (4, 1, (4, 1)), (4, 3, (2, 3)), (4, 99, (1, 4)), (1, 4, (1, 1))],
)
def test_panel_grid_honors_columns_and_caps_at_the_panel_count(n_panels, columns, expected):
    """A large value means 'one row' whatever the group count, so a style
    written for four groups still reads for one."""
    assert panel_grid(n_panels, columns) == expected


@pytest.mark.parametrize(
    "n_frames,expected", [(1, 1), (4, 2), (8, 3), (11, 4), (12, 4), (17, 5), (19, 5)]
)
def test_square_columns(n_frames, expected):
    assert square_columns(n_frames) == expected


def test_block_layout_keeps_each_block_on_its_own_rows():
    grid, cells = block_layout(2, 4, columns=2)
    assert grid == (4, 2)
    assert cells[:4] == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert cells[4:] == [(2, 0), (2, 1), (3, 0), (3, 1)]


def test_map_aspect_is_height_over_width():
    assert map_aspect(24, 26) == pytest.approx(24 / 26)
    assert map_aspect(24, 26, 2.0) == pytest.approx(24 / 52)
    # A zero ratio is treated as 1.0 rather than dividing by zero.
    assert map_aspect(10, 10, 0.0) == pytest.approx(1.0)


# -- the canvas ------------------------------------------------------------


def test_panel_width_comes_from_the_style():
    canvas = Canvas(4, style=ViewStyle(panel_width=3.0))
    assert canvas.panels[0].figure.get_figwidth() == pytest.approx(6.0)  # two columns


def test_panel_columns_widens_the_figure_into_one_row():
    canvas = Canvas(4, style=ViewStyle(panel_width=3.0, panel_columns=4))
    figure = canvas.panels[0].figure
    assert figure.get_figwidth() == pytest.approx(12.0)
    # Every panel shares a row, so their vertical extents overlap.
    tops = {round(panel.ax.get_position().y1, 6) for panel in canvas.panels}
    assert len(tops) == 1


def test_explicit_grid_overrides_panel_columns():
    """A collage places its own panels; the style must not move them."""
    grid, cells = block_layout(2, 3, columns=3)
    canvas = Canvas(len(cells), grid=grid, cells=cells, style=ViewStyle(panel_columns=1))
    rows = {cell[0] for cell in cells}
    assert len(rows) == 2
    assert len(canvas.panels) == 6


def test_zero_panels_is_an_error():
    with pytest.raises(ValueError):
        Canvas(0)


def test_cells_must_match_the_panel_count():
    with pytest.raises(ValueError):
        Canvas(3, grid=(2, 2), cells=[(0, 0), (0, 1)])


def test_figure_is_idempotent():
    """A caller may take the figure and keep drawing on it, so a second call
    must not re-run the correction pass."""
    canvas = Canvas(1, style=ViewStyle())
    draw_into(canvas.panels[0], canvas.style)
    assert canvas.figure() is canvas.figure()


def test_colorbar_snaps_to_the_panels_it_reads():
    style = ViewStyle(panel_columns=2)
    canvas = Canvas(2, style=style)
    mappable = None
    for panel in canvas.panels:
        mappable = draw_into(panel, style)
    bar = canvas.colorbar(mappable, canvas.panels)
    canvas.figure()
    box = bar.ax.get_position()
    boxes = [panel.ax.get_position() for panel in canvas.panels]
    assert box.y0 == pytest.approx(min(b.y0 for b in boxes), abs=1e-6)
    assert box.y1 == pytest.approx(max(b.y1 for b in boxes), abs=1e-6)


def test_title_and_caption_are_drawn_only_when_given():
    style = ViewStyle()
    with_text = Canvas(1, style=style, title="T", caption="C")
    without = Canvas(1, style=style)
    assert with_text.panels[0].figure._suptitle is not None
    assert without.panels[0].figure._suptitle is None


def test_value_text_is_resized_against_the_drawn_width():
    """fit_values defers sizing to the correction pass, which only a drawn
    figure can settle."""
    style = ViewStyle(show_values=True, value_size=None)
    canvas = Canvas(1, style=style)
    panel = canvas.panels[0]
    grid = np.arange(4.0).reshape(2, 2)
    draw.cells(panel.ax, grid, SPEC, style)
    draw.cell_values(panel.ax, grid, SPEC, style, lambda v: f"{v:.2f}")
    panel.fit_values(2)
    before = panel.ax.texts[0].get_fontsize()
    canvas.figure()
    assert panel.ax.texts[0].get_fontsize() != before


def test_fixed_value_size_is_left_alone():
    style = ViewStyle(show_values=True, value_size=9.0)
    canvas = Canvas(1, style=style)
    panel = canvas.panels[0]
    panel.fit_values(2)
    assert panel.value_columns is None
