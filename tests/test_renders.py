"""Tests for the renderers.

Every artist is tested by direct call as well as through panel_figure. That
split matters: a NameError once sat in draw_core_slice's default-color branch
through several rounds of visual checking, because the figure path resolves
colors before drawing and never reached it.
"""

import numpy as np
import pytest
from conftest import make_axial_slice, make_core_slice, make_surface_slice
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.figure import Figure

from vera_core.data.analysis.color import ColorSpec
from vera_core.data.renders.axial_view import (
    assembly_edges,
    axial_aspect,
    axial_columns,
    axial_view_figure,
    draw_axial_slice,
)
from vera_core.data.renders.core_surface_view import (
    LABELS_ACROSS_A_NODE,
    draw_surface_slice,
    surface_columns,
    surface_polygons,
    surface_view_figure,
)
from vera_core.data.renders.core_view import (
    MAX_LABEL_SIDE,
    core_columns,
    core_image,
    core_view_figure,
    draw_core_slice,
    image_side,
)
from vera_core.data.renders.layout import DARK, LIGHT, ViewStyle

VALUES = ViewStyle(show_values=True)


def fresh_axes() -> Axes:
    return Figure().add_subplot()


# -- core ------------------------------------------------------------------


class TestCoreImage:
    def test_two_d_cell_flattens_to_pin_resolution(self):
        slice_ = make_core_slice(cell=(2, 2), rows=3, cols=4)
        assert core_image(slice_).shape == (6, 8)

    def test_one_d_cell_is_reshaped_to_a_square(self):
        """NODAL and COMP_NODAL produce a 1-D cell, which as_image() rejects."""
        slice_ = make_core_slice(cell=(4,), rows=3, cols=4)
        assert core_image(slice_).shape == (6, 8)
        assert image_side(slice_) == 2

    def test_cell_less_data_passes_through(self):
        slice_ = make_core_slice(cell=(), rows=3, cols=4)
        assert core_image(slice_).shape == (3, 4)
        assert image_side(slice_) == 1

    def test_columns_counts_pins_not_assemblies(self):
        assert core_columns(make_core_slice(cell=(2, 2), cols=4)) == 8


class TestDrawCoreSlice:
    def test_draws_with_no_explicit_color(self):
        """The direct path resolves its own default; the figure path never
        reaches that branch."""
        slice_ = make_core_slice()
        mappable = draw_core_slice(fresh_axes(), slice_)
        assert (mappable.norm.vmin, mappable.norm.vmax) == slice_.value_range

    def test_explicit_color_wins(self):
        spec = ColorSpec(0.0, 10.0)
        mappable = draw_core_slice(fresh_axes(), make_core_slice(), spec)
        assert (mappable.norm.vmin, mappable.norm.vmax) == spec.range

    def test_nan_is_painted_as_the_background(self):
        """Empty positions, out-of-range ids, non-fuel pins and threshold
        rejects all arrive as NaN, so one bad color covers all four."""
        mappable = draw_core_slice(fresh_axes(), make_core_slice(), style=ViewStyle())
        bad = mappable.get_cmap().get_bad()
        assert tuple(np.round(bad[:3], 4)) == pytest.approx(
            tuple(np.round(_rgb(DARK.background), 4))
        )

    def test_row_zero_is_at_the_top(self):
        ax = fresh_axes()
        draw_core_slice(ax, make_core_slice())
        lower, upper = ax.get_ylim()
        assert lower > upper, "a map reads top-down"

    def test_non_square_pins_invert_the_aspect(self):
        """aspect_ratio is dx/dy of a pin; set_aspect takes y-unit over
        x-unit, so passing it straight through squashes the map."""
        ax = fresh_axes()
        draw_core_slice(ax, make_core_slice(aspect_ratio=2.0))
        assert ax.get_aspect() == pytest.approx(0.5)

    def test_values_are_drawn_on_a_small_assembly(self):
        ax = fresh_axes()
        draw_core_slice(ax, make_core_slice(cell=(2, 2)), style=VALUES)
        assert ax.texts

    def test_values_are_suppressed_on_a_wide_assembly(self):
        ax = fresh_axes()
        side = MAX_LABEL_SIDE + 1
        draw_core_slice(ax, make_core_slice(cell=(side, side)), style=VALUES)
        assert not ax.texts, "text is illegible once an assembly is this wide"

    def test_no_text_sits_on_a_nan_cell(self):
        ax = fresh_axes()
        draw_core_slice(ax, make_core_slice(), style=VALUES)
        assert all(text.get_text() for text in ax.texts)

    def test_axis_labels_can_be_turned_off(self):
        ax = fresh_axes()
        draw_core_slice(ax, make_core_slice(), style=ViewStyle(show_axis_labels=False))
        assert list(ax.get_xticks()) == []


# -- surface ---------------------------------------------------------------


class TestSurfacePolygons:
    def test_four_triangles_per_node(self):
        slice_ = make_surface_slice(side=2, rows=2, cols=2)
        polygons, values, positions = surface_polygons(slice_)
        filled = 3  # one grid position is empty
        assert len(polygons) == filled * slice_.n_nodes * 4
        assert len(values) == len(polygons) == len(positions)

    def test_every_triangle_has_three_vertices(self):
        polygons, _, _ = surface_polygons(make_surface_slice())
        assert {len(p) for p in polygons} == {3}

    def test_triangles_meet_at_the_node_centre(self):
        polygons, _, _ = surface_polygons(make_surface_slice(side=1, rows=1, cols=1))
        centres = {tuple(np.round(p[-1], 6)) for p in polygons[:4]}
        assert len(centres) == 1

    def test_empty_positions_are_skipped(self):
        slice_ = make_surface_slice(rows=2, cols=2)
        polygons, _, _ = surface_polygons(slice_)
        assert len(polygons) < 4 * slice_.n_nodes * 4

    def test_columns_counts_thirds_of_a_node(self):
        """The binding constraint is one node's east label meeting the next
        node's west label, a third of a node apart."""
        slice_ = make_surface_slice(side=2, cols=4)
        assert surface_columns(slice_) == 4 * 2 * LABELS_ACROSS_A_NODE


class TestDrawSurfaceSlice:
    def test_draws_with_no_explicit_color(self):
        slice_ = make_surface_slice()
        mappable = draw_surface_slice(fresh_axes(), slice_)
        assert (mappable.norm.vmin, mappable.norm.vmax) == slice_.value_range

    def test_adds_a_polygon_collection(self):
        ax = fresh_axes()
        draw_surface_slice(ax, make_surface_slice())
        assert any(isinstance(c, PolyCollection) for c in ax.collections)

    def test_one_label_per_face(self):
        ax = fresh_axes()
        slice_ = make_surface_slice(side=1, rows=2, cols=2)
        draw_surface_slice(ax, slice_, style=VALUES)
        assert len(ax.texts) == 3 * 4  # three filled positions, four faces each

    def test_row_zero_is_at_the_top(self):
        ax = fresh_axes()
        draw_surface_slice(ax, make_surface_slice())
        lower, upper = ax.get_ylim()
        assert lower > upper


# -- axial -----------------------------------------------------------------


class TestAxialGeometry:
    def test_assembly_edges_drop_sub_columns(self):
        slice_ = make_axial_slice(cell_width=2, n_cols=7)
        assert len(assembly_edges(slice_)) == 8

    def test_columns_counts_values_not_assemblies(self):
        assert axial_columns(make_axial_slice(cell_width=2, n_cols=7)) == 14

    def test_aspect_is_the_real_centimetre_ratio(self):
        slice_ = make_axial_slice()
        x_lo, x_hi, y_lo, y_hi = slice_.extent
        assert axial_aspect(slice_) == pytest.approx((y_hi - y_lo) / (x_hi - x_lo))


class TestDrawAxialSlice:
    def test_draws_with_no_explicit_color(self):
        slice_ = make_axial_slice()
        mappable = draw_axial_slice(fresh_axes(), slice_)
        assert (mappable.norm.vmin, mappable.norm.vmax) == slice_.value_range

    def test_elevation_runs_upward(self):
        """Unlike a map, this axis is a physical height."""
        ax = fresh_axes()
        draw_axial_slice(ax, make_axial_slice())
        lower, upper = ax.get_ylim()
        assert lower < upper

    def test_aspect_is_locked_at_one(self):
        ax = fresh_axes()
        draw_axial_slice(ax, make_axial_slice())
        assert ax.get_aspect() == pytest.approx(1.0)

    def test_cells_follow_the_non_uniform_mesh(self):
        ax = fresh_axes()
        slice_ = make_axial_slice()
        draw_axial_slice(ax, slice_)
        (mesh,) = [c for c in ax.collections]
        drawn = np.unique(np.round(mesh.get_coordinates()[:, 0, 1], 6))
        assert np.allclose(drawn, slice_.y_edges)

    def test_values_are_suppressed_on_a_wide_cell(self):
        ax = fresh_axes()
        draw_axial_slice(ax, make_axial_slice(cell_width=17), style=VALUES)
        assert not ax.texts

    def test_values_are_drawn_on_a_narrow_cell(self):
        ax = fresh_axes()
        draw_axial_slice(ax, make_axial_slice(cell_width=2), style=VALUES)
        assert ax.texts


# -- figures ---------------------------------------------------------------


@pytest.mark.parametrize(
    "build, make",
    [
        (core_view_figure, make_core_slice),
        (surface_view_figure, make_surface_slice),
        (axial_view_figure, make_axial_slice),
    ],
    ids=["core", "surface", "axial"],
)
class TestPanelFigures:
    def test_empty_slice_list_raises(self, build, make):
        with pytest.raises(ValueError, match="no slices"):
            build([])

    def test_one_group_gives_one_panel_and_one_colorbar(self, build, make):
        figure = build([make()])
        assert len(figure.axes) == 2

    def test_four_groups_give_four_panels(self, build, make):
        groups = [make(group=g, n_groups=4) for g in range(4)]
        assert len(build(groups).axes) == 8

    def test_group_titles_appear_only_when_grouped(self, build, make):
        single = build([make()])
        grouped = build([make(group=g, n_groups=2) for g in range(2)])
        assert not any(ax.get_title() for ax in single.axes)
        assert any("Group" in ax.get_title() for ax in grouped.axes)

    def test_colorbar_matches_the_map_height(self, build, make):
        """A locked aspect shrinks the map inside its cell while the bar fills
        it, so the bar has to be snapped after the layout resolves."""
        figure = build([make()])
        plot, bar = figure.axes[0].get_position(), figure.axes[1].get_position()
        assert bar.y0 == pytest.approx(plot.y0, abs=1e-6)
        assert bar.height == pytest.approx(plot.height, abs=1e-6)

    def test_theme_sets_the_figure_facecolor(self, build, make):
        figure = build([make()], style=ViewStyle(theme=LIGHT))
        assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)

    def test_value_text_is_sized_to_the_measured_map(self, build, make):
        """Sizing from the panel width overflows, because the row labels and
        the colorbar take part of it."""
        seeded = build([make()], style=ViewStyle(show_values=True, value_size=None))
        drawn = [t for ax in seeded.axes for t in ax.texts]
        if not drawn:
            pytest.skip("this view does not label at the fixture's cell width")
        assert len({round(t.get_fontsize(), 3) for t in drawn}) == 1

    def test_explicit_value_size_is_left_alone(self, build, make):
        figure = build([make()], style=ViewStyle(show_values=True, value_size=9.0))
        drawn = [t for ax in figure.axes for t in ax.texts]
        if not drawn:
            pytest.skip("this view does not label at the fixture's cell width")
        assert {t.get_fontsize() for t in drawn} == {9.0}


def _rgb(hex_color: str) -> tuple[float, float, float]:
    from matplotlib.colors import to_rgb

    return to_rgb(hex_color)
