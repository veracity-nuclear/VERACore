from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

from ..analysis.color import ColorSpec, resolve_color_specs
from ..analysis.info import create_info
from ..analysis.surface_slice import (
    SurfaceRequest,
    SurfaceSlice,
    surface_slices,
)
from ..dtypes import VeraDataset
from ..thresholds import ThresholdCondition
from .layout import (
    DEFAULT_DPI,
    FALLBACK_VALUE_SIZE,
    Selection,
    View,
    ViewStyle,
    colormap,
    contrast_color,
    draw_axis_labels,
    draw_grid,
    frame_axes,
    panel_figure,
    value_formatter,
    write_figure,
)

LABEL_INSET = 1 / 6
"""Where a face's text sits between the node edge and its centre, as a
fraction of the node."""

LABELS_ACROSS_A_NODE = 3
"""Label columns per node, for text sizing. The binding constraint is not the
two labels inside a node but the gap between one node's east label and the
next node's west label: their centres are LABEL_INSET from either side of the
shared edge, so each label gets a third of a node."""


def node_triangles(x0: float, y0: float, step: float):
    """The four faces of one node, in W, N, E, S order.

    Yields (vertices, label_x, label_y).
    """
    x1, y1 = x0 + step, y0 + step
    cx, cy = x0 + step / 2, y0 + step / 2
    near, far = step * LABEL_INSET, step * (1 - LABEL_INSET)
    yield [(x0, y0), (x0, y1), (cx, cy)], x0 + near, cy  # W
    yield [(x0, y0), (x1, y0), (cx, cy)], cx, y0 + near  # N
    yield [(x1, y0), (x1, y1), (cx, cy)], x0 + far, cy  # E
    yield [(x0, y1), (x1, y1), (cx, cy)], cx, y0 + far  # S


def surface_polygons(slice_: SurfaceSlice):
    """Every face of every node as (vertices, values, label positions)."""
    side = slice_.node_side
    polygons, values, positions = [], [], []
    for (row, col, node_y, node_x), faces in _nodes(slice_):
        x0 = col * side + node_x
        y0 = row * side + node_y
        for (vertices, label_x, label_y), value in zip(
            node_triangles(x0, y0, 1.0), faces, strict=False
        ):
            polygons.append(vertices)
            values.append(value)
            positions.append((label_x, label_y))
    return polygons, np.array(values, dtype=float), positions


def _nodes(slice_: SurfaceSlice):
    """(row, col, node_y, node_x), face values for every node carrying data."""
    n_rows, n_cols = slice_.grid_shape
    side = slice_.node_side
    for row in range(n_rows):
        for col in range(n_cols):
            payload = slice_.data[row, col]
            for node_y in range(side):
                for node_x in range(side):
                    faces = payload[node_y, node_x]
                    if np.isnan(faces).all():
                        continue
                    yield (row, col, node_y, node_x), faces


def surface_columns(slice_: SurfaceSlice) -> int:
    """Label columns across the map. Two per node, since west and east text
    sit side by side within one node."""
    return slice_.grid_shape[1] * slice_.node_side * LABELS_ACROSS_A_NODE


def _draw_values(
    ax: Axes,
    values: np.ndarray,
    positions,
    mappable: ScalarMappable,
    style: ViewStyle,
):
    """Numeric text per face, colored for contrast against its triangle."""
    size = style.value_size or FALLBACK_VALUE_SIZE
    finite = values[~np.isnan(values)]
    write = value_formatter(finite, style)
    for value, (x, y) in zip(values, positions, strict=False):
        if np.isnan(value):
            continue
        ax.text(
            x,
            y,
            write(value),
            ha="center",
            va="center",
            fontsize=size,
            color=contrast_color(mappable.to_rgba(value), style.theme),
        )


def draw_surface_slice(
    ax: Axes,
    slice_: SurfaceSlice,
    color: ColorSpec | None = None,
    style: ViewStyle | None = None,
) -> ScalarMappable:
    """Render one surface slice into ax and return its mappable."""
    if style is None:
        style = ViewStyle()
    spec = resolve_color_specs([slice_], color)[0]
    side = slice_.node_side
    n_rows, n_cols = slice_.grid_shape
    width, height = n_cols * side, n_rows * side

    polygons, values, positions = surface_polygons(slice_)
    faces = PolyCollection(
        polygons,
        array=np.ma.masked_invalid(values),
        cmap=colormap(spec, style.theme),
        norm=Normalize(vmin=spec.vmin, vmax=spec.vmax),
        edgecolors=style.theme.edge,
        linewidths=style.edge_width,
    )
    ax.add_collection(faces)
    frame_axes(ax, (0, width), (height, 0), 1.0 / slice_.aspect_ratio, style)

    if style.show_grid:
        draw_grid(ax, n_rows, n_cols, side, style)
    if style.show_axis_labels:
        draw_axis_labels(ax, slice_, side, style)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if style.show_values:
        _draw_values(ax, values, positions, faces, style)
    return faces


def surface_view_figure(slices: list[SurfaceSlice], **kwargs):
    """Figure for one request's groups. kwargs go to panel_figure()."""
    return panel_figure(slices, draw_surface_slice, surface_columns, **kwargs)


def save_surface_view(path, slices: list[SurfaceSlice], *, dpi: int = DEFAULT_DPI, **kwargs):
    """Build and write in one call. Format follows the suffix."""
    return write_figure(surface_view_figure(slices, **kwargs), path, dpi)


class SurfaceView(View[SurfaceRequest]):
    """Core surface maps for one loaded source."""

    request_type = SurfaceRequest

    def select(
        self,
        array: str | VeraDataset,
        *,
        state: int = 0,
        z: int = 0,
        src_id: str | None = None,
        thresholds: Sequence[ThresholdCondition] = (),
        mask_reflected: bool = True,
        group: int | None = None,
    ) -> Selection[SurfaceRequest]:
        """A surface map at one dataset, state and axial level.

        array           dataset name, or the VeraDataset itself
        state           state-point index
        z               axial level index
        src_id          source id, when the request outlives this view
        thresholds      conditions that blank values before rendering
        mask_reflected  drop reflected assemblies
        group           one energy group, or None for every group
        """
        return Selection(
            self,
            SurfaceRequest(
                array=array,
                state=state,
                z=z,
                src_id=src_id,
                thresholds=thresholds,
                mask_reflected=mask_reflected,
                group=group,
            ),
        )

    build_slices = staticmethod(surface_slices)
    build_info = staticmethod(create_info)
    draw = staticmethod(draw_surface_slice)
    columns = staticmethod(surface_columns)
