from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..analysis.axial_slice import (
    X_AXIS,
    AxialRequest,
    AxialSlice,
    axial_slices,
)
from ..analysis.color import ColorSpec, resolve_color_specs
from ..analysis.info import create_info
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
    frame_axes,
    panel_figure,
    value_formatter,
    write_figure,
)

ELEVATION_LABEL = "Elevation (cm)"


def assembly_edges(slice_: AxialSlice) -> np.ndarray:
    """x boundaries between assemblies, dropping the sub-column divisions."""
    return slice_.x_edges[:: slice_.cell_width]


def axial_columns(slice_: AxialSlice) -> int:
    """Label columns across the cut: one per value, not one per assembly."""
    return slice_.n_cols * slice_.cell_width


def axial_aspect(slice_: AxialSlice) -> float:
    """Drawn height over width. Both axes are cm, so this is the real ratio."""
    x_lo, x_hi, y_lo, y_hi = slice_.extent
    width = x_hi - x_lo
    return (y_hi - y_lo) / width if width else 1.0


def _draw_axis_labels(ax: Axes, slice_: AxialSlice, style: ViewStyle):
    """Assembly labels below the cut, elevation up the left in cm."""
    edges = assembly_edges(slice_)
    centers = (edges[:-1] + edges[1:]) / 2
    ax.set_xticks(centers, labels=list(slice_.x_labels)[: slice_.n_cols])
    ax.set_ylabel(ELEVATION_LABEL, color=style.theme.foreground)
    ax.tick_params(
        which="major",
        length=0,
        colors=style.theme.foreground,
        labelsize=style.axis_label_size,
    )


def _draw_values(ax: Axes, slice_: AxialSlice, mappable: ScalarMappable, style: ViewStyle):
    """Numeric text per value, colored for contrast against its cell."""
    size = style.value_size or FALLBACK_VALUE_SIZE
    write = value_formatter(slice_.finite(), style)
    x_centers = (slice_.x_edges[:-1] + slice_.x_edges[1:]) / 2
    y_centers = slice_.elevations
    image = slice_.as_image()
    for (row, col), value in np.ndenumerate(image):
        if np.isnan(value):
            continue
        ax.text(
            x_centers[col],
            y_centers[row],
            write(value),
            ha="center",
            va="center",
            fontsize=size,
            color=contrast_color(mappable.to_rgba(value), style.theme),
        )


def draw_axial_slice(
    ax: Axes,
    slice_: AxialSlice,
    color: ColorSpec | None = None,
    style: ViewStyle | None = None,
) -> ScalarMappable:
    """Render one axial cut into ax and return its mappable for a colorbar."""
    if style is None:
        style = ViewStyle()
    spec = resolve_color_specs([slice_], color)[0]
    x_lo, x_hi, y_lo, y_hi = slice_.extent

    mesh = ax.pcolormesh(
        slice_.x_edges,
        slice_.y_edges,
        np.ma.masked_invalid(slice_.as_image()),
        cmap=colormap(spec, style.theme),
        norm=Normalize(vmin=spec.vmin, vmax=spec.vmax),
        shading="flat",
    )
    frame_axes(ax, (x_lo, x_hi), (y_lo, y_hi), 1.0, style)

    if style.show_grid:
        _draw_assembly_boundaries(ax, slice_, style)
    if style.show_axis_labels:
        _draw_axis_labels(ax, slice_, style)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if style.show_values and slice_.labelable:
        _draw_values(ax, slice_, mesh, style)
    return mesh


def _draw_assembly_boundaries(ax: Axes, slice_: AxialSlice, style: ViewStyle):
    """One vertical line per assembly boundary. Layer boundaries are left to
    the color breaks."""
    ax.set_xticks(assembly_edges(slice_), minor=True)
    ax.grid(
        which="minor",
        axis="x",
        color=style.theme.grid,
        linewidth=style.grid_width,
        alpha=0.9,
    )
    ax.tick_params(which="minor", length=0)


def axial_view_figure(slices: list[AxialSlice], **kwargs):
    """Figure for one request's groups. kwargs go to panel_figure()."""
    return panel_figure(slices, draw_axial_slice, axial_columns, axial_aspect, **kwargs)


def save_axial_view(path, slices: list[AxialSlice], *, dpi: int = DEFAULT_DPI, **kwargs):
    """Build and write in one call. Format follows the suffix."""
    return write_figure(axial_view_figure(slices, **kwargs), path, dpi)


class AxialView(View[AxialRequest]):
    """Axial cuts for one loaded source.

    An AxialRequest takes axis, assembly and pin rather than z: a cut spans
    every axial level, so there is no single level to select.
    """

    request_type = AxialRequest

    def select(
        self,
        array: str | VeraDataset,
        *,
        state: int = 0,
        assembly: int = 0,
        pin: int = 0,
        axis: str = X_AXIS,
        src_id: str | None = None,
        thresholds: Sequence[ThresholdCondition] = (),
        mask_reflected: bool = True,
        group: int | None = None,
    ) -> Selection[AxialRequest]:
        """An axial cut through one assembly, at one dataset and state.

        There is no z: a cut spans every axial level.

        array           dataset name, or the VeraDataset itself
        state           state-point index
        assembly        assembly the cut passes through, from
                        core.reduced_core_map_assembly(i, j)
        pin             pin across the cut: j for an x cut, i for a y cut
        axis            X_AXIS along a core row, Y_AXIS down a column
        src_id          source id, when the request outlives this view
        thresholds      conditions that blank values before rendering
        mask_reflected  drop reflected assemblies
        group           one energy group, or None for every group
        """
        return Selection(
            self,
            AxialRequest(
                array=array,
                state=state,
                assembly=assembly,
                pin=pin,
                axis=axis,
                src_id=src_id,
                thresholds=thresholds,
                mask_reflected=mask_reflected,
                group=group,
            ),
        )

    build_slices = staticmethod(axial_slices)
    build_info = staticmethod(create_info)
    draw = staticmethod(draw_axial_slice)
    columns = staticmethod(axial_columns)
    aspect = staticmethod(axial_aspect)
