from typing import Sequence

import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from ..analysis.color import ColorSpec, resolve_color_specs
from ..analysis.core_slice import (
    CoreSlice,
    SliceRequest,
    assembly_side,
    core_slices,
)
from ..analysis.info import create_info
from ..dtypes import VeraDataset
from ..thresholds import ThresholdCondition
from .layout import (
    DEFAULT_DPI,
    FALLBACK_VALUE_SIZE,
    Selection,
    View,
    colormap,
    contrast_color,
    draw_axis_labels,
    draw_grid,
    frame_axes,
    panel_figure,
    value_formatter,
    write_figure,
)
from .styles import (
    DARK,
    LIGHT,
    ViewStyle,
    ViewTheme,
)

MAX_LABEL_SIDE = 2
"""Widest assembly that can legibly carry per-pin value text."""

# Kept for callers written against the pre-layout.py names. DARK and LIGHT
# are re-exported from layout for the same reason.
CoreTheme = ViewTheme
__all__ = [
    "DARK",
    "LIGHT",
    "CoreTheme",
    "CoreView",
    "core_view_figure",
    "save_core_view",
    "draw_core_slice",
]


def core_image(slice_: CoreSlice) -> np.ndarray:
    """The slice as one 2-D pin-resolution array."""
    data = slice_.data
    if len(slice_.cell_shape) == 1:
        side = assembly_side(slice_.cell_shape[0])
        data = data.reshape(*slice_.grid_shape, side, side)
    if data.ndim == 2:
        return data
    n_rows, n_cols, cell_h, cell_w = data.shape
    return data.transpose(0, 2, 1, 3).reshape(n_rows * cell_h, n_cols * cell_w)


def image_side(slice_: CoreSlice) -> int:
    """Pins across one assembly in the rendered image, 1 for cell-less data."""
    cell = slice_.cell_shape
    if len(cell) == 0:
        return 1
    return assembly_side(cell[0]) if len(cell) == 1 else cell[0]


def core_columns(slice_: CoreSlice) -> int:
    """Pins across the whole map, which is one label column each."""
    return slice_.grid_shape[1] * image_side(slice_)


def _draw_values(ax: Axes, image: np.ndarray, mappable: ScalarMappable, style: ViewStyle):
    """Numeric text per pin, colored for contrast against the cell."""
    size = style.value_size or FALLBACK_VALUE_SIZE
    finite = image[~np.isnan(image)]
    write = value_formatter(finite, style)
    for (y, x), value in np.ndenumerate(image):
        if np.isnan(value):
            continue
        ax.text(
            x + 0.5,
            y + 0.5,
            write(value),
            ha="center",
            va="center",
            fontsize=size,
            color=contrast_color(mappable.to_rgba(value), style.theme),
        )


def draw_core_slice(
    ax: Axes,
    slice_: CoreSlice,
    color: ColorSpec | None = None,
    style: ViewStyle | None = None,
) -> ScalarMappable:
    """Render one slice into ax and return its mappable for a colorbar."""
    if style is None:
        style = ViewStyle()
    spec = resolve_color_specs([slice_], color)[0]
    image = core_image(slice_)
    side = image_side(slice_)
    height, width = image.shape

    mappable = ax.imshow(
        image,
        cmap=colormap(spec, style.theme),
        norm=Normalize(vmin=spec.vmin, vmax=spec.vmax),
        extent=(0, width, height, 0),
        interpolation="nearest",
    )
    frame_axes(ax, (0, width), (height, 0), 1.0 / slice_.aspect_ratio, style)

    if style.show_grid:
        draw_grid(ax, *slice_.grid_shape, side, style)
    if style.show_axis_labels:
        draw_axis_labels(ax, slice_, side, style)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
    if style.show_values and side <= MAX_LABEL_SIDE:
        _draw_values(ax, image, mappable, style)
    return mappable


def core_view_figure(slices: list[CoreSlice], **kwargs):
    """Figure for one request's groups. kwargs go to panel_figure()."""
    return panel_figure(slices, draw_core_slice, core_columns, **kwargs)


def save_core_view(path, slices: list[CoreSlice], *, dpi: int = DEFAULT_DPI, **kwargs):
    """Build and write in one call. Format follows the suffix."""
    return write_figure(core_view_figure(slices, **kwargs), path, dpi)


class CoreView(View[SliceRequest]):
    """Core maps for one loaded source."""

    request_type = SliceRequest

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
    ) -> Selection[SliceRequest]:
        """A core map at one dataset, state and axial level.

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
            SliceRequest(
                array=array,
                state=state,
                z=z,
                src_id=src_id,
                thresholds=thresholds,
                mask_reflected=mask_reflected,
                group=group,
            ),
        )

    build_slices = staticmethod(core_slices)
    build_info = staticmethod(create_info)
    draw = staticmethod(draw_core_slice)
    columns = staticmethod(core_columns)
