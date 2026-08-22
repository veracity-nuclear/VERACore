"""Artists: functions that draw into one Axes."""

import re
from collections.abc import Callable, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure
from matplotlib.patches import Polygon, Rectangle

from ..analysis.color import ColorSpec
from .styles import CHAR_WIDTH_RATIO, ViewStyle, ViewTheme

FALLBACK_VALUE_SIZE = 6.0
MIN_VALUE_SIZE = 3.5
MAX_VALUE_SIZE = 22.0

LABEL_POINTS_PER_PANEL = 130.0
"""Total width in points that axis labels may occupy across a panel."""

COLORBAR_FRACTION = 0.06
COLORBAR_PAD = 0.03
BAR_TITLE_SIZE = 11.0


# -- text ------------------------------------------------------------------

_EXPONENT = re.compile(r"e([+-])0*(\d+)")


def tidy_exponent(text: str) -> str:
    """1e-05 -> 1e-5, 1.2e+06 -> 1.2e6. Python's padded exponent wastes two
    characters per label, which matters inside a cell."""
    return _EXPONENT.sub(lambda m: "e-" + m[2] if m[1] == "-" else "e" + m[2], text)


def value_formatter(values: Sequence[float], style: ViewStyle) -> Callable[[float], str]:
    """A formatter for one panel's cells. If any value needs scientific
    notation, every cell uses it."""
    digits = max(style.decimals, 1)
    if style.value_format == "fixed":
        return lambda value: f"{value:.{style.decimals}f}"
    scientific = style.value_format == "sci" or any(
        "e" in f"{value:.{digits}g}" for value in values
    )
    if scientific:
        return lambda value: tidy_exponent(f"{value:.{digits - 1}e}")
    return lambda value: f"{value:.{digits}g}"


def auto_value_size(map_width_pt: float, n_columns: int, label_chars: int) -> float:
    """Point size that fits label_chars across one cell of a map."""
    cell_points = map_width_pt / max(n_columns, 1)
    fitted = cell_points * 0.8 / (CHAR_WIDTH_RATIO * max(label_chars, 1))
    return float(np.clip(fitted, MIN_VALUE_SIZE, MAX_VALUE_SIZE))


def fit_axis_label_size(n_labels: int, style: ViewStyle) -> float:
    """Label size that fits n_labels across a panel, never above the style's."""
    fitted = LABEL_POINTS_PER_PANEL / max(n_labels, 1)
    return min(style.axis_label_size, max(fitted, 4.0))


def contrast_color(rgba, theme: ViewTheme) -> str:
    """Black or white, whichever reads on the color underneath."""
    if rgba is None:
        return theme.foreground
    red, green, blue = rgba[:3]
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "#000000" if luminance > 0.55 else "#ffffff"


def colormap(color: ColorSpec, theme: ViewTheme):
    """The named colormap with NaN painted as the page background."""
    return mpl.colormaps[color.cmap].with_extremes(bad=theme.background)


# -- artists ---------------------------------------------------------------


def frame_axes(
    ax: Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    aspect: float,
    style: ViewStyle,
):
    """Common axes setup: background, limits, locked aspect, no spines."""
    ax.set_facecolor(style.theme.background)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect(aspect)
    for spine in ax.spines.values():
        spine.set_visible(False)


def cells(
    ax: Axes,
    grid: np.ndarray,
    color: ColorSpec,
    style: ViewStyle,
    *,
    aspect_ratio: float = 1.0,
) -> ScalarMappable:
    """A grid of colored cells, one per array entry, origin at the top left.

    Data coordinates are cells, so a caller places ticks and text in cell
    units without knowing the pixel size. aspect_ratio is dx / dy of one
    cell, so a wide cell makes a short panel.
    """
    n_rows, n_cols = grid.shape
    image = ax.imshow(
        grid,
        cmap=colormap(color, style.theme),
        vmin=color.vmin,
        vmax=color.vmax,
        interpolation="nearest",
        extent=(0.0, float(n_cols), float(n_rows), 0.0),
    )
    frame_axes(ax, (0.0, n_cols), (n_rows, 0.0), 1.0 / (aspect_ratio or 1.0), style)
    return image


def block_grid(ax: Axes, n_rows: int, n_cols: int, side: int, style: ViewStyle):
    """One line every `side` cells: the boundaries between assemblies."""
    ax.set_xticks(np.arange(n_cols + 1) * side, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) * side, minor=True)
    ax.grid(which="minor", color=style.theme.grid, linewidth=style.grid_width, alpha=0.9)
    ax.tick_params(which="minor", length=0)


def axis_labels(
    ax: Axes,
    x_labels: Sequence,
    y_labels: Sequence,
    side: int,
    style: ViewStyle,
):
    """Column labels above, row labels to the left, centered on their block.

    Labels are already in map order, so no reversal: x_labels[0] is leftmost.
    """
    n_cols, n_rows = len(x_labels), len(y_labels)
    ax.set_xticks(np.arange(n_cols) * side + side / 2, labels=[str(x) for x in x_labels])
    ax.set_yticks(np.arange(n_rows) * side + side / 2, labels=[str(y) for y in y_labels])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(
        which="major",
        length=0,
        colors=style.theme.foreground,
        labelsize=fit_axis_label_size(max(n_cols, n_rows), style),
    )


def cell_values(
    ax: Axes,
    grid: np.ndarray,
    color: ColorSpec,
    style: ViewStyle,
    write: Callable[[float], str],
):
    """One label per finite cell, colored to read against its cell.

    Sized from the style; Panel.fit_values() corrects that once the figure is
    drawn and the real panel width is known.
    """
    cmap = colormap(color, style.theme)
    size = style.value_size or FALLBACK_VALUE_SIZE
    for (row, col), value in np.ndenumerate(grid):
        if not np.isfinite(value):
            continue
        ax.text(
            col + 0.5,
            row + 0.5,
            write(float(value)),
            ha="center",
            va="center",
            fontsize=size,
            color=contrast_color(cmap(float(color.normalize(value))), style.theme),
        )


FACE_CORNERS = (
    ((0.0, 0.0), (0.0, 1.0)),
    ((0.0, 0.0), (1.0, 0.0)),
    ((1.0, 0.0), (1.0, 1.0)),
    ((0.0, 1.0), (1.0, 1.0)),
)
"""The two corners of each face, W N E S, as offsets within one cell. A face
is drawn as those corners and the cell center, so the four of them cut the
cell into triangles meeting in the middle."""

FACE_ANCHORS = ((1 / 6, 0.5), (0.5, 1 / 6), (5 / 6, 0.5), (0.5, 5 / 6))
"""Where a face's label sits, a third of the way out from the center: far
enough apart that four labels in one cell do not collide."""


def face_cells(
    ax: Axes,
    faces: np.ndarray,
    color: ColorSpec,
    style: ViewStyle,
    *,
    aspect_ratio: float = 1.0,
) -> ScalarMappable:
    """A grid of cells cut into four triangles, one per lateral face.

    faces is (n_rows, n_cols, 4), the last axis ordered W N E S. Coordinates
    are cells with the origin at the top left, as in cells(), so a caller
    places ticks and text the same way for either.

    A face that is not finite is left undrawn rather than painted the
    background color, so an empty core position shows nothing at all.
    """
    n_rows, n_cols = faces.shape[:2]
    polygons, values = [], []
    for (row, col, face), value in np.ndenumerate(faces):
        if not np.isfinite(value):
            continue
        corners = FACE_CORNERS[face]
        polygons.append([(col + dx, row + dy) for dx, dy in corners] + [(col + 0.5, row + 0.5)])
        values.append(value)
    collection = PolyCollection(
        polygons,
        array=np.asarray(values, dtype=float),
        cmap=colormap(color, style.theme),
        norm=mpl.colors.Normalize(vmin=color.vmin, vmax=color.vmax),
        edgecolors=style.theme.edge,
        linewidths=style.edge_width,
    )
    ax.add_collection(collection)
    frame_axes(ax, (0.0, n_cols), (n_rows, 0.0), 1.0 / (aspect_ratio or 1.0), style)
    return collection


def face_values(
    ax: Axes,
    faces: np.ndarray,
    color: ColorSpec,
    style: ViewStyle,
    write: Callable[[float], str],
):
    """One label per finite face, inside its triangle. The counterpart of
    cell_values for a map drawn by face_cells."""
    cmap = colormap(color, style.theme)
    size = style.value_size or FALLBACK_VALUE_SIZE
    for (row, col, face), value in np.ndenumerate(faces):
        if not np.isfinite(value):
            continue
        offset_x, offset_y = FACE_ANCHORS[face]
        ax.text(
            col + offset_x,
            row + offset_y,
            write(float(value)),
            ha="center",
            va="center",
            fontsize=size,
            color=contrast_color(cmap(float(color.normalize(value))), style.theme),
        )


def highlight_span(
    ax: Axes,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    style: ViewStyle,
):
    """An outline around a rectangle of the map, in data coordinates."""
    (x0, x1), (y0, y1) = x_span, y_span
    ax.add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor=style.theme.highlight,
            linewidth=style.highlight_width,
            zorder=5,
        )
    )


def highlight_block(ax: Axes, row: int, col: int, side: int, style: ViewStyle):
    """An outline around one block of `side` cells: the chosen assembly."""
    highlight_span(ax, (col * side, (col + 1) * side), (row * side, (row + 1) * side), style)


def highlight_face(ax: Axes, row: int, col: int, face: int, style: ViewStyle):
    """An outline around one face's triangle in one cell: the chosen surface.

    face indexes FACE_CORNERS, so a caller passes the same order it drew in.
    """
    corners = FACE_CORNERS[face]
    ax.add_patch(
        Polygon(
            [(col + dx, row + dy) for dx, dy in corners] + [(col + 0.5, row + 0.5)],
            closed=True,
            fill=False,
            edgecolor=style.theme.highlight,
            linewidth=style.highlight_width,
            zorder=6,
        )
    )


# -- artists on an explicit mesh --------------------------------------------


def mesh_cells(
    ax: Axes,
    grid: np.ndarray,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    color: ColorSpec,
    style: ViewStyle,
    *,
    aspect: float = 1.0,
) -> ScalarMappable:
    """A grid of cells on explicit boundaries, for a map whose rows are not
    all the same height: an axial cut, where a layer is as tall as its mesh.

    Data coordinates are whatever the edges are measured in, cm for an axial
    cut, so a caller places ticks and text in those units too. grid is
    (len(y_edges) - 1, len(x_edges) - 1), row 0 against the first edge.
    aspect is the vertical scale over the horizontal one: 1.0 draws both in
    the same units, which is what makes an elevation read as a height.
    """
    mesh = ax.pcolormesh(
        np.asarray(x_edges, dtype=float),
        np.asarray(y_edges, dtype=float),
        grid,
        cmap=colormap(color, style.theme),
        vmin=color.vmin,
        vmax=color.vmax,
        shading="flat",
    )
    frame_axes(ax, (x_edges[0], x_edges[-1]), (y_edges[0], y_edges[-1]), aspect, style)
    return mesh


def mesh_grid(
    ax: Axes,
    x_lines: Sequence[float],
    y_lines: Sequence[float],
    style: ViewStyle,
):
    """Lines at explicit positions: block_grid for a map whose cells are not
    all the same size. Either sequence may be empty."""
    ax.set_xticks(list(x_lines), minor=True)
    ax.set_yticks(list(y_lines), minor=True)
    ax.grid(which="minor", color=style.theme.grid, linewidth=style.grid_width, alpha=0.9)
    ax.tick_params(which="minor", length=0)


def mesh_axis_labels(
    ax: Axes,
    x_labels: Sequence,
    x_centers: Sequence[float],
    style: ViewStyle,
    y_title: str = "",
):
    """Column labels above at given positions, a continuous scale down the
    left. The vertical axis of a cut is a measurement, not a row index, so it
    keeps matplotlib's own ticks instead of one label per row."""
    ax.set_xticks(list(x_centers), labels=[str(label) for label in x_labels])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(
        axis="x",
        which="major",
        length=0,
        colors=style.theme.foreground,
        labelsize=fit_axis_label_size(len(x_labels), style),
    )
    ax.tick_params(
        axis="y",
        which="major",
        length=3,
        colors=style.theme.foreground,
        labelsize=style.axis_label_size,
    )
    if y_title:
        ax.set_ylabel(y_title, color=style.theme.foreground, fontsize=style.axis_label_size)


def mesh_values(
    ax: Axes,
    grid: np.ndarray,
    x_edges: Sequence[float],
    y_edges: Sequence[float],
    color: ColorSpec,
    style: ViewStyle,
    write: Callable[[float], str],
):
    """One label per finite cell, centered in it. cell_values for a map drawn
    by mesh_cells."""
    cmap = colormap(color, style.theme)
    size = style.value_size or FALLBACK_VALUE_SIZE
    x_edges = np.asarray(x_edges, dtype=float)
    y_edges = np.asarray(y_edges, dtype=float)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    for (row, col), value in np.ndenumerate(grid):
        if not np.isfinite(value):
            continue
        ax.text(
            x_centers[col],
            y_centers[row],
            write(float(value)),
            ha="center",
            va="center",
            fontsize=size,
            color=contrast_color(cmap(float(color.normalize(value))), style.theme),
        )


# -- artists for a plot ----------------------------------------------------

LINE_CMAP = "tab10"
"""Where the line cycle comes from. Lines are separate series, not points on
a scale, so they take a qualitative map rather than the data's own."""

TICK_LABEL_RATIO = 0.8
"""Tick text against the style's axis label size."""

LEGEND_SIZE = 8.0


def line_colors(n_lines: int, cmap: str = LINE_CMAP) -> list:
    """n colors from a qualitative map, repeating once it runs out."""
    colors = mpl.colormaps[cmap].colors
    return [colors[index % len(colors)] for index in range(n_lines)]


def plot_axes(
    ax: Axes,
    style: ViewStyle,
    *,
    x_label: str = "",
    y_label: str = "",
):
    """A framed plot: themed background, spines, ticks and grid.

    frame_axes() locks an aspect and hides every spine, which is right for a
    map and wrong for a plot, where the axes are a scale to read against.
    """
    ax.set_facecolor(style.theme.background)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(style.theme.grid)
    ax.tick_params(
        colors=style.theme.foreground,
        labelsize=style.axis_label_size * TICK_LABEL_RATIO,
    )
    if style.show_grid:
        ax.grid(color=style.theme.grid, linewidth=style.grid_width, alpha=0.4)
        ax.set_axisbelow(True)
    if style.show_axis_labels:
        ax.set_xlabel(x_label, color=style.theme.foreground, fontsize=style.axis_label_size)
        ax.set_ylabel(y_label, color=style.theme.foreground, fontsize=style.axis_label_size)


def lines(
    ax: Axes,
    traces: Sequence,
    style: ViewStyle,
    *,
    colors: Sequence | None = None,
    width: float = 1.6,
    marker_size: float = 4.0,
) -> list:
    """One line per trace, each ((x, y), label, mode).

    A non-finite value breaks the line rather than joining across it, which
    is what draws a slice built of region segments as one bar per region.
    Returns the handles, for a legend the caller places.
    """
    traces = list(traces)
    colors = list(colors) if colors is not None else line_colors(len(traces))
    handles = []
    for index, ((x, y), label, mode) in enumerate(traces):
        (handle,) = ax.plot(
            x,
            y,
            label=label,
            color=colors[index % len(colors)],
            linewidth=0.0 if mode == "markers" else width,
            linestyle="none" if mode == "markers" else "-",
            marker="o" if "markers" in mode else None,
            markersize=marker_size,
        )
        handles.append(handle)
    return handles


def rule(ax: Axes, y: float, style: ViewStyle, *, color: str = "", label: str = ""):
    """A dashed line across the plot at one height: the chosen layer."""
    return ax.axhline(
        y,
        color=color or style.theme.highlight,
        linestyle="--",
        linewidth=style.highlight_width / 2,
        label=label or None,
        zorder=1.5,
    )


def legend(figure: Figure, handles: Sequence, style: ViewStyle, *, columns: int = 1):
    """A legend under the whole figure, below the caption.

    A plot names its lines where a map names its colors with a bar, so this
    is the counterpart of colorbar() and sits in the same place. It is
    anchored just below the figure rather than given to the layout engine,
    which would otherwise put it in the strip the caption already holds; the
    crop on save is what brings it back into the image.
    """
    return figure.legend(
        handles=list(handles),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.0),
        bbox_transform=figure.transFigure,
        ncols=columns,
        frameon=False,
        labelcolor=style.theme.foreground,
        fontsize=LEGEND_SIZE,
    )


def colorbar(
    figure: Figure,
    mappable: ScalarMappable,
    ax: Axes | Sequence[Axes],
    units: str = "",
    style: ViewStyle | None = None,
    title: str = "",
) -> Colorbar:
    """Vertical colorbar beside ax, endpoints only by default.

    Several axes may be given, for one bar reading a shared scale. Views
    normally reach this through Canvas.colorbar(), which also snaps the bar
    to the drawn maps.
    """
    if style is None:
        style = ViewStyle()
    color_range = mappable.norm.vmin, mappable.norm.vmax
    bar = figure.colorbar(mappable, ax=ax, fraction=COLORBAR_FRACTION, pad=COLORBAR_PAD)
    if style.endpoint_ticks_only:
        bar.set_ticks(list(color_range), labels=[tidy_exponent(f"{v:g}") for v in color_range])
    bar.ax.tick_params(length=0, colors=style.theme.foreground)
    bar.outline.set_visible(False)
    if units:
        # Anchored left, not centered: a units string is wider than the bar,
        # and centering spills it over whatever sits to the left.
        bar.ax.set_xlabel(units, color=style.theme.foreground, labelpad=8, loc="left")
    if title:
        # Names what this bar reads, for a figure holding more than one.
        bar.ax.set_title(title, color=style.theme.foreground, fontsize=BAR_TITLE_SIZE)
    return bar
