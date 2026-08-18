"""Artists: functions that draw into one Axes.

The bottom layer of the renders package. Nothing here knows what a VERA
dataset is, and nothing here owns a figure. Every function takes an Axes a
caller has already made, so a view composes only the parts it needs.

    draw.py       artists, one Axes at a time      <- you are here
    canvas.py     the figure and its panels
    view.py       the selection interface
    *_view.py     one concrete view, wiring the three together
"""

import re
from collections.abc import Callable, Sequence

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

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
