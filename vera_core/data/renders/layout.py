import dataclasses
import inspect
import math
import re
from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Protocol

import matplotlib as mpl
import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from ..analysis.color import ColorScope, ColorSource, ColorSpec, resolve_color_specs
from .styles import ViewStyle, ViewTheme

DEFAULT_DPI = 200
PANEL_WIDTH_IN = 5.0
CAPTION_ALLOWANCE_IN = 0.5
TITLE_ALLOWANCE_IN = 0.4
TITLE_SIZE = 14.0
PAD_INCHES = 0.2
"""The figure is sized generously and cropped on save: a locked aspect makes
the drawn height impossible to predict exactly."""

FALLBACK_VALUE_SIZE = 6.0
CHAR_WIDTH_RATIO = 0.62
"""Width of a digit over its point size, for fitting text to a cell."""

COLORBAR_FRACTION = 0.06
COLORBAR_PAD = 0.03
TITLE_PAD = 24
"""Group titles clear the column labels, which sit on the top axis."""


class PanelSlice(Protocol):
    """The attributes panel_figure() reads. Structural, so a slice type does
    not have to inherit anything."""

    grid_shape: tuple[int, int]
    aspect_ratio: float
    units: str
    group: int | None
    slice_value_range: tuple[float, float]
    group_value_range: tuple[float, float]
    dataset_value_range: tuple[float, float]

    def finite(self) -> np.ndarray: ...


DrawSlice = Callable[[Axes, PanelSlice, ColorSpec, "ViewStyle"], ScalarMappable]
"""Draws one slice into an Axes and returns its mappable."""

CountColumns = Callable[[PanelSlice], int]
"""How many label columns the slice spans, for sizing value text. A core map
counts pins; a surface map counts thirds of a node, since adjacent east and
west labels sit that far apart."""

PanelAspect = Callable[[PanelSlice], float]
"""Drawn height over drawn width of one panel, for sizing the figure."""


def map_aspect(slice_: PanelSlice) -> float:
    """Panel aspect for a view laid out on the assembly grid."""
    n_rows, n_cols = slice_.grid_shape
    return n_rows / (n_cols * slice_.aspect_ratio)


_EXPONENT = re.compile(r"e([+-])0*(\d+)")


def tidy_exponent(text: str) -> str:
    """1e-05 -> 1e-5, 1.2e+06 -> 1.2e6. Python's padded exponent wastes two
    characters per label, which matters inside a cell."""
    return _EXPONENT.sub(lambda m: "e-" + m[2] if m[1] == "-" else "e" + m[2], text)


def value_formatter(values: Sequence[float], style: ViewStyle) -> Callable[[float], str]:
    """A formatter for one panel's cells. If any
    value needs scientific notation, every cell uses it.
    """
    digits = max(style.decimals, 1)
    if style.value_format == "fixed":
        return lambda value: f"{value:.{style.decimals}f}"
    scientific = style.value_format == "sci" or any(
        "e" in f"{value:.{digits}g}" for value in values
    )
    if scientific:
        return lambda value: tidy_exponent(f"{value:.{digits - 1}e}")
    return lambda value: f"{value:.{digits}g}"


def auto_value_size(map_width_pt: float, n_cols: int, label_chars: int) -> float:
    """Point size that fits label_chars across one cell of the map."""
    cell_points = map_width_pt / max(n_cols, 1)
    fitted = cell_points * 0.8 / (CHAR_WIDTH_RATIO * max(label_chars, 1))
    return float(np.clip(fitted, 3.5, 22.0))


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


def draw_grid(ax: Axes, n_rows: int, n_cols: int, side: int, style: ViewStyle):
    """One line per assembly boundary, across the full bounding grid."""
    ax.set_xticks(np.arange(n_cols + 1) * side, minor=True)
    ax.set_yticks(np.arange(n_rows + 1) * side, minor=True)
    ax.grid(which="minor", color=style.theme.grid, linewidth=style.grid_width, alpha=0.9)
    ax.tick_params(which="minor", length=0)


LABEL_POINTS_PER_PANEL = 130.0
"""Total width in points that axis labels may occupy across a panel."""


def fit_axis_label_size(n_labels: int, style: ViewStyle) -> float:
    """Label size that fits n_labels across a panel, never above the style's."""
    fitted = LABEL_POINTS_PER_PANEL / max(n_labels, 1)
    return min(style.axis_label_size, max(fitted, 4.0))


def draw_axis_labels(ax: Axes, slice_: PanelSlice, side: int, style: ViewStyle):
    """Column letters above, row numbers to the left, centered on assemblies.

    Labels are already in map order, so no reversal: x_labels[0] is leftmost.
    """
    n_rows, n_cols = slice_.grid_shape
    ax.set_xticks(np.arange(n_cols) * side + side / 2, labels=list(slice_.x_labels)[:n_cols])
    ax.set_yticks(np.arange(n_rows) * side + side / 2, labels=list(slice_.y_labels)[:n_rows])
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(
        which="major",
        length=0,
        colors=style.theme.foreground,
        labelsize=style.axis_label_size,
    )


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


def draw_colorbar(
    figure: Figure,
    mappable: ScalarMappable,
    ax: Axes,
    units: str = "",
    style: ViewStyle | None = None,
) -> Colorbar:
    """Vertical colorbar beside ax, endpoints only by default."""
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
    return bar


def finalize_layout(
    figure: Figure, panels: list[tuple[Axes, Axes, int]], rescale_values: bool
) -> None:
    """Resolve the layout, then correct what only measurement can settle.

    Two things cannot be computed before the layout runs. A map has a locked
    aspect, so it shrinks inside its cell while the colorbar fills that cell;
    the bar has to be snapped to the map's drawn box. And value text sized
    against the panel width overflows, because the row labels and the colorbar
    take part of that width. Both are fixed here from measurements.
    """
    figure.draw_without_rendering()
    measured = [
        (ax.get_position(), cax.get_position(), ax.get_window_extent().width)
        for ax, cax, _ in panels
    ]
    figure.set_layout_engine("none")
    for (ax, cax, n_cols), (ax_box, cax_box, width_px) in zip(panels, measured, strict=False):
        cax.set_position([cax_box.x0, ax_box.y0, cax_box.width, ax_box.height])
        if not rescale_values or not ax.texts:
            continue
        width_pt = width_px * 72.0 / figure.dpi
        chars = max(len(text.get_text()) for text in ax.texts)
        for text in ax.texts:
            text.set_fontsize(auto_value_size(width_pt, n_cols, chars))


def request_title(request) -> str:
    """A heading for one request.

    'PIN POWERS' for a single source, 'PIN POWERS | vera2' when the request
    names one.
    """
    name = getattr(request, "array_name", None) or str(getattr(request, "array", ""))
    heading = name.replace("_", " ").upper()
    src_id = getattr(request, "src_id", None)
    return heading if src_id is None else f"{heading} | {src_id}"


def panel_grid(n_groups: int) -> tuple[int, int]:
    """(n_rows, n_cols) of panels: one row up to two groups, then 2 wide"""
    n_cols = 1 if n_groups == 1 else 2
    return math.ceil(n_groups / n_cols), n_cols


def _sized_style(
    style: ViewStyle,
    slices: Sequence[PanelSlice],
    columns: CountColumns,
    panel_width: float,
) -> ViewStyle:
    """Seed an automatic value size from the widest label any cell will carry."""
    if not style.show_values or style.value_size is not None:
        return style
    widest = 1
    for slice_ in slices:
        finite = slice_.finite()
        write = value_formatter(finite, style)
        widest = max(widest, *(len(write(v)) for v in finite), 1)
    seed = auto_value_size(panel_width * 72.0, columns(slices[0]), widest)
    return replace(style, value_size=seed)


def panel_figure(
    slices: Sequence[PanelSlice],
    draw: DrawSlice,
    columns: CountColumns,
    aspect: PanelAspect = map_aspect,
    *,
    info=None,
    title: str | None = None,
    color: ColorSource = None,
    style: ViewStyle | None = None,
    color_scope: ColorScope = ColorScope.SLICE,
    panel_width: float = PANEL_WIDTH_IN,
) -> Figure:
    """Lay out one panel per group and draw each with the view's artist.

    A title sits above every panel; group titles label each one; the caption
    runs below. Group titles appear only for multi-group datasets.
    """
    if style is None:
        style = ViewStyle()
    if not slices:
        raise ValueError("no slices to render")

    n_rows, n_cols = panel_grid(len(slices))
    panel_aspect = aspect(slices[0])
    figure = Figure(
        figsize=(
            n_cols * panel_width,
            n_rows * panel_width * panel_aspect
            + CAPTION_ALLOWANCE_IN
            + (TITLE_ALLOWANCE_IN if title else 0.0),
        ),
        facecolor=style.theme.background,
        layout="constrained",
    )
    FigureCanvasAgg(figure)

    grid = figure.add_gridspec(n_rows, n_cols)
    specs = resolve_color_specs(slices, color, scope=color_scope)
    auto_values = style.show_values and style.value_size is None
    style = _sized_style(style, slices, columns, panel_width)
    panels = []

    for index, (slice_, spec) in enumerate(zip(slices, specs, strict=False)):
        row, col = divmod(index, n_cols)
        ax = figure.add_subplot(grid[row, col])
        mappable = draw(ax, slice_, spec, style)
        if len(slices) > 1:
            ax.set_title(f"Group {index + 1}", color=style.theme.foreground, pad=TITLE_PAD)
        bar = draw_colorbar(figure, mappable, ax, slice_.units, style)
        panels.append((ax, bar.ax, columns(slice_)))

    if title:
        figure.suptitle(title, color=style.theme.foreground, fontsize=TITLE_SIZE)
    if info is not None:
        figure.supxlabel(info.caption(), color=style.theme.foreground, fontsize=9)
    finalize_layout(figure, panels, auto_values)
    return figure


def write_figure(figure: Figure, path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """Write a built figure, cropping the slack the locked aspect leaves.

    The facecolor is passed explicitly.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        path,
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=PAD_INCHES,
    )
    return path


class Selection[RequestT]:
    """One request bound to the view that can render it."""

    def __init__(self, view: "View[RequestT]", request: RequestT):
        self.view = view
        self.request = request
        self.title: str | bool = False

    def replace(self, **changes) -> "Selection[RequestT]":
        """A selection for the same view with some request fields changed.

        base = view.select("pin_powers", state=2, z=0)
        for z in range(n_layers):
            base.replace(z=z).savefig(f"z{z}.png")
        """
        return Selection(self.view, dataclasses.replace(self.request, **changes))

    def slices(self) -> list:
        """The renderable groups for this request."""
        return self.view.slices(self.request)

    def figure(
        self,
        *,
        title: str | bool = False,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        **kwargs,
    ) -> Figure:
        """The rendered figure. kwargs go to View.figure()."""
        title = title if title else self.title
        return self.view.figure(
            self.request,
            title=self.title,
            style=style,
            color=color,
            color_scope=color_scope,
            **kwargs,
        )

    def savefig(
        self,
        path: str | Path,
        *,
        title: str | bool = False,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        **kwargs,
    ) -> Path:
        """Render and write to path. Format follows the suffix."""
        title = title if title else self.title
        return self.view.savefig(
            self.request,
            path,
            title=title,
            style=style,
            color=color,
            color_scope=color_scope,
            **kwargs,
        )

    def __repr__(self) -> str:
        return f"<{type(self.view).__name__} selection {self.request.label()}>"


class View[RequestT]:
    """Base for render views, bound to one loaded source.

    A subclass sets six hooks and inherits select(), slices(), figure() and
    savefig():

        request_type   the frozen request dataclass select() builds
        build_slices   (source, request) -> list of slices
        build_info     (source, request) -> caption info
        draw           (ax, slice, spec, style) -> mappable
        columns        (slice) -> label columns spanned
        aspect         (slice) -> drawn height over width
    """

    request_type: type
    build_slices: Callable
    build_info: Callable
    draw: DrawSlice
    columns: CountColumns
    aspect: PanelAspect = staticmethod(map_aspect)

    def __init__(
        self,
        source,
        *,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope = ColorScope.SLICE,
    ):
        if style is None:
            style = ViewStyle()
        self.source = source
        self.style = style
        self.color = color
        self.color_scope = color_scope

    def select(self, array, **kwargs) -> Selection[RequestT]:
        """Build this view's request and bind it, ready to render."""
        return Selection(self, self.request_type(array=array, **kwargs))

    def slices(self, request: RequestT) -> list:
        """The renderable groups for one request."""
        found = self.build_slices(self.source, request)
        if not found:
            raise ValueError(f"{request.label()} has no view of this kind")
        if request.group is None:
            return found
        return [s for s in found if s.group == request.group]

    def figure(
        self,
        request: RequestT,
        *,
        title: str | bool = False,
        caption: bool = True,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        panel_width: float = PANEL_WIDTH_IN,
    ) -> Figure:
        """The rendered figure, for callers that want to add to it or embed it."""
        return panel_figure(
            self.slices(request),
            self.draw,
            self.columns,
            self.aspect,
            info=self.build_info(self.source, request) if caption else None,
            title=request_title(request) if title is True else (title or None),
            color=color if color is not None else self.color,
            style=style or self.style,
            color_scope=self.color_scope if color_scope is None else color_scope,
            panel_width=panel_width,
        )

    def savefig(
        self,
        request: RequestT,
        path: str | Path,
        *,
        title: str | bool = False,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        dpi: int = DEFAULT_DPI,
        **kwargs,
    ) -> Path:
        """Render one request and write it to path.

        Format follows the suffix: .png, .pdf, .svg.
        """
        return write_figure(
            self.figure(
                request, title=title, style=style, color=color, color_scope=color_scope, **kwargs
            ),
            path,
            dpi,
        )


def assert_select_matches(view_type: type) -> None:
    """Raise if a view's select() has drifted from its request dataclass."""
    fields = {f.name for f in dataclasses.fields(view_type.request_type)}
    params = set(inspect.signature(view_type.select).parameters) - {"self"}
    problems = []
    if fields - params:
        problems.append(f"select() is missing {sorted(fields - params)}")
    if params - fields:
        problems.append(f"select() has {sorted(params - fields)}, not on the request")
    if problems:
        raise AssertionError(f"{view_type.__name__}: {'; '.join(problems)}")
