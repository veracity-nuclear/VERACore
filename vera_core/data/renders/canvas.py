"""The figure and its panels.

A Canvas is one output image. It holds one Panel per group, sizes the figure,
and runs the correction pass that only a drawn figure can settle. It does not
know what a panel contains: a view draws into panel.ax with the artists in
draw.py and tells the panel about the two things that need measuring.

    draw.py       artists, one Axes at a time
    canvas.py     the figure and its panels        <- you are here
    view.py       the selection interface
    *_view.py     one concrete view, wiring the three together
"""

import math
from pathlib import Path

from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from matplotlib.figure import Figure

from . import draw
from .styles import ViewStyle

DEFAULT_DPI = 200
PANEL_WIDTH_IN = 5.0
CAPTION_ALLOWANCE_IN = 0.5
TITLE_ALLOWANCE_IN = 0.4
TITLE_SIZE = 14.0
CAPTION_SIZE = 9.0
PAD_INCHES = 0.2
"""The figure is sized generously and cropped on save: a locked aspect makes
the drawn height impossible to predict exactly."""

PANEL_TITLE_PAD = 24
"""Panel titles clear the column labels, which sit on the top axis."""


def map_aspect(n_rows: int, n_cols: int, aspect_ratio: float = 1.0) -> float:
    """Drawn height over drawn width of a map laid out on a cell grid.

    What a view passes as Canvas(panel_aspect=...). aspect_ratio is dx / dy
    of one cell, so a wide cell makes a short panel.
    """
    return n_rows / (n_cols * (aspect_ratio or 1.0))


def panel_grid(n_panels: int) -> tuple[int, int]:
    """(n_rows, n_cols) of panels: one row up to two, then two wide."""
    n_cols = 1 if n_panels == 1 else 2
    return math.ceil(n_panels / n_cols), n_cols


class Panel:
    """One subplot of a Canvas.

    A view draws into `ax` with the artists in draw.py, and calls the methods
    here for the parts the canvas has to settle by measurement.
    """

    def __init__(self, figure: Figure, ax: Axes, style: ViewStyle):
        self.figure = figure
        self.ax = ax
        self.style = style
        self.colorbar_ax: Axes | None = None
        self.value_columns: int | None = None

    def title(self, text: str) -> None:
        self.ax.set_title(text, color=self.style.theme.foreground, pad=PANEL_TITLE_PAD)

    def colorbar(self, mappable: ScalarMappable, units: str = "") -> Colorbar:
        """A colorbar for this panel, snapped to the drawn map on finish."""
        bar = draw.colorbar(self.figure, mappable, self.ax, units, self.style)
        self.colorbar_ax = bar.ax
        return bar

    def fit_values(self, n_columns: int) -> None:
        """Resize this panel's text on finish so n_columns labels fit across
        the drawn map. Ignored when the style fixes value_size."""
        if self.style.value_size is None:
            self.value_columns = n_columns

    def _settle(self, ax_box, width_pt: float) -> None:
        """Correct what the measured layout revealed.

        A map has a locked aspect, so it shrinks inside its cell while the
        colorbar fills that cell; the bar is snapped back to the map's box.
        Value text sized against the nominal panel width overflows, because
        the row labels and the colorbar take part of that width.
        """
        if self.colorbar_ax is not None:
            box = self.colorbar_ax.get_position()
            self.colorbar_ax.set_position([box.x0, ax_box.y0, box.width, ax_box.height])
        if not self.value_columns or not self.ax.texts:
            return
        chars = max(len(text.get_text()) for text in self.ax.texts)
        size = draw.auto_value_size(width_pt, self.value_columns, chars)
        for text in self.ax.texts:
            text.set_fontsize(size)


class Canvas:
    """One output image: equally sized panels, a heading and a caption.

    canvas = Canvas(slice_.n_groups, panel_aspect=..., style=style)
    for group, panel in enumerate(canvas.panels):
        ...                     # draw into panel.ax
    canvas.save("out.png")
    """

    def __init__(
        self,
        n_panels: int,
        *,
        panel_aspect: float = 1.0,
        style: ViewStyle | None = None,
        title: str | None = None,
        caption: str | None = None,
        panel_width: float = PANEL_WIDTH_IN,
    ):
        if n_panels < 1:
            raise ValueError("a canvas needs at least one panel")
        self.style = style or ViewStyle()
        self.panel_width = panel_width
        n_rows, n_cols = panel_grid(n_panels)
        self._figure = Figure(
            figsize=(
                n_cols * panel_width,
                n_rows * panel_width * panel_aspect
                + CAPTION_ALLOWANCE_IN
                + (TITLE_ALLOWANCE_IN if title else 0.0),
            ),
            facecolor=self.style.theme.background,
            layout="constrained",
        )
        FigureCanvasAgg(self._figure)
        grid = self._figure.add_gridspec(n_rows, n_cols)
        self.panels = [
            Panel(self._figure, self._figure.add_subplot(grid[divmod(index, n_cols)]), self.style)
            for index in range(n_panels)
        ]
        if title:
            self._figure.suptitle(title, color=self.style.theme.foreground, fontsize=TITLE_SIZE)
        if caption:
            self._figure.supxlabel(
                caption, color=self.style.theme.foreground, fontsize=CAPTION_SIZE
            )
        self._settled = False

    def figure(self) -> Figure:
        """The finished matplotlib figure: layout resolved, panels corrected.

        Idempotent, so a caller may take the figure and keep drawing on it.
        """
        if self._settled:
            return self._figure
        self._figure.draw_without_rendering()
        measured = [
            (panel.ax.get_position(), panel.ax.get_window_extent().width) for panel in self.panels
        ]
        self._figure.set_layout_engine("none")
        for panel, (box, width_px) in zip(self.panels, measured, strict=True):
            panel._settle(box, width_px * 72.0 / self._figure.dpi)
        self._settled = True
        return self._figure

    def save(self, path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
        """Write the finished figure. Format follows the suffix."""
        return write_figure(self.figure(), path, dpi)


def write_figure(figure: Figure, path: str | Path, dpi: int = DEFAULT_DPI) -> Path:
    """Write any figure, cropping the slack a locked aspect leaves.

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
