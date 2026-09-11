"""The figure and its panels.

A Canvas is one output image. It holds one Panel per group, sizes the figure,
and runs the correction pass that only a drawn figure can settle. It does not
know what a panel contains.
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


def square_columns(n_frames: int) -> int:
    """Columns that keep a block as square as it can be: ceil(sqrt(n)).

    8 frames go 3 wide, 11 or 12 go 4 wide, 17 through 19 go 5. Integer
    arithmetic, so no float rounding decides a boundary case.
    """
    return math.isqrt(max(n_frames, 1) - 1) + 1


def block_layout(n_blocks: int, n_frames: int, columns: int | None = None):
    """Grid and cell placement for n_blocks blocks of n_frames panels each.

    A collage draws one block per group: every frame of a group sits together,
    wrapping at `columns`, and the next group starts on a fresh row so a
    block is never split across another group's. columns defaults to a
    square block.

    Returns ((n_rows, n_cols), cells), cells being (row, col) per panel in
    block-major order.
    """
    columns = max(1, min(columns or square_columns(n_frames), n_frames))
    block_rows = math.ceil(n_frames / columns)
    cells = [
        (block * block_rows + frame // columns, frame % columns)
        for block in range(n_blocks)
        for frame in range(n_frames)
    ]
    return (n_blocks * block_rows, columns), cells


class Panel:
    """One subplot of a Canvas.

    A view draws into `ax` with the artists in draw.py, and calls the methods
    here for the parts the canvas has to settle by measurement.
    """

    def __init__(self, canvas: "Canvas", ax: Axes, style: ViewStyle):
        self.canvas = canvas
        self.figure = canvas._figure
        self.ax = ax
        self.style = style
        self.value_columns: int | None = None

    def title(self, text: str) -> None:
        self.ax.set_title(text, color=self.style.theme.foreground, pad=PANEL_TITLE_PAD)

    def colorbar(self, mappable: ScalarMappable, units: str = "") -> Colorbar:
        """A colorbar for this panel alone. Canvas.colorbar() gives one bar
        to several panels, which is what a shared scale needs."""
        return self.canvas.colorbar(mappable, [self], units)

    def fit_values(self, n_columns: int) -> None:
        """Resize this panel's text on finish so n_columns labels fit across
        the drawn map. Ignored when the style fixes value_size."""
        if self.style.value_size is None:
            self.value_columns = n_columns

    def _settle_values(self, width_pt: float) -> None:
        """Resize value text against the width the panel actually got.

        Text sized against the nominal panel width overflows, because the row
        labels and the colorbar take part of that width.
        """
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

    Panels fill a two-wide grid by default. Pass grid and cells to place them
    yourself, as a collage does to keep each group's frames together.
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
        grid: tuple[int, int] | None = None,
        cells: list[tuple[int, int]] | None = None,
    ):
        if n_panels < 1:
            raise ValueError("a canvas needs at least one panel")
        self.style = style or ViewStyle()
        self.panel_width = panel_width
        n_rows, n_cols = grid or panel_grid(n_panels)
        if cells is None:
            cells = [divmod(index, n_cols) for index in range(n_panels)]
        if len(cells) != n_panels:
            raise ValueError(f"{len(cells)} cells for {n_panels} panels")
        self._colorbars: list[tuple[Axes, list[Panel]]] = []
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
        spec = self._figure.add_gridspec(n_rows, n_cols)
        self.panels = [
            Panel(self, self._figure.add_subplot(spec[cell]), self.style) for cell in cells
        ]
        if title:
            self._figure.suptitle(title, color=self.style.theme.foreground, fontsize=TITLE_SIZE)
        if caption:
            self._figure.supxlabel(
                caption, color=self.style.theme.foreground, fontsize=CAPTION_SIZE
            )
        self._settled = False

    def colorbar(
        self,
        mappable: ScalarMappable,
        panels: "list[Panel]",
        units: str = "",
        title: str = "",
    ) -> Colorbar:
        """One colorbar for a set of panels, snapped to them on finish.

        Give a collage's whole group its bar in one call: the panels share a
        scale, so they share the bar that reads it. title names which panels
        those are, for a figure holding more than one bar.
        """
        bar = draw.colorbar(
            self._figure, mappable, [panel.ax for panel in panels], units, self.style, title
        )
        self._colorbars.append((bar.ax, list(panels)))
        return bar

    def figure(self) -> Figure:
        """The finished matplotlib figure: layout resolved, panels corrected.

        Idempotent, so a caller may take the figure and keep drawing on it.
        """
        if self._settled:
            return self._figure
        self._figure.draw_without_rendering()
        boxes = {id(panel): panel.ax.get_position() for panel in self.panels}
        widths = {
            id(panel): panel.ax.get_window_extent().width * 72.0 / self._figure.dpi
            for panel in self.panels
        }
        self._figure.set_layout_engine("none")
        for panel in self.panels:
            panel._settle_values(widths[id(panel)])
        for colorbar_ax, panels in self._colorbars:
            self._snap(colorbar_ax, [boxes[id(panel)] for panel in panels])
        self._settled = True
        return self._figure

    @staticmethod
    def _snap(colorbar_ax: Axes, panel_boxes) -> None:
        """Pull a colorbar back onto the maps it reads.

        A map has a locked aspect, so it shrinks inside its cell while the
        bar fills the cell; the bar is sized to span its panels instead.
        """
        box = colorbar_ax.get_position()
        bottom = min(panel_box.y0 for panel_box in panel_boxes)
        top = max(panel_box.y1 for panel_box in panel_boxes)
        colorbar_ax.set_position([box.x0, bottom, box.width, top - bottom])

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
