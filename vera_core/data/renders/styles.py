from dataclasses import dataclass, replace

CHAR_WIDTH_RATIO = 0.62
"""Width of a digit over its point size, for fitting text to a cell."""


@dataclass(frozen=True)
class ViewTheme:
    """Colors for one rendering. background is also the NaN color, or empty
    positions and guide tubes show as a different shade than the page."""

    background: str
    foreground: str
    grid: str
    highlight: str
    edge: str
    """Outline between adjacent faces or cells drawn as polygons."""

    def with_background(self, color: str) -> "ViewTheme":
        return replace(self, background=color)


DARK = ViewTheme(
    background="#121212",
    foreground="#ffffff",
    grid="#6b6b6b",
    highlight="#ffffff",
    edge="#000000",
)

LIGHT = ViewTheme(
    background="#ffffff",
    foreground="#202020",
    grid="#9a9a9a",
    highlight="#000000",
    edge="#404040",
)


@dataclass(frozen=True)
class ViewStyle:
    """Presentation choices that are neither data nor color mapping."""

    theme: ViewTheme = DARK
    show_grid: bool = True
    show_axis_labels: bool = True
    show_values: bool = False
    decimals: int = 2
    value_format: str = "auto"
    """How cell text is written, with decimals as the precision.

        'auto'   significant figures, switching the whole panel to
                 scientific notation if any one value needs it
        'sci'    always scientific
        'fixed'  decimal places, matching the trame Decimals select, and
                 printing 0.00 for anything below that precision
    """

    endpoint_ticks_only: bool = True
    """Colorbar shows only vmin and vmax, matching the trame editor. Set False
    for intermediate ticks, which reads better in a report."""

    axis_label_size: float = 13.0
    value_size: float | None = None
    """Points. None lets panel_figure size text to the cell."""

    grid_width: float = 0.8
    edge_width: float = 0.5
    highlight_width: float = 2.0
