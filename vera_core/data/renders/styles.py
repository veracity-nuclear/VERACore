from dataclasses import dataclass, replace

from ..analysis.color import DEFAULT_CMAP, ColorScope

CHAR_WIDTH_RATIO = 0.62
"""Width of a digit over its point size, for fitting text to a cell."""

PANEL_WIDTH_IN = 5.0


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
    """Every rendering choice that does not depend on the data drawn.

    One style can be reused for any dataset. What is drawn is chosen by
    View.select(); a fixed colorbar range is RenderOptions.color.
    """

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
        'fixed'  decimal places, and printing 0.00 for anything below that precision
    """

    endpoint_ticks_only: bool = True
    """Colorbar shows only vmin and vmax. Set False
    for intermediate ticks, which reads better in a report."""

    axis_label_size: float = 13.0
    value_size: float | None = None
    """Points. None lets panel_figure size text to the cell."""

    grid_width: float = 0.8
    edge_width: float = 0.5
    highlight_width: float = 2.0

    # -- color -------------------------------------------------------------

    cmap: str = DEFAULT_CMAP
    """Colormap for every group whose color does not name one."""

    color_scope: ColorScope = ColorScope.SLICE_ALL
    """How much of the data one colorbar spans; see ColorScope. A plain
    string such as "dataset_group" is accepted."""

    # -- layout ------------------------------------------------------------

    show_caption: bool = True
    panel_width: float = PANEL_WIDTH_IN
    """Inches across one panel, before cropping."""

    panel_columns: int | None = None
    """Panels per row. None keeps the default: one row up to two panels,
    then two wide. Any value at or above the panel count gives one row."""

    unit_label: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "color_scope", ColorScope(self.color_scope))
        if self.panel_width <= 0:
            raise ValueError(f"panel_width must be positive, got {self.panel_width}")
        if self.panel_columns is not None and self.panel_columns < 1:
            raise ValueError(f"panel_columns must be at least 1, got {self.panel_columns}")
