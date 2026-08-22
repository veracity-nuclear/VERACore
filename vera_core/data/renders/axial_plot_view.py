"""The axial plot: one line per dataset against elevation, on one panel."""

from collections.abc import Sequence

import numpy as np

from ..analysis.axial_lines import AxialLines
from ..analysis.vera_slices import convert_ji_to_node
from ..dtypes import VeraDim
from . import draw
from .canvas import Canvas, block_layout
from .view import RenderOptions, Selection, View

PLOT_ASPECT = 1.15
"""Drawn height over drawn width. A profile is read up the page, so the
panel is a little taller than it is wide."""

ELEVATION_TITLE = "Elevation (cm)"

LEGEND_COLUMNS = 1
"""One line's name carries its source, dataset, units, position and group,
so two of them will not sit side by side."""


class AxialLinesView(View):
    """Plots one or more datasets against elevation, over one or more sources.

    Built with a single source by Vera, and with several by VeraSet. The
    lines carry their source's name, so a plot over several reads as a
    comparison without the caller labelling anything.
    """

    def __init__(self, sources, options: RenderOptions | None = None):
        self.sources = list(sources) if isinstance(sources, (list, tuple)) else [sources]
        if not self.sources:
            raise ValueError("an axial plot needs at least one source")
        super().__init__(self.sources[0], options)

    def select(
        self,
        arrays: str | Sequence[str],
        *,
        assembly: int = 0,
        pin: tuple[int, int] = (0, 0),
        node: int | None = None,
        surface: int = 0,
        state: int | None = None,
        layer: int | None = None,
        src_id: str | None = None,
    ) -> Selection:
        """Bind one or more arrays and a position, ready to render.

        Every array is read at the same position, from every source, and all
        of them land on one panel; for arrays at different positions, render
        two selections and combine them yourself.

            arrays    one dataset name, or several
            assembly  which assembly the line is read from
            pin       (row, column) within it, the (j, i) the web view picks
            node      which node, derived from pin when left out
            surface   which face, for surface data
            layer     draws a dashed line at that level's elevation, the
                      marker the web view puts at the selected layer
            src_id    only labels the heading
        """
        names = (arrays,) if isinstance(arrays, str) else tuple(arrays)
        if not names:
            raise ValueError("an axial plot needs at least one array")
        return Selection(
            self,
            arrays=names,
            assembly=assembly,
            pin=tuple(pin),
            node=int(convert_ji_to_node(*pin)) if node is None else int(node),
            surface=surface,
            state=state,
            layer=layer,
            src_id=src_id,
        )

    def indices(self, selection: Selection) -> dict:
        """The position, as the dimensions AxialLines indexes on."""
        row, column = selection.pin
        return {
            VeraDim.PIN_Y: row,
            VeraDim.PIN_X: column,
            VeraDim.NODE: int(selection.node),
            VeraDim.SURFACE: selection.surface,
            VeraDim.ASSEMBLY: selection.assembly,
        }

    def build_slice(self, selection: Selection) -> AxialLines:
        names = list(selection.arrays)
        index = self.indices(selection)
        lines = AxialLines.create_axial_lines(
            self.sources,
            [names for _ in self.sources],
            [[index for _ in names] for _ in self.sources],
            state=selection.state,
        )
        if not lines.axial_arrays:
            raise ValueError(f"{selection.label()} has no axial lines")
        return lines

    # -- sweeps ------------------------------------------------------------

    def shared_color(self, slices, scope=None):
        """Lines carry no scale, so a sweep has none to hold constant. The
        frames of a collage still share an axis range, which set_ylim gives
        them from each frame's own extents."""
        return None

    def sweep_values(self, over: str, selection: Selection) -> Sequence:
        """`layer` moves the marker, which is the only axial choice here: the
        lines themselves always span every level."""
        if over == "layer":
            return self.axial_levels(selection)
        return super().sweep_values(over, selection)

    def axial_levels(self, selection: Selection) -> range:
        """Levels of the first array. Several arrays on one plot may sit on
        different meshes, and the marker can only be at one height."""
        dtype = self.source.get_dataset_dtype(selection.arrays[0], selection.state)
        if not dtype.has_axial_dim():
            raise ValueError(
                f"{selection.arrays[0]!r} reads as {dtype}, which has no axial dimension;"
                " check the name or pass values"
            )
        return range(len(self.elevations(selection)))

    def elevations(self, selection: Selection) -> np.ndarray:
        """The mesh the marker is placed on, one height per level."""
        dtype = self.source.get_dataset_dtype(selection.arrays[0], selection.state)
        means = np.asarray(self.source.core.get_axial_mesh_means(dataset_type=dtype))
        # A continuous detector's mesh is intervals, not heights.
        return means.mean(axis=1) if means.ndim == 2 else means

    # -- labels ------------------------------------------------------------

    def default_title(self, selection: Selection) -> str:
        """'PIN POWERS', or 'PIN POWERS | vera2' when a source is named."""
        heading = " | ".join(name.replace("_", " ").upper() for name in selection.arrays)
        return heading if selection.src_id is None else f"{heading} | {selection.src_id}"

    def caption(self, lines_: AxialLines, selection: Selection) -> str:
        """What is held fixed. The lines name themselves, so the caption does
        not repeat the datasets or the sources."""
        row, column = selection.pin
        state = lines_.states[0] if lines_.states else selection.state
        parts = [
            f"State {state}",
            f"Assembly {selection.assembly}",
            f"Pin ({column + 1},{row + 1})",
        ]
        if len(self.sources) > 1:
            parts.append(f"{len(self.sources)} sources")
        return " · ".join(parts)

    def collage_caption(self, slices, selections: list[Selection], over: str) -> str:
        held = [f"Assembly {selections[0].assembly}"]
        if over != "state":
            held.append(f"State {slices[0].states[0] if slices[0].states else '-'}")
        return " · ".join([*held, f"{len(selections)} {over}s"])

    # -- drawing -----------------------------------------------------------

    def draw_plot(self, panel, lines_: AxialLines, selection: Selection, style) -> list:
        """One AxialLines into one panel: the whole of the drawing.

        Both render() and render_collage() go through here, so a frame of a
        collage and a still of the same plot are the same picture. Returns
        the line handles, which the caller names in a legend.
        """
        handles = draw.lines(panel.ax, lines_.lines(), style)
        draw.plot_axes(panel.ax, style, y_label=ELEVATION_TITLE)
        panel.ax.set_ylim(*lines_.axial_extents())
        if selection.layer is not None:
            elevations = self.elevations(selection)
            draw.rule(panel.ax, float(elevations[selection.layer]), style)
        return handles

    def render(self, lines_: AxialLines, selection: Selection, options: RenderOptions) -> Canvas:
        canvas = Canvas(
            1,
            panel_aspect=PLOT_ASPECT,
            style=options.style,
            title=options.resolved_title(self, selection),
            caption=self.caption(lines_, selection) if options.caption else None,
            panel_width=options.panel_width,
        )
        panel = canvas.panels[0]
        handles = self.draw_plot(panel, lines_, selection, options.style)
        draw.legend(panel.figure, handles, options.style, columns=LEGEND_COLUMNS)
        return canvas

    def render_collage(
        self,
        slices: "list[AxialLines]",
        selections: list[Selection],
        options: RenderOptions,
        *,
        over: str,
        columns: int = None,
    ) -> Canvas:
        """Every frame on its own panel, under one legend.

        The frames hold the same lines at different states or with the marker
        at a different level, so naming them once is enough.
        """
        n_frames = len(slices)
        grid, cells = block_layout(1, n_frames, columns)
        canvas = Canvas(
            len(cells),
            grid=grid,
            cells=cells,
            panel_aspect=PLOT_ASPECT,
            style=options.style,
            title=options.resolved_title(self, selections[0]),
            caption=self.collage_caption(slices, selections, over) if options.caption else None,
            panel_width=options.panel_width,
        )
        handles = []
        for panel, lines_, selection in zip(canvas.panels, slices, selections, strict=True):
            handles = self.draw_plot(panel, lines_, selection, options.style)
            panel.title(self.frame_title(selection, over))
        draw.legend(canvas.panels[0].figure, handles, options.style, columns=LEGEND_COLUMNS)
        return canvas
