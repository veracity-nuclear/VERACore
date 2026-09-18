"""The axial view: one vertical cut through the core, a panel per group."""

from collections.abc import Sequence
from typing import Literal

from matplotlib.cm import ScalarMappable

from ..analysis.axial_slice import AxialSlice
from ..analysis.color import resolve_color_specs
from ..thresholds import ThresholdCondition
from . import draw
from .canvas import Canvas, block_layout
from .view import RenderOptions, Selection, View

ELEVATION_TITLE = "Elevation (cm)"


class AxialView(View):
    """Renders one vertical cut of one dataset, a panel per energy group."""

    def select(
        self,
        array: str,
        *,
        axis: Literal["x", "y"] = "x",
        assembly: int = 0,
        pin: int = 0,
        state: int | None = None,
        group: int | None = None,
        src_id: str | None = None,
        thresholds: Sequence[ThresholdCondition] = (),
        highlight: tuple[int, int] | None = None,
    ) -> Selection:
        """Bind one array, cut and state, ready to render.

        An x cut runs along the core row holding `assembly` and is indexed by
        the selected pin's j; a y cut runs along the column. group picks one
        energy group, None renders all of them. highlight outlines one cell,
        given as (z, column) with z counted from the bottom, the way the web
        view marks the selected one. src_id only labels the heading.
        """
        return Selection(
            self,
            array=array,
            axis=axis,
            assembly=assembly,
            pin=pin,
            state=state,
            group=group,
            src_id=src_id,
            thresholds=tuple(thresholds),
            highlight=highlight,
        )

    def build_slice(self, selection: Selection) -> AxialSlice:
        slice_ = AxialSlice.create_axial_slice(
            self.source,
            selection.array,
            selection.pin,
            selection.assembly,
            selection.axis,
            state=selection.state,
            thresholds_to_apply=selection.thresholds,
        )
        if slice_ is None:
            raise ValueError(f"{selection.label()} has no axial view")
        return slice_ if selection.group is None else slice_.group_slice(selection.group)

    def default_title(self, selection: Selection) -> str:
        """'PIN POWERS', or 'PIN POWERS | vera2' when a source is named."""
        heading = selection.array.replace("_", " ").upper()
        return heading if selection.src_id is None else f"{heading} | {selection.src_id}"

    def caption(self, slice_: AxialSlice) -> str:
        return slice_.info.caption()

    def collage_caption(self, slices, selections: list[Selection], over: str) -> str:
        """What every frame has in common. The swept choice is left out: the
        frame titles carry it."""
        held = [f"{slices[0].axis.upper()} cut"]
        if over != "assembly":
            held.append(f"Assembly {selections[0].assembly}")
        if over != "pin":
            held.append(f"Pin {selections[0].pin}")
        if over != "state":
            held.append(f"State {slices[0].state}")
        return " · ".join([*held, f"{len(selections)} {over}s"])

    def panel_aspect(self, slice_: AxialSlice) -> float:
        """Drawn height over drawn width. Both are centimeters, so a cut is
        as tall against its width as the core is."""
        x_lo, x_hi, y_lo, y_hi = slice_.extent
        return (y_hi - y_lo) / (x_hi - x_lo)

    def draw_map(
        self,
        panel,
        slice_: AxialSlice,
        group: int,
        spec,
        style,
        highlight: tuple[int, int] | None = None,
    ) -> ScalarMappable:
        """One group of one slice into one panel: the whole of the drawing.

        Both render() and render_collage() go through here, so a frame of a
        collage and a still of the same cut are the same picture.
        """
        columns = slice_.column_edges()
        grid = slice_.to_grid(group)
        mappable = draw.mesh_cells(panel.ax, grid, columns, slice_.y_edges, spec, style)
        if style.show_grid:
            draw.mesh_grid(panel.ax, slice_.x_edges, (), style)
        if style.show_axis_labels:
            centers = (slice_.x_edges[:-1] + slice_.x_edges[1:]) / 2
            draw.mesh_axis_labels(
                panel.ax, slice_.x_labels, centers, style, y_title=ELEVATION_TITLE
            )
        if highlight is not None:
            level, cell = highlight
            draw.highlight_span(
                panel.ax,
                (slice_.x_edges[cell], slice_.x_edges[cell + 1]),
                (slice_.y_edges[level], slice_.y_edges[level + 1]),
                style,
            )
        if style.show_values and slice_.labelable:
            write = draw.value_formatter(slice_.finite(group), style)
            draw.mesh_values(panel.ax, grid, columns, slice_.y_edges, spec, style, write)
            panel.fit_values(len(columns) - 1)
        return mappable

    def render(self, slice_: AxialSlice, selection: Selection, options: RenderOptions) -> Canvas:
        style = options.style
        canvas = Canvas(
            slice_.n_groups,
            panel_aspect=self.panel_aspect(slice_),
            style=style,
            title=options.resolved_title(self, selection),
            caption=self.caption(slice_) if style.show_caption else None,
        )
        specs = resolve_color_specs(slice_, options.color, scope=style.color_scope, cmap=style.cmap)
        for group, (panel, spec) in enumerate(zip(canvas.panels, specs, strict=True)):
            mappable = self.draw_map(panel, slice_, group, spec, style, selection.highlight)
            if slice_.n_groups > 1:
                panel.title(f"Group {group + 1}")
            canvas.colorbar(mappable, [panel], units=style.unit_label or slice_.units)
        return canvas

    def render_collage(
        self,
        slices: "list[AxialSlice]",
        selections: list[Selection],
        options: RenderOptions,
        *,
        over: str,
        columns: int = None,
    ) -> Canvas:
        """A block of frames per group, each block on its own scale.

        Panels are laid out block-major, so canvas.panels[group] holds that
        group's frames in order, and one colorbar serves each block. A cut is
        tall and narrow, so a block runs along one row unless the caller asks
        for a different width; a square block would leave most of the page
        empty.
        """
        style = options.style
        first = slices[0]
        n_groups, n_frames = first.n_groups, len(slices)
        grid, cells = block_layout(n_groups, n_frames, columns or n_frames)
        canvas = Canvas(
            len(cells),
            grid=grid,
            cells=cells,
            panel_aspect=self.panel_aspect(first),
            style=style,
            title=options.resolved_title(self, selections[0]),
            caption=self.collage_caption(slices, selections, over) if style.show_caption else None,
        )
        specs = resolve_color_specs(first, options.color, scope=style.color_scope, cmap=style.cmap)
        for group in range(n_groups):
            block = canvas.panels[group * n_frames : (group + 1) * n_frames]
            mappable = None
            for panel, slice_, selection in zip(block, slices, selections, strict=True):
                mappable = self.draw_map(
                    panel, slice_, group, specs[group], style, selection.highlight
                )
                panel.title(self.frame_title(selection, over))
            canvas.colorbar(
                mappable,
                block,
                units=style.unit_label or first.units,
                title=f"Group {group + 1}" if n_groups > 1 else "",
            )
        return canvas
