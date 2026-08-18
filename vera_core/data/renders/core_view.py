"""The core map view: one axial layer on the assembly grid, a panel per group."""

from collections.abc import Sequence

from matplotlib.cm import ScalarMappable

from ..analysis.color import resolve_color_specs
from ..analysis.core_slice import CoreSlice
from ..thresholds import ThresholdCondition
from . import draw
from .canvas import Canvas, block_layout, map_aspect
from .view import RenderOptions, Selection, View

MAX_VALUE_SIDE = 2
"""Beyond nodal (2x2) there are too many values in an assembly to label."""


class CoreView(View):
    """Renders one axial layer of one dataset, a panel per energy group."""

    def select(
        self,
        array: str,
        *,
        z: int = 0,
        state: int | None = None,
        group: int | None = None,
        src_id: str | None = None,
        thresholds: Sequence[ThresholdCondition] = (),
    ) -> Selection:
        """Bind one array, layer and state, ready to render.

        group picks one energy group, None renders all of them. src_id only
        labels the heading.
        """
        return Selection(
            self,
            array=array,
            z=z,
            state=state,
            group=group,
            src_id=src_id,
            thresholds=tuple(thresholds),
        )

    def build_slice(self, selection: Selection) -> CoreSlice:
        slice_ = CoreSlice.create_core_slice(
            self.source,
            selection.array,
            selection.z,
            state_idx=selection.state,
            thresholds=selection.thresholds,
        )
        if slice_ is None:
            raise ValueError(f"{selection.label()} has no core view")
        return slice_ if selection.group is None else slice_.group_slice(selection.group)

    def default_title(self, selection: Selection) -> str:
        """'PIN POWERS', or 'PIN POWERS | vera2' when a source is named."""
        heading = selection.array.replace("_", " ").upper()
        return heading if selection.src_id is None else f"{heading} | {selection.src_id}"

    def caption(self, slice_: CoreSlice, selection: Selection) -> str:
        """Override to name exposure or elevation from the source, as the
        web view's footer does."""
        return f"State {slice_.state} · Axial - {selection.z}"

    def collage_caption(self, slices, selections: list[Selection], over: str) -> str:
        """What every frame has in common. The swept choice is left out: the
        frame titles carry it."""
        held = [f"Axial - {selections[0].z}"] if over != "z" else []
        if over != "state":
            held.append(f"State {slices[0].state}")
        return " · ".join([*held, f"{len(selections)} {over}s"])

    def draw_map(self, panel, slice_: CoreSlice, group: int, spec, style) -> ScalarMappable:
        """One group of one slice into one panel: the whole of the drawing.

        Both render() and render_collage() go through here, so a frame of a
        collage and a still of the same layer are the same picture.
        """
        n_rows, n_cols = slice_.grid_shape
        side = slice_.assembly_side
        grid = slice_.to_grid(group)
        mappable = draw.cells(panel.ax, grid, spec, style, aspect_ratio=slice_.aspect_ratio)
        if style.show_grid:
            draw.block_grid(panel.ax, n_rows, n_cols, side, style)
        if style.show_axis_labels:
            draw.axis_labels(panel.ax, slice_.x_labels, slice_.y_labels, side, style)
        if style.show_values and side <= MAX_VALUE_SIDE:
            write = draw.value_formatter(slice_.finite(group), style)
            draw.cell_values(panel.ax, grid, spec, style, write)
            panel.fit_values(n_cols * side)
        return mappable

    def render(self, slice_: CoreSlice, selection: Selection, options: RenderOptions) -> Canvas:
        n_rows, n_cols = slice_.grid_shape
        canvas = Canvas(
            slice_.n_groups,
            panel_aspect=map_aspect(n_rows, n_cols, slice_.aspect_ratio),
            style=options.style,
            title=options.resolved_title(self, selection),
            caption=self.caption(slice_, selection) if options.caption else None,
            panel_width=options.panel_width,
        )
        specs = resolve_color_specs(slice_, options.color, scope=options.color_scope)
        for group, (panel, spec) in enumerate(zip(canvas.panels, specs, strict=True)):
            mappable = self.draw_map(panel, slice_, group, spec, options.style)
            if slice_.n_groups > 1:
                panel.title(f"Group {group + 1}")
            canvas.colorbar(mappable, [panel], units=slice_.units)
        return canvas

    def render_collage(
        self,
        slices: "list[CoreSlice]",
        selections: list[Selection],
        options: RenderOptions,
        *,
        over: str,
        columns: int = None,
    ) -> Canvas:
        """A block of frames per group, each block on its own scale.

        Panels are laid out block-major, so canvas.panels[group] holds that
        group's frames in order, and one colorbar serves each block.
        """
        first = slices[0]
        n_rows, n_cols = first.grid_shape
        n_groups, n_frames = first.n_groups, len(slices)
        grid, cells = block_layout(n_groups, n_frames, columns)
        canvas = Canvas(
            len(cells),
            grid=grid,
            cells=cells,
            panel_aspect=map_aspect(n_rows, n_cols, first.aspect_ratio),
            style=options.style,
            title=options.resolved_title(self, selections[0]),
            caption=self.collage_caption(slices, selections, over) if options.caption else None,
            panel_width=options.panel_width,
        )
        specs = resolve_color_specs(first, options.color, scope=options.color_scope)
        for group in range(n_groups):
            block = canvas.panels[group * n_frames : (group + 1) * n_frames]
            mappable = None
            for panel, slice_, selection in zip(block, slices, selections, strict=True):
                mappable = self.draw_map(panel, slice_, group, specs[group], options.style)
                panel.title(self.frame_title(selection, over))
            canvas.colorbar(
                mappable,
                block,
                units=first.units,
                title=f"Group {group + 1}" if n_groups > 1 else "",
            )
        return canvas
