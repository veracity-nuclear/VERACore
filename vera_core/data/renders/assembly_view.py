"""The assembly view: one assembly's lattice at one layer, a panel per group."""

from collections.abc import Sequence

from matplotlib.cm import ScalarMappable

from ..analysis.assembly_slice import AssemblySlice
from ..analysis.color import resolve_color_specs
from ..thresholds import ThresholdCondition
from . import draw
from .canvas import Canvas, block_layout, map_aspect
from .view import RenderOptions, Selection, View

PIN_SIDE = 1
"""One cell per value: an assembly lattice has no blocks within it."""


class AssemblyView(View):
    """Renders one assembly of one dataset, a panel per energy group."""

    def select(
        self,
        array: str,
        *,
        assembly: int = 0,
        z: int = 0,
        state: int | None = None,
        group: int | None = None,
        src_id: str | None = None,
        thresholds: Sequence[ThresholdCondition] = (),
        highlight: tuple[int, int] | None = None,
    ) -> Selection:
        """Bind one assembly, layer and state, ready to render.

        group picks one energy group, None renders all of them. highlight
        outlines one cell, given as its (row, column) in the lattice, the way
        the web view marks the selected pin. z is ignored for radial data,
        which is already collapsed over it. src_id only labels the heading.
        """
        return Selection(
            self,
            array=array,
            assembly=assembly,
            z=z,
            state=state,
            group=group,
            src_id=src_id,
            thresholds=tuple(thresholds),
            highlight=highlight,
        )

    def build_slice(self, selection: Selection) -> AssemblySlice:
        slice_ = AssemblySlice.create_assembly_slice(
            self.source,
            selection.array,
            selection.z,
            selection.assembly,
            state=selection.state,
            thresholds_to_apply=selection.thresholds or None,
        )
        if slice_ is None:
            raise ValueError(f"{selection.label()} has no assembly view")
        return slice_ if selection.group is None else slice_.group_slice(selection.group)

    def default_title(self, selection: Selection) -> str:
        """'PIN POWERS', or 'PIN POWERS | vera2' when a source is named."""
        heading = selection.array.replace("_", " ").upper()
        return heading if selection.src_id is None else f"{heading} | {selection.src_id}"

    def caption(self, slice_: AssemblySlice) -> str:
        return slice_.info.caption()

    def collage_caption(self, slices, selections: list[Selection], over: str) -> str:
        """What every frame has in common. The swept choice is left out: the
        frame titles carry it."""
        held = []
        if over != "assembly":
            held.append(f"Assembly {selections[0].assembly}")
        if over != "z":
            held.append(f"Axial - {selections[0].z}")
        if over != "state":
            held.append(f"State {slices[0].state}")
        return " · ".join([*held, f"{len(selections)} {over}s"])

    def draw_map(
        self,
        panel,
        slice_: AssemblySlice,
        group: int,
        spec,
        style,
        highlight: tuple[int, int] | None = None,
    ) -> ScalarMappable:
        """One group of one slice into one panel: the whole of the drawing.

        Both render() and render_collage() go through here, so a frame of a
        collage and a still of the same assembly are the same picture.
        """
        side = slice_.side
        grid = slice_.to_grid(group)
        mappable = draw.cells(panel.ax, grid, spec, style, aspect_ratio=slice_.aspect_ratio)
        if style.show_grid:
            draw.block_grid(panel.ax, side, side, PIN_SIDE, style)
        if style.show_axis_labels:
            draw.axis_labels(panel.ax, slice_.x_labels, slice_.y_labels, PIN_SIDE, style)
        if highlight is not None:
            draw.highlight_block(panel.ax, *highlight, PIN_SIDE, style)
        if style.show_values:
            write = draw.value_formatter(slice_.finite(group), style)
            draw.cell_values(panel.ax, grid, spec, style, write)
            panel.fit_values(side)
        return mappable

    def render(self, slice_: AssemblySlice, selection: Selection, options: RenderOptions) -> Canvas:
        style = options.style
        side = slice_.side
        canvas = Canvas(
            slice_.n_groups,
            panel_aspect=map_aspect(side, side, slice_.aspect_ratio),
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
        slices: "list[AssemblySlice]",
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
        style = options.style
        first = slices[0]
        side = first.side
        n_groups, n_frames = first.n_groups, len(slices)
        grid, cells = block_layout(n_groups, n_frames, columns)
        canvas = Canvas(
            len(cells),
            grid=grid,
            cells=cells,
            panel_aspect=map_aspect(side, side, first.aspect_ratio),
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
