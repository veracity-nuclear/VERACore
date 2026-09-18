"""The assembly surface view: one assembly's lateral faces at one layer."""

from matplotlib.cm import ScalarMappable

from ..analysis.assembly_surface_slice import AssemblySurfaceSlice
from ..analysis.color import resolve_color_specs
from . import draw
from .canvas import Canvas, block_layout, map_aspect
from .view import RenderOptions, Selection, View

NODE_SIDE = 1
"""One cell per node: an assembly's lattice has no blocks within it."""

LABELS_PER_CELL = 3
"""Labels across one cell. The face anchors sit at thirds, west and east on
the outside with north and south between them, so a cell is three labels
wide and each one has to fit in a third of it."""


class AssemblySurfaceView(View):
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
        highlight: tuple | None = None,
    ) -> Selection:
        """Bind one assembly, layer and state, ready to render.

        group picks one energy group, None renders all of them. highlight
        marks what the web view marks: (row, column) outlines that cell, and
        (row, column, face) outlines one of its triangles as well, face being
        one of the slice's face names. src_id only labels the heading.
        """
        return Selection(
            self,
            array=array,
            assembly=assembly,
            z=z,
            state=state,
            group=group,
            src_id=src_id,
            highlight=highlight,
        )

    def build_slice(self, selection: Selection) -> AssemblySurfaceSlice:
        slice_ = AssemblySurfaceSlice.create_assemlby_surface_slice(
            vera_source=self.source,
            selected_array=selection.array,
            z=selection.z,
            assembly_id=selection.assembly,
            state=selection.state,
        )
        if slice_ is None:
            raise ValueError(f"{selection.label()} has no assembly surface view")
        return slice_ if selection.group is None else slice_.group_slice(selection.group)

    def default_title(self, selection: Selection) -> str:
        """'NODAL XS SFLX', or 'NODAL XS SFLX | vera2' when a source is named."""
        heading = selection.array.replace("_", " ").upper()
        return heading if selection.src_id is None else f"{heading} | {selection.src_id}"

    def caption(self, slice_: AssemblySurfaceSlice) -> str:
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

    @staticmethod
    def highlit_face(selection: Selection) -> str | None:
        """The face named by a (row, column, face) highlight, if there is one."""
        highlight = selection.highlight
        return highlight[2] if highlight is not None and len(highlight) > 2 else None

    def draw_map(
        self,
        panel,
        slice_: AssemblySurfaceSlice,
        group: int,
        spec,
        style,
        highlight: tuple | None = None,
    ) -> ScalarMappable:
        """One group of one slice into one panel: the whole of the drawing.

        Both render() and render_collage() go through here, so a frame of a
        collage and a still of the same assembly are the same picture.
        """
        side = slice_.side
        grid = slice_.to_grid(group)
        mappable = draw.face_cells(panel.ax, grid, spec, style, aspect_ratio=slice_.aspect_ratio)
        if style.show_grid:
            draw.block_grid(panel.ax, side, side, NODE_SIDE, style)
        if style.show_axis_labels:
            draw.axis_labels(panel.ax, slice_.x_labels, slice_.y_labels, NODE_SIDE, style)
        if highlight is not None:
            row, col = highlight[0], highlight[1]
            draw.highlight_block(panel.ax, row, col, NODE_SIDE, style)
            if len(highlight) > 2:
                face = highlight[2]
                if face not in slice_.faces:
                    raise ValueError(
                        f"{face!r} is not a face; expected one of {', '.join(slice_.faces)}"
                    )
                draw.highlight_face(panel.ax, row, col, slice_.faces.index(face), style)
        if style.show_values:
            write = draw.value_formatter(slice_.finite(group), style)
            draw.face_values(panel.ax, grid, spec, style, write)
            panel.fit_values(side * LABELS_PER_CELL)
        return mappable

    def render(
        self, slice_: AssemblySurfaceSlice, selection: Selection, options: RenderOptions
    ) -> Canvas:
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
        slices: "list[AssemblySurfaceSlice]",
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
