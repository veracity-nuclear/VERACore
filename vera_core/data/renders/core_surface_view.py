"""The core surface view: lateral face data on the assembly grid, a panel per
group."""

from matplotlib.cm import ScalarMappable

from ..analysis.color import resolve_color_specs
from ..analysis.surface_slice import SurfaceSlice
from . import draw
from .canvas import Canvas, block_layout, map_aspect
from .view import RenderOptions, Selection, View

MAX_VALUE_SIDE = 2
"""Beyond 2x2 nodes there are too many faces in an assembly to label."""

LABELS_PER_CELL = 3
"""Labels across one cell. The face anchors sit at thirds, west and east on
the outside with north and south between them, so a cell is three labels
wide and each one has to fit in a third of it."""


class SurfaceView(View):
    """Renders one axial layer of one dataset, a panel per energy group."""

    def select(
        self,
        array: str,
        *,
        z: int = 0,
        state: int | None = None,
        group: int | None = None,
        src_id: str | None = None,
        highlight: tuple[int, int] | None = None,
    ) -> Selection:
        """Bind one array, layer and state, ready to render.

        group picks one energy group, None renders all of them. highlight
        outlines one assembly, given as its (row, column) on the map, the way
        the web view marks the selected one. src_id only labels the heading.
        """
        return Selection(
            self,
            array=array,
            z=z,
            state=state,
            group=group,
            src_id=src_id,
            highlight=highlight,
        )

    def build_slice(self, selection: Selection) -> SurfaceSlice:
        slice_ = SurfaceSlice.create_surface_slice(
            self.source,
            selection.array,
            selection.z,
            state=selection.state,
        )
        if slice_ is None:
            raise ValueError(f"{selection.label()} has no surface view")
        return slice_ if selection.group is None else slice_.group_slice(selection.group)

    def default_title(self, selection: Selection) -> str:
        """'NODAL XS SFLX', or 'NODAL XS SFLX | vera2' when a source is named."""
        heading = selection.array.replace("_", " ").upper()
        return heading if selection.src_id is None else f"{heading} | {selection.src_id}"

    def caption(self, slice_: SurfaceSlice) -> str:
        return slice_.info.caption()

    def collage_caption(self, slices, selections: list[Selection], over: str) -> str:
        """What every frame has in common. The swept choice is left out: the
        frame titles carry it."""
        held = [f"Axial - {selections[0].z}"] if over != "z" else []
        if over != "state":
            held.append(f"State {slices[0].state}")
        return " · ".join([*held, f"{len(selections)} {over}s"])

    def draw_map(
        self,
        panel,
        slice_: SurfaceSlice,
        group: int,
        spec,
        style,
        highlight: tuple[int, int] | None = None,
    ) -> ScalarMappable:
        """One group of one slice into one panel: the whole of the drawing.

        Both render() and render_collage() go through here, so a frame of a
        collage and a still of the same layer are the same picture.
        """
        n_rows, n_cols = slice_.grid_shape
        side = slice_.node_side
        grid = slice_.to_grid(group)
        mappable = draw.face_cells(panel.ax, grid, spec, style, aspect_ratio=slice_.aspect_ratio)
        if style.show_grid:
            draw.block_grid(panel.ax, n_rows, n_cols, side, style)
        if style.show_axis_labels:
            draw.axis_labels(panel.ax, slice_.x_labels, slice_.y_labels, side, style)
        if highlight is not None:
            draw.highlight_block(panel.ax, *highlight, side, style)
        if style.show_values and side <= MAX_VALUE_SIDE:
            write = draw.value_formatter(slice_.finite(group), style)
            draw.face_values(panel.ax, grid, spec, style, write)
            panel.fit_values(n_cols * side * LABELS_PER_CELL)
        return mappable

    def render(self, slice_: SurfaceSlice, selection: Selection, options: RenderOptions) -> Canvas:
        style = options.style
        n_rows, n_cols = slice_.grid_shape
        canvas = Canvas(
            slice_.n_groups,
            panel_aspect=map_aspect(n_rows, n_cols, slice_.aspect_ratio),
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
        slices: "list[SurfaceSlice]",
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
        n_rows, n_cols = first.grid_shape
        n_groups, n_frames = first.n_groups, len(slices)
        grid, cells = block_layout(n_groups, n_frames, columns)
        canvas = Canvas(
            len(cells),
            grid=grid,
            cells=cells,
            panel_aspect=map_aspect(n_rows, n_cols, first.aspect_ratio),
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
