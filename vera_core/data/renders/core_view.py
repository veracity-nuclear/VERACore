"""The core map view: one axial layer on the assembly grid, a panel per group.

The matplotlib counterpart of the trame core view. Both read a CoreSlice, so
what a report shows and what the web view shows cannot drift apart. This is
the layer that knows both halves: it reads VERA data and calls the artists.

    draw.py       artists, one Axes at a time
    canvas.py     the figure and its panels
    view.py       the selection interface
    core_view.py  one concrete view, wiring the three together  <- here
"""

from collections.abc import Sequence

from ..analysis.color import resolve_color_specs
from ..analysis.core_slice import CoreSlice
from ..thresholds import ThresholdCondition
from . import draw
from .canvas import Canvas, map_aspect
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
        dataset_range: bool = False,
    ) -> Selection:
        """Bind one array, layer and state, ready to render.

        group picks one energy group, None renders all of them. src_id only
        labels the heading. dataset_range reads the whole dataset's extent,
        which ColorScope.DATASET needs and nothing else does.
        """
        return Selection(
            self,
            array=array,
            z=z,
            state=state,
            group=group,
            src_id=src_id,
            thresholds=tuple(thresholds),
            dataset_range=dataset_range,
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

    def render(self, slice_: CoreSlice, selection: Selection, options: RenderOptions) -> Canvas:
        style = options.style
        n_rows, n_cols = slice_.grid_shape
        side = slice_.assembly_side
        canvas = Canvas(
            slice_.n_groups,
            panel_aspect=map_aspect(n_rows, n_cols, slice_.aspect_ratio),
            style=style,
            title=options.resolved_title(self, selection),
            caption=self.caption(slice_, selection) if options.caption else None,
            panel_width=options.panel_width,
        )
        specs = resolve_color_specs(slice_, options.color, scope=options.color_scope)

        for group, (panel, spec) in enumerate(zip(canvas.panels, specs, strict=True)):
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
            if slice_.n_groups > 1:
                panel.title(f"Group {group + 1}")
            panel.colorbar(mappable, units=slice_.units)
        return canvas
