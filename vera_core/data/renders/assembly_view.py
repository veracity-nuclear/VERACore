# from dataclasses import replace
# from typing import Sequence

# import numpy as np
# from matplotlib.axes import Axes
# from matplotlib.cm import ScalarMappable
# from matplotlib.colors import Normalize

# from ..analysis.assembly_slice import (
#     AssemblyRequest,
#     AssemblySlice,
#     assembly_slices,
# )
# from ..analysis.color import ColorSpec, resolve_color_specs
# from ..analysis.info import create_info
# from ..dtypes import VeraDataset
# from ..thresholds import ThresholdCondition
# from .layout import (
#     DEFAULT_DPI,
#     FALLBACK_VALUE_SIZE,
#     Selection,
#     View,
#     ViewStyle,
#     colormap,
#     contrast_color,
#     draw_axis_labels,
#     draw_grid,
#     fit_axis_label_size,
#     frame_axes,
#     panel_figure,
#     value_formatter,
#     write_figure,
# )

# PIN_CELL = 1
# """One label column per pin, so the grid helpers step by one."""


# def assembly_columns(slice_: AssemblySlice) -> int:
#     """Label columns across the lattice: one per pin."""
#     return slice_.side


# def axis_label_size(slice_: AssemblySlice, style: ViewStyle) -> float:
#     """Label size that fits one number per pin, never above the style's."""
#     return fit_axis_label_size(slice_.side, style)


# def _draw_values(ax: Axes, slice_: AssemblySlice, mappable: ScalarMappable, style: ViewStyle):
#     """Numeric text per pin, colored for contrast against the cell."""
#     size = style.value_size or FALLBACK_VALUE_SIZE
#     write = value_formatter(slice_.finite(), style)
#     for (row, col), value in np.ndenumerate(slice_.data):
#         if np.isnan(value):
#             continue
#         ax.text(
#             col + 0.5,
#             row + 0.5,
#             write(value),
#             ha="center",
#             va="center",
#             fontsize=size,
#             color=contrast_color(mappable.to_rgba(value), style.theme),
#         )


# def draw_assembly_slice(
#     ax: Axes,
#     slice_: AssemblySlice,
#     color: ColorSpec | None = None,
#     style: ViewStyle | None = None,
# ) -> ScalarMappable:
#     """Render one assembly lattice into ax and return its mappable."""
#     if style is None:
#         style = ViewStyle()
#     spec = resolve_color_specs([slice_], color)[0]
#     n_rows, n_cols = slice_.grid_shape

#     mappable = ax.imshow(
#         slice_.data,
#         cmap=colormap(spec, style.theme),
#         norm=Normalize(vmin=spec.vmin, vmax=spec.vmax),
#         extent=(0, n_cols, n_rows, 0),
#         interpolation="nearest",
#     )
#     frame_axes(ax, (0, n_cols), (n_rows, 0), 1.0 / slice_.aspect_ratio, style)

#     if style.show_grid:
#         draw_grid(ax, n_rows, n_cols, PIN_CELL, style)
#     if style.show_axis_labels:
#         fitted = replace(style, axis_label_size=axis_label_size(slice_, style))
#         draw_axis_labels(ax, slice_, PIN_CELL, fitted)
#     else:
#         ax.set_xticks([])
#         ax.set_yticks([])
#     if style.show_values:
#         _draw_values(ax, slice_, mappable, style)
#     return mappable


# def assembly_view_figure(slices: list[AssemblySlice], **kwargs):
#     """Figure for one request's groups. kwargs go to panel_figure()."""
#     return panel_figure(slices, draw_assembly_slice, assembly_columns, **kwargs)


# def save_assembly_view(path, slices: list[AssemblySlice], *, dpi: int = DEFAULT_DPI, **kwargs):
#     """Build and write in one call. Format follows the suffix."""
#     return write_figure(assembly_view_figure(slices, **kwargs), path, dpi)


# class AssemblyView(View[AssemblyRequest]):
#     """Pin lattices for one loaded source."""

#     request_type = AssemblyRequest

#     def select(
#         self,
#         array: str | VeraDataset,
#         *,
#         state: int = 0,
#         assembly: int = 0,
#         z: int = 0,
#         src_id: str | None = None,
#         thresholds: Sequence[ThresholdCondition] = (),
#         mask_reflected: bool = True,
#         group: int | None = None,
#     ) -> Selection[AssemblyRequest]:
#         """One assembly's pin lattice, at one dataset, state and level.

#         array           dataset name, or the VeraDataset itself
#         state           state-point index
#         assembly        assembly index, from
#                         core.reduced_core_map_assembly(i, j)
#         z               axial level index, ignored for radial datasets
#         src_id          source id, when the request outlives this view
#         thresholds      conditions that blank values before rendering
#         mask_reflected  drop reflected assemblies
#         group           one energy group, or None for every group
#         """
#         return Selection(
#             self,
#             AssemblyRequest(
#                 array=array,
#                 state=state,
#                 assembly=assembly,
#                 z=z,
#                 src_id=src_id,
#                 thresholds=thresholds,
#                 mask_reflected=mask_reflected,
#                 group=group,
#             ),
#         )

#     build_slices = staticmethod(assembly_slices)
#     build_info = staticmethod(create_info)
#     draw = staticmethod(draw_assembly_slice)
#     columns = staticmethod(assembly_columns)
