# from dataclasses import replace
# from typing import Sequence

# import numpy as np
# from matplotlib.axes import Axes
# from matplotlib.cm import ScalarMappable
# from matplotlib.collections import PolyCollection
# from matplotlib.colors import Normalize

# from ..analysis.assembly_surface_slice import (
#     AssemblySurfaceRequest,
#     AssemblySurfaceSlice,
#     assembly_surface_slices,
# )
# from ..analysis.color import ColorSpec, resolve_color_specs
# from ..analysis.info import create_info
# from ..dtypes import VeraDataset
# from ..thresholds import ThresholdCondition
# from .core_surface_view import LABELS_ACROSS_A_NODE, node_triangles
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

# NODE_CELL = 1
# """One grid cell per node, so the grid helpers step by one."""


# def assembly_surface_polygons(slice_: AssemblySurfaceSlice):
#     """Every face of every node as (vertices, values, label positions)."""
#     polygons, values, positions = [], [], []
#     for (row, col), _ in np.ndenumerate(slice_.data[..., 0]):
#         faces = slice_.data[row, col]
#         if np.isnan(faces).all():
#             continue
#         for (vertices, label_x, label_y), value in zip(
#             node_triangles(col, row, 1.0), faces, strict=False
#         ):
#             polygons.append(vertices)
#             values.append(value)
#             positions.append((label_x, label_y))
#     return polygons, np.array(values, dtype=float), positions


# def assembly_surface_columns(slice_: AssemblySurfaceSlice) -> int:
#     """Label columns across the assembly, for sizing value text."""
#     return slice_.side * LABELS_ACROSS_A_NODE


# def axis_label_size(slice_: AssemblySurfaceSlice, style: ViewStyle) -> float:
#     """Label size that fits one number per node, never above the style's."""
#     return fit_axis_label_size(slice_.side, style)


# def _draw_values(
#     ax: Axes,
#     values: np.ndarray,
#     positions,
#     mappable: ScalarMappable,
#     style: ViewStyle,
# ):
#     """Numeric text per face, colored for contrast against its triangle."""
#     size = style.value_size or FALLBACK_VALUE_SIZE
#     write = value_formatter(values[~np.isnan(values)], style)
#     for value, (x, y) in zip(values, positions, strict=False):
#         if np.isnan(value):
#             continue
#         ax.text(
#             x,
#             y,
#             write(value),
#             ha="center",
#             va="center",
#             fontsize=size,
#             color=contrast_color(mappable.to_rgba(value), style.theme),
#         )


# def draw_assembly_surface_slice(
#     ax: Axes,
#     slice_: AssemblySurfaceSlice,
#     color: ColorSpec | None = None,
#     style: ViewStyle | None = None,
# ) -> ScalarMappable:
#     """Render one assembly's node faces into ax and return its mappable."""
#     if style is None:
#         style = ViewStyle()
#     spec = resolve_color_specs([slice_], color)[0]
#     n_rows, n_cols = slice_.grid_shape

#     polygons, values, positions = assembly_surface_polygons(slice_)
#     faces = PolyCollection(
#         polygons,
#         array=np.ma.masked_invalid(values),
#         cmap=colormap(spec, style.theme),
#         norm=Normalize(vmin=spec.vmin, vmax=spec.vmax),
#         edgecolors=style.theme.edge,
#         linewidths=style.edge_width,
#     )
#     ax.add_collection(faces)
#     frame_axes(ax, (0, n_cols), (n_rows, 0), 1.0 / slice_.aspect_ratio, style)

#     if style.show_grid:
#         draw_grid(ax, n_rows, n_cols, NODE_CELL, style)
#     if style.show_axis_labels:
#         fitted = replace(style, axis_label_size=axis_label_size(slice_, style))
#         draw_axis_labels(ax, slice_, NODE_CELL, fitted)
#     else:
#         ax.set_xticks([])
#         ax.set_yticks([])
#     if style.show_values:
#         _draw_values(ax, values, positions, faces, style)
#     return faces


# def assembly_surface_figure(slices: list[AssemblySurfaceSlice], **kwargs):
#     """Figure for one request's groups. kwargs go to panel_figure()."""
#     return panel_figure(slices, draw_assembly_surface_slice, assembly_surface_columns, **kwargs)


# def save_assembly_surface_view(
#     path, slices: list[AssemblySurfaceSlice], *, dpi: int = DEFAULT_DPI, **kwargs
# ):
#     """Build and write in one call. Format follows the suffix."""
#     return write_figure(assembly_surface_figure(slices, **kwargs), path, dpi)


# class AssemblySurfaceView(View[AssemblySurfaceRequest]):
#     """Node surface maps for one loaded source."""

#     request_type = AssemblySurfaceRequest

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
#     ) -> Selection[AssemblySurfaceRequest]:
#         """One assembly's node faces, at one dataset, state and level.

#         array           dataset name, or the VeraDataset itself
#         state           state-point index
#         assembly        assembly index, from
#                         core.reduced_core_map_assembly(i, j)
#         z               axial level index
#         src_id          source id, when the request outlives this view
#         thresholds      conditions that blank values before rendering
#         mask_reflected  drop reflected assemblies
#         group           one energy group, or None for every group
#         """
#         return Selection(
#             self,
#             AssemblySurfaceRequest(
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

#     build_slices = staticmethod(assembly_surface_slices)
#     build_info = staticmethod(create_info)
#     draw = staticmethod(draw_assembly_surface_slice)
#     columns = staticmethod(assembly_surface_columns)
