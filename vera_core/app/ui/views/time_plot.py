import numpy as np
import plotly.graph_objects as go
from trame.ui.html import DivLayout
from trame.widgets import html, plotly, vuetify

from vera_core.data.dtypes import Surface, VeraDtype
from vera_core.data.registry import VeraDataRegistry

from ..helpers import convert_ji_to_node, get_safe_idxs, is_non_active_view

SEP = "\x1f"


def option_for(view_id):
    return {
        "name": f"time_plot_{view_id}",
        "label": "Time Plot",
        "multi_picker": True,
        "icon": "mdi-chart-line",
        "allowed_categories": [dtype.title for dtype in VeraDtype if dtype != VeraDtype.UNKNOWN],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]
    time_axis_key = f"selected_time_axis_{view_id}"
    time_axes_options_key = f"time_axes_{view_id}"
    state[time_axis_key] = "state_count"
    state[time_axes_options_key] = ["state_count"]

    selected_set_key = f"multi_selected_{view_id}"

    update_fn_name = f"update_time_plot_{view_id}"

    def create_line():
        figure = go.Figure()
        for token in state[selected_set_key]:
            identifier = ""
            src_id, array_name = token.split(SEP, 1)
            src = registry.get(src_id)
            time_axis = src.time_axes()[state[time_axis_key]]
            array_shape = np.shape(src.get_dataset(array_name, mask_reflected=False))
            array_dtype = src.get_dataset_dtype(array_name)
            is_comp = array_dtype.is_computational()
            indices = get_safe_idxs(view_id, state, registry, src_id, array_name)
            if not indices:
                continue
            ny, nx, nax, nass, _, _ = indices
            assembly_label = src.core.reduced_core_map_label(nass, is_comp=is_comp)
            axial_label = src.core.get_axial_mesh_means(dataset_type=array_dtype)[nax]
            units = src.get_dataset_units(array_name)
            units_label = f" ({units}) " if units != "unitless" else ""
            indices_list = []
            match array_dtype:
                case VeraDtype.PIN | VeraDtype.CHANNEL:
                    indices_list.append((ny, nx, nax, nass))
                    identifier = f" | {assembly_label} @({nx + 1},{ny + 1}) z = {axial_label}"
                case VeraDtype.POINT_DETECTOR:
                    indices_list.append((nax, nass))
                    identifier = f" | {assembly_label} z = {axial_label}"
                case VeraDtype.AXIAL:
                    indices_list.append((nax))
                    identifier = f" | z = {axial_label}"
                case VeraDtype.RADIAL | VeraDtype.CHANNEL_RADIAL:
                    indices_list.append((ny, nx, nass))
                    identifier = f" | {assembly_label} @({nx + 1},{ny + 1})"
                case VeraDtype.RADIAL_ASSEMBLY | VeraDtype.RADIAL_POINT_DETECTOR:
                    indices_list.append((nass))
                    identifier = f" | {assembly_label}"
                case VeraDtype.SCALAR:
                    indices_list.append((0))
                case VeraDtype.RADIAL_NODE:
                    node_idx = convert_ji_to_node(ny, nx)
                    indices_list.append((node_idx, nass))
                    identifier = f" | {assembly_label} @(NODE {node_idx + 1})"
                case VeraDtype.COMP_NODAL | VeraDtype.NODAL:
                    node_idx = convert_ji_to_node(ny, nx)
                    indices_list.append((node_idx, nax, nass))
                    identifier = f" | {assembly_label} @(NODE {node_idx + 1}) | z = {axial_label}"
                case VeraDtype.ASSEMBLY | VeraDtype.COMP_ASSY:
                    indices_list.append((0, nax, nass))
                    identifier = f" | {assembly_label} | z = {axial_label}"
                case VeraDtype.COMP_ASSY_ENERGY | VeraDtype.COMP_NODAL_ENERGY:
                    node_idx = (
                        convert_ji_to_node(ny, nx)
                        if array_dtype == VeraDtype.COMP_NODAL_ENERGY
                        else 0
                    )
                    num_energy_groups = array_shape[0]
                    for n_group in range(num_energy_groups):
                        indices_list.append((n_group, node_idx, nax, nass))
                    identifier = f" | {assembly_label} @(NODE {node_idx + 1}) | z = {axial_label}"
                case VeraDtype.COMP_ASSY_SURFACE | VeraDtype.COMP_NODAL_SURFACE:
                    selected_surface = state.selected_surface
                    num_energy_groups = array_shape[1]
                    nodal_idx = (
                        0
                        if array_dtype == VeraDtype.COMP_ASSY_SURFACE
                        else convert_ji_to_node(ny, nx)
                    )
                    for group_n in range(num_energy_groups):
                        indices_list.append((selected_surface, group_n, nodal_idx, nax, nass))
                    surface_label = f" {Surface(state.selected_surface).str}"
                    identifier = f" | {assembly_label} @(NODE {nodal_idx + 1}{surface_label})"
                case _:
                    continue
            for idx_n, indices in enumerate(indices_list):
                group_label = "" if len(indices_list) <= 1 else f" GROUP {idx_n + 1}"
                values = [x.get(array_name)[indices] for x in src.states if array_name in x]
                figure.add_trace(
                    go.Scatter(
                        x=time_axis,
                        y=values,
                        mode="lines",
                        name=f"{src_id} | {array_name.replace('_', ' ').title()}{units_label}{group_label + identifier}",
                    )
                )

        # add_vline only spans y in [0, 1], so draw the marker manually.
        float_info = np.finfo(np.float64)
        axis = state[time_axis_key]
        x_val = registry.time_axis_value(axis, int(state["selected_time"]))
        x = [x_val] * 2
        figure.add_trace(
            go.Scatter(
                x=x,
                y=[float_info.min, float_info.max],
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_dark" if state["dark_mode"] else "plotly",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
            ),
        )
        return figure

    @state.change("src_tree_meta")
    def update_time_axes_options(**kwargs):
        state[time_axes_options_key] = registry.shared_time_axes()

    @state.change(
        selected_set_key,
        "max_time",
        "selected_assembly_ij",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
        time_axis_key,
        "dark_mode",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if is_non_active_view(state, view_id, option) or registry.default_src is None:
            return
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line())

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; width: 100%;display: flex; flex-direction: column;"
        style = "; ".join(
            [
                "width: 100%",
                "height: 100%",
                "user-select: none",
            ]
        )
        with html.Div(style="flex: 1; min-height: 0; width: 100%;"):
            figure = plotly.Figure(
                display_logo=False,
                display_mode_bar=False,
                style=style,
            )
            setattr(ctrl, update_fn_name, figure.update)
        with html.Div(
            style=(
                "flex: 0 0 auto; display: flex; align-items: center;"
                "justify-content: center; gap: 6px; padding: 4px 0;"
            )
        ):
            html.Span("X-Axis:", classes="text-caption text--secondary")
            vuetify.VSelect(
                v_model=(time_axis_key,),
                items=(time_axes_options_key,),
                hide_details=True,
                dense=True,
                prepend_outer_icon="mdi-axis-x-arrow",
                style="max-width: 220px;",
            )
