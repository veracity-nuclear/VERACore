import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from trame.ui.html import DivLayout
from trame.widgets import plotly

from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype
from ..helpers import is_non_active_view, get_safe_idxs

SEP = "\x1f"

def option_for(view_id):
    return {
        "name": f"axial_plot_{view_id}",
        "label": "Axial Plot",
        "multi_picker" : True,
        "icon": "mdi-align-horizontal-center",
        "allowed_categories": [VeraDtype.PIN.title, VeraDtype.CHANNEL.title, VeraDtype.AXIAL.title, VeraDtype.ASSEMBLY.title]
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_set_key = f"multi_selected_{view_id}"

    update_fn_name = f"update_axial_plot_{view_id}"

    def create_line():
        figure = go.Figure()
        for token in state[selected_set_key]:
            src_id, array_name = token.split(SEP, 1)
            src = registry.get(src_id)
            full_array = src.array(array_name)
            array_dtype : VeraDtype = full_array.dataset_type
            j, i, layer, assy, _, _ = get_safe_idxs(view_id, state, registry, src_id, array_name)
            assembly_label = src.core.reduced_core_map_label(assy, is_comp=array_dtype.is_computational())
            identifier : str = ""
            match array_dtype:
                case VeraDtype.PIN | VeraDtype.CHANNEL:
                    axial_array = full_array[j, i, :, assy]
                    identifier = f" | {assembly_label} @({i + 1},{j + 1})"
                case VeraDtype.ASSEMBLY:
                    axial_array = full_array[:, assy]
                    identifier = f" | {assembly_label}"
                case VeraDtype.AXIAL:
                    axial_array = full_array
                case _:
                    raise RuntimeError(f"Axial Plot cannot visualize datasets of type {str(array_dtype)}")
            figure.add_trace(
                go.Scatter(
                    x=axial_array,
                    y=src.core.axial_mesh_means,
                    mode="lines",
                    name=f"{src_id} | {array_name.replace('_', ' ').title()}{identifier}",
                )
            )

        # add_hline only spans x in [0, 1], so draw the layer marker manually.
        float_info = np.finfo(np.float64)
        figure.add_trace(
            go.Scatter(
                x=[float_info.min, float_info.max],
                y=[registry.default_src.core.axial_mesh_means[state.selected_layer]] * 2,
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(margin=dict(t=0, b=0, l=0, r=0), 
                             legend=dict(orientation="h",
                                         yanchor="top",
                                         y=-0.1,
                                         xanchor="center",
                                         x=0.5,)
                            ,)
        return figure

    @state.change(
        selected_set_key,
        "selected_assembly",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line())

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; width: 100%;"
        style = "; ".join([
            "width: 100%",
            "height: 100%",
            "user-select: none",
        ])
        figure = plotly.Figure(
            display_logo=False,
            display_mode_bar=False,
            style=style,
        )
        setattr(ctrl, update_fn_name, figure.update)