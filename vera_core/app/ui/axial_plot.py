import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from trame.ui.html import DivLayout
from trame.widgets import plotly


def option_for(view_id):
    return {
        "name": f"axial_plot_{view_id}",
        "label": "Axial Plot",
        "icon": "mdi-align-horizontal-center",
    }


def initialize(server, vera_out_file, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    update_fn_name = f"update_axial_plot_{view_id}"

    def create_line(selected_array, indices=(0, 0, 0, 0)):
        selected_j, selected_i, selected_layer, selected_assembly = indices

        full_array = vera_out_file.array(selected_array)
        array = full_array[selected_j, selected_i, :, selected_assembly]

        figure = px.line(
            x=array,
            y=vera_out_file.core.axial_mesh_means,
            labels={"x": selected_array, "y": "Axial (cm)"},
        )

        # add_hline only plots x==0 to x==1, so draw the marker manually.
        float_info = np.finfo(np.float64)
        figure.add_trace(
            go.Scatter(
                x=[float_info.min, float_info.max],
                y=[vera_out_file.core.axial_mesh_means[selected_layer]] * 2,
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        return figure

    @state.change(
        selected_array_key,
        "selected_assembly",
        "selected_layer",
        "selected_i",
        "selected_j",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if state[f"grid_view_{view_id}"]["name"] != option["name"]:
            return
        selected_array = state[selected_array_key]
        indices = (
            int(state.selected_j),
            int(state.selected_i),
            int(state.selected_layer),
            int(state.selected_assembly),
        )
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line(selected_array, indices))

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