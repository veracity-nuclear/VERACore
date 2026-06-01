import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from trame.ui.html import DivLayout
from trame.widgets import plotly

from vera_core.app.core import VeraDataRegistry, VeraDataSource
from ..helpers import is_non_active_view

def option_for(view_id):
    return {
        "name": f"time_plot_{view_id}",
        "label": "Time Plot",
        "multi_picker" : False,
        "icon": "mdi-chart-line",
    }


def initialize(server, registry : VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_file_key = f"selected_file_{view_id}"
    update_fn_name = f"update_time_plot_{view_id}"

    def create_line(vera_source : VeraDataSource, selected_array, indices=(0, 0, 0, 0)):
        exposures = [np.asarray(x.exposure).item() for x in vera_source.states]

        if selected_array == "pin_volumes":
            # Volumes don't change with time, so this is a flat line.
            pin_volumes = vera_source.core.pin_volumes
            array = [pin_volumes[indices] for _ in vera_source.states]
        else:
            array = [getattr(x, selected_array)[indices] for x in vera_source.states]
        figure = px.line(
            x=exposures,
            y=array,
            labels={"x": "exposure", "y": selected_array},
        )

        # add_vline only plots y==0 to y==1, so draw the marker manually.
        float_info = np.finfo(np.float64)
        figure.add_trace(
            go.Scatter(
                x=[np.asarray(vera_source.active_state.exposure).item()] * 2,
                y=[float_info.min, float_info.max],
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        return figure

    @state.change(
        selected_array_key,
        selected_file_key,
        "selected_assembly",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}"
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        selected_file = state[selected_file_key]
        selected_array = state[selected_array_key]
        indices = (
            int(state.selected_j),
            int(state.selected_i),
            int(state.selected_layer),
            int(state.selected_assembly),
        )
        vera_source = registry.get(selected_file)
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line(vera_source, selected_array, indices))

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