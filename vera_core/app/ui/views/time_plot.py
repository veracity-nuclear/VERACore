import numpy as np
import plotly.graph_objects as go

from trame.ui.html import DivLayout
from trame.widgets import plotly

from vera_core.app.core import VeraDataRegistry, VeraDataSource
from ..helpers import is_non_active_view

SEP = "\x1f"

def option_for(view_id):
    return {
        "name": f"time_plot_{view_id}",
        "label": "Time Plot",
        "multi_picker": True,
        "icon": "mdi-chart-line",
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_set_key = f"multi_selected_{view_id}"

    update_fn_name = f"update_time_plot_{view_id}"

    def create_line(indices=(0, 0, 0, 0)):
        figure = go.Figure()
        for token in state[selected_set_key]:
            src_id, array_name = token.split(SEP, 1)
            src = registry.get(src_id)
            exposures = [np.asarray(x.exposure).item() for x in src.states]
            if array_name == "pin_volumes":
                # Volumes don't change with time, so this is a flat line.
                pin_volumes = src.core.pin_volumes
                values = [pin_volumes[indices] for _ in src.states]
            else:
                values = [getattr(x, array_name)[indices] for x in src.states]
            figure.add_trace(
                go.Scatter(
                    x=exposures,
                    y=values,
                    mode="lines",
                    name=f"{src_id} | {array_name.replace('_', ' ').title()}",
                )
            )

        # add_vline only spans y in [0, 1], so draw the marker manually.
        float_info = np.finfo(np.float64)
        figure.add_trace(
            go.Scatter(
                x=[np.asarray(registry.default_src.active_state.exposure).item()] * 2,
                y=[float_info.min, float_info.max],
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        return figure

    @state.change(
        selected_set_key,
        "max_time",
        "selected_assembly",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = (
            int(state.selected_j),
            int(state.selected_i),
            int(state.selected_layer),
            int(state.selected_assembly),
        )
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line(indices))

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