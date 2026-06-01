import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from trame.ui.html import DivLayout
from trame.widgets import plotly

from vera_core.app.core import VeraDataRegistry, VeraDataSource
from ..features import FileMenu
from ..helpers import is_non_active_view

SEP = "\x1f"

def option_for(view_id):
    return {
        "name": f"axial_plot_{view_id}",
        "label": "Axial Plot",
        "multi_picker" : True,
        "icon": "mdi-align-horizontal-center",
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_file_key = f"selected_file_{view_id}"
    my_arrays = f"axial_array_list_{view_id}"
    my_arrays_labels = f"axial_array_labels_{view_id}"
    state[my_arrays] = []
    state[my_arrays_labels] = "Select datasets"

    update_fn_name = f"update_axial_plot_{view_id}"

    @ctrl.set(f"toggle_axial_array_{view_id}")
    def toggle_axial_array(file, key):
        array_token = f"{file}{SEP}{key}"
        current = list(state[my_arrays])
        if array_token in current:
            current.remove(array_token)
        else:
            current.append(array_token)
        state[my_arrays] = current
        state[my_arrays_labels] = (
            f"{len(current)} selected" if current else "Select datasets"
        )

    def create_line(source : VeraDataSource, selected_array, indices=(0, 0, 0, 0)):
        selected_j, selected_i, selected_layer, selected_assembly = indices

        arrays_to_plot = state[my_arrays]
        figure = go.Figure()
        for token in arrays_to_plot:
            file_name, array_name = token.split(SEP, 1)
            full_array = registry.get(file_name).array(array_name)
            axial_array = full_array[selected_j, selected_i, :, selected_assembly]
            figure.add_trace(
                go.Scatter(
                    x=axial_array,
                    y=source.core.axial_mesh_means,
                    mode="lines",
                    name=f"{file_name} | {array_name.replace("_", " ").title()}",
                )
            )

        # add_hline only plots x==0 to x==1, so draw the marker manually.
        float_info = np.finfo(np.float64)
        figure.add_trace(
            go.Scatter(
                x=[float_info.min, float_info.max],
                y=[source.core.axial_mesh_means[selected_layer]] * 2,
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(margin=dict(t=0, b=0, l=0, r=0))
        return figure

    @state.change(
        my_arrays,
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
        source = registry.get(selected_file)
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line(source, selected_array, indices))

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; width: 100%;"

        FileMenu.build_file_dataset_multi_picker(ctrl, my_arrays_labels, my_arrays, f"toggle_axial_array_{view_id}")
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