import numpy as np
import operator

from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDatasetType, VeraDataSource
from vera_core.app.core.thresholds import apply_thresholds
from ..helpers import format_label

def option_for(view_id):
    return {
        "name": f"core_view_{view_id}",
        "label": "Core View",
        "icon": "mdi-chart-pie",
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]
    state[f"core_readout_{view_id}"] = {}

    selected_array_key = f"selected_array_{view_id}"
    selected_file_key = f"selected_file_{view_id}"
    core_assemblies_key = f"core_assemblies_{view_id}"
    state.setdefault(core_assemblies_key, [])

    @state.change(selected_array_key, selected_file_key, "selected_layer", "thresholds", f"grid_view_{view_id}")
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_view(**kwargs):
        if state[f"grid_view_{view_id}"]["name"] != option["name"]:
            return
        selected_array = state[selected_array_key]
        selected_file = state[selected_file_key]
        thres_key = format_label(selected_file, selected_array)
        selected_layer = int(state.selected_layer)

        vera_source : VeraDataSource = registry.get(selected_file)
        array = vera_source.array(selected_array)
        is_assembly_average = array.dataset_type == VeraDatasetType.ASSEMBLY 
        if array.ndim == 4:
            layer_array = array[:, :, selected_layer].swapaxes(0, 2).swapaxes(1, 2).copy()
            control_rod_positions = vera_source.core.control_rod_positions
            rod_rows, rod_cols = control_rod_positions
            layer_array[:, rod_rows, rod_cols] = np.nan
            state[f"core_readout_{view_id}"] = {"values": []}
        elif is_assembly_average:
            layer_array = array[selected_layer, :]
            state[f"core_readout_{view_id}"] = {"values": layer_array.tolist(),}
        

        thres = state["thresholds"]
        if thres.get(thres_key):
            layer_array = apply_thresholds(layer_array, thres[thres_key])
        reduced_core_map = vera_source.core.reduced_core_map
        core_width = reduced_core_map.shape[0]
        result = []
        labels = []
        for i in range(core_width):
            line = []
            if is_assembly_average:
                labels_line = []
                labels.append(labels_line)
            result.append(line)
            for j in range(core_width):
                index = reduced_core_map[i, j] - 1
                if index == -1:
                    continue   
                if is_assembly_average:
                    line.append([float(layer_array[index])])
                    labels_line.append(np.round(layer_array[index], 2))
                else:
                    line.append(np.ravel(layer_array[index]).tolist())               

        state[core_assemblies_key] = result
        # ... after building `result` and computing assembly_means for the avg case ...
        state[f"core_labels_{view_id}"] = labels

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                vera.CoreView(
                    v_if=(f"{core_assemblies_key} && {core_assemblies_key}.length",),
                    value=(core_assemblies_key, []),
                    labels=(f"core_labels_{view_id}", []),
                    selected_i=("selected_assembly_ij.i",),
                    selected_j=("selected_assembly_ij.j",),
                    color_preset="jet",
                    color_range=(f"color_range_{view_id}", [0, 3]),
                    click="selected_assembly_ij = $event",
                    busy=("trame__busy",),
                )
            html.Div(
                f"{{{{ selected_assembly != null && core_readout_{view_id}"
                f" ? core_readout_{view_id}.values[selected_assembly] : '' }}}}",
                classes="text-caption text-center",
                style="flex: 0 0 auto; padding: 4px 0;",
            )
        with html.Div(style=(
            "flex: 0 0 auto; width: 70px; padding: 4px 0;"
            "display: flex; align-self: stretch;"
        )):
            vera.VerticalColorMapEditor(
                v_model=f"color_range_{view_id}",
                color_preset="jet",
            )