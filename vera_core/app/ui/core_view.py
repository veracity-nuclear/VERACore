import numpy as np

from trame.ui.html import DivLayout
from vera_core.widgets import vera


def option_for(view_id):
    return {
        "name": f"core_view_{view_id}",
        "label": "Core View",
        "icon": "mdi-chart-pie",
    }


def initialize(server, vera_out_file, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    core_assemblies_key = f"core_assemblies_{view_id}"
    state.setdefault(core_assemblies_key, [])

    @state.change(selected_array_key, "selected_layer")
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_view(**kwargs):
        selected_array = state[selected_array_key]
        selected_layer = int(state.selected_layer)

        array = vera_out_file.array(selected_array)
        layer_array = array[:, :, selected_layer].swapaxes(0, 2).swapaxes(1, 2)

        control_rod_positions = vera_out_file.core.control_rod_positions
        reduced_core_map = vera_out_file.core.reduced_core_map
        core_width = reduced_core_map.shape[0]

        result = []
        for i in range(core_width):
            line = []
            result.append(line)
            for j in range(core_width):
                index = reduced_core_map[i, j] - 1
                if index == -1:
                    continue
                assembly_array = layer_array[index].copy()
                assembly_array[control_rod_positions] = np.nan
                line.append(np.ravel(assembly_array).tolist())

        state[core_assemblies_key] = result

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%;"
        vera.CoreView(
            value=(core_assemblies_key, []),
            selected_i=("selected_assembly_ij.i",),
            selected_j=("selected_assembly_ij.j",),
            color_preset="jet",
            color_range=("color_range", [0, 3]),
            click="selected_assembly_ij = $event",
            busy=("trame__busy",),
        )