import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import html
from vera_core.widgets import vera
import operator

THRESHOLD_OPS = {
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

def apply_thresholds(array, conditions):
    keep = np.ones(array.shape, dtype=bool)
    for c in conditions:
        print(c)
        keep &= THRESHOLD_OPS[c["op"]](array, c["value"])
    return np.where(keep, array, np.nan)

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

    @state.change(selected_array_key, "selected_layer", "thresholds")
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_view(**kwargs):
        if state[f"grid_view_{view_id}"]["name"] != option["name"]:
            return
        selected_array = state[selected_array_key]
        selected_layer = int(state.selected_layer)

        array = vera_out_file.array(selected_array)
        layer_array = array[:, :, selected_layer].swapaxes(0, 2).swapaxes(1, 2).copy()

        control_rod_positions = vera_out_file.core.control_rod_positions
        rod_rows, rod_cols = control_rod_positions
        layer_array[:, rod_rows, rod_cols] = np.nan
        thres = state["thresholds"]
        if thres.get(selected_array):
            layer_array = apply_thresholds(layer_array, thres[selected_array])
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
                line.append(np.ravel(layer_array[index]).tolist())

        state[core_assemblies_key] = result

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: column;"
        with html.Div(style="flex: 1; min-height: 0; position: relative;"):
            vera.CoreView(
                value=(core_assemblies_key, []),
                selected_i=("selected_assembly_ij.i",),
                selected_j=("selected_assembly_ij.j",),
                color_preset="jet",
                color_range=(f"color_range_{view_id}", [0, 3]),
                click="selected_assembly_ij = $event",
                busy=("trame__busy",),
            )
        with html.Div(style="flex: 0 0 auto; padding: 4px 0;"):
            vera.ColorMapEditor(
                v_model=f"color_range_{view_id}",
                color_preset="jet",
            )