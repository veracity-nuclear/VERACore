import numpy as np
import operator

from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDtype, VeraDataSource, VeraDataset
from vera_core.app.core.thresholds import apply_thresholds
from ..helpers import format_label, is_non_active_view, set_info

def option_for(view_id):
    return {
        "name": f"core_view_{view_id}",
        "label": "Core View",
        "multi_picker" : False,
        "icon": "mdi-chart-pie",
        "allowed_categories": [VeraDtype.PIN.title, VeraDtype.CHANNEL.title, 
                               VeraDtype.ASSEMBLY.title, VeraDtype.RADIAL.title, 
                               VeraDtype.RADIAL_ASSEMBLY.title, VeraDtype.COMP_NODAL.title]
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    core_assemblies_key = f"core_assemblies_{view_id}"
    aspect_ratio_key = f"aspect_ratio_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(core_assemblies_key, [])
    state.setdefault(aspect_ratio_key, 1)
    

    def _vis_pin_level_data(src, dataset):
        reduced_core_map = src.core.reduced_core_map
        core_width = reduced_core_map.shape[0]
        result = []
        for i in range(core_width):
            line = []
            result.append(line)
            for j in range(core_width):
                index = reduced_core_map[i, j] - 1
                if index == -1:
                    continue   
                line.append(np.ravel(dataset[index]).tolist())               
        state[core_assemblies_key] = result
        state[f"core_labels_{view_id}"] = []
    
    def _vis_assembly_level_data(src, dataset):
        reduced_core_map = src.core.reduced_core_map
        core_width = reduced_core_map.shape[0]
        result = []
        labels = []
        for i in range(core_width):
            line = []
            labels_line = []
            labels.append(labels_line)
            result.append(line)
            for j in range(core_width):
                index = reduced_core_map[i, j] - 1
                if index == -1:
                    continue   
                line.append([float(dataset[index])])
                labels_line.append(np.round(dataset[index], 2))
        state[core_assemblies_key] = result
        state[f"core_labels_{view_id}"] = labels
    
    NUM_NODES = 4
    def _vis_nodal_level_data(src : VeraDataSource, dataset : VeraDataset):
        cm = src.core.comp_core_map if dataset.dataset_type.is_computational() else src.core.reduced_core_map
        core_width = cm.shape[0]
        result = []
        for i in range(core_width):
            line = []
            result.append(line)
            for j in range(core_width):
                index = cm[i, j] - 1
                if index == -1:
                    continue   
                line.append(list(dataset[:, index]))
        state[core_assemblies_key] = result
        state[f"core_labels_{view_id}"] = []
    


    @state.change(selected_array_key, selected_src_key, "selected_layer", "thresholds", f"grid_view_{view_id}", lock_flag)
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        selected_src_id = state[selected_src_key]
        selected_array = state[selected_array_key]
        thres_key = format_label(selected_src_id, selected_array)
        selected_layer = int(state.selected_layer)

        vera_source : VeraDataSource = registry.get(selected_src_id)
        state[aspect_ratio_key] = vera_source.core.aspect_ratio
        array = vera_source.array(selected_array)
        array_dtype = array.dataset_type
        if str(array_dtype).upper() not in option_for(0)["allowed_categories"]:
            return
        match array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                layer_array = array[:, :, selected_layer].swapaxes(0, 2).swapaxes(1, 2).copy()
            case VeraDtype.ASSEMBLY:
                layer_array = array[selected_layer, :]
            case VeraDtype.RADIAL:
                layer_array = array.swapaxes(0, 2).swapaxes(1, 2).copy()
            case VeraDtype.RADIAL_ASSEMBLY:
                layer_array = array
            case VeraDtype.COMP_NODAL:
                layer_array = array[:, selected_layer, :]
                _vis_nodal_level_data(vera_source, layer_array)
                return
            case _:
                raise RuntimeError(f"Core View cannot visualize a dataset of type {str(array_dtype)} ")
        # nan out control rods for dataset with pin level data
        if array_dtype in (VeraDtype.PIN, VeraDtype.RADIAL):
            control_rod_positions = vera_source.core.control_rod_positions
            rod_rows, rod_cols = control_rod_positions
            layer_array[:, rod_rows, rod_cols] = np.nan
        # set labels if array_dtype is assembly valued dtype
        is_assembly_avg = array_dtype in (VeraDtype.ASSEMBLY, VeraDtype.RADIAL_ASSEMBLY)
        thres = state["thresholds"]
        if thres.get(thres_key):
            layer_array = apply_thresholds(layer_array, thres[thres_key])
        reduced_core_map = vera_source.core.reduced_core_map
        core_width = reduced_core_map.shape[0]
        result = []
        labels = []
        for i in range(core_width):
            line = []
            if is_assembly_avg:
                labels_line = []
                labels.append(labels_line)
            result.append(line)
            for j in range(core_width):
                index = reduced_core_map[i, j] - 1
                if index == -1:
                    continue   
                if is_assembly_avg:
                    line.append([float(layer_array[index])])
                    labels_line.append(np.round(layer_array[index], 2))
                else:
                    line.append(np.ravel(layer_array[index]).tolist())               
        state[core_assemblies_key] = result
        state[f"core_labels_{view_id}"] = labels
        set_info(state, vera_source, view_id)

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
                    aspect_ratio=(aspect_ratio_key, 1),
                    color_preset="jet",
                    color_range=(f"color_range_{view_id}", [0, 3]),
                    click="selected_assembly_ij = $event",
                    busy=("trame__busy",),
                )
            html.Div(
                "Exposure {{ " + info + ".Exposure }}"
                " · ({{ " + info + ".Assembly }})"
                " · Axial - {{ " + info + ".Layer }}",
                classes="text-caption text-center",
            )
        with html.Div(style=(
            "flex: 0 0 auto; width: 70px; padding: 4px 0;"
            "display: flex; align-self: stretch;"
        )):
            vera.VerticalColorMapEditor(
                v_model=f"color_range_{view_id}",
                color_preset="jet",
            )