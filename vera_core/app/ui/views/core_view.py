import numpy as np
import math

from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDtype, VeraDataSource, VeraDataset, MAX_NUM_GROUPS, NUM_NODES
from vera_core.app.core.thresholds import apply_thresholds
from ..helpers import format_label, is_non_active_view, set_info, get_safe_idxs

def option_for(view_id):
    return {
        "name": f"core_view_{view_id}",
        "label": "Core View",
        "multi_picker" : False,
        "icon": "mdi-chart-pie",
        "allowed_categories": [VeraDtype.PIN.title, 
                               VeraDtype.CHANNEL.title, 
                               VeraDtype.ASSEMBLY.title, 
                               VeraDtype.RADIAL.title, 
                               VeraDtype.RADIAL_ASSEMBLY.title, 
                               VeraDtype.COMP_NODAL.title,
                               VeraDtype.COMP_NODAL_ENERGY.title, 
                               VeraDtype.COMP_ASSY.title,
                               VeraDtype.COMP_ASSY_ENERGY.title, 
                               VeraDtype.NODAL.title,
                               VeraDtype.DETECTOR.title,
                               VeraDtype.RADIAL_DETECTOR.title]
    }

def _nan_out_control_rods(array : np.ndarray, control_rod_positions):
    if control_rod_positions is None:
        return array
    rod_rows, rod_cols = control_rod_positions
    new_array = array.copy()
    new_array[:, rod_rows, rod_cols] = np.nan
    return new_array

def _assembly_side(n: int) -> int:
    """Pin-side length for a cell of n values. n must be a perfect square."""
    if n <= 0:
        return 0
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"assembly cell length {n} is not a perfect square")
    return side


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"

    n_groups_key = f"n_groups_{view_id}"
    state.setdefault(n_groups_key, 0)
    group_keys = [f"core_assemblies_{view_id}_{g}" for g in range(MAX_NUM_GROUPS)]
    label_keys = [f"core_labels_{view_id}_{g}" for g in range(MAX_NUM_GROUPS)]
    x_label_key = f"core_view_x_labels_{view_id}"
    y_label_key = f"core_view_y_labels_{view_id}"
    core_cols_key = f"core_cols_{view_id}"
    assembly_size_key = f"assembly_size_{view_id}"
    
    for gk in group_keys:
        state.setdefault(gk, [])
    for lk in label_keys:
        state.setdefault(lk, [])

    aspect_ratio_key = f"aspect_ratio_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(aspect_ratio_key, 1)
    state.setdefault(x_label_key, [])
    state.setdefault(y_label_key, [])
    state.setdefault(core_cols_key, 1)
    state.setdefault(assembly_size_key, 1)

    def _vis_pin_level_data(src : VeraDataSource, dataset : VeraDataset):
        cm = src.core.get_map(dataset)
        is_assembly_avg = dataset.is_assembly()
        core_width = cm.shape[0]
        result = []
        labels = []
        for i in range(core_width):
            line = [None] * core_width
            result.append(line)
            if is_assembly_avg:
                labels_line = [None] * core_width
                labels.append(labels_line)
            for j in range(core_width):
                index = cm[i, j] - 1
                if index == -1:
                    continue
                if is_assembly_avg:
                    line[j] = [float(dataset[index])]
                    labels_line[j] = np.round(dataset[index], 2)
                else:
                    line[j] = np.ravel(dataset[index]).tolist()
        return result, labels        
    
    @state.change("selected_assembly_ij")
    def update_info(**kwargs):
        set_info(view_id, state, registry)

    @state.change(selected_array_key, selected_src_key, "selected_layer", "thresholds", f"grid_view_{view_id}", lock_flag)
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        _, _, selected_layer, _, selected_src_id, selected_array = indices
        thres_key = format_label(selected_src_id, selected_array)
        vera_source : VeraDataSource = registry.get(selected_src_id)
        core = vera_source.core
        state[aspect_ratio_key] = vera_source.core.aspect_ratio
        raw_array = vera_source.array(selected_array)
        raw_array_dtype = raw_array.dataset_type
        is_comp = raw_array_dtype.is_computational()
        if raw_array_dtype.title not in option_for(0)["allowed_categories"]:
            return
        layer_arrays = []
        match raw_array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                layer_arrays.append(raw_array[:, :, selected_layer].swapaxes(0, 2).swapaxes(1, 2))
            case VeraDtype.ASSEMBLY | VeraDtype.DETECTOR:
                layer_arrays.append(raw_array[selected_layer, :])
            case VeraDtype.COMP_ASSY:
                layer_arrays.append(raw_array[0, selected_layer, :])
            case VeraDtype.RADIAL:
                layer_arrays.append(raw_array.swapaxes(0, 2).swapaxes(1, 2))
            case VeraDtype.RADIAL_ASSEMBLY, VeraDtype.RADIAL_DETECTOR:
                layer_arrays.append(raw_array)
            case VeraDtype.COMP_NODAL | VeraDtype.NODAL:
                layer_arrays.append(raw_array[:, selected_layer, :].swapaxes(0, 1))
            case VeraDtype.COMP_ASSY_ENERGY:
                num_energy_groups = np.shape(raw_array)[0]
                for energy_group in range(num_energy_groups):
                    layer_arrays.append(raw_array[energy_group, 0, selected_layer, :])
            case VeraDtype.COMP_NODAL_ENERGY:
                num_energy_groups = np.shape(raw_array)[0]
                for energy_group in range(num_energy_groups):
                    layer_arrays.append(raw_array[energy_group, :, selected_layer, :].swapaxes(0, 1))
            case _:
                raise RuntimeError(f"Core View cannot visualize a dataset of type {str(raw_array_dtype)} ")
        state[x_label_key] = (core.comp_core_map_column_labels if raw_array_dtype.is_computational() else
            core.reduced_core_map_column_labels)
        start_idx = (core.comp_map_start_index if raw_array_dtype.is_computational() else
            vera_source.core.reduced_core_map_start_index)
        
        num_rows = 0
        for idx, layer_array in enumerate(layer_arrays):
            if raw_array_dtype in (VeraDtype.PIN, VeraDtype.RADIAL):
                layer_array = _nan_out_control_rods(layer_array, vera_source.core.control_rod_positions)
            thres = state["thresholds"]
            if thres.get(thres_key):
                layer_array = apply_thresholds(layer_array, thres[thres_key])        
            result, labels = _vis_pin_level_data(src=vera_source, dataset=layer_array)
            num_rows = len(result) 
            state[f"core_assemblies_{view_id}_{idx}"] = result
            state[f"core_labels_{view_id}_{idx}"] = labels
        if is_comp:
            state[y_label_key] = [start_idx + row + 1 for row in range(num_rows)]
        else:
            state[y_label_key] = core.reduced_core_map_row_labels
        num_groups = len(layer_arrays)
        state[core_cols_key] = core.comp_core_map.shape[0] if is_comp else core.reduced_core_map.shape[0]
        
        sample = next((c for row in result for c in row if isinstance(c, list) and c), None)
        state[f"assembly_size_{view_id}"] = _assembly_side(len(sample)) if sample else 0
        
        for idx in range(num_groups, MAX_NUM_GROUPS):
            state[f"core_assemblies_{view_id}_{idx}"] = []
            state[f"core_labels_{view_id}_{idx}"] = []
        state[n_groups_key] = num_groups
        set_info(view_id, state, registry)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            # Row of up to 4 group views; wraps to a 2x2 grid when >2 groups.
            with html.Div(style=(
                "flex: 1; min-height: 0;"
                "display: flex; flex-direction: row; flex-wrap: wrap;"
            )):
                for g in range(MAX_NUM_GROUPS):
                    with html.Div(
                        v_if=(f"{n_groups_key} > {g}",),
                        style=(
                            "flex: 1 1 45%; min-width: 0; min-height: 0;"
                            "display: flex; flex-direction: column; position: relative;"
                        ),
                    ):
                        html.Div(
                            f"Group {g + 1}",
                            v_if=(f"{n_groups_key} > 1",),
                            classes="text-caption text-center font-weight-medium",
                            style="flex: 0 0 auto;",
                        )
                        # Core + its own colorbar, side by side.
                        with html.Div(style=(
                            "flex: 1; min-height: 0;"
                            "display: flex; flex-direction: row;"
                        )):
                            with html.Div(style="flex: 1; min-width: 0; min-height: 0; position: relative;"):
                                vera.CoreView(
                                    value=(group_keys[g], []),
                                    labels=(label_keys[g], []),
                                    selected_i=("selected_assembly_ij.i",),
                                    selected_j=("selected_assembly_ij.j",),
                                    aspect_ratio=(aspect_ratio_key, 1),
                                    x_labels=(f"{x_label_key}",),
                                    y_labels=(f"{y_label_key}",),
                                    assembly_size = (assembly_size_key,),
                                    core_cols = (core_cols_key,),
                                    color_preset="jet",
                                    color_range=(f"color_range_{view_id}_{g}", [0, 3]),
                                    click="selected_assembly_ij = $event",
                                    dark=("dark_mode",),
                                    busy=("trame__busy",),
                                )
                            with html.Div(style=(
                                "flex: 0 0 auto; width: 70px; padding: 4px 0;"
                                "display: flex; align-self: stretch;"
                            )):
                                vera.VerticalColorMapEditor(
                                    v_model=f"color_range_{view_id}_{g}",
                                    color_preset="jet",
                                    units=(f"color_units_{view_id}",),
                                )
            html.Div(
                "Exposure {{ " + info + ".Exposure }}"
                " · ({{ " + info + ".Assembly }})"
                " · Axial - {{ " + info + ".Layer }}",
                classes="text-caption text-center",
            )