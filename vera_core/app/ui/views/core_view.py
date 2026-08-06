from typing import Sequence

import numpy as np
from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.data.dtypes import MAX_NUM_GROUPS, VeraDtype
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry
from vera_core.data.thresholds import ThresholdCondition, apply_thresholds
from vera_core.widgets import vera

from ..helpers import format_label, get_safe_idxs, is_non_active_view, set_info
from ._core_grid import (
    assembly_side,
    core_labels,
    format_for_vis,
    nan_out_non_fuel_locs,
)

ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.PIN,
    VeraDtype.CHANNEL,
    VeraDtype.ASSEMBLY,
    VeraDtype.RADIAL,
    VeraDtype.RADIAL_ASSEMBLY,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
    VeraDtype.COMP_ASSY,
    VeraDtype.COMP_ASSY_ENERGY,
    VeraDtype.NODAL,
    VeraDtype.POINT_DETECTOR,
    VeraDtype.RADIAL_POINT_DETECTOR,
]

MAX_LABEL_SIDE = 2


def option_for(view_id):
    return {
        "name": f"core_view_{view_id}",
        "label": "Core View",
        "multi_picker": False,
        "icon": "mdi-chart-pie",
        "allowed_categories": [dtype.title for dtype in ALLOWED_DTYPES],
    }


def create_core_view(
    vera_source: VeraDataSource,
    dataset_name: str,
    z: int,
    thresholds: Sequence[ThresholdCondition] = [],
):
    if z < 0:
        raise RuntimeError(f"z must be < 0, z = {z}")
    dataset = vera_source.get_dataset(dataset_name)
    ds_dtype = dataset.dataset_type
    if ds_dtype not in ALLOWED_DTYPES:
        return tuple()
    is_comp = ds_dtype.is_computational()
    layer_list = []
    match ds_dtype:
        case VeraDtype.PIN | VeraDtype.CHANNEL:
            layer_list.append(dataset[:, :, z].swapaxes(0, 2).swapaxes(1, 2))
        case VeraDtype.POINT_DETECTOR:
            layer_list.append(dataset[z, :])
        case VeraDtype.ASSEMBLY | VeraDtype.COMP_ASSY:
            layer_list.append(dataset[0, z, :])
        case VeraDtype.RADIAL:
            layer_list.append(dataset.swapaxes(0, 2).swapaxes(1, 2))
        case VeraDtype.RADIAL_ASSEMBLY | VeraDtype.RADIAL_POINT_DETECTOR:
            layer_list.append(dataset)
        case VeraDtype.COMP_NODAL | VeraDtype.NODAL:
            layer_list.append(dataset[:, z, :].swapaxes(0, 1))
        case VeraDtype.COMP_ASSY_ENERGY:
            num_energy_groups = np.shape(dataset)[0]
            for energy_group in range(num_energy_groups):
                layer_list.append(dataset[energy_group, 0, z, :])
        case VeraDtype.COMP_NODAL_ENERGY:
            num_energy_groups = np.shape(dataset)[0]
            for energy_group in range(num_energy_groups):
                layer_list.append(dataset[energy_group, :, z, :].swapaxes(0, 1))
        case _:
            raise RuntimeError(f"Core View cannot visualize a dataset of type {str(ds_dtype)} ")
    core = vera_source.core
    results = []
    for layer in layer_list:
        if ds_dtype.has_fuel_pins():
            layer = nan_out_non_fuel_locs(layer, vera_source, z, ds_dtype == VeraDtype.RADIAL)
        if thresholds:
            layer = apply_thresholds(layer, thresholds)
        # The label values are the cell values, so the second return is unused.
        formatted_result, _ = format_for_vis(src=vera_source, dataset=layer)
        results.append(formatted_result)

    sample = next((c for row in formatted_result for c in row if isinstance(c, list) and c), None)
    assembly_side_size = assembly_side(len(sample)) if sample else 0
    x_labels, y_labels, max_core_cols = core_labels(core, is_comp)
    return (
        results,
        assembly_side_size,
        x_labels,
        y_labels,
        max_core_cols,
    )


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"

    n_groups_key = f"n_groups_{view_id}"
    state.setdefault(n_groups_key, 0)
    group_keys = [f"core_assemblies_{view_id}_{g}" for g in range(MAX_NUM_GROUPS)]
    x_label_key = f"core_view_x_labels_{view_id}"
    y_label_key = f"core_view_y_labels_{view_id}"
    core_cols_key = f"core_cols_{view_id}"
    assembly_size_key = f"assembly_size_{view_id}"

    show_labels_key = f"assembly_show_labels_{view_id}"
    state.setdefault(show_labels_key, False)
    decimals_key = f"assembly_decimals_{view_id}"
    state.setdefault(decimals_key, 2)

    for gk in group_keys:
        state.setdefault(gk, [])

    aspect_ratio_key = f"aspect_ratio_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(aspect_ratio_key, 1)
    state.setdefault(x_label_key, [])
    state.setdefault(y_label_key, [])
    state.setdefault(core_cols_key, 1)
    state.setdefault(assembly_size_key, 1)

    @state.change("selected_assembly_ij")
    def update_info(**kwargs):
        set_info(view_id, state, registry)

    @state.change(
        selected_array_key,
        selected_src_key,
        "selected_layer",
        "thresholds",
        f"grid_view_{view_id}",
        lock_flag,
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        _, _, selected_layer, _, selected_src_id, selected_array = indices
        thres_key = format_label(selected_src_id, selected_array)
        thresholds_to_apply = state["thresholds"].get(thres_key, [])
        vera_source: VeraDataSource = registry.get(selected_src_id)
        state[aspect_ratio_key] = vera_source.core.aspect_ratio
        vis_state = create_core_view(
            vera_source, selected_array, selected_layer, thresholds_to_apply
        )
        if not vis_state:
            return
        results, assembly_side_size, xlabels, ylabels, max_core_cols = vis_state
        num_groups = len(results)
        for idx in range(MAX_NUM_GROUPS):
            state[f"core_assemblies_{view_id}_{idx}"] = [] if idx >= num_groups else results[idx]
        state[n_groups_key] = num_groups
        state[assembly_size_key] = assembly_side_size
        state[x_label_key] = xlabels
        state[y_label_key] = ylabels
        state[core_cols_key] = max_core_cols
        set_info(view_id, state, registry)

    can_label = f"{assembly_size_key} > 0 && {assembly_size_key} <= {MAX_LABEL_SIDE}"

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=("flex: 1; min-width: 0;display: flex; flex-direction: column;")):
            # Row of up to 4 group views; wraps to a 2x2 grid when >2 groups.
            with html.Div(
                style=(
                    "flex: 1; min-height: 0;display: flex; flex-direction: row; flex-wrap: wrap;"
                )
            ):
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
                        with html.Div(
                            style=("flex: 1; min-height: 0;display: flex; flex-direction: row;")
                        ):
                            with html.Div(
                                style="flex: 1; min-width: 0; min-height: 0; position: relative;"
                            ):
                                vera.CoreView(
                                    value=(group_keys[g], []),
                                    selected_i=("selected_assembly_ij.i",),
                                    selected_j=("selected_assembly_ij.j",),
                                    aspect_ratio=(aspect_ratio_key, 1),
                                    x_labels=(f"{x_label_key}",),
                                    y_labels=(f"{y_label_key}",),
                                    assembly_size=(assembly_size_key,),
                                    core_cols=(core_cols_key,),
                                    color_preset="jet",
                                    color_range=(f"color_range_{view_id}_{g}", [0, 3]),
                                    click="selected_assembly_ij = $event",
                                    dark=("dark_mode",),
                                    busy=("trame__busy",),
                                    show_labels=(show_labels_key, False),
                                    decimals=(decimals_key, 2),
                                )
                            with html.Div(
                                style=(
                                    "flex: 0 0 auto; width: 70px; padding: 4px 0;"
                                    "display: flex; align-self: stretch;"
                                )
                            ):
                                vera.VerticalColorMapEditor(
                                    v_model=f"color_range_{view_id}_{g}",
                                    color_preset="jet",
                                    units=(f"color_units_{view_id}",),
                                )
            # Footer: caption, values toggle and decimals selector on one line.
            with html.Div(
                style=(
                    "flex: 0 0 auto; display: flex; align-items: center;"
                    "justify-content: center; gap: 16px;"
                    "min-height: 44px; padding: 6px 16px;"
                )
            ):
                html.Div(
                    "Exposure {{ " + info + ".Exposure }}"
                    " · ({{ " + info + ".Assembly }})"
                    " · Axial - {{ " + info + ".Layer }}",
                    classes="text-caption",
                )
                vuetify.VCheckbox(
                    v_if=(can_label,),
                    v_model=show_labels_key,
                    label="Show values",
                    dense=True,
                    hide_details=True,
                    classes="ma-0 pa-0 text-caption",
                    style="flex: 0 0 auto;",
                )
                vuetify.VSelect(
                    v_if=(f"{show_labels_key} && ({can_label})",),
                    v_model=decimals_key,
                    items=("[0,1,2,3,4]",),
                    label="Decimals",
                    dense=True,
                    hide_details=True,
                    style="flex: 0 0 auto; max-width: 90px;",
                )
