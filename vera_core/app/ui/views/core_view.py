from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.data.analysis.core_slice import CoreSlice
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry
from vera_core.data.renders import CoreView, Selection
from vera_core.widgets import vera

from ..helpers import (
    format_label,
    get_safe_idxs,
    get_thresholds,
    is_non_active_view,
    pick_group,
    set_info,
)
from . import MAX_VIS_GROUPS
from .save_image import notification, register_photo_state, take_photo

MAX_LABEL_SIDE = 2


def option_for(view_id):
    return {
        "name": f"core_view_{view_id}",
        "label": "Core View",
        "multi_picker": False,
        "icon": "mdi-chart-pie",
        "allowed_categories": [dtype.title for dtype in CoreSlice.ALLOWED_DTYPES],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    selected_group_key = f"selected_group_{view_id}"

    n_groups_key = f"n_groups_{view_id}"
    state.setdefault(n_groups_key, 0)
    group_keys = [f"core_assemblies_{view_id}_{g}" for g in range(MAX_VIS_GROUPS)]
    x_label_key = f"core_view_x_labels_{view_id}"
    y_label_key = f"core_view_y_labels_{view_id}"
    core_cols_key = f"core_cols_{view_id}"
    assembly_size_key = f"assembly_size_{view_id}"

    show_labels_key = f"assembly_show_labels_{view_id}"
    state.setdefault(show_labels_key, False)
    decimals_key = f"assembly_decimals_{view_id}"
    state.setdefault(decimals_key, 2)

    msg_key, msg_show_key = register_photo_state(state, view_id, option["name"])

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

    saved_sel: Selection | None = None

    @state.change("selected_assembly_ij")
    def update_info(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        set_info(view_id, state, registry)

    @state.change(
        selected_array_key,
        selected_src_key,
        selected_group_key,
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
        _, _, selected_layer, _, selected_src_id, selected_array, time, _ = indices
        vera_source: VeraDataSource = registry.get(selected_src_id)
        state[aspect_ratio_key] = vera_source.core.aspect_ratio
        thresholds_to_apply = get_thresholds(state, view_id)
        core_slice = CoreSlice.create_core_slice(
            vera_source,
            selected_array,
            selected_layer,
            thresholds=thresholds_to_apply,
            state_idx=time,
        )
        if not core_slice:
            return

        cv = CoreView(source=vera_source)
        sel = cv.select(
            selected_array,
            z=selected_layer,
            state=vera_source.active_state_index,
            thresholds=thresholds_to_apply,
        )
        sel_group = state[selected_group_key]
        sel.title = format_label(selected_src_id, selected_array, sel_group)
        nonlocal saved_sel
        saved_sel = sel

        results = core_slice.serialize_data_groups()
        results = pick_group(results, sel_group)
        num_groups = len(results)
        for idx in range(MAX_VIS_GROUPS):
            state[f"core_assemblies_{view_id}_{idx}"] = [] if idx >= num_groups else results[idx][0]
        state[n_groups_key] = num_groups
        state[assembly_size_key] = core_slice.assembly_side
        state[x_label_key] = core_slice.x_labels
        state[y_label_key] = core_slice.y_labels
        state[core_cols_key] = core_slice.max_columns
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
                for g in range(MAX_VIS_GROUPS):
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
                                    color_preset=("color_preset",),
                                    color_range=(f"color_range_{view_id}_{g}", [0, 1]),
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
                                    color_preset=("color_preset",),
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
                take_photo(
                    state=state,
                    saved_sel=lambda: saved_sel,
                    show_labels_key=show_labels_key,
                    decimals_key=decimals_key,
                    msg_key=msg_key,
                    msg_show_key=msg_show_key,
                    n_groups_key=n_groups_key,
                )
            notification(msg_key, msg_show_key)
