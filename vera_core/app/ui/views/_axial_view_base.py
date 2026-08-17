from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

# from vera_core.data.renders import AxialView, Selection
from vera_core.data.analysis.axial_slice import AxialSlice
from vera_core.data.dtypes import VeraDtype
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry
from vera_core.widgets import vera

from ..helpers import (
    format_label,
    get_safe_idxs,
    is_non_active_view,
    set_info,
)
from .save_image import register_photo_state

MAX_VIS_GROUPS = 4
FALLBACK_DISPLAY_SIZE = 17
X_SCALE = 3.0


_AXIS_OPTIONS = {
    "x": {
        "label": "X Axial View",
        "icon": "mdi-border-horizontal",
    },
    "y": {
        "label": "Y Axial View",
        "icon": "mdi-border-vertical",
    },
}


def option_for(view_id, axis):
    cfg = _AXIS_OPTIONS[axis]
    return {
        "name": f"{axis}_axial_view_{view_id}",
        "label": cfg["label"],
        "multi_picker": False,
        "icon": cfg["icon"],
        "allowed_categories": [
            VeraDtype.PIN.title,
            VeraDtype.CHANNEL.title,
            VeraDtype.ASSEMBLY.title,
            VeraDtype.COMP_NODAL.title,
            VeraDtype.COMP_NODAL_ENERGY.title,
            VeraDtype.COMP_ASSY.title,
            VeraDtype.COMP_ASSY_ENERGY.title,
            VeraDtype.NODAL.title,
        ],
    }


def build_axial_view(server, registry: VeraDataRegistry, view_id, axis):
    state, ctrl = server.state, server.controller
    is_x = axis == "x"

    option = option_for(view_id, axis)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"

    prefix = f"{axis}_axial_core"
    core_keys = [f"{prefix}_{view_id}_{g}" for g in range(MAX_VIS_GROUPS)]
    size_x_keys = [f"{prefix}_size_x_{view_id}_{g}" for g in range(MAX_VIS_GROUPS)]
    size_y_key = f"{prefix}_size_y_{view_id}"
    label_x_key = f"{prefix}_label_x_{view_id}"
    label_y_key = f"{prefix}_label_y_{view_id}"
    y_scale_key = f"{prefix}_y_scale_{view_id}"
    label_count_key = f"{prefix}_label_count_{view_id}"
    show_labels_key = f"{prefix}_show_labels_{view_id}"
    decimals_key = f"{prefix}_decimals_{view_id}"
    selected_layer_key = f"selected_layer_{view_id}"
    n_groups_key = f"n_groups_{view_id}"
    info = f"label_info_{view_id}"

    msg_key, msg_show_key = register_photo_state(state, view_id, option["name"])

    pin_key = "selected_j" if is_x else "selected_i"

    for ck in core_keys:
        state.setdefault(ck, [])
    for sk in size_x_keys:
        state.setdefault(sk, [])
    state.setdefault(size_y_key, [])
    state.setdefault(label_y_key, [])
    state.setdefault(label_x_key, [])
    state.setdefault(selected_layer_key, 0)
    state.setdefault(n_groups_key, 0)
    state.setdefault(y_scale_key, 3)
    state.setdefault(label_count_key, 0)
    state.setdefault(show_labels_key, False)
    state.setdefault(decimals_key, 2)

    # saved_sel: Selection | None = None

    def axial_cell_selected(layer, clicked_idx):
        if is_x:
            assembly_i = clicked_idx
            assembly_j = state.selected_assembly_ij["j"]
        else:
            assembly_i = state.selected_assembly_ij["i"]
            assembly_j = clicked_idx
        src_id = state[selected_src_key]
        array_dtype = registry.get_ds_dtype(src_id, state[selected_array_key])
        state.selected_layer = registry.src_axial_idx_to_global_idx(src_id, array_dtype, layer)
        state.selected_assembly_ij = {"i": assembly_i, "j": assembly_j}

    @state.change("selected_layer")
    def update_axial_selected_layer(**kwargs):
        src_id = state[selected_src_key]
        array_dtype = registry.get_ds_dtype(src_id, state[selected_array_key])
        state[selected_layer_key] = registry.global_axial_idx_to_src_idx(
            src_id, array_dtype, state.selected_layer
        )

    @state.change(
        selected_array_key,
        selected_src_key,
        "selected_assembly_ij",
        pin_key,
        f"grid_view_{view_id}",
        f"locked_{view_id}",
        "thresholds",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_axial_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return

        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        if is_x:
            selected_pin, _, _, selected_assembly, src_id, selected_array = indices
        else:
            _, selected_pin, _, selected_assembly, src_id, selected_array = indices

        vera_source: VeraDataSource = registry.get(state[selected_src_key])

        thres_key = format_label(src_id, selected_array)
        thres = state["thresholds"].get(thres_key)

        axial_slice = AxialSlice.create_axial_slice(
            vera_source=vera_source,
            selected_array=selected_array,
            pin=selected_pin,
            assembly_id=selected_assembly,
            dim="x" if is_x else "y",
            thresholds_to_apply=thres,
        )
        if not axial_slice:
            return
        # ax = AxialView(vera_source)
        # sel = ax.select(
        #     selected_array,
        #     state=vera_source.active_state_index,
        #     axis="x" if is_x else "y",
        #     assembly=selected_assembly,
        #     pin=selected_pin,
        #     thresholds=thres,
        # )
        # sel.title = format_label(src_id, selected_array)
        # nonlocal saved_sel
        # saved_sel = sel

        images = axial_slice.serialize_data_groups()
        x_sizes = axial_slice.x_size.tolist()
        for g, image in enumerate(images):
            state[core_keys[g]] = image
            state[size_x_keys[g]] = x_sizes

        for g in range(axial_slice.n_groups, MAX_VIS_GROUPS):
            state[core_keys[g]] = []
            state[size_x_keys[g]] = []

        state[label_count_key] = axial_slice.cell_width
        state[size_y_key] = axial_slice.y_size.tolist()
        state[label_y_key] = axial_slice.y_labels
        state[y_scale_key] = axial_slice.y_scale

        state[label_x_key] = axial_slice.x_labels

        state[n_groups_key] = axial_slice.n_groups
        set_info(view_id, state, registry)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=("flex: 1; min-width: 0;display: flex; flex-direction: column;")):
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
                        with html.Div(
                            style=("flex: 1; min-height: 0;display: flex; flex-direction: row;")
                        ):
                            with html.Div(
                                style="flex: 1; min-width: 0; min-height: 0; position: relative;"
                            ):
                                axial_kwargs = dict(
                                    value=(core_keys[g], []),
                                    color_preset="jet",
                                    color_range=(f"color_range_{view_id}_{g}", [0, 3]),
                                    x_sizes=(size_x_keys[g], []),
                                    y_sizes=(size_y_key, []),
                                    x_labels=(label_x_key, []),
                                    y_labels=(label_y_key, []),
                                    selected_i=(f"selected_assembly_ij.{'i' if is_x else 'j'}",),
                                    selected_j=(
                                        f"{label_y_key}.length - {selected_layer_key} - 1",
                                    ),
                                    click=(
                                        axial_cell_selected,
                                        f"[{label_y_key}.length - $event.j - 1, $event.i]",
                                    ),
                                    x_scale=(str(X_SCALE),),
                                    y_scale=(y_scale_key,),
                                    label_count=(label_count_key, 0),
                                    show_labels=(show_labels_key, False),
                                    decimals=(decimals_key, 2),
                                    busy=("trame__busy",),
                                    dark=("dark_mode",),
                                )
                                vera.AxialView(**axial_kwargs)
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
                    "Exposure {{ " + info + ".Exposure }} · ({{ " + info + ".Assembly }})",
                    classes="text-caption",
                )
                vuetify.VCheckbox(
                    v_if=(f"{label_count_key} > 0",),
                    v_model=show_labels_key,
                    label="Show values",
                    dense=True,
                    hide_details=True,
                    classes="ma-0 pa-0 text-caption",
                    style="flex: 0 0 auto;",
                )
                vuetify.VSelect(
                    v_if=(f"{show_labels_key} && {label_count_key} > 0",),
                    v_model=decimals_key,
                    items=("[0,1,2,3,4]",),
                    label="Decimals",
                    dense=True,
                    hide_details=True,
                    style="flex: 0 0 auto; max-width: 90px;",
                )
            #     take_photo(
            #         state=state,
            #         saved_sel=lambda: saved_sel,
            #         show_labels_key=show_labels_key,
            #         decimals_key=decimals_key,
            #         msg_key=msg_key,
            #         msg_show_key=msg_show_key,
            #         n_groups_key=n_groups_key,
            #     )
            # notification(msg_key, msg_show_key)
