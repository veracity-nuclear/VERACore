from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.data.analysis.surface_slice import SurfaceSlice
from vera_core.data.dtypes import MAX_NUM_GROUPS
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry

# from vera_core.data.renders import Selection, SurfaceView
from vera_core.widgets import vera

from ..helpers import get_safe_idxs, is_non_active_view, pick_group, set_info
from .save_image import register_photo_state


def option_for(view_id):
    return {
        "name": f"core_surface_view_{view_id}",
        "label": "Core Surface View",
        "multi_picker": False,
        "icon": "mdi-vector-square",
        "allowed_categories": [dtype.title for dtype in SurfaceSlice.ALLOWED_DTYPES],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    selected_group_key = f"selected_group_{view_id}"

    n_groups_key = f"n_groups_{view_id}"
    group_keys = [f"core_surface_cells_{view_id}_{g}" for g in range(MAX_NUM_GROUPS)]
    x_label_key = f"core_surface_x_labels_{view_id}"
    y_label_key = f"core_surface_y_labels_{view_id}"
    aspect_ratio_key = f"aspect_ratio_{view_id}"
    show_labels_key = f"surface_show_labels_{view_id}"
    decimals_key = f"surface_decimals_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(n_groups_key, 0)
    for gk in group_keys:
        state.setdefault(gk, [])
    state.setdefault(x_label_key, [])
    state.setdefault(y_label_key, [])
    state.setdefault(aspect_ratio_key, 1)
    state.setdefault(show_labels_key, False)
    state.setdefault(decimals_key, 2)

    msg_key, msg_show_key = register_photo_state(state, view_id, option["name"])

    # saved_sel: Selection | None = None

    @state.change(
        selected_array_key,
        selected_src_key,
        selected_group_key,
        "selected_layer",
        f"grid_view_{view_id}",
        lock_flag,
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_surface_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        _, _, selected_layer, _, selected_src_id, selected_array, time, _ = indices
        vera_source: VeraDataSource = registry.get(selected_src_id)
        core_surface_slice = SurfaceSlice.create_surface_slice(
            vera_source=vera_source,
            selected_array=selected_array,
            z=selected_layer,
            state=time,
        )
        if not core_surface_slice:
            return

        # sel = SurfaceView(vera_source).select(
        #     array, state=vera_source.active_state_index, z=selected_layer
        # )
        # sel.title = format_label(selected_src_id, selected_array)
        # nonlocal saved_sel
        # saved_sel = sel
        images = core_surface_slice.serialize_dataset_groups()
        sel_group = state[selected_group_key]
        images = pick_group(images, sel_group)
        for g, image in enumerate(images):
            if g >= MAX_NUM_GROUPS:
                break
            # (4_faces, n_nodes, nass) for this energy group + layer
            state[group_keys[g]] = image

        for g in range(len(images), MAX_NUM_GROUPS):
            state[group_keys[g]] = []

        state[x_label_key] = core_surface_slice.x_labels
        state[y_label_key] = core_surface_slice.y_labels
        state[n_groups_key] = len(images)
        set_info(view_id, state, registry)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=("flex: 1; min-width: 0;display: flex; flex-direction: column;")):
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
                        # Surface view + its own colorbar, side by side.
                        with html.Div(
                            style=("flex: 1; min-height: 0;display: flex; flex-direction: row;")
                        ):
                            with html.Div(
                                style="flex: 1; min-width: 0; min-height: 0; position: relative;"
                            ):
                                vera.SurfaceView(
                                    value=(group_keys[g], []),
                                    selected_i=("selected_assembly_ij.i",),
                                    selected_j=("selected_assembly_ij.j",),
                                    x_labels=(x_label_key, []),
                                    y_labels=(y_label_key, []),
                                    aspect_ratio=(aspect_ratio_key, 1),
                                    color_preset=("color_preset",),
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
                    v_model=show_labels_key,
                    label="Show values",
                    dense=True,
                    hide_details=True,
                    classes="ma-0 pa-0 text-caption",
                    style="flex: 0 0 auto;",
                )
                vuetify.VSelect(
                    v_model=decimals_key,
                    v_if=(show_labels_key,),
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
