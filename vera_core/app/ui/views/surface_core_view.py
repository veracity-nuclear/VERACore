import numpy as np
from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.data.dtypes import MAX_NUM_GROUPS, VeraDtype
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry
from vera_core.data.renders import Selection, SurfaceView
from vera_core.widgets import vera

from ..helpers import format_label, get_safe_idxs, is_non_active_view, set_info
from .save_image import notification, register_photo_state, take_photo

# Lateral faces are the first four of [W, N, E, S, T, B]
LATERAL_FACE_SLICE = slice(0, 4)


def option_for(view_id):
    return {
        "name": f"core_surface_view_{view_id}",
        "label": "Core Surface View",
        "multi_picker": False,
        "icon": "mdi-vector-square",
        "allowed_categories": [
            VeraDtype.COMP_ASSY_SURFACE.title,
            VeraDtype.COMP_NODAL_SURFACE.title,
        ],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"

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

    saved_sel: Selection | None = None

    def _build_cells(radial_adf, core_map, n_nodes):
        """Lay out ADF into value[j][i] = list-of-nodes, each node = [w,n,e,s]..
        For assembly ADF n_nodes == 1; for nodal n_nodes == 4.
        """
        core_width = core_map.shape[0]
        grid = []
        for row in range(core_width):
            line = []
            grid.append(line)
            for col in range(core_width):
                assembly_idx = int(core_map[row, col]) - 1
                if assembly_idx < 0:
                    line.append([])  # empty position
                    continue
                # Each node -> [w, n, e, s] as python floats.
                nodes = []
                for node in range(n_nodes):
                    faces = radial_adf[:, node, assembly_idx]
                    nodes.append([float(v) for v in faces])
                line.append(nodes)
        return grid

    @state.change(
        selected_array_key,
        selected_src_key,
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
        _, _, selected_layer, _, selected_src_id, selected_array = indices
        vera_source: VeraDataSource = registry.get(selected_src_id)
        core = vera_source.core
        state[aspect_ratio_key] = core.aspect_ratio

        array = vera_source.get_dataset(selected_array)
        array_dtype = array.dataset_type
        if array_dtype.title not in option_for(0)["allowed_categories"]:
            return

        is_comp = array_dtype.is_computational()
        core_map = core.comp_core_map if is_comp else core.reduced_core_map

        n_energy = array.shape[1]
        n_nodes = array.shape[2]

        sel = SurfaceView(vera_source).select(
            array, state=vera_source.active_state_index, z=selected_layer
        )
        sel.title = format_label(selected_src_id, selected_array)
        nonlocal saved_sel
        saved_sel = sel

        for g in range(n_energy):
            # (4_faces, n_nodes, nass) for this energy group + layer
            radial = np.asarray(array[LATERAL_FACE_SLICE, g, :, selected_layer, :])
            grid = _build_cells(radial, core_map, n_nodes)
            state[group_keys[g]] = grid

        for g in range(n_energy, MAX_NUM_GROUPS):
            state[group_keys[g]] = []

        state[x_label_key] = (
            core.comp_core_map_column_labels if is_comp else core.reduced_core_map_column_labels
        )
        start_idx = core.comp_map_start_index if is_comp else core.reduced_core_map_start_index
        state[y_label_key] = [start_idx + row + 1 for row in range(core_map.shape[0])]

        state[n_groups_key] = n_energy
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
