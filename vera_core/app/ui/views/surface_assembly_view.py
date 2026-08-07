import numpy as np
from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.data.dtypes import MAX_NUM_GROUPS, VeraDtype
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry
from vera_core.widgets import vera

from ..helpers import get_safe_idxs, is_non_active_view, set_info

# Lateral faces are the first four of [W, N, E, S, T, B]
LATERAL_FACE_SLICE = slice(0, 4)


def option_for(view_id):
    return {
        "name": f"assembly_surface_view_{view_id}",
        "label": "Assembly Surface View",
        "multi_picker": False,
        "icon": "mdi-arrow-expand-all",
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
    group_keys = [f"assy_surface_cells_{view_id}_{g}" for g in range(MAX_NUM_GROUPS)]
    decimals_key = f"assembly_decimals_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(n_groups_key, 0)
    state.setdefault(decimals_key, 2)
    for gk in group_keys:
        state.setdefault(gk, [])

    def _build_cells(radial_adf, n_nodes, side):
        """value[idx] = [w, n, e, s] per pin/node cell."""
        cells = []
        for node in range(n_nodes):
            faces = radial_adf[:, node]
            cells.append([float(v) for v in faces])
        return cells

    @state.change(
        selected_array_key,
        selected_src_key,
        "selected_assembly_ij",
        "selected_layer",
        f"grid_view_{view_id}",
        lock_flag,
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_assembly_surface_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        _, _, selected_layer, selected_assembly, selected_src_id, selected_array = indices
        vera_source: VeraDataSource = registry.get(selected_src_id)
        array = vera_source.get_dataset(selected_array)
        array_dtype = array.dataset_type
        if array_dtype.title not in option_for(0)["allowed_categories"]:
            return

        # ADF shape: (6_faces, n_energy, n_nodes, nax, nass)
        n_energy = array.shape[1]
        n_nodes = array.shape[2]
        side = int(round(np.sqrt(n_nodes)))  # 1 assembly, 2 nodal

        for g in range(n_energy):
            # (4_faces, n_nodes) for this assembly, group, layer
            radial = np.asarray(array[LATERAL_FACE_SLICE, g, :, selected_layer, selected_assembly])
            cells = _build_cells(radial, n_nodes, side)
            state[group_keys[g]] = cells

        for g in range(n_energy, MAX_NUM_GROUPS):
            state[group_keys[g]] = []

        state[n_groups_key] = n_energy
        set_info(view_id, state, registry)

    def on_surface_click(i, j, surface):
        """Committed click: set the selected pin and the global surface."""
        state.selected_i = i
        state.selected_j = j
        state.selected_surface = int(surface)

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
                        # Assembly surface view + its own colorbar, side by side.
                        with html.Div(
                            style=("flex: 1; min-height: 0;display: flex; flex-direction: row;")
                        ):
                            with html.Div(
                                style="flex: 1; min-width: 0; min-height: 0; position: relative;"
                            ):
                                vera.AssemblySurfaceView(
                                    value=(group_keys[g], []),
                                    selected_i=("selected_i", 7),
                                    selected_j=("selected_j", 7),
                                    selected_surface=("selected_surface", -1),
                                    color_preset="jet",
                                    color_range=(f"color_range_{view_id}_{g}", [0, 3]),
                                    click=(
                                        on_surface_click,
                                        "[$event.i, $event.j, $event.surface]",
                                    ),
                                    dark=("dark_mode",),
                                    busy=("trame__busy",),
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
            # Footer: caption with the decimals selector alongside.
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
                    " · Axial - {{ " + info + ".Layer }}"
                    " · Surface - {{ ['W','N','E','S'][selected_surface] || '-' }}",
                    classes="text-caption",
                )
                vuetify.VSelect(
                    v_model=decimals_key,
                    items=("[0,1,2,3,4]",),
                    label="Decimals",
                    dense=True,
                    hide_details=True,
                    style="flex: 0 0 auto; max-width: 90px;",
                )
