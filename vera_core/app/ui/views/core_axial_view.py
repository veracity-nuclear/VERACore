import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDtype
from ..helpers import is_non_active_view, is_view_locked


def option_for(view_id):
    return {
        "name": f"core_axial_view_{view_id}",
        "label": "Core Axial View",
        "multi_picker": False,
        "icon": "mdi-chart-line-variant",
        "allowed_categories": [VeraDtype.ASSEMBLY.title],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    core_axials_key = f"core_axials_{view_id}"
    xrange_key = f"core_axial_xrange_{view_id}"
    aspect_ratio_key = f"aspect_ratio_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(core_axials_key, [])
    state.setdefault(xrange_key, [0.0, 1.0])
    state.setdefault(aspect_ratio_key, 1)

    @state.change(
        selected_array_key,
        selected_src_key,
        "selected_layer",
        f"grid_view_{view_id}",
        lock_flag,
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_core_axial(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        src = registry.get(state[selected_src_key])
        if src is None:
            return
        array = src.array(state[selected_array_key])
        if array.dataset_type != VeraDtype.ASSEMBLY:
            return

        rcm = src.core.reduced_core_map
        profiles = {}
        for j in range(rcm.shape[0]):
            for i in range(rcm.shape[1]):
                assy_id = int(rcm[j, i]) - 1
                if assy_id < 0:
                    continue
                profiles[(j, i)] = np.asarray(array[:, assy_id])

        if profiles:
            all_vals = np.concatenate(list(profiles.values()))
            x_min, x_max = float(np.nanmin(all_vals)), float(np.nanmax(all_vals))
        else:
            x_min, x_max = 0.0, 1.0

        cell_grid = []
        for j in range(rcm.shape[0]):
            row = []
            cell_grid.append(row)
            for i in range(rcm.shape[1]):
                prof = profiles.get((j, i))
                row.append(prof.tolist() if prof is not None else None)

        state[core_axials_key] = cell_grid
        state[xrange_key] = [x_min, x_max]
        state[aspect_ratio_key] = float(src.core.aspect_ratio)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                vera.CoreAxialView(
                    v_if=(f"{core_axials_key} && {core_axials_key}.length",),
                    value=(core_axials_key, []),
                    x_range=(xrange_key, [0.0, 1.0]),
                    selected_i=("selected_assembly_ij.i",),
                    selected_j=("selected_assembly_ij.j",),
                    aspect_ratio=(aspect_ratio_key, 1),
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