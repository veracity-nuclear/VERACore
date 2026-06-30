import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import html
from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype
from ..helpers import is_non_active_view, get_safe_idxs, set_info, convert_ji_to_node

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
    core_key = f"{prefix}_{view_id}"
    size_x_key = f"{prefix}_size_x_{view_id}"
    size_y_key = f"{prefix}_size_y_{view_id}"
    label_x_key = f"{prefix}_label_x_{view_id}"
    label_y_key = f"{prefix}_label_y_{view_id}"
    selected_layer_key = f"selected_layer_{view_id}"
    info = f"label_info_{view_id}"

    pin_key = "selected_j" if is_x else "selected_i"

    state.setdefault(core_key, [])
    state.setdefault(size_x_key, [])
    state.setdefault(size_y_key, [])
    state.setdefault(label_y_key, [])
    state.setdefault(selected_layer_key, 0)
    state.setdefault(label_x_key, [])

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
        "selected_assembly",
        pin_key,
        f"grid_view_{view_id}",
        f"locked_{view_id}",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_axial_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return

        selected_array = state[selected_array_key]
        if is_x:
            selected_pin, _, _, selected_assembly, _, _ = get_safe_idxs(view_id, state, registry)
        else:
            _, selected_pin, _, selected_assembly, _, _ = get_safe_idxs(view_id, state, registry)

        vera_source: VeraDataSource = registry.get(state[selected_src_key])
        core = vera_source.core
        array = vera_source.array(selected_array)
        array_dtype: VeraDtype = array.dataset_type
        is_comp = array_dtype.is_computational()

        if str(array_dtype).upper() not in option["allowed_categories"]:
            return

        if is_x:
            assembly_indices = core.row_assembly_indices(selected_assembly, is_comp)
        else:
            assembly_indices = core.col_assembly_indices(selected_assembly, is_comp)

        # Build image_data as (axial_levels, columns). `cell_width` is how many
        # values span one assembly cell: pins for PIN/CHANNEL, 1 for ASSEMBLY,
        # 2 nodes for COMP_NODAL.
        is_assembly = array_dtype == VeraDtype.ASSEMBLY
        is_nodal = array_dtype == VeraDtype.COMP_NODAL

        match array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                cell_width = array.shape[0]
                if is_x:
                    image_data = array[selected_pin, :, :, assembly_indices]
                else:
                    image_data = array[:, selected_pin, :, assembly_indices]
                image_data = np.vstack(image_data).T

            case VeraDtype.ASSEMBLY:
                cell_width = core.pin_volumes.shape[0]  # full-width uniform cell
                image_data = np.vstack(array[:, assembly_indices])

            case VeraDtype.COMP_NODAL:
                nodal = array[:, :, assembly_indices]  # (nodes, nax, ncols)
                n_nodes = nodal.shape[0]

                if n_nodes == 1:
                    cell_width = core.pin_volumes.shape[0]
                    image_data = np.vstack(nodal[0])
                else:
                    if is_x:
                        node = int(convert_ji_to_node(selected_pin, 0))  # j picks the row
                        node_pair = (0, 1) if node in (0, 1) else (2, 3)
                    else:
                        node = int(convert_ji_to_node(0, selected_pin))  # i picks the col
                        node_pair = (0, 2) if node in (0, 2) else (1, 3)
                    cell_width = len(node_pair)
                    sel = nodal[list(node_pair)]
                    sel = np.transpose(sel, (1, 2, 0))
                    image_data = sel.reshape(sel.shape[0], -1)

            case _:
                raise RuntimeError(
                    f"Axial view cannot visualize datasets of type {str(array_dtype)}"
                )

        # Reverse axial axis so level 0 renders at the bottom.
        image_data = image_data[::-1, :]
        nb_lines = image_data.shape[0]
        if is_assembly:
            nb_cols = image_data.shape[1]
        else:
            nb_cols = image_data.shape[1] // cell_width

        # Axial sizing/labels follow the computational mesh for comp datasets.
        mesh_pixels = core.comp_axial_mesh_pixels if is_comp else core.axial_mesh_pixels
        mesh_means = core.comp_axial_mesh_means if is_comp else core.axial_mesh_means
        label_y = [np.round(m, 1) for m in mesh_means][::-1]

        state[size_y_key] = mesh_pixels.tolist()
        if is_nodal and n_nodes > 1:
            data_width = cell_width
            display_width = core.pin_volumes.shape[0]
        else:
            data_width = cell_width
            display_width = cell_width

        state[size_x_key] = [display_width for _ in range(nb_cols)]
        state[label_y_key] = label_y

        grid = []
        for j in range(nb_lines):
            row = []
            grid.append(row)
            for i in range(nb_cols):
                if is_assembly:
                    row.append(np.full(data_width, image_data[j, i]).tolist())
                else:
                    cell = image_data[j, i * data_width:(i + 1) * data_width]
                    row.append(np.ravel(cell).tolist())
        
        if is_x:
            state[label_x_key] = core.reduced_core_map_column_labels if not is_comp else core.comp_core_map_column_labels
        else:
            start_x = core.reduced_core_map_start_index + 1 if not is_comp else core.comp_map_start_index + 1
            stop_x = nb_cols + start_x + 1
            state[label_x_key] = list(range(start_x, stop_x))
        state[core_key] = grid
        set_info(view_id, state, registry)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                axial_kwargs = dict(
                    value=(core_key, []),
                    color_preset="jet",
                    color_range=(f"color_range_{view_id}", [0, 3]),
                    x_sizes=(size_x_key, []),
                    y_sizes=(size_y_key, []),
                    x_labels=(label_x_key, []),
                    y_labels=(label_y_key, []),
                    selected_i=(f"selected_assembly_ij.{'i' if is_x else 'j'}",),
                    selected_j=(f"{label_y_key}.length - {selected_layer_key} - 1",),
                    click=(
                        axial_cell_selected,
                        f"[{label_y_key}.length - $event.j - 1, $event.i]",
                    ),
                    x_scale=("3",),
                    y_scale=("3",),
                    busy=("trame__busy",),
                )
                vera.AxialView(**axial_kwargs)
            html.Div(
                "Exposure {{ " + info + ".Exposure }}"
                " · ({{ " + info + ".Assembly }})",
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