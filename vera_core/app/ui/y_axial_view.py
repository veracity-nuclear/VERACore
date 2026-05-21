import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import html
from vera_core.widgets import vera


def option_for(view_id):
    return {
        "name": f"y_axial_view_{view_id}",
        "label": "Y Axial View",
        "icon": "mdi-border-vertical",
    }


def initialize(server, vera_out_file, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    core_key = f"y_axial_core_{view_id}"
    size_x_key = f"y_axial_core_size_x_{view_id}"
    size_y_key = f"y_axial_core_size_y_{view_id}"
    label_x_key = f"y_axial_core_label_x_{view_id}"
    label_y_key = f"y_axial_core_label_y_{view_id}"

    # Convert these to 1-based indexing.
    start_x = vera_out_file.core.reduced_core_map_start_index + 1
    stop_x = len(vera_out_file.core.core_map) + 1

    state.setdefault(core_key, [])
    state.setdefault(size_x_key, [])
    state.setdefault(size_y_key, [])
    state.setdefault(label_x_key, list(range(start_x, stop_x)))
    state.setdefault(label_y_key, [])

    def axial_cell_selected(layer, assembly_j):
        assembly_i = state.selected_assembly_ij["i"]
        state.selected_assembly = vera_out_file.core.reduced_core_map_assembly(
            assembly_i, assembly_j
        )
        state.selected_layer = layer

    @state.change(
        selected_array_key,
        "selected_assembly",
        "selected_i",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_axial_view(**kwargs):
        if state[f"grid_view_{view_id}"]["name"] != option["name"]:
            return
        selected_array = state[selected_array_key]
        selected_assembly = int(state.selected_assembly)
        selected_i = int(state.selected_i)

        col_assembly_indices = vera_out_file.core.col_assembly_indices(
            selected_assembly
        )
        array = vera_out_file.array(selected_array)
        assembly_size = array.shape[0]

        # Numpy puts the indexing subspace on the front.
        image_data = array[:, selected_i, :, col_assembly_indices]
        image_data = np.vstack(image_data).T

        # Reverse y-axis since ax.invert_yaxis() doesn't apply here.
        image_data = image_data[::-1, :]

        nb_lines = image_data.shape[0]
        nb_cols = int(image_data.shape[1] / assembly_size)

        size_y = vera_out_file.core.axial_mesh_pixels.tolist()
        label_y = [i + 1 for i in range(len(size_y))]
        label_y.reverse()

        state[size_y_key] = size_y
        state[size_x_key] = [assembly_size for _ in range(nb_cols)]
        state[label_y_key] = label_y

        core = []
        for j in range(nb_lines):
            line = []
            core.append(line)
            for i in range(nb_cols):
                assembly = image_data[
                    j, slice(i * assembly_size, (i + 1) * assembly_size)
                ]
                line.append(np.ravel(assembly).tolist())
        state[core_key] = core

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: column;"
        with html.Div(style="flex: 1; min-height: 0; position: relative;"):
            vera.AxialView(
                value=(core_key, []),
                color_preset="jet",
                color_range=(f"color_range_{view_id}", [0, 3]),
                x_sizes=(size_x_key, []),
                y_sizes=(size_y_key, []),
                x_labels=(label_x_key, list(range(start_x, stop_x))),
                y_labels=(label_y_key, []),
                selected_i=("selected_assembly_ij.j",),
                selected_j=(f"{label_y_key}.length - selected_layer - 1",),
                click=(
                    axial_cell_selected,
                    f"[{label_y_key}.length - $event.j - 1, $event.i]",
                ),
                x_scale=("3",),
                y_scale=("3",),
                busy=("trame__busy",),
            )
        vera.ColorMapEditor(
            v_model=f"color_range_{view_id}",
            color_preset="jet",
        )