import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import html
from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype
from ..helpers import is_non_active_view, make_safe_index, set_info

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
        "allowed_categories": ["PIN", "CHANNEL", "ASSEMBLY"],
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
    info = f"label_info_{view_id}"

    pin_key = "selected_j" if is_x else "selected_i"

    state.setdefault(core_key, [])
    state.setdefault(size_x_key, [])
    state.setdefault(size_y_key, [])
    state.setdefault(label_y_key, [])
    if not is_x:
        state.setdefault(label_x_key, [])

    def axial_cell_selected(layer, clicked_idx):
        src = registry.get(state[selected_src_key])
        if is_x:
            assembly_i = clicked_idx
            assembly_j = state.selected_assembly_ij["j"]
        else:
            assembly_i = state.selected_assembly_ij["i"]
            assembly_j = clicked_idx
        state.selected_assembly = src.core.reduced_core_map_assembly(
            assembly_i, assembly_j
        )
        state.selected_layer = layer

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
        
        selected_assembly = int(state.selected_assembly)
        selected_pin = int(state[pin_key])

        vera_source: VeraDataSource = registry.get(state[selected_src_key])
        array = vera_source.array(selected_array)
        array_dtype = array.dataset_type

        if str(array_dtype).upper() not in option_for(0, "x")["allowed_categories"]:
            return

        if is_x:
            assembly_indices = vera_source.core.row_assembly_indices(selected_assembly)
        else:
            assembly_indices = vera_source.core.col_assembly_indices(selected_assembly)

        # Clamp indices against the core shape. Slot order is (j-pin, i-pin,
        # axial, assembly); each view clamps the pin slot it actually uses.
        if is_x:
            selected_pin, _, _, selected_assembly = make_safe_index(
                selected_pin, 0, 0, selected_assembly,
                array_dtype, vera_source.core_shape,
            )
        else:
            _, selected_pin, _, selected_assembly = make_safe_index(
                0, selected_pin, 0, selected_assembly,
                array_dtype, vera_source.core_shape,
            )

        match array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                assembly_size = array.shape[0]
                # Numpy puts the indexing subspace on the front.
                if is_x:
                    image_data = array[selected_pin, :, :, assembly_indices]
                else:
                    image_data = array[:, selected_pin, :, assembly_indices]
                image_data = np.vstack(image_data).T
            case VeraDtype.ASSEMBLY:
                assembly_size = 1
                image_data = array[:, assembly_indices]
                image_data = np.vstack(image_data)
            case _:
                raise RuntimeError(
                    f"Axial view cannot visualize datasets of type {str(array_dtype)}"
                )

        # Reverse y-axis since ax.invert_yaxis() doesn't apply here.
        image_data = image_data[::-1, :]
        nb_lines = image_data.shape[0]
        nb_cols = int(image_data.shape[1] / assembly_size)

        size_y = vera_source.core.axial_mesh_pixels.tolist()
        label_y = [i + 1 for i in range(len(size_y))]
        label_y.reverse()
        if array_dtype == VeraDtype.ASSEMBLY:
            assembly_size = vera_source.core.pin_volumes.shape[0]

        state[size_y_key] = size_y
        state[size_x_key] = [assembly_size for _ in range(nb_cols)]
        state[label_y_key] = label_y

        if not is_x:
            # Column labels are the core-row numbers for this source. Computed
            # here (not at init) so it is safe when no file is loaded yet.
            start_x = vera_source.core.reduced_core_map_start_index + 1
            stop_x = len(vera_source.core.core_map) + 1
            state[label_x_key] = list(range(start_x, stop_x))

        core = []
        for j in range(nb_lines):
            line = []
            core.append(line)
            for i in range(nb_cols):
                if array_dtype == VeraDtype.ASSEMBLY:
                    line.append(np.full((17,), fill_value=image_data[j, i]).tolist())
                else:
                    assembly = image_data[
                        j, slice(i * assembly_size, (i + 1) * assembly_size)
                    ]
                    line.append(np.ravel(assembly).tolist())
        state[core_key] = core
        set_info(state, vera_source, view_id)

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
                    y_labels=(label_y_key, []),
                    selected_i=(f"selected_assembly_ij.{'i' if is_x else 'j'}",),
                    selected_j=(f"{label_y_key}.length - selected_layer - 1",),
                    click=(
                        axial_cell_selected,
                        f"[{label_y_key}.length - $event.j - 1, $event.i]",
                    ),
                    x_scale=("3",),
                    y_scale=("3",),
                    busy=("trame__busy",),
                )
                if not is_x:
                    axial_kwargs["x_labels"] = (label_x_key, [])
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