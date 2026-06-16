import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import vuetify

from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype
from ..helpers import is_non_active_view, make_safe_index

def option_for(view_id):
    return {
        "name": f"table_view_{view_id}",
        "label": "Table View",
        "multi_picker" : False,
        "icon": "mdi-table",
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    headers_key = f"table_view_headers_{view_id}"
    rows_key = f"table_view_rows_{view_id}"
    state.setdefault(headers_key, [])
    state.setdefault(rows_key, [])

    @state.change(
        selected_array_key,
        selected_src_key,
        "selected_assembly",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_table(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        selected_src_id = state[selected_src_key]
        selected_array = state[selected_array_key]
        selected_assy = int(state.selected_assembly)
        selected_layer = int(state.selected_layer)
        selected_i = int(state.selected_i)
        selected_j = int(state.selected_j)

        src : VeraDataSource = registry.get(selected_src_id)
        array = src.array(selected_array)
        array_dtype = array.dataset_type
        selected_j, selected_i, selected_layer, selected_assembly = make_safe_index(selected_j, selected_i, selected_layer, selected_assy, array_dtype, src.core_shape)
        match array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                indices = (selected_j, selected_i, selected_layer, selected_assembly)
            case VeraDtype.ASSEMBLY:
                indices = (selected_layer, selected_assembly)
            case VeraDtype.AXIAL:
                indices = (selected_layer)
            case VeraDtype.RADIAL | VeraDtype.CHANNEL_RADIAL:
                indices = (selected_j, selected_i, selected_assembly)
            case VeraDtype.RADIAL_ASSEMBLY:
                indices = (selected_assembly)
            case VeraDtype.SCALAR:
                indices = (0)
            case _:
                raise RuntimeError(f"Table view cannot visualize datasets of type {str(array_dtype)}")
        value = array[indices]

        data_dict = {selected_array.replace("_", " ").title(): value}
        for scalar_dataset in src.active_state.scalar_datasets:
            data_dict[scalar_dataset.replace("_", " ").title()] = np.asarray(
                src.array(scalar_dataset)
            ).item()

        # Round floats so we don't display too many sig figs (7 matches veraview).
        sig_figs = 7
        data_dict = {
            k: float(f"{v:0.{sig_figs}g}")
            for k, v in data_dict.items()
            if isinstance(v, float)
        }

        axial_value = src.core.axial_mesh_means[selected_layer]
        assembly_label = src.core.reduced_core_map_label(selected_assembly)
        columns = [
            "Dataset",
            f"Assembly {assembly_label}; Axial {axial_value:0.6g} cm; Pin ({selected_i + 1}, {selected_j + 1})",
        ]

        headers = [{"text": x, "value": x} for x in columns]
        rows = [dict(zip(columns, entry)) for entry in data_dict.items()]

        state[headers_key] = headers
        state[rows_key] = rows

    table_options = {
        "headers": (headers_key, []),
        "items": (rows_key, []),
        "classes": "mt-n2",
        "disable_sort": True,
        "dense": True,
        "disable_pagination": True,
        "hide_default_footer": True,
    }

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; width: 100%;"
        vuetify.VDataTable(**table_options)