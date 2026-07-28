import numpy as np

from trame.ui.html import DivLayout
from trame.widgets import vuetify

from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype, NUM_NODES, Surface
from ..helpers import is_non_active_view, get_safe_idxs, convert_ji_to_node

def option_for(view_id):
    return {
        "name": f"table_view_{view_id}",
        "label": "Table View",
        "multi_picker" : False,
        "icon": "mdi-table",
        "allowed_categories": [dtype.title for dtype in VeraDtype if dtype != VeraDtype.UNKNOWN],
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
        "selected_assembly_ij",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
        "selected_surface"
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_table(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        (selected_j, selected_i, selected_layer, 
         selected_assembly, selected_src_id, selected_array) = indices

        src : VeraDataSource = registry.get(selected_src_id)
        array = src.array(selected_array)
        array_dtype = array.dataset_type
        indices_list = []
        match array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                indices_list.append((selected_j, selected_i, selected_layer, selected_assembly))
            case VeraDtype.POINT_DETECTOR:
                indices_list.append((selected_layer, selected_assembly))
            case VeraDtype.NODAL:
                indices_list.append((convert_ji_to_node(selected_j, selected_i), selected_layer, selected_assembly))
            case VeraDtype.RADIAL_NODE:
                indices_list.append((convert_ji_to_node(selected_j, selected_i), selected_assembly))
            case VeraDtype.COMP_ASSY | VeraDtype.ASSEMBLY:
                indices_list.append((0, selected_layer, selected_assembly))
            case VeraDtype.AXIAL:
                indices_list.append((selected_layer))
            case VeraDtype.RADIAL | VeraDtype.CHANNEL_RADIAL:
                indices_list.append((selected_j, selected_i, selected_assembly))
            case VeraDtype.RADIAL_ASSEMBLY | VeraDtype.RADIAL_POINT_DETECTOR:
                indices_list.append((selected_assembly))
            case VeraDtype.SCALAR:
                indices_list.append((0))
            case VeraDtype.COMP_NODAL:
                indices_list.append((convert_ji_to_node(selected_j, selected_i), selected_layer, selected_assembly))
            case VeraDtype.COMP_NODAL_ENERGY | VeraDtype.COMP_ASSY_ENERGY:
                num_energy_groups = array.shape[0]
                idx = convert_ji_to_node(selected_j, selected_i) if array_dtype == VeraDtype.COMP_NODAL_ENERGY else 0
                for group_n in range(num_energy_groups):
                    indices_list.append((group_n, idx, selected_layer, selected_assembly))
            case VeraDtype.COMP_ASSY_SURFACE | VeraDtype.COMP_NODAL_SURFACE:
                selected_surface = state.selected_surface
                num_energy_groups = array.shape[1]
                nodal_idx = 0 if array_dtype == VeraDtype.COMP_ASSY_SURFACE else convert_ji_to_node(selected_j, selected_i)
                for group_n in range(num_energy_groups):
                    indices_list.append((selected_surface, group_n, nodal_idx, selected_layer, selected_assembly))
            case _:
                raise RuntimeError(f"Table view cannot visualize datasets of type {str(array_dtype)}")
        data_dict = {}
        base_label = selected_array.replace('_', ' ').title()
        for group_n, indices in enumerate(indices_list):
            group_label = "" if len(indices_list) <= 1 else f" GROUP {group_n + 1}"
            value = array[indices]
            data_dict[f"{base_label}{group_label}"] = value
        for scalar_dataset in src.active_state.scalar_datasets:
            data_dict[scalar_dataset.replace("_", " ").title()] = np.asarray(
                src.array(scalar_dataset)
            ).item()

        # Round floats so we don't display too many sig figs (7 matches veraview).
        sig_figs = 7
        data_dict = {
            k : (float(f"{v:0.{sig_figs}g}") if isinstance(v, float) else v) 
            for k, v in data_dict.items()
        }
        is_comp = array_dtype.is_computational()

        axial_value = src.core.get_axial_mesh_means(dataset_type=array_dtype)[selected_layer]
        assembly_label = src.core.reduced_core_map_label(selected_assembly, is_comp)
        pin_label = f"Pin ({selected_i + 1}, {selected_j + 1})"
        surface_label = f" {Surface(state.selected_surface).str}"
        node_label = f"Node {convert_ji_to_node(selected_j, selected_i) + 1}"
        is_node = array_dtype.is_nodal()
        is_surface = array_dtype.is_surface()
        label = pin_label if not is_node else (node_label + (surface_label if is_surface else ""))
        columns = [
            "Dataset",
            f"Assembly {assembly_label}; Axial {axial_value:0.6g} cm; {label}",
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