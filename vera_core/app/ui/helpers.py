import numpy as np

from trame_server.core import State
from vera_core.app.core import VeraDataSource, VeraDtype, VeraOutCore, VeraDataRegistry, NUM_NODES

def format_label(file : str, key : str):
    return f"{file} | {key.replace('_', ' ').upper()}"

def array_range(array):
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo == hi:
        eps = max(abs(hi) * 1e-9, 1e-12)
        return (lo, hi + eps)
    return (lo, hi)

def get_next_y_from_layout(layout):
    next_y = 0
    for item in layout:
        y, h = item.get("y", 0), item.get("h", 1)
        if y + h > next_y:
            next_y = y + h
    return next_y

def is_view_locked(state, view_id):
    return bool(state[f"locked_{view_id}"])

def is_non_active_view(state : State, view_id : int, option : dict[str, str]) -> bool:
    return state[f"grid_view_{view_id}"]["name"] != option["name"] or is_view_locked(state, view_id)

def set_info(view_id : int, state : State, registry : VeraDataRegistry):
    j, i, layer, assy, src_id, ds_name = get_safe_idxs(view_id, state, registry)
    vera_source = registry.get(src_id)
    dtype = vera_source.array_dtype(ds_name)
    is_comp = dtype.is_computational()
    axial_mesh = vera_source.core.axial_mesh_means if not is_comp else vera_source.core.comp_axial_mesh_means
    state[f"label_info_{view_id}"] = {
            "Exposure": np.round(vera_source.active_state.exposure[0], decimals=3),
            "Assembly": vera_source.core.reduced_core_map_label(assy, is_comp),
            "Layer": np.round(axial_mesh[layer], decimals=2),
            "Pin_x" : int(i),
            "Pin_y" : int(j),
        }

def _get_assy_idx(ds_dtype : VeraDtype, state : State):
    is_comp = ds_dtype.is_computational()
    if is_comp and hasattr(state, "selected_comp_assembly"):
        return int(state["selected_comp_assembly"])
    elif not is_comp and hasattr(state, "selected_assembly"):
        return int(state["selected_assembly"])
    else:
        raise RuntimeError("Unable to determine which assembly idx to use")

def get_safe_idxs(view_id : int, state : State, registry : VeraDataRegistry, sel_src_id : str | None = None, sel_dataset_name : str | None = None):
    """Returns (selected_j, selected_i, selected_layer, selected_assembly, src_id, dataset_name)"""
    dataset_name = state[f"selected_array_{view_id}"] if not sel_dataset_name else sel_dataset_name
    src_id = state[f"selected_src_id_{view_id}"] if not sel_src_id else sel_src_id
    vera_source = registry.get(src_id)
    if vera_source is None:
        raise RuntimeError("Invalid source id")
    core = vera_source.core
    vdtype = vera_source.array_dtype(dataset_name)
    if vdtype == VeraDtype.UNKNOWN:
        raise RuntimeError("Unknown dtype of selected dataset")

    sel_j = int(state.selected_j)
    sel_i = int(state.selected_i)
    sel_assy = _get_assy_idx(vdtype, state)
    sel_layer = registry.global_axial_idx_to_src_idx(src_id, vdtype, int(state.selected_layer))

    core_shape = core.comp_core_shape if core.has_comp_core() and vdtype.is_computational() else core.core_shape
    if len(core_shape) != 4:
        raise ValueError("core_shape must have 4 dim: npy, npx, nax, nass")
    npy, npx, nax, nass = core_shape
    ncy, ncx = npy + 1, npx + 1

    safe_y = ncy if vdtype.is_channel() else npy
    safe_x = ncx if vdtype.is_channel() else npx
    
    safe_j = int(np.clip(sel_j, 0, safe_y - 1))
    safe_i = int(np.clip(sel_i, 0, safe_x - 1))
    safe_layer = int(np.clip(sel_layer, 0, nax - 1))
    safe_assy_idx = int(np.clip(sel_assy, 0, nass - 1))

    return safe_j, safe_i, safe_layer, safe_assy_idx, src_id, dataset_name

def convert_ji_to_node(selected_j, selected_i):
    return np.clip((selected_i + selected_j * int(NUM_NODES / 2)), 0, NUM_NODES - 1)