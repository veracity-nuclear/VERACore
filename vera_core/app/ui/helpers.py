import numpy as np

from trame_server.core import State
from vera_core.app.core import VeraDataSource, VeraDtype, VeraOutCore

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

def set_info(state : State, vera_source : VeraDataSource, view_id : int):
    state[f"label_info_{view_id}"] = {
            "Exposure": np.round(vera_source.active_state.exposure[0], decimals=3),
            "Assembly": vera_source.core.reduced_core_map_label(state.selected_assembly),
            "Layer": vera_source.core.axial_mesh_means[state.selected_layer],
            "Pin_x" : int(state.selected_i),
            "Pin_y" : int(state.selected_j),
        }

def get_assy_idx(ds_dtype : VeraDtype, state : State):
    is_comp = ds_dtype.is_computational()
    if is_comp and hasattr(state, "selected_comp_assembly"):
        return int(state["selected_comp_assembly"])
    elif not is_comp and hasattr(state, "selected_assembly"):
        return int(state["selected_assembly"])
    else:
        raise RuntimeError("Unable to determine which assembly idx to use")
    
def make_safe_index(selected_j, selected_i, selected_layer, selected_assy, dataset_type : VeraDtype, core : VeraOutCore):
    core_shape = core.comp_core_shape if core.has_comp_core() and dataset_type.is_computational() else core.core_shape
    if len(core_shape) != 4:
        raise ValueError("core_shape must have 4 dim: npy, npx, nax, nass")
    selected_j = int(selected_j)
    selected_i = int(selected_i)
    selected_layer = int(selected_layer)
    selected_assy = int(selected_assy)
    npy, npx, nax, nass = core_shape
    ncy, ncx = npy + 1, npx + 1

    safe_y = ncy if dataset_type.is_channel() else npy
    safe_x = ncx if dataset_type.is_channel() else npx
    j = int(np.clip(selected_j, 0, safe_y - 1))
    i = int(np.clip(selected_i, 0, safe_x - 1))
    layer = int(np.clip(selected_layer, 0, nax - 1))
    assy_idx = int(np.clip(selected_assy, 0, nass - 1))
    return j, i, layer, assy_idx