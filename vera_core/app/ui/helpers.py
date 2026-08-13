import functools

import numpy as np
from trame_server.core import State

from vera_core.data.dtypes import NUM_NODES, VeraDtype
from vera_core.data.model import VeraOutCore
from vera_core.data.registry import VeraDataRegistry


def format_label(file: str, key: str):
    return f"{key.replace('_', ' ').upper()} | {file}"


def get_next_y_from_layout(layout):
    next_y = 0
    for item in layout:
        y, h = item.get("y", 0), item.get("h", 1)
        if y + h > next_y:
            next_y = y + h
    return next_y


def is_view_locked(state, view_id):
    return bool(state[f"locked_{view_id}"])


def is_non_active_view(state: State, view_id: int, option: dict[str, str]) -> bool:
    return state[f"grid_view_{view_id}"]["name"] != option["name"] or is_view_locked(state, view_id)


def _layer_elevation(axial_mesh, layer):
    mesh = np.asarray(axial_mesh, dtype=float)
    if mesh.ndim == 2:
        mesh = mesh.mean(axis=1)
    if not 0 <= layer < mesh.shape[0]:
        return None
    return float(np.round(mesh[layer], 2))


def set_info(view_id: int, state: State, registry: VeraDataRegistry):
    if is_view_locked(state=state, view_id=view_id):
        return
    indices = get_safe_idxs(view_id, state, registry)
    if not indices:
        return
    j, i, layer, assy, src_id, ds_name = indices
    vera_source = registry.get(src_id)
    dtype = vera_source.get_dataset_dtype(ds_name)
    is_comp = dtype.is_computational()
    axial_mesh = vera_source.core.get_axial_mesh_means(dataset_type=dtype)
    exposure = vera_source.active_state.get("exposure", None)
    state[f"label_info_{view_id}"] = {
        "Exposure": np.round(exposure[0], decimals=3) if exposure is not None else "not recorded",
        "Assembly": vera_source.core.reduced_core_map_label(assy, is_comp),
        "Layer": _layer_elevation(axial_mesh, layer),
        "Pin_x": int(i),
        "Pin_y": int(j),
    }


def _get_assy_idx(ds_dtype: VeraDtype, state: State, src_core: VeraOutCore):
    is_comp = ds_dtype.is_computational()
    is_detector = ds_dtype.is_detector()
    i, j = state.selected_assembly_ij["i"], state.selected_assembly_ij["j"]
    assy = src_core.reduced_core_map_assembly(i, j, is_comp=is_comp, is_detector=is_detector)
    return assy


def get_safe_idxs(
    view_id: int,
    state: State,
    registry: VeraDataRegistry,
    sel_src_id: str | None = None,
    sel_dataset_name: str | None = None,
) -> tuple | None:
    """Returns (selected_j, selected_i, selected_layer, selected_assembly, src_id, dataset_name)"""
    dataset_name = state[f"selected_array_{view_id}"] if not sel_dataset_name else sel_dataset_name
    src_id = state[f"selected_src_id_{view_id}"] if not sel_src_id else sel_src_id
    vera_source = registry.get(src_id)
    if vera_source is None:
        return None
    core = vera_source.core
    vdtype = vera_source.get_dataset_dtype(dataset_name)
    if vdtype == VeraDtype.UNKNOWN:
        return None
    sel_assy = _get_assy_idx(vdtype, state, core)
    if sel_assy < 0:
        return None
    sel_j = int(state.selected_j)
    sel_i = int(state.selected_i)
    sel_layer = registry.global_axial_idx_to_src_idx(src_id, vdtype, int(state.selected_layer))

    core_shape = core.get_core_shape(dataset_type=vdtype)
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

    return (safe_j, safe_i, safe_layer, safe_assy_idx, src_id, dataset_name)


def convert_ji_to_node(selected_j, selected_i):
    return np.clip((selected_i + selected_j * int(NUM_NODES / 2)), 0, NUM_NODES - 1)


def has_src(registry: VeraDataRegistry, *args, **kwargs):
    return registry.default_src_id is not None


def requires_src(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not has_src(*args, **kwargs):
            return
        return func(*args, **kwargs)

    return wrapper


def default_dataset_name(names: dict) -> str | None:
    """Source's default dataset, preferring pin_powers, then the pin category,
    then the first available. names maps category_title -> dataset_name."""
    if not names:
        return None
    if "pin_powers" in names.values():
        return "pin_powers"
    pin = VeraDtype.PIN.title
    if pin in names:
        return names[pin]
    return next(iter(names.values()))
