import numpy as np
from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.data.dtypes import VeraDtype
from vera_core.data.registry import VeraDataRegistry
from vera_core.widgets import vera

from ..helpers import get_time, is_non_active_view

ALLOWED_DTYPES = [
    VeraDtype.ASSEMBLY,
    VeraDtype.POINT_DETECTOR,
    VeraDtype.CONTINOUS_DETECTOR,
]


def _wire(values):
    out = []
    for value in values:
        if value is None or not np.isfinite(value):
            out.append(None)
        else:
            out.append(float(value))
    return out


def _mesh_geometry(dtype, mesh):
    if dtype is VeraDtype.CONTINOUS_DETECTOR:
        y = []
        for low, high in mesh:
            y.extend([low, high, None])
        return {"kind": "segments", "y": _wire(y[:-1])}
    kind = "points" if dtype is VeraDtype.POINT_DETECTOR else "line"
    return {"kind": kind, "y": _wire(mesh)}


def _cell_x(dtype, values):
    """Per-cell x coordinates, index-aligned with the shared mesh y."""
    if dtype is VeraDtype.CONTINOUS_DETECTOR:
        x = []
        for value in values:
            x.extend([value, value, None])
        return _wire(x[:-1])
    return _wire(values)


def _series_mean(dtype, values, mesh):
    finite = np.isfinite(values)
    if not finite.any():
        return None

    if dtype is VeraDtype.CONTINOUS_DETECTOR:
        spans = (mesh[:, 1] - mesh[:, 0])[finite]
        if spans.sum() <= 0:
            return None
        return float(np.average(values[finite], weights=spans))

    return float(values[finite].mean())


def _profile_matrix(array):
    data = np.asarray(array, dtype=float)
    if data.ndim == 3 and data.shape[0] == 1:
        data = data[0]
    if data.ndim != 2:
        raise RuntimeError(
            f"Core axial view expects (n_axial, n_assembly) values, got shape {data.shape}"
        )
    return data


def _axial_mesh(src, dtype):
    mesh = np.asarray(src.core.get_axial_mesh_means(dataset_type=dtype), dtype=float)

    if dtype is VeraDtype.CONTINOUS_DETECTOR:
        if mesh.ndim != 2 or mesh.shape[1] < 2:
            raise RuntimeError(
                f"Continuous detector mesh must be (n_region, >=2), got shape {mesh.shape}"
            )
        return np.column_stack((mesh[:, 0], mesh[:, -1]))

    if mesh.ndim != 1:
        raise RuntimeError(f"{dtype.title} mesh must be 1d elevations, got shape {mesh.shape}")
    return mesh


def _finite_range(values, fallback=(0.0, 1.0)):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return list(fallback)
    return [float(finite.min()), float(finite.max())]


def option_for(view_id):
    return {
        "name": f"core_axial_view_{view_id}",
        "label": "Core Axial View",
        "multi_picker": False,
        "icon": "mdi-chart-line-variant",
        "owns_color_bar": True,
        "allowed_categories": [dtype.title for dtype in ALLOWED_DTYPES],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    core_axials_key = f"core_axials_{view_id}"
    xrange_key = f"core_axial_xrange_{view_id}"
    yrange_key = f"core_axial_yrange_{view_id}"
    xlabels_key = f"core_axial_xlabels_{view_id}"
    ylabels_key = f"core_axial_ylabels_{view_id}"
    mesh_key = f"core_axial_mesh_{view_id}"
    aspect_ratio_key = f"aspect_ratio_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"

    state.setdefault(core_axials_key, [])
    state.setdefault(xrange_key, [0.0, 1.0])
    state.setdefault(yrange_key, [0.0, 1.0])
    state.setdefault(mesh_key, {"kind": "line", "y": []})
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
        array = src.get_dataset(state[selected_array_key], state_idx=get_time(state, view_id))
        core = src.core
        dtype = array.dataset_type
        if dtype not in ALLOWED_DTYPES:
            return
        data = _profile_matrix(array)
        mesh = _axial_mesh(src, dtype)
        if data.shape[0] != mesh.shape[0]:
            raise RuntimeError(
                f"{dtype.title}: {data.shape[0]} values per assembly does not match "
                f"{mesh.shape[0]} mesh entries"
            )
        core_map = src.core.get_map(dataset_type=dtype)
        grid = []
        for j in range(core_map.shape[0]):
            row = []
            grid.append(row)
            for i in range(core_map.shape[1]):
                assembly = int(core_map[j, i]) - 1
                if assembly < 0:
                    row.append(None)
                    continue
                values = data[:, assembly]
                row.append(
                    {
                        "x": _cell_x(dtype, values),
                        "mean": _series_mean(dtype, values, mesh),
                    }
                )
        means = [c["mean"] for row in grid for c in row if c and c["mean"] is not None]
        state[f"color_range_{view_id}_0"] = [min(means), max(means)]
        is_comp = dtype.is_computational()
        state[xlabels_key] = (
            core.comp_core_map_column_labels if is_comp else core.reduced_core_map_column_labels
        )
        state[ylabels_key] = (
            core.comp_core_map_row_labels if is_comp else core.reduced_core_map_row_labels
        )
        state[mesh_key] = _mesh_geometry(dtype, mesh)
        state[core_axials_key] = grid
        state[xrange_key] = _finite_range(data)
        state[yrange_key] = _finite_range(mesh)
        state[aspect_ratio_key] = float(src.core.aspect_ratio)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style="flex: 1; min-width: 0; display: flex; flex-direction: column;"):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                vera.CoreAxialView(
                    v_if=(
                        f"{core_axials_key} && {core_axials_key}.length && {mesh_key} && {mesh_key}.y && {mesh_key}.y.length",
                    ),
                    value=(core_axials_key, []),
                    x_range=(xrange_key, [0.0, 1.0]),
                    y_range=(yrange_key, [0.0, 1.0]),
                    mesh=(mesh_key, {"kind": "line", "y": []}),
                    selected_i=("selected_assembly_ij.i",),
                    selected_j=("selected_assembly_ij.j",),
                    aspect_ratio=(aspect_ratio_key, 1),
                    color_preset="jet",
                    color_range=(f"color_range_{view_id}_0", [0.0, 1.0]),
                    dark=("$vuetify.theme.dark",),
                    x_labels=(xlabels_key,),
                    y_labels=(ylabels_key,),
                    click="selected_assembly_ij = $event",
                    busy=("trame__busy",),
                )
            html.Div(
                "Exposure {{ " + info + ".Exposure }} · ({{ " + info + ".Assembly }})",
                classes="text-caption text-center",
            )
        with html.Div(
            style="flex: 0 0 auto; width: 70px; padding: 4px 0; display: flex; align-self: stretch;"
        ):
            vera.VerticalColorMapEditor(
                v_model=f"color_range_{view_id}_0",
                color_preset="jet",
                units=(f"color_units_{view_id}",),
            )
