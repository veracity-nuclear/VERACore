import numpy as np
import plotly.graph_objects as go
from trame.ui.html import DivLayout
from trame.widgets import html, plotly, vuetify

from vera_core.app.core import Surface, VeraDataRegistry, VeraDataset, VeraDtype

from ..helpers import convert_ji_to_node, get_safe_idxs, is_non_active_view

ALL_ASSEMBLIES, ASSEMBLY, PIN = "all_assemblies", "assembly", "pin"
LAYER, ALL = "layer", "all"

RADIAL_ITEMS = [
    {"text": "All assemblies", "value": ALL_ASSEMBLIES},
    {"text": "Selected assembly", "value": ASSEMBLY},
    {"text": "Selected pin", "value": PIN},
]

AXIAL_ITEMS = [
    {"text": "Selected layer", "value": LAYER},
    {"text": "All layers", "value": ALL},
]

PIN4, NODE3, FLAT2, AXIAL1 = "pin4", "node3", "flat2", "axial1"


def option_for(view_id):
    return {
        "name": f"histogram_{view_id}",
        "label": "Histogram",
        "icon": "mdi-chart-histogram",
        "allowed_categories": [
            VeraDtype.PIN.title,
            VeraDtype.CHANNEL.title,
            VeraDtype.AXIAL.title,
            VeraDtype.ASSEMBLY.title,
            VeraDtype.COMP_NODAL.title,
            VeraDtype.COMP_NODAL_ENERGY.title,
            VeraDtype.COMP_NODAL_SURFACE.title,
            VeraDtype.COMP_ASSY_SURFACE.title,
            VeraDtype.COMP_ASSY.title,
            VeraDtype.COMP_ASSY_ENERGY.title,
            VeraDtype.NODAL.title,
            VeraDtype.POINT_DETECTOR.title,
            VeraDtype.CONTINOUS_DETECTOR.title,
        ],
    }


def flat_finite_nonzero(values):
    flat = np.asarray(values).ravel()
    return flat[np.isfinite(flat) & (flat != 0.0)]


def group_arrays(full_array, dtype: VeraDtype, surface: int):
    match dtype:
        case VeraDtype.PIN | VeraDtype.CHANNEL:
            return [("", full_array, PIN4)]
        case VeraDtype.ASSEMBLY | VeraDtype.COMP_ASSY | VeraDtype.NODAL | VeraDtype.COMP_NODAL:
            return [("", full_array, NODE3)]
        case VeraDtype.POINT_DETECTOR | VeraDtype.CONTINOUS_DETECTOR:
            return [("", full_array, FLAT2)]
        case VeraDtype.AXIAL:
            return [("", full_array, AXIAL1)]
        case VeraDtype.COMP_ASSY_ENERGY | VeraDtype.COMP_NODAL_ENERGY:
            return [(f" GROUP {g + 1}", full_array[g], NODE3) for g in range(full_array.shape[0])]
        case VeraDtype.COMP_ASSY_SURFACE | VeraDtype.COMP_NODAL_SURFACE:
            return [
                (f" GROUP {g + 1}", full_array[surface, g], NODE3)
                for g in range(full_array.shape[1])
            ]
        case _:
            return []


def scoped(arr: VeraDataset, layout, radial, axial, j: int, i: int, assy: int, layer: int):
    z = slice(None) if axial == ALL else layer
    match layout:
        case "axial1":
            return arr[:]
        case "flat2":
            return arr[z, :] if radial == ALL_ASSEMBLIES else arr[z, assy]
        case "node3":
            if radial == PIN:
                node = 0 if arr.shape[0] == 1 else convert_ji_to_node(j, i)
                return arr[node, z, assy]
            return arr[:, z, :] if radial == ALL_ASSEMBLIES else arr[:, z, assy]
        case _:
            if radial == PIN:
                return arr[j, i, z, assy]
            return arr[:, :, z, :] if radial == ALL_ASSEMBLIES else arr[:, :, z, assy]


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]
    state.setdefault("histogram_radial_items", RADIAL_ITEMS)
    state.setdefault("histogram_axial_items", AXIAL_ITEMS)
    state.setdefault(f"hist_radial_{view_id}", ALL_ASSEMBLIES)
    state.setdefault(f"hist_axial_{view_id}", LAYER)

    update_fn_name = f"update_histogram_{view_id}"

    def create_histogram():
        figure = go.Figure()
        src_id = state[f"selected_src_id_{view_id}"]
        array_name = state[f"selected_array_{view_id}"]
        src = registry.get(src_id)
        if src is None or not array_name:
            return figure
        full_array = src.get_dataset(array_name)

        units = full_array.physical_units
        units_label = f" ({units})" if units != "unitless" else ""
        array_dtype: VeraDtype = full_array.dataset_type

        indices = get_safe_idxs(view_id, state, registry, src_id, array_name)
        if not indices:
            return figure
        j, i, layer, assy, _, _ = indices

        radial = state[f"hist_radial_{view_id}"]
        axial = state[f"hist_axial_{view_id}"]
        surface = state.selected_surface
        groups = group_arrays(full_array, array_dtype, surface)

        for suffix, arr, layout in groups:
            scoped_arr = scoped(arr, layout, radial, axial, j, i, assy, layer)
            binned = flat_finite_nonzero(scoped_arr)
            if binned.size < 2:
                continue
            figure.add_trace(
                go.Histogram(
                    x=binned,
                    histnorm="percent",
                    opacity=1.0 if len(groups) == 1 else 0.6,
                    name=f"{array_name.replace('_', ' ').title()}{suffix}",
                )
            )

        x_title = f"{array_name.replace('_', ' ').title()}{units_label}"
        if array_dtype != VeraDtype.AXIAL:
            assembly_label = src.core.reduced_core_map_label(assy, array_dtype.is_computational())
            region = {
                ALL_ASSEMBLIES: "All assemblies",
                ASSEMBLY: f"{assembly_label}",
                PIN: f"{assembly_label} @({i + 1},{j + 1})",
            }[radial]
            if axial == ALL:
                extent = "All Layers"
            else:
                mesh = src.core.get_axial_mesh_means(dataset_type=array_dtype)
                extent = f"{float(np.round(mesh[layer], decimals=2))} cm"
            surface_label = ""
            if array_dtype in (
                VeraDtype.COMP_ASSY_SURFACE,
                VeraDtype.COMP_NODAL_SURFACE,
            ):
                surface_label = f", {Surface(surface).str}"
            x_title = f"{x_title} - {region}, {extent}{surface_label}"

        if not figure.data:
            figure.add_annotation(
                text="Not enough samples for this scope",
                showarrow=False,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
            )

        figure.update_layout(
            barmode="overlay",
            margin=dict(t=10, b=0, l=0, r=0),
            template="plotly_dark" if state["dark_mode"] else "plotly",
            xaxis_title=x_title,
            yaxis_title="Frequency (%)",
            showlegend=len(figure.data) > 1,
            legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="center", x=0.5),
        )
        return figure

    @state.change(
        f"selected_array_{view_id}",
        f"selected_src_id_{view_id}",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
        f"hist_radial_{view_id}",
        f"hist_axial_{view_id}",
        "selected_layer",
        "selected_surface",
        "selected_assembly_ij",
        "selected_i",
        "selected_j",
        "dark_mode",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if is_non_active_view(state, view_id, option) or registry.default_src is None:
            return
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_histogram())

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "; ".join(
            [
                "height: 100%",
                "width: 100%",
                "display: flex",
                "flex-direction: column",
            ]
        )
        with html.Div(classes="d-flex", style="flex: 0 0 auto; gap: 8px;"):
            vuetify.VSelect(
                v_model=(f"hist_radial_{view_id}",),
                items=("histogram_radial_items",),
                dense=True,
                hide_details=True,
                classes="pt-0 mt-0 text-caption",
            )
            vuetify.VSelect(
                v_model=(f"hist_axial_{view_id}",),
                items=("histogram_axial_items",),
                dense=True,
                hide_details=True,
                classes="pt-0 mt-0 text-caption",
            )
        figure = plotly.Figure(
            display_logo=False,
            display_mode_bar=False,
            style="width: 100%; flex: 1 1 0; min-height: 0; user-select: none;",
        )
        setattr(ctrl, update_fn_name, figure.update)
