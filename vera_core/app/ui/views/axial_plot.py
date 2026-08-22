import numpy as np
import plotly.graph_objects as go
from trame.ui.html import DivLayout
from trame.widgets import plotly

from vera_core.data.analysis.axial_lines import AxialLines
from vera_core.data.dtypes import VeraDim
from vera_core.data.registry import VeraDataRegistry

from ..helpers import convert_ji_to_node, get_safe_idxs, is_non_active_view

SEP = "\x1f"

print(type(AxialLines.ALLOWED_DTYPES))


def option_for(view_id):
    return {
        "name": f"axial_plot_{view_id}",
        "label": "Axial Plot",
        "multi_picker": True,
        "icon": "mdi-align-horizontal-center",
        "allowed_categories": [vdtype.title for vdtype in AxialLines.ALLOWED_DTYPES],
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    selected_set_key = f"multi_selected_{view_id}"

    update_fn_name = f"update_axial_plot_{view_id}"

    def create_line():
        figure = go.Figure()
        vera_sources = []
        dataset_names = []
        all_indices = []
        for token in state[selected_set_key]:
            src_id, array_name = token.split(SEP, 1)
            src = registry.get(src_id)
            indices = get_safe_idxs(view_id, state, registry, src_id, array_name)
            if not indices:
                continue
            j, i, layer, assy, _, _ = indices
            vera_sources.append(src)
            dataset_names.append(array_name)
            all_indices.append(
                {
                    VeraDim.PIN_Y: j,
                    VeraDim.PIN_X: i,
                    VeraDim.NODE: convert_ji_to_node(j, i),
                    VeraDim.SURFACE: int(state.selected_surface),
                    VeraDim.AXIAL: layer,
                    VeraDim.ASSEMBLY: assy,
                }
            )

        axial_lines = AxialLines.create_axial_lines(vera_sources, dataset_names, all_indices)
        for (x, y), identifier, mode in axial_lines.lines():
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode=mode,
                    name=identifier,
                )
            )

        # add_hline only spans x in [0, 1], so draw the layer marker manually.
        float_info = np.finfo(np.float64)
        figure.add_trace(
            go.Scatter(
                x=[float_info.min, float_info.max],
                y=[registry.global_axial_mesh[state.selected_layer]] * 2,
                mode="lines",
                line=go.scatter.Line(color="red", dash="dash"),
                showlegend=False,
            )
        )

        figure.update_layout(
            margin=dict(t=0, b=0, l=0, r=0),
            template="plotly_dark" if state["dark_mode"] else "plotly",
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.1,
                xanchor="center",
                x=0.5,
            ),
        )
        return figure

    @state.change(
        selected_set_key,
        "selected_assembly_ij",
        "selected_layer",
        "selected_i",
        "selected_j",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
        "dark_mode",
        "selected_surface",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def on_cell_change(**kwargs):
        if is_non_active_view(state, view_id, option) or registry.default_src is None:
            return
        update_fn = getattr(ctrl, update_fn_name, None)
        if update_fn is not None:
            update_fn(create_line())

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; width: 100%;"
        style = "; ".join(
            [
                "width: 100%",
                "height: 100%",
                "user-select: none",
            ]
        )
        figure = plotly.Figure(
            display_logo=False,
            display_mode_bar=False,
            style=style,
        )
        setattr(ctrl, update_fn_name, figure.update)
