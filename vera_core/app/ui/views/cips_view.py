"""CIPS view: up to three assembly-scalar metrics stacked inside one core cell."""

from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDtype, VeraDataSource, MAX_NUM_GROUPS
from vera_core.app.core.thresholds import apply_thresholds
from ..helpers import format_label, is_non_active_view, set_info, get_safe_idxs, array_range
from ._core_grid import core_labels, format_for_vis, nan_out_non_fuel_locs


MULTI_SEP = "\x1f"

MAX_METRICS = min(3, MAX_NUM_GROUPS)

ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.RADIAL_ASSEMBLY,
    VeraDtype.ASSEMBLY,
]

CIPS_ROLES = [
    ("Crud Mass", ("crud_mass",)),
    ("Boron Mass", ("boron_mass", "boron_deposit")),
    ("Crud Thickness", ("crud_thick", "thickness")),
]

def option_for(view_id):
    return {
        "name": f"cips_view_{view_id}",
        "label": "CIPS View",
        "multi_picker": True,
        "owns_color_bar": True,
        "icon": "mdi-layers-triple",
        "allowed_categories": [dtype.title for dtype in ALLOWED_DTYPES],
    }


def _role_rank(array_name: str) -> int:
    low = array_name.lower()
    for rank, (_, fragments) in enumerate(CIPS_ROLES):
        if any(fragment in low for fragment in fragments):
            return rank
    return len(CIPS_ROLES)


def _role_label(array_name: str) -> str:
    rank = _role_rank(array_name)
    if rank < len(CIPS_ROLES):
        return CIPS_ROLES[rank][0]
    return array_name.replace("_", " ").title()


def _ordered_pairs(tokens):
    """(src_id, array_name) pairs in role order, capped at MAX_METRICS.

    sort is stable, so unrecognized names keep their selection order and land
    after the known roles.
    """
    pairs = [tuple(token.split(MULTI_SEP, 1)) for token in tokens if MULTI_SEP in token]
    pairs.sort(key=lambda pair: _role_rank(pair[1]))
    return pairs[:MAX_METRICS]


def _metric_values(vera_source: VeraDataSource, array_name: str, z: int, thresholds):
    """One metric as a core-shaped grid of floats/None, plus its value range."""
    dataset = vera_source.array(array_name)
    ds_dtype = dataset.dataset_type
    match ds_dtype:
        case VeraDtype.RADIAL_ASSEMBLY:
            layer = dataset
        case VeraDtype.ASSEMBLY:
            layer = dataset[0, z, :]
        case _:
            raise RuntimeError(f"CIPS View cannot visualize datasets of type {str(ds_dtype)}")
    if ds_dtype.has_fuel_pins():
        layer = nan_out_non_fuel_locs(layer, vera_source, z, ds_dtype == VeraDtype.RADIAL_ASSEMBLY)
    if thresholds:
        layer = apply_thresholds(layer, thresholds)
    _, values = format_for_vis(src=vera_source, dataset=layer)
    if not values:
        raise RuntimeError(f"{array_name} did not resolve to one value per assembly")
    return values, array_range(layer)


def create_cips_view(registry: VeraDataRegistry, tokens, z: int, thresholds_state: dict):
    """Build the full widget payload. Returns None when nothing is selectable."""
    if z < 0:
        raise RuntimeError(f"z must be >= 0, z = {z}")
    metrics = []
    ranges = []
    source = None
    for src_id, array_name in _ordered_pairs(tokens):
        vera_source = registry.get(src_id)
        if vera_source is None or vera_source.array_dtype(array_name) not in ALLOWED_DTYPES:
            continue
        thresholds = thresholds_state.get(format_label(src_id, array_name), [])
        values, value_range = _metric_values(vera_source, array_name, z, thresholds)
        metrics.append({
            "index": len(metrics),
            "label": _role_label(array_name),
            "name": array_name,
            "src_id": src_id,
            "units": vera_source.array_units(array_name),
            "values": values,
        })
        ranges.append(value_range)
        source = source or vera_source
    if source is None:
        return None
    x_labels, y_labels, max_core_cols = core_labels(source.core, is_comp=False)
    return {
        "metrics": metrics,
        "ranges": ranges,
        "x_labels": x_labels,
        "y_labels": y_labels,
        "core_cols": max_core_cols,
        "aspect_ratio": source.core.aspect_ratio,
    }


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]

    multi_key = f"multi_selected_{view_id}"
    metrics_key = f"cips_metrics_{view_id}"
    n_metrics_key = f"cips_n_metrics_{view_id}"
    mode_key = f"cips_color_mode_{view_id}"
    primary_key = f"cips_primary_{view_id}"
    decimals_key = f"cips_decimals_{view_id}"
    x_label_key = f"core_view_x_labels_{view_id}"
    y_label_key = f"core_view_y_labels_{view_id}"
    core_cols_key = f"core_cols_{view_id}"
    aspect_ratio_key = f"aspect_ratio_{view_id}"
    lock_flag = f"locked_{view_id}"
    info = f"label_info_{view_id}"
    range_keys = [f"color_range_{view_id}_{g}" for g in range(MAX_METRICS)]
    units_keys = [f"color_units_{view_id}_{g}" for g in range(MAX_METRICS)]

    state.setdefault(metrics_key, [])
    state.setdefault(n_metrics_key, 0)
    state.setdefault(mode_key, "primary")
    state.setdefault(primary_key, 0)
    state.setdefault(decimals_key, 2)
    state.setdefault(x_label_key, [])
    state.setdefault(y_label_key, [])
    state.setdefault(core_cols_key, 1)
    state.setdefault(aspect_ratio_key, 1)
    for key in units_keys:
        state.setdefault(key, "unitless")

    last_names = None

    def _rescale_allowed(names):
        """Rescale on a new metric set, or on any update while unlocked."""
        nonlocal last_names
        changed = names != last_names
        last_names = names
        return changed or not state[lock_flag]

    @state.change("selected_assembly_ij")
    def update_info(**kwargs):
        set_info(view_id, state, registry)

    @state.change(multi_key, "selected_layer", "thresholds", f"grid_view_{view_id}", lock_flag)
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_cips_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        _, _, selected_layer, _, _, _ = indices
        payload = create_cips_view(registry, state[multi_key], selected_layer, state["thresholds"])
        if payload is None:
            return
        metrics = payload["metrics"]
        names = [metric["name"] for metric in metrics]
        if _rescale_allowed(names):
            for g, value_range in enumerate(payload["ranges"]):
                state[range_keys[g]] = value_range
        for g in range(MAX_METRICS):
            state[units_keys[g]] = metrics[g]["units"] if g < len(metrics) else "unitless"
        state[metrics_key] = metrics
        state[n_metrics_key] = len(metrics)
        if state[primary_key] >= len(metrics):
            state[primary_key] = 0
        state[x_label_key] = payload["x_labels"]
        state[y_label_key] = payload["y_labels"]
        state[core_cols_key] = payload["core_cols"]
        state[aspect_ratio_key] = payload["aspect_ratio"]
        set_info(view_id, state, registry)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style=(
                "flex: 1; min-height: 0;"
                "display: flex; flex-direction: row;"
            )):
                with html.Div(style="flex: 1; min-width: 0; min-height: 0; position: relative;"):
                    vera.CipsCoreView(
                        metrics=(metrics_key, []),
                        color_ranges=("[" + ", ".join(range_keys) + "]",),
                        color_mode=(mode_key,),
                        primary_index=(primary_key,),
                        color_preset="jet",
                        selected_i=("selected_assembly_ij.i",),
                        selected_j=("selected_assembly_ij.j",),
                        x_labels=(x_label_key,),
                        y_labels=(y_label_key,),
                        core_cols=(core_cols_key,),
                        aspect_ratio=(aspect_ratio_key, 1),
                        decimals=(decimals_key, 2),
                        click="selected_assembly_ij = $event",
                        dark=("dark_mode",),
                        busy=("trame__busy",),
                    )
                with html.Div(
                    style=(
                        "flex: 0 0 auto; width: 76px; padding: 4px 0;"
                        "display: flex; flex-direction: column;"
                    ),
                ):
                    html.Div(
                        "{{ " + f"{metrics_key}[0].label" + " }}" if len(state[metrics_key]) > 0 else "",
                        classes="text-caption text-center font-weight-medium",
                        style="flex: 0 0 auto;",
                    )
                    with html.Div(style="flex: 1; min-height: 0; display: flex; align-self: stretch;"):
                        vera.VerticalColorMapEditor(
                            v_model=range_keys[0],
                            color_preset="jet",
                            units=(units_keys[0],),
                        )
            # Footer: centered caption, view controls pinned right.
            with html.Div(style=(
                "flex: 0 0 auto; position: relative;"
                "display: flex; align-items: center; justify-content: center;"
                "min-height: 44px; padding: 6px 16px;"
            )):
                html.Div(
                    "Exposure {{ " + info + ".Exposure }}"
                    " · ({{ " + info + ".Assembly }})"
                    " · Axial - {{ " + info + ".Layer }}",
                    classes="text-caption text-center",
                )
                with html.Div(
                    style=(
                        "display: flex; gap: 8px; align-items: center;"
                    ),
                ):
                    vuetify.VSelect(
                        v_model=mode_key,
                        items=("['primary', 'banded']",),
                        label="Color",
                        dense=True, hide_details=True,
                        style="max-width: 110px;",
                    )
                    vuetify.VSelect(
                        v_model=primary_key,
                        items=(metrics_key,),
                        item_text="label",
                        item_value="index",
                        label="Color by",
                        v_if=(f"{mode_key} === 'primary' && {n_metrics_key} > 1",),
                        dense=True, hide_details=True,
                        style="max-width: 140px;",
                    )
                    vuetify.VSelect(
                        v_model=decimals_key,
                        items=("[0,1,2,3,4]",),
                        label="Decimals",
                        dense=True, hide_details=True,
                        style="max-width: 90px;",
                    )