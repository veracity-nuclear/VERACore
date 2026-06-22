import numpy as np
from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype
from vera_core.app.core.thresholds import apply_thresholds
from ..helpers import format_label, is_non_active_view, make_safe_index, set_info


def option_for(view_id):
    return {
    "name": f"assembly_view_{view_id}",
    "label": "Assembly View",
    "multi_picker" : False,
    "icon": "mdi-dots-grid",
    "allowed_categories": ["PIN", "CHANNEL", "RADIAL"],
}


def initialize(server, registry: VeraDataRegistry, view_id):
    state, ctrl = server.state, server.controller

    # if OPTION not in state.grid_options:
    #     state.grid_options.append(OPTION)

    # A cache of assembly images.
    option = option_for(view_id)
    state[f"grid_options_{view_id}"] = state[f"grid_options_{view_id}"] + [option]
    cached_assembly_images = {}

    selected_array_key = f"selected_array_{view_id}"
    selected_src_key = f"selected_src_id_{view_id}"
    assembly_array = f"assembly_array_{view_id}"
    state.setdefault(assembly_array, [])
    info = f"label_info_{view_id}"


    @state.change(
        "assembly_view_size",
        selected_array_key,
        selected_src_key,
        "selected_assembly",
        "selected_layer",
        "color_range",
        "thresholds",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_assembly_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        selected_src_id = state[selected_src_key]
        selected_time = state["selected_time"]
        selected_layer = int(state["selected_layer"])
        selected_assembly = int(state["selected_assembly"])
        selected_array = state[selected_array_key]
        thres_key = format_label(selected_src_id, selected_array)

        thres = state["thresholds"]
        thres_hash = 0
        if thres.get(thres_key):
            for condition in thres[thres_key]:
                thres_hash += hash(condition["op"]) + hash(condition["value"])
        image_data = None
        

        # Extract from cache if possible
        cache_key = (selected_time, selected_array, selected_assembly, selected_layer, thres_hash, selected_src_id)
        if cache_key in cached_assembly_images:
            # Shortcut if we have a cache. We might still need to redraw
            # if the figure size was updated.
            image_data = cached_assembly_images[cache_key]

        # Extract data from H5 + add to cache
        if image_data is None:
            vera_source : VeraDataSource = registry.get(selected_src_id)
            array = vera_source.array(selected_array)
            array_dtype : VeraDtype = array.dataset_type
            if str(array_dtype).upper() not in option_for(0)["allowed_categories"]:
                return
            match array_dtype:
                case VeraDtype.PIN | VeraDtype.CHANNEL:
                    image_data = array[:, :, selected_layer, selected_assembly].copy()
                case VeraDtype.RADIAL:
                    image_data = array[:, :, selected_assembly].copy()
                case _:
                    raise RuntimeError(f"Assembly View cannot visualize datasets of type {str(array_dtype)}")
            if thres.get(thres_key):
                image_data = apply_thresholds(image_data, thres[thres_key])
                
            if array_dtype in (VeraDtype.PIN, VeraDtype.RADIAL):
                control_rod_positions = vera_source.core.control_rod_positions
                # Make control rod positions equal to nan
                image_data[control_rod_positions] = np.nan

            # Only allow one image in the cache
            MAX_ITEMS_IN_CACHE = 1
            while len(cached_assembly_images) >= MAX_ITEMS_IN_CACHE:
                cached_assembly_images.pop(next(iter(cached_assembly_images)))

            cached_assembly_images[cache_key] = image_data
            set_info(state, vera_source, view_id)

        # Update the client
        state[assembly_array] = np.ravel(image_data).tolist()

    # UI content
    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style="flex: 1; min-height: 0; position: relative;"):
                vera.AssemblyView(
                    value=(f"assembly_array_{view_id}", []),
                    selected_i=("selected_i", 7),
                    selected_j=("selected_j", 7),
                    color_preset="jet",
                    color_range=(f"color_range_{view_id}", [0, 3]),
                    click="setAll({ selected_i: $event.i, selected_j: $event.j})",
                    busy=("trame__busy",),
                )
            html.Div(
                "Exposure {{ " + info + ".Exposure }}"
                " · ({{ " + info + ".Assembly }})"
                " · Axial - {{ " + info + ".Layer }}",
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
