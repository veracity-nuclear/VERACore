import numpy as np
from trame.ui.html import DivLayout
from trame.widgets import html

from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry, VeraDataSource, VeraDtype, MAX_NUM_GROUPS
from vera_core.app.core.thresholds import apply_thresholds
from ..helpers import format_label, is_non_active_view, get_safe_idxs, set_info

def option_for(view_id):
    return {
    "name": f"assembly_view_{view_id}",
    "label": "Assembly View",
    "multi_picker" : False,
    "icon": "mdi-dots-grid",
    "allowed_categories": [VeraDtype.PIN.title, VeraDtype.CHANNEL.title, VeraDtype.RADIAL.title, VeraDtype.COMP_NODAL.title,
                           VeraDtype.COMP_NODAL_ENERGY.title]
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
    n_groups_key = f"n_groups_{view_id}"
    state.setdefault(n_groups_key, 0)
    assembly_keys = [f"assembly_array_{view_id}_{g}" for g in range(MAX_NUM_GROUPS)]
    for ak in assembly_keys:
        state.setdefault(ak, [])
    info = f"label_info_{view_id}"

    @state.change(
        "assembly_view_size",
        selected_array_key,
        selected_src_key,
        "selected_assembly_ij",
        "selected_layer",
        "thresholds",
        f"grid_view_{view_id}",
        f"locked_{view_id}",
    )
    @ctrl.add("on_vera_out_active_state_index_changed")
    def update_assembly_view(**kwargs):
        if is_non_active_view(state, view_id, option):
            return
        indices = get_safe_idxs(view_id, state, registry)
        if not indices:
            return
        _, _, selected_layer, selected_assembly, selected_src_id, selected_array = indices
        selected_time = state["selected_time"]
        vera_source : VeraDataSource = registry.get(selected_src_id)
        thres_key = format_label(selected_src_id, selected_array)
        thres = state["thresholds"]
        thres_hash = 0
        if thres.get(thres_key):
            for condition in thres[thres_key]:
                thres_hash += hash(condition["op"]) + hash(condition["value"])
        images_dataset = None
        
        # Extract from cache if possible
        cache_key = (selected_time, selected_array, selected_assembly, selected_layer, thres_hash, selected_src_id)
        if cache_key in cached_assembly_images:
            # Shortcut if we have a cache. We might still need to redraw
            # if the figure size was updated.
            images_dataset = cached_assembly_images[cache_key]

        # Extract data from H5 + add to cache
        if images_dataset is None:
            array = vera_source.array(selected_array)
            array_dtype : VeraDtype = array.dataset_type
            if str(array_dtype).upper() not in option_for(0)["allowed_categories"]:
                return
            match array_dtype:
                case VeraDtype.PIN | VeraDtype.CHANNEL:
                    images_dataset = [array[:, :, selected_layer, selected_assembly].copy()]
                case VeraDtype.RADIAL:
                    images_dataset = [array[:, :, selected_assembly].copy()]
                case VeraDtype.COMP_NODAL:
                    images_dataset = [array[:, selected_layer, selected_assembly]]
                case VeraDtype.COMP_NODAL_ENERGY:
                    num_energy_groups = np.shape(array)[0]
                    images_dataset = [array[energy_group, :, selected_layer, selected_assembly] for energy_group in range(num_energy_groups)]
                case _:
                    raise RuntimeError(f"Assembly View cannot visualize datasets of type {str(array_dtype)}")
            if thres.get(thres_key):
                for idx, image in enumerate(images_dataset):
                    images_dataset[idx] = apply_thresholds(image, thres[thres_key])
                
            if array_dtype in (VeraDtype.PIN, VeraDtype.RADIAL):
                control_rod_positions = vera_source.core.control_rod_positions
                # Make control rod positions equal to nan
                for image in images_dataset:
                    image[control_rod_positions] = np.nan

            # Only allow one image in the cache
            MAX_ITEMS_IN_CACHE = 1
            while len(cached_assembly_images) >= MAX_ITEMS_IN_CACHE:
                cached_assembly_images.pop(next(iter(cached_assembly_images)))

            cached_assembly_images[cache_key] = images_dataset
            set_info(view_id, state, registry)

        # Update the client
        for idx, image in enumerate(images_dataset):
            state[f"assembly_array_{view_id}_{idx}"] = np.ravel(image).tolist()
        num_groups = len(images_dataset)
        for idx in range(num_groups, MAX_NUM_GROUPS):
            state[f"assembly_array_{view_id}_{idx}"] = []
        state[n_groups_key] = num_groups

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=(
            "flex: 1; min-width: 0;"
            "display: flex; flex-direction: column;"
        )):
            with html.Div(style=(
                "flex: 1; min-height: 0;"
                "display: flex; flex-direction: row; flex-wrap: wrap;"
            )):
                for g in range(MAX_NUM_GROUPS):
                    with html.Div(
                        v_if=(f"{n_groups_key} > {g}",),
                        style=(
                            "flex: 1 1 45%; min-width: 0; min-height: 0;"
                            "display: flex; flex-direction: column; position: relative;"
                        ),
                    ):
                        html.Div(
                            f"Group {g + 1}",
                            v_if=(f"{n_groups_key} > 1",),
                            classes="text-caption text-center font-weight-medium",
                            style="flex: 0 0 auto;",
                        )
                        # Assembly view + its own colorbar, side by side.
                        with html.Div(style=(
                            "flex: 1; min-height: 0;"
                            "display: flex; flex-direction: row;"
                        )):
                            with html.Div(style="flex: 1; min-width: 0; min-height: 0; position: relative;"):
                                vera.AssemblyView(
                                    value=(assembly_keys[g], []),
                                    selected_i=("selected_i", 7),
                                    selected_j=("selected_j", 7),
                                    color_preset="jet",
                                    color_range=(f"color_range_{view_id}_{g}", [0, 3]),
                                    click="setAll({ selected_i: $event.i, selected_j: $event.j})",
                                    dark=("dark_mode",),
                                    busy=("trame__busy",),
                                )
                            with html.Div(style=(
                                "flex: 0 0 auto; width: 70px; padding: 4px 0;"
                                "display: flex; align-self: stretch;"
                            )):
                                vera.VerticalColorMapEditor(
                                    v_model=f"color_range_{view_id}_{g}",
                                    color_preset="jet",
                                )
            html.Div(
                "Exposure {{ " + info + ".Exposure }}"
                " · ({{ " + info + ".Assembly }})"
                " · Axial - {{ " + info + ".Layer }}",
                classes="text-caption text-center",
            )