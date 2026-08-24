from trame.ui.html import DivLayout
from trame.widgets import html, vuetify

from vera_core.data.analysis.assembly_slice import AssemblySlice
from vera_core.data.dtypes import MAX_NUM_GROUPS
from vera_core.data.model import VeraDataSource
from vera_core.data.registry import VeraDataRegistry

# from vera_core.data.renders import AssemblyView, Selection
from vera_core.widgets import vera

from ..helpers import get_safe_idxs, get_thresholds, is_non_active_view, set_info
from .save_image import register_photo_state


def option_for(view_id):
    return {
        "name": f"assembly_view_{view_id}",
        "label": "Assembly View",
        "multi_picker": False,
        "icon": "mdi-dots-grid",
        "allowed_categories": [dtype.title for dtype in AssemblySlice.ALLOWED_DTYPES],
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
    decimals_key = f"assembly_decimals_{view_id}"
    state.setdefault(decimals_key, 2)

    msg_key, msg_show_key = register_photo_state(state, view_id, option["name"])

    # saved_sel: Selection | None = None

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
        _, _, selected_layer, selected_assembly, selected_src_id, selected_array, time, _ = indices
        selected_time = state["selected_time"]
        vera_source: VeraDataSource = registry.get(selected_src_id)
        thres_hash = 0
        thresholds_to_apply = get_thresholds(state, view_id)
        for condition in thresholds_to_apply:
            thres_hash += hash(condition["op"]) + hash(condition["value"])

        images_dataset = None

        # Extract from cache if possible
        cache_key = (
            selected_time,
            selected_array,
            selected_assembly,
            selected_layer,
            thres_hash,
            selected_src_id,
        )
        if cache_key in cached_assembly_images:
            # Shortcut if we have a cache. We might still need to redraw
            # if the figure size was updated.
            images_dataset = cached_assembly_images[cache_key]

        # Extract data from H5 + add to cache
        if images_dataset is None:
            assembly_slice = AssemblySlice.create_assembly_slice(
                vera_source=vera_source,
                selected_array=selected_array,
                z=selected_layer,
                assembly_id=selected_assembly,
                state=time,
                thresholds_to_apply=thresholds_to_apply,
            )
            if not assembly_slice:
                return
            # sel = AssemblyView(vera_source).select(
            #     array,
            #     state=vera_source.active_state_index,
            #     assembly=selected_assembly,
            #     z=selected_layer,
            #     thresholds=thresholds_to_apply,
            # )
            # sel.title = format_label(selected_src_id, selected_array)
            # nonlocal saved_sel
            # saved_sel = sel

            # Only allow one image in the cache
            MAX_ITEMS_IN_CACHE = 1
            while len(cached_assembly_images) >= MAX_ITEMS_IN_CACHE:
                cached_assembly_images.pop(next(iter(cached_assembly_images)))
            images_dataset = assembly_slice.serialize_data_groups()
            cached_assembly_images[cache_key] = images_dataset

        # Update the client
        for idx, image in enumerate(images_dataset):
            state[f"assembly_array_{view_id}_{idx}"] = image
        num_groups = len(images_dataset)
        for idx in range(num_groups, MAX_NUM_GROUPS):
            state[f"assembly_array_{view_id}_{idx}"] = []
        state[n_groups_key] = num_groups
        set_info(view_id, state, registry)

    with DivLayout(server, template_name=option["name"]) as layout:
        layout.root.style = "height: 100%; display: flex; flex-direction: row;"
        with html.Div(style=("flex: 1; min-width: 0;display: flex; flex-direction: column;")):
            with html.Div(
                style=(
                    "flex: 1; min-height: 0;display: flex; flex-direction: row; flex-wrap: wrap;"
                )
            ):
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
                        with html.Div(
                            style=("flex: 1; min-height: 0;display: flex; flex-direction: row;")
                        ):
                            with html.Div(
                                style="flex: 1; min-width: 0; min-height: 0; position: relative;"
                            ):
                                vera.AssemblyView(
                                    value=(assembly_keys[g], []),
                                    selected_i=("selected_i", 7),
                                    selected_j=("selected_j", 7),
                                    color_preset="jet",
                                    color_range=(f"color_range_{view_id}_{g}", [0, 3]),
                                    click="setAll({ selected_i: $event.i, selected_j: $event.j})",
                                    dark=("dark_mode",),
                                    busy=("trame__busy",),
                                    decimals=(decimals_key, 2),
                                )
                            with html.Div(
                                style=(
                                    "flex: 0 0 auto; width: 70px; padding: 4px 0;"
                                    "display: flex; align-self: stretch;"
                                )
                            ):
                                vera.VerticalColorMapEditor(
                                    v_model=f"color_range_{view_id}_{g}",
                                    color_preset="jet",
                                    units=(f"color_units_{view_id}",),
                                )
            with html.Div(
                style=(
                    "flex: 0 0 auto; position: relative;"
                    "display: flex; align-items: center; justify-content: center;"
                    "min-height: 44px; padding: 6px 16px;"
                )
            ):
                html.Div(
                    "Exposure {{ " + info + ".Exposure }}"
                    " · ({{ " + info + ".Assembly }})"
                    " · Axial - {{ " + info + ".Layer }}",
                    classes="text-caption text-center text-truncate",
                    style="max-width: calc(100% - 120px);",
                )
                with html.Div(
                    style="position: absolute; right: 16px; top: 50%; transform: translateY(-50%);",
                ):
                    vuetify.VSelect(
                        v_model=decimals_key,
                        items=("[0,1,2,3,4]",),
                        label="Decimals",
                        dense=True,
                        hide_details=True,
                        style="max-width: 72px;",
                    )
                #     take_photo(
                #         state=state,
                #         saved_sel=lambda: saved_sel,
                #         show_labels_key=True,
                #         decimals_key=decimals_key,
                #         msg_key=msg_key,
                #         msg_show_key=msg_show_key,
                #         n_groups_key=n_groups_key,
                #     )
                # notification(msg_key, msg_show_key)
