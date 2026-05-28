import numpy as np

from trame_server.core import Server
from vera_core.app.core import VeraDataRegistry

from .features import derive, threshold, diff
from .layout import build_layout
from .helpers import  format_label, get_next_y_from_layout, array_range
from .views import (
    assembly_view,
    axial_plot,
    core_view,
    empty,
    table_view,
    time_plot,
    x_axial_view,
    y_axial_view,
    volume_view,
)

DEFAULT_NB_ROWS = 8
NUM_VIEW_SLOTS = 10

VIEW_MODULES = [
    empty,
    core_view,
    assembly_view,
    axial_plot,
    table_view,
    time_plot,
    volume_view,
    x_axial_view,
    y_axial_view,
]

def initialize(server : Server, registry: VeraDataRegistry):
    state, ctrl = server.state, server.controller
    state.trame__title = "VERACore"

    state.setdefault("grid_item_dirty_key", 0)

    # FIXME: For our example, fix this to match VeraView.
    # Come up with a way to autogenerate it.
    state.color_range = (0.0273, 1.95)
    state.selected_layer = 24
    state.selected_assembly = 36
    state.selected_time = 0
    state.max_time = max(0, len(registry.default_source.states) - 1)
    state.file_tree = {fid: [{"text": k.replace("_", " ").title(), "value": k} for k in keys]
                            for fid, keys in registry.full_core_keys().items()}
    state.selected_array = "pin_powers"
    state.selected_file = registry.default 
    state.global_label = "Select a global dataset"
    # FIXME ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    diff.register_diff_state_ctrl(state, ctrl, registry)
    threshold.register_threshold_state_ctrl(state, ctrl, registry)
    derive.register_derived_state_ctrl(state, ctrl, registry)

    def _recompute_card_range(view_id):
        selected_array = state[f"selected_array_{view_id}"]
        selected_file = state[f"selected_file_{view_id}"]
        array = registry.get(selected_file).array(selected_array)
        state[f"color_range_{view_id}"] = array_range(array)
       
    @state.change("selected_time")
    def selected_time_changed(selected_time, **kwargs):
        selected_time = int(selected_time)
        registry.change_active_state(selected_time)

        ctrl.on_vera_out_active_state_index_changed(
            selected_time=selected_time, **kwargs
        )
        # Automatically normalize color scale to current state
        for view_id in all_view_ids:
            _recompute_card_range(view_id)
        # Keep the global range in sync with the toolbar selection (for volume view).
        global_array = registry.get(state.selected_file).array(state.selected_array)
        state.color_range = array_range(global_array)
    
    @state.change("selected_array")
    def toolbar_array_changed(selected_array, **kwargs):
        # Global color_range still drives volume view.
        array = registry.get(state.selected_file).array(selected_array)
        state.color_range = array_range(array)

    # Keep selected_assembly and selected_assembly_ij in sync
    @state.change("selected_assembly_ij")
    def selected_assembly_ij_changed(selected_assembly_ij, **kwargs):
        i, j = selected_assembly_ij["i"], selected_assembly_ij["j"]
        state.selected_assembly = registry.default_source.core.reduced_core_map_assembly(i, j)

    @state.change("selected_assembly")
    def selected_assembly_changed(selected_assembly, **kwargs):
        i, j = registry.default_source.core.reduced_core_map_ij(selected_assembly)
        state.selected_assembly_ij = dict(i=i, j=j)
    
    @ctrl.set("card_selected_array_changed")
    def card_selected_array_changed(view_id, selected_array):
        if not selected_array:
            return
        state[f"selected_array_{view_id}"] = selected_array
    
    @ctrl.set("global_array_change")
    def global_array_change(selected_file: str, selected_array: str):
        if not selected_file or not selected_array:
            return
        print("in global array changed")
        state.selected_array = selected_array
        state.selected_file = selected_file
        for view_id in all_view_ids:
            state[f"selected_file_{view_id}"] = selected_file
            state[f"selected_array_{view_id}"] = selected_array
            state[f"selected_label_{view_id}"] = format_label(selected_file, selected_array)

    # Initialize all visualizations
    state.setdefault("grid_layout", [])

    # Reserve the various views
    all_view_ids = [f"{v+1}" for v in range(NUM_VIEW_SLOTS)]

    for view_id in all_view_ids:
        state[f"grid_options_{view_id}"] = []
        state[f"selected_array_{view_id}"] = "pin_powers"
        state[f"selected_file_{view_id}"] = registry.default
        
        state[f"color_range_{view_id}"] = (0.0, 1.0)  
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)

        state[f"selected_label_{view_id}"] = format_label(registry.default, 'pin_powers')
        for module in VIEW_MODULES:
            module.initialize(server, registry, view_id)

    @ctrl.set("grid_add_view")
    def add_view():
        next_view_id = available_view_ids.pop()
        next_y = get_next_y_from_layout(state.grid_layout)
        state.grid_layout.append(
            dict(x=0, w=12, h=DEFAULT_NB_ROWS, y=next_y, i=next_view_id)
        )
        state.dirty("grid_layout")

    @ctrl.set("grid_remove_view")
    def remove_view(view_id):
        available_view_ids.append(view_id)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state.grid_layout = list(
            filter(lambda item: item.get("i") != view_id, state.grid_layout)
        )

    @ctrl.set("pick_file_dataset")
    def pick_file_dataset(view_id, file, key):
        print("here", view_id, file, key)
        state[f"selected_file_{view_id}"] = file
        state[f"selected_array_{view_id}"] = key
        state[f"selected_label_{view_id}"] = f"{file} | {key.replace('_', ' ').upper()}"

    def _make_array_watcher(view_id):
        @state.change(f"selected_array_{view_id}")
        def _on_card_array_change(**kwargs):
            _recompute_card_range(view_id)
        return _on_card_array_change

    for view_id in all_view_ids:
        _make_array_watcher(view_id)
        _recompute_card_range(view_id)  # initial population

    available_view_ids = list(all_view_ids)

    def place(module, x, y, w, h):
        view_id = available_view_ids.pop(0)
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = module.option_for(view_id)

    place(x_axial_view,   0,  0, 3, 17)
    place(core_view,      6,  0, 3,  9)
    place(assembly_view,  9,  0, 3,  9)
    place(axial_plot,     6,  9, 3,  8)
    place(time_plot,      9,  9, 3,  8)
    place(volume_view,    3,  0, 3, 17)
    place(table_view,     0, 17, 6, 10)

    build_layout(server, state, ctrl, registry) # vue ui is built here
            