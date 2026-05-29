import functools
import numpy as np

from trame_server.core import Server
from trame.app.asynchronous import StateQueue
from vera_core.app.core import VeraDataRegistry

from .features import DeriveMenu, DiffMenu, ThresholdMenu, FileMenu, StreamMenu
from .layout import build_layout
from .helpers import format_label, get_next_y_from_layout, array_range
from .views import (
    assembly_view,
    axial_plot,
    core_view,
    empty,
    table_view,
    time_plot,
    x_axial_view,
    # y_axial_view,
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
    # y_axial_view,
]


def _center_assembly(reduced_core_map):
    """Return the 0-based index of the loaded assembly nearest the core center."""
    positions = np.argwhere(reduced_core_map > 0)
    center = (np.array(reduced_core_map.shape) - 1) / 2.0
    nearest = positions[np.argmin(np.linalg.norm(positions - center, axis=1))]
    i, j = nearest
    return int(reduced_core_map[i, j]) - 1


def initialize(server: Server, registry: VeraDataRegistry, state_queue : StateQueue):
    state, ctrl = server.state, server.controller
    state.trame__title = "VERACore"

    # --- Neutral defaults (valid with or without a loaded file) ---
    state.setdefault("grid_item_dirty_key", 0)
    state.setdefault("grid_layout", [])
    state.setdefault("has_data", False)
    state.setdefault("global_label", "Select a global dataset")
    state.setdefault("color_range", (0.0, 1.0))
    state.setdefault("selected_time", 0)
    state.setdefault("max_time", 0)
    state.setdefault("selected_assembly_ij", dict(i=0, j=0))

    DiffMenu.register_diff_state_ctrl(state, ctrl, registry)
    ThresholdMenu.register_threshold_state_ctrl(state, ctrl, registry)
    DeriveMenu.register_derived_state_ctrl(state, ctrl, registry)
    FileMenu.register_file_menu_state_ctrl(state, ctrl, registry)
    StreamMenu.register_stream_menu_state_ctrl(state, ctrl, registry, state_queue)

    all_view_ids = [f"{v + 1}" for v in range(NUM_VIEW_SLOTS)]
    available_view_ids = list(all_view_ids)

    def has_source():
        return registry.default is not None
    
    def requires_source(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not has_source():
                return
            return func(*args, **kwargs)
        return wrapper

    @requires_source
    def _recompute_card_range(view_id):
        if state[f"selected_file_{view_id}"] != state.selected_ft_source:
            return
        source = registry.get(state[f"selected_file_{view_id}"])
        if source is None:
            return
        array = source.array(state[f"selected_array_{view_id}"])
        state[f"color_range_{view_id}"] = array_range(array)

    # --- Data-dependent watchers (no-op until a source is loaded) ---
    @state.change("selected_time", "selected_ft_source")
    @requires_source
    def selected_time_changed(selected_time, selected_ft_source, **kwargs):
        if selected_ft_source not in registry.source_ids():
            return
        selected_time = int(selected_time)
        registry.change_active_state(selected_ft_source, selected_time)
        ctrl.on_vera_out_active_state_index_changed(
            selected_time=selected_time, **kwargs
        )
        # Normalize color scale to the current state.
        for view_id in all_view_ids:
            _recompute_card_range(view_id)
        global_array = registry.get(selected_ft_source).array(state.selected_array)
        state.color_range = array_range(global_array)
    
    @state.change("selected_ft_source")
    def selected_ft_source_changed(selected_ft_source, **kwargs):
        if selected_ft_source not in registry.source_ids():
            return
        source = registry.get(selected_ft_source)
        state.max_time = max(0, len(source.states) - 1)
        print(state.max_time)

    @state.change("selected_array")
    @requires_source
    def toolbar_array_changed(selected_array, **kwargs):
        array = registry.get(state.selected_file).array(selected_array)
        state.color_range = array_range(array)

    # Keep selected_assembly and selected_assembly_ij in sync.
    @state.change("selected_assembly_ij")
    @requires_source
    def selected_assembly_ij_changed(selected_assembly_ij, **kwargs):
        i, j = selected_assembly_ij["i"], selected_assembly_ij["j"]
        state.selected_assembly = registry.default_source.core.reduced_core_map_assembly(i, j)

    @state.change("selected_assembly")
    @requires_source
    def selected_assembly_changed(selected_assembly, **kwargs):
        i, j = registry.default_source.core.reduced_core_map_ij(int(selected_assembly))
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
        state.selected_array = selected_array
        state.selected_file = selected_file
        for view_id in all_view_ids:
            state[f"selected_file_{view_id}"] = selected_file
            state[f"selected_array_{view_id}"] = selected_array
            state[f"selected_label_{view_id}"] = format_label(selected_file, selected_array)

    for view_id in all_view_ids:
        state[f"grid_options_{view_id}"] = []
        state[f"selected_array_{view_id}"] = "pin_powers"
        state[f"selected_file_{view_id}"] = registry.default
        state[f"color_range_{view_id}"] = (0.0, 1.0)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state[f"selected_label_{view_id}"] = format_label(registry.default, "pin_powers")
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
        state[f"selected_file_{view_id}"] = file
        state[f"selected_array_{view_id}"] = key
        state[f"selected_label_{view_id}"] = format_label(file, key)

    def _make_array_watcher(view_id):
        @state.change(f"selected_array_{view_id}")
        def _on_card_array_change(**kwargs):
            _recompute_card_range(view_id)
        return _on_card_array_change

    for view_id in all_view_ids:
        _make_array_watcher(view_id)

    def place(module, x, y, w, h):
        view_id = available_view_ids.pop(0)
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = module.option_for(view_id)

    activation = {"done": False}

    def activate_source():
        """Run the data-dependent setup once, when the first source exists.

        Invoked immediately when a file is supplied at startup, or by the file
        menu the first time a file is opened from the UI.
        """
        if activation["done"] or not has_source():
            return

        source = registry.default_source
        default_id = registry.default
        array = source.array("pin_powers")
        ny, nx, nz = array.shape[0], array.shape[1], array.shape[2]

        # Selections derived from the loaded file rather than hardcoded.
        state.selected_file = default_id
        state.selected_array = "pin_powers"
        state.selected_layer = nz // 2
        state.selected_ft_source = registry.default
        state.selected_i = (nx // 2) - (1 if nx // 2 >= 1 else 0) # not a center pin
        state.selected_j = (ny // 2) - (1 if ny // 2 >= 1 else 0)
        state.selected_assembly = _center_assembly(source.core.reduced_core_map)
        state.color_range = array_range(array)
        state.max_time = max(0, len(source.states) - 1)
        state.selected_time = 0

        for view_id in all_view_ids:
            state[f"selected_file_{view_id}"] = default_id
            state[f"selected_array_{view_id}"] = "pin_powers"
            state[f"selected_label_{view_id}"] = format_label(default_id, "pin_powers")
            _recompute_card_range(view_id)

        # Default arrangement of views.
        place(x_axial_view,   0,  0, 3, 17)
        place(core_view,      6,  0, 3,  9)
        place(assembly_view,  9,  0, 3,  9)
        place(axial_plot,     6,  9, 3,  8)
        place(time_plot,      9,  9, 3,  8)
        place(volume_view,    3,  0, 3, 17)
        place(table_view,     0, 17, 6, 10)
        state.dirty("grid_layout")

        state.has_data = True
        activation["done"] = True

    ctrl.activate_source = activate_source

    build_layout(server, state, ctrl, registry)  # vue ui is built here

    if has_source():
        activate_source()