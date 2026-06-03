import functools
import numpy as np

from trame_server.core import Server
from trame.app.asynchronous import StateQueue
from vera_core.app.core import VeraDataRegistry

from .features import DeriveMenu, DiffMenu, ThresholdMenu, FileMenu, StreamMenu, DatasetPicker
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
MULTI_SEP = "\x1f"


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

    def has_src():
        return registry.default_src_id is not None

    state.setdefault("grid_item_dirty_key", 0)
    state.setdefault("grid_layout", [])
    state.setdefault("has_data", has_src())
    state.setdefault("selected_time", 0)
    state.setdefault("max_time", 0)
    state.setdefault("selected_assembly_ij", dict(i=0, j=0))

    # initialize UI state for each feature
    DatasetPicker.register_dataset_picker_state(state, registry) # the File and Stream menu relies on dataset picker state
    DiffMenu.register_diff_state_ctrl(state, ctrl, registry)
    ThresholdMenu.register_threshold_state_ctrl(state, ctrl, registry)
    DeriveMenu.register_derived_state_ctrl(state, ctrl, registry)
    FileMenu.register_file_menu_state_ctrl(state, ctrl, registry)
    StreamMenu.register_stream_menu_state_ctrl(state, ctrl, registry, state_queue)

    all_view_ids = [f"{v + 1}" for v in range(NUM_VIEW_SLOTS)]
    available_view_ids = list(all_view_ids)

    # wrapper for guarding against an empty registry
    def requires_src(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not has_src():
                return
            return func(*args, **kwargs)
        return wrapper

    @requires_src
    def _recompute_card_range(view_id):
        """Rescales the each views colorbar"""
        src = registry.get(state[f"selected_src_id_{view_id}"])
        if src is None:
            return
        array = src.array(state[f"selected_array_{view_id}"])
        state[f"color_range_{view_id}"] = array_range(array)

    @state.change("selected_time")
    @requires_src
    def selected_time_changed(selected_time, **kwargs):
        """Fans out selected vera state changes to each view"""
        selected_time = int(selected_time)
        registry.change_all_active_state(selected_time)
        ctrl.on_vera_out_active_state_index_changed(
            selected_time=selected_time, **kwargs
        )
        # Normalize color scale to the current state.
        for view_id in all_view_ids:
            _recompute_card_range(view_id)

    @state.change("selected_assembly_ij")
    @requires_src
    def selected_assembly_ij_changed(selected_assembly_ij, **kwargs):
        """Keep selected_assembly and selected_assembly_ij in sync."""
        i, j = selected_assembly_ij["i"], selected_assembly_ij["j"]
        new_assembly = registry.default_src.core.reduced_core_map_assembly(i, j)
        if state.selected_assembly != new_assembly:
            state.selected_assembly = new_assembly

    @state.change("selected_assembly")
    @requires_src
    def selected_assembly_changed(selected_assembly, **kwargs):
        """Keep selected_assembly and selected_assembly_ij in sync."""
        i, j = registry.default_src.core.reduced_core_map_ij(int(selected_assembly))
        new_ij = {"i": i, "j": j}
        if state.selected_assembly_ij != new_ij:
            state.selected_assembly_ij = new_ij

    # initilize state for each each view template. Each grid card owns a view_id, so the templates initiliazed with that specific view_id belong to the card
    for view_id in all_view_ids:
        state[f"grid_options_{view_id}"] = []
        state[f"selected_array_{view_id}"] = ""
        state[f"selected_src_id_{view_id}"] = registry.default_src_id
        state[f"color_range_{view_id}"] = (0.0, 1.0)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state[f"selected_label_{view_id}"] = format_label(registry.default_src_id, "pin_powers")
        state[f"multi_selected_{view_id}"] = []
        state[f"multi_label_{view_id}"] = "Select datasets"
        for module in VIEW_MODULES:
            module.initialize(server, registry, view_id)

    @ctrl.set("grid_add_view")
    def add_view():
        """add a new card view to the grid layout"""
        next_view_id = available_view_ids.pop()
        next_y = get_next_y_from_layout(state.grid_layout)
        state.grid_layout.append(
            dict(x=0, w=12, h=DEFAULT_NB_ROWS, y=next_y, i=next_view_id)
        )
        state.dirty("grid_layout")

    @ctrl.set("grid_remove_view")
    def remove_view(view_id):
        """remove card view from grid"""
        available_view_ids.append(view_id)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state.grid_layout = list(
            filter(lambda item: item.get("i") != view_id, state.grid_layout)
        )

    @ctrl.set("select_dataset")
    def select_dataset(view_id, src_id, array_name):
        """Changes the dataset being visualized by the card view with view_id
        Args:
            view_id: specifies which card view's state is being updated
            src_id: the identifier of the source in the registry the dataset is in (since multiple sources could have datasets with identical names)
            array_name: the name of the new dataset to be visualized
        """
        state[f"selected_src_id_{view_id}"] = src_id
        state[f"selected_array_{view_id}"] = array_name
        state[f"selected_label_{view_id}"] = format_label(src_id, array_name)
    
    @ctrl.set("toggle_multi_array")
    def toggle_multi_array(view_id, src_id, array_name):
        """Toggle a (source, array) pair in a multi-picker view's selection set.
        Currently axial_plot and time_plot only views that support visualizing multiple array/datasets at a time
        """
        selected_key = f"multi_selected_{view_id}"
        label_key = f"multi_label_{view_id}"
        token = f"{src_id}{MULTI_SEP}{array_name}" # note that this is a string so trame can easily serialize state
        current = list(state[selected_key])
        if token in current:
            current.remove(token)
        else:
            current.append(token)
        state[selected_key] = current
        state[label_key] = f"{len(current)} selected" if current else "Select datasets"

    def _make_array_watcher(view_id):
        """factory function for creating a state watcher that keeps the color bar in sync with the card view's visualized dataset"""
        @state.change(f"selected_array_{view_id}", f"selected_src_id_{view_id}")
        def _on_card_array_change(**kwargs):
            _recompute_card_range(view_id)
        return _on_card_array_change

    for view_id in all_view_ids:
        _make_array_watcher(view_id)

    def place(module, x, y, w, h):
        """helper function for intializing UI"""
        view_id = available_view_ids.pop(0)
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = module.option_for(view_id)

    activation_done = False

    def activate_src():
        """Run the data-dependent setup once, when the first src exists.

        Invoked immediately when a file is supplied at startup, or by the file
        menu the first time a file is opened from the UI.
        """
        nonlocal activation_done
        if activation_done or not has_src():
            return

        src = registry.default_src
        default_id = registry.default_src_id
        array = src.array("pin_powers")
        ny, nx, nz = array.shape[0], array.shape[1], array.shape[2]

        """
        state.selected_layer : state for tracking which axial_plane is selected 

        state.selected_i : state for tracking the x-index of the selected pin 
        state.selected_j : state for tracking the y-index of the selected pin
        
        state.selected_assembly : state for tracking id of the selected assembly
        state.selected_assembly_ij : state for tracking the row and col of the selected assembly in the core_map

        state.max_time : state for tracking the maximum state number of all sources in the registry
        state.selected_time : state for tracking the STATE_n being visualized, i.e. if state.selected_time == 2, STATE_0002 in the vera source is being visualized
        """
        state.selected_layer = nz // 2
        state.selected_i = (nx // 2) - (1 if nx // 2 >= 1 else 0) # not a center pin
        state.selected_j = (ny // 2) - (1 if ny // 2 >= 1 else 0)
        state.selected_assembly = _center_assembly(src.core.reduced_core_map)
        assembly_i, assembly_j = src.core.reduced_core_map_ij(state.selected_assembly)
        state.selected_assembly_ij = {"i": assembly_i, "j": assembly_j}
        state.max_time = registry.max_state
        state.selected_time = 0
        for view_id in all_view_ids:
            state[f"selected_src_id_{view_id}"] = default_id
            state[f"selected_array_{view_id}"] = "pin_powers"
            state[f"selected_label_{view_id}"] = format_label(default_id, "pin_powers")
            state[f"multi_selected_{view_id}"] = [f"{default_id}{MULTI_SEP}{"pin_powers"}"]
            state[f"multi_label_{view_id}"] = "1 Selected"
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
        for view_id in all_view_ids:
            if state[f"grid_view_{view_id}"]["name"] == volume_view.option_for(view_id)["name"]:
                getattr(ctrl, f"reset_volume_{view_id}_camera")()
        activation_done = True

    ctrl.activate_src = activate_src
   
    build_layout(server, state, ctrl, registry)  # vue ui is built here

    if has_src():
        # automatically run data dependent setup if source was provied from command line
        activate_src()