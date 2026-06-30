import functools
import numpy as np

from trame_server.core import Server
from trame.app.asynchronous import StateQueue
from vera_core.app.core import VeraDataRegistry, VeraDtype

from .features import DeriveMenu, DiffMenu, ThresholdMenu, FileMenu, StreamMenu, DatasetPicker, LocateMenu
from .layout import build_layout
from .helpers import format_label, get_next_y_from_layout, array_range, is_view_locked
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
    core_axial_view,
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
    y_axial_view,
    core_axial_view
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
    state.setdefault("grid_rebuild_key", 0)
    state.setdefault("has_data", has_src())
    state.setdefault("selected_time", 0)
    state.setdefault("max_time", 0)
    state.setdefault("selected_layer", 0)
    state.setdefault("max_layer", 0)
    state.setdefault("selected_assembly_ij", dict(i=0, j=0))

    # initialize UI state for each feature
    DatasetPicker.register_dataset_picker_state(state, registry) # the File and Stream menu relies on dataset picker state
    DiffMenu.register_diff_state_ctrl(state, ctrl, registry)
    ThresholdMenu.register_threshold_state_ctrl(state, ctrl, registry)
    DeriveMenu.register_derived_state_ctrl(state, ctrl, registry)
    FileMenu.register_file_menu_state_ctrl(state, ctrl, registry)
    StreamMenu.register_stream_menu_state_ctrl(state, ctrl, registry, state_queue)
    LocateMenu.register_locate_state_ctrl(state, ctrl, registry)

    all_view_ids = [f"{v + 1}" for v in range(NUM_VIEW_SLOTS)]
    available_view_ids = list(all_view_ids)
    """
    A view_id is an id assigned to each card that namespaces its per-view state.
    """
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
            if not is_view_locked(state, view_id):
                _recompute_card_range(view_id)

    @state.change("selected_assembly_ij")
    @requires_src
    def selected_assembly_ij_changed(selected_assembly_ij, **kwargs):
        """Keep selected_assembly and selected_assembly_ij in sync."""
        i, j = selected_assembly_ij["i"], selected_assembly_ij["j"]
        core = registry.default_src.core
        new_assembly = core.reduced_core_map_assembly(i, j)
        if new_assembly >= 0 and state.selected_assembly != new_assembly:
            state.selected_assembly = new_assembly
        if not core.has_comp_core() or not hasattr(state, "selected_comp_assembly"):
            return
        new_comp_assembly = core.reduced_core_map_assembly(i, j, is_comp=True)
        if new_comp_assembly != state.selected_comp_assembly:
            state.selected_comp_assembly = new_comp_assembly
        

    @ctrl.set("sync_ij_with_core_assembly")
    @requires_src
    def sync_core_assembly(core_assembly_idx):
        core = registry.default_src.core
        i, j = core.reduced_core_map_ij(core_assembly_idx)
        state.selected_assembly_ij = {"i":i, "j":j}
        assert core.reduced_core_map_assembly(i,j) == core_assembly_idx
    
    @ctrl.set("sync_ij_with_comp_assembly")
    @requires_src
    def sync_comp_assembly(comp_assembly_idx):
        core = registry.default_src.core
        i, j = core.reduced_core_map_ij(comp_assembly_idx)
        state.selected_assembly_ij = {"i":i, "j":j}
        assert core.comp_core_map_assembly(i,j) == comp_assembly_idx

        
    @state.change("src_tree_meta")
    def refresh_max_state(**kwargs):
        max_state = registry.max_state
        if max_state > state["max_time"]:
            state["max_time"] = max_state

    # initialize state for each each view template. Each view_id has a full set of the templates initiliazed with that specific view_id
    for view_id in all_view_ids:
        #
        # Per view_id state
        # each card gets assigned a view_id, this defines which state the card uses

        # state.selected_src_id_{{view_id}} : state that stores the selected dataset source id (id in a registry) that 
        #     the view is reading its datasets from
        # state.selected_array_{{view_id}} : state that stores the selected dataset name that the view
        #     is visualizing
        # state.locked_{{view_id}} : state that stores whether the view responds/updates to global changes to 
        #     selected_i, selected_j, selected_layer, selected_assembly. If true the view is "locked" and
        #     will not change until unlocked.
        # state.label_info_{{view_id}} : state that stores the selected_i, selected_j, selected_layer, and selected_assembly
        #     and exposure of the data being visualized by the view
        # state.selected_label_{{view_id}} : state the stores a formatted label of the view's selected source and dataset
        # state.multi_selected_{{view_id}} : state the stores a list of source-datasets that are being visualized by the view,
        #     note that only view's that support visualizing multiple datasets at a time use this
        # state.multi_label_{{view_id}} : state the stores a string label of how many datasets are being visualized by the view,
        #     only used by view's that support visualizing multiple datasets.
        #
        state[f"grid_options_{view_id}"] = []
        state[f"selected_array_{view_id}"] = ""
        state[f"selected_src_id_{view_id}"] = registry.default_src_id
        state[f"color_range_{view_id}"] = (0.0, 1.0)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state[f"selected_label_{view_id}"] = format_label(registry.default_src_id, "pin_powers")
        state[f"locked_{view_id}"] = False
        state[f"label_info_{view_id}"] = {"Exposure": "-", "Assembly": "-", "Layer": "-", "Pin_x" : "-", "Pin_y" : "-"}
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
        state.grid_rebuild_key += 1

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
        @state.change(f"selected_array_{view_id}", f"selected_src_id_{view_id}", f"locked_{view_id}")
        def _on_card_array_change(**kwargs):
            _recompute_card_range(view_id)
        return _on_card_array_change

    for view_id in all_view_ids:
        _make_array_watcher(view_id)

    def place(module, x, y, w, h, default_datasets_names : dict, default_id):
        """helper function for intializing UI"""

        if "allowed_categories" in module.option_for(0):
            module_allowed_categories = module.option_for(0)["allowed_categories"]
            available_categories = set(default_datasets_names).intersection(module_allowed_categories)
        else:
            available_categories = set(default_datasets_names)
        if not available_categories:
            return
        default_dataset_name = default_datasets_names.get(next(iter(available_categories)))
        if VeraDtype.PIN.title in available_categories:
            default_dataset_name = default_datasets_names[VeraDtype.PIN.title]

        view_id = available_view_ids.pop(0)
        state[f"selected_src_id_{view_id}"] = default_id
        state[f"selected_array_{view_id}"] = default_dataset_name
        state[f"selected_label_{view_id}"] = format_label(default_id, default_dataset_name)
        state[f"multi_selected_{view_id}"] = [f"{default_id}{MULTI_SEP}{default_dataset_name}"] # a seperator must be used instead of a tuple since the trame state needs to serializable
        state[f"multi_label_{view_id}"] = "1 Selected"
        _recompute_card_range(view_id)
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
        default_names = src.default_datasets()
        default_name = default_names[next(iter(default_names))]
        if VeraDtype.PIN.title in default_names:
            default_name = default_names[str(VeraDtype.PIN)]

        core_shape = src.core.core_shape
        ny, nx, nz = core_shape[0], core_shape[1], core_shape[2]

        # Global State
        # state.selected_layer : state for tracking which axial_plane is selected 
        # state.selected_i : state for tracking the x-index of the selected pin 
        # state.selected_j : state for tracking the y-index of the selected pin
        # state.selected_assembly : state for tracking id of the selected assembly
        # state.selected_assembly_ij : state for tracking the row and col of the selected assembly in the core_map
        # state.max_time : state for tracking the maximum state number of all sources in the registry
        # state.selected_time : state for tracking the STATE_n being visualized, i.e. if state.selected_time == 2, STATE_0002 in the vera source is being visualized
        state.selected_layer = nz // 2
        state.max_layer = len(registry.global_axial_mesh) - 1
        state.selected_i = (nx // 2) - (1 if nx // 2 >= 1 else 0) # not a center pin
        state.selected_j = (ny // 2) - (1 if ny // 2 >= 1 else 0)
        state.selected_assembly = _center_assembly(src.core.reduced_core_map)
        if src.core.has_comp_core():
            state.selected_comp_assembly = src.core.assy_to_comp_assy(state.selected_assembly)
            assert src.core.comp_assy_to_assy(state.selected_comp_assembly) == state.selected_assembly
        assembly_i, assembly_j = src.core.reduced_core_map_ij(state.selected_assembly)
        state.selected_assembly_ij = {"i": assembly_i, "j": assembly_j}
        state.max_time = registry.max_state
        state.selected_time = 0
        for view_id in all_view_ids:
            state[f"selected_src_id_{view_id}"] = default_id
            state[f"selected_array_{view_id}"] = default_name
            state[f"selected_label_{view_id}"] = format_label(default_id, default_name)
            state[f"multi_selected_{view_id}"] = [f"{default_id}{MULTI_SEP}{default_name}"] # a seperator must be used instead of a tuple since the trame state needs to serializable
            state[f"multi_label_{view_id}"] = "1 Selected"
            _recompute_card_range(view_id)
        # Default arrangement of views.
        place(x_axial_view,   0,  0, 3, 17, default_names, default_id)
        place(core_view,      6,  0, 4,  10, default_names, default_id)
        place(assembly_view,  9,  0, 3,  9, default_names, default_id)
        place(axial_plot,     6,  9, 3,  8, default_names, default_id)
        place(time_plot,      9,  9, 3,  8, default_names, default_id)
        place(volume_view,    3,  0, 3, 17, default_names, default_id)
        place(table_view,     0, 17, 6, 10, default_names, default_id)
        state.dirty("grid_layout")
        state.has_data = True
        # for view_id in all_view_ids:
        #     if state[f"grid_view_{view_id}"]["name"] == volume_view.option_for(view_id)["name"]:
        #         getattr(ctrl, f"reset_volume_{view_id}_camera")()
        activation_done = True

    ctrl.activate_src = activate_src
   
    build_layout(server, state, ctrl, registry)  # vue ui is built here

    if has_src():
        # automatically run data dependent setup if source was provied from command line
        activate_src()