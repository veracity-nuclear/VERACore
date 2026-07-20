import functools, json
from pathlib import Path
from dataclasses import asdict
import numpy as np

from trame_server.core import Server
from trame.app.asynchronous import StateQueue
from vera_core.app.core import (VeraDataRegistry, VeraDtype, 
                                VeraOutFile, MAX_NUM_GROUPS, 
                                LATERAL_SURACES, Session, 
                                ViewSession, recipe_sources)

from .features import DeriveMenu, DiffMenu, ThresholdMenu, FileMenu, StreamMenu, DatasetPicker, LocateMenu, SaveSession
from .layout import build_layout
from .helpers import format_label, get_next_y_from_layout, array_range, is_view_locked, default_dataset_name
from .views import (
    surface_core_view,
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
    surface_assembly_view
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
    core_axial_view,
    surface_core_view,
    surface_assembly_view,
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

    state.setdefault("grid_item_dirty_key", 0)
    state.setdefault("grid_layout", [])
    state.setdefault("grid_rebuild_key", 0)
    state.setdefault("has_data", registry.has_src())
    state.setdefault("selected_time", 0)
    state.setdefault("max_time", 0)
    state.setdefault("selected_layer", 0)
    state.setdefault("max_layer", 0)
    state.setdefault("selected_assembly_ij", dict(i=0, j=0))
    state.setdefault("selected_surface", 0)
    state.setdefault("dark_mode", True)
    state.setdefault("recipes", [])

    all_view_ids = [f"{v + 1}" for v in range(NUM_VIEW_SLOTS)]

    # initialize UI state for each feature
    DatasetPicker.register_dataset_picker_state(state, registry) # the File and Stream menu relies on dataset picker state
    DiffMenu.register_diff_state_ctrl(state, ctrl, registry)
    ThresholdMenu.register_threshold_state_ctrl(state, ctrl, registry)
    DeriveMenu.register_derived_state_ctrl(state, ctrl, registry)
    FileMenu.register_file_menu_state_ctrl(state, ctrl, registry)
    FileMenu.register_session_state_ctrl(state, ctrl, registry)
    SaveSession.register_session_menu_state_ctrl(state, ctrl, registry, all_view_ids)
    StreamMenu.register_stream_menu_state_ctrl(state, ctrl, registry, state_queue)
    LocateMenu.register_locate_state_ctrl(state, ctrl, registry)

    available_view_ids = list(all_view_ids)
    """
    A view_id is an id assigned to each card that namespaces its per-view state.
    """
    # wrapper for guarding against an empty registry
    def requires_src(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not registry.has_src():
                return
            return func(*args, **kwargs)
        return wrapper

    @requires_src
    def _recompute_card_range(view_id):
        """Rescales the each views colorbar"""
        src = registry.get(state[f"selected_src_id_{view_id}"])
        if src is None:
            return
        array_name = state[f"selected_array_{view_id}"]
        if not array_name:
            return
        array = src.array(array_name)
        if array.dataset_type in (VeraDtype.COMP_ASSY_ENERGY, VeraDtype.COMP_NODAL_ENERGY):
            group_arrays = [array[g] for g in range(array.shape[0])]
        elif array.dataset_type in (VeraDtype.COMP_ASSY_SURFACE, VeraDtype.COMP_NODAL_SURFACE):
            group_arrays = [array[LATERAL_SURACES, g] for g in range(array.shape[1])]
        else:
            group_arrays = [array]
        for g, group_array in enumerate(group_arrays):
            state[f"color_range_{view_id}_{g}"] = array_range(group_array)

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
        #     selected_i, selected_j, selected_layer, selected_assembly_ij. If true the view is "locked" and
        #     will not change until unlocked.
        # state.label_info_{{view_id}} : state that stores the selected_i, selected_j, selected_layer, and selected_assembly_ij
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
        for g in range(MAX_NUM_GROUPS):
            state[f"color_range_{view_id}_{g}"] = (0.0, 1.0)
        state[f"color_units_{view_id}"] = "unitless"
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
            if (src := registry.get(state[f"selected_src_id_{view_id}"])) is not None:
                state[f"color_units_{view_id}"] = src.array_units(state[f"selected_array_{view_id}"])
        return _on_card_array_change
    
    def _make_option_watcher(view_id):
        @state.change(f"grid_view_{view_id}")
        @requires_src
        def _on_grid_option_change(**kwargs):
            # FIXME, need to add this to multi select views 
            option = state[f"grid_view_{view_id}"]
            src_id = state[f"selected_src_id_{view_id}"]
            sel_ds = state[f"selected_array_{view_id}"]
            dtype = registry.get(src_id).array_dtype(sel_ds)
            if ("allowed_categories" in option and dtype.title in option["allowed_categories"]) or "allowed_categories" not in option:
                return
            default_datasets_names = registry.get(src_id).default_datasets()
            allowed_categories = option["allowed_categories"]
            available_categories = sorted(set(default_datasets_names).intersection(allowed_categories))
            if len(available_categories) == 0:
                return
            new_ds_name = default_datasets_names[available_categories[0]]
            state[f"selected_array_{view_id}"] = new_ds_name
            state[f"selected_label_{view_id}"] = format_label(src_id, new_ds_name)
        return _on_grid_option_change

    for view_id in all_view_ids:
        _make_array_watcher(view_id)
        _make_option_watcher(view_id)
    
    def _default_array_for_option(src_id, option):
        """Dataset on src_id allowed by `option`, preferring pin_powers. None if unsatisfiable."""
        names = registry.get(src_id).default_datasets()
        allowed = option.get("allowed_categories")
        candidates = set(names) if allowed is None else set(names).intersection(allowed)
        if not candidates:
            return None
        pin = VeraDtype.PIN.title
        if pin in candidates:
            return "pin_powers" if "pin_powers" in names.values() else names[pin]
        return names[sorted(candidates)[0]]

    def place(module, x, y, w, h, default_id):
        default_dataset_name = _default_array_for_option(default_id, module.option_for(0))
        if default_dataset_name is None:
            return
        view_id = available_view_ids.pop(0)
        state[f"selected_src_id_{view_id}"] = default_id
        state[f"selected_array_{view_id}"] = default_dataset_name
        state[f"selected_label_{view_id}"] = format_label(default_id, default_dataset_name)
        state[f"multi_selected_{view_id}"] = [f"{default_id}{MULTI_SEP}{default_dataset_name}"]
        state[f"multi_label_{view_id}"] = "1 Selected"
        _recompute_card_range(view_id)
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = module.option_for(view_id)

    def _clear_view_source(view_id):
        state[f"selected_src_id_{view_id}"] = None
        state[f"selected_array_{view_id}"] = ""
        state[f"selected_label_{view_id}"] = "No dataset"

    activation_done = False

    @ctrl.set("remove_source")
    def _remove_source(src_id: str):
        if src_id not in registry:
            return
        registry.remove_src(src_id)
        state.recipes = [r for r in state.recipes if src_id not in recipe_sources(r)]

        fallback_id = registry.default_src_id
        for view_id in all_view_ids:
            # scrub dead tokens from multi-select views
            prefix = f"{src_id}{MULTI_SEP}"
            current = state[f"multi_selected_{view_id}"]
            kept = [t for t in current if not t.startswith(prefix)]
            if kept != current:
                state[f"multi_selected_{view_id}"] = kept
                state[f"multi_label_{view_id}"] = f"{len(kept)} selected" if kept else "Select datasets"

            # primary source reference
            if state[f"selected_src_id_{view_id}"] != src_id:
                continue

            new_array = _default_array_for_option(fallback_id, state[f"grid_view_{view_id}"]) if fallback_id else None
            if new_array is not None:
                select_dataset(view_id, fallback_id, new_array)
            else:
                _clear_view_source(view_id)

        if not registry.has_src():
            state.has_data = False
            state.grid_layout = []
            nonlocal available_view_ids, activation_done
            available_view_ids = list(all_view_ids)
            activation_done = False

        DatasetPicker.refresh_src_tree(state, registry)
        state.grid_rebuild_key += 1

    @ctrl.set("load_session")
    def _load_session(in_path: str):
        print("here", in_path)
        data = json.loads(Path(in_path).read_text())
        session = Session(**{
            **data,
            "views": [ViewSession(**v) for v in data["views"]],
            "recipes": data.get("recipes", []),
        })
        for id, path in session.file_paths.items():
            if not Path(path).is_file():
                state.file_error = f"Session file not found: {path}"
                return
        registry.clear()
        core_overrides = session.core_overrides or {}
        for src_id, path in session.file_paths.items():
            registry.add_src(VeraOutFile(path, core_overrides=core_overrides.get(path, {})), src_id=src_id)
        if session.default_src_id in registry:
            registry.default_src_id = session.default_src_id

        # replay recipes in creation order (= dependency order) before views reference them
        state.recipes = []
        present = set(session.file_paths)
        recipe_errors = []
        for r in session.recipes:
            missing = recipe_sources(r) - present
            if missing:
                recipe_errors.append(f"{r['name']}: missing source(s) {', '.join(missing)}")
                continue
            try:
                registry.apply_recipe(r)
                state.recipes = state.recipes + [r]
            except Exception as e:
                recipe_errors.append(f"{r['name']}: {e}")
        if recipe_errors:
            state.file_error = "Some datasets failed to rebuild: " + "; ".join(recipe_errors)

        DatasetPicker.refresh_src_tree(state, registry)
        state.grid_layout = [dict(v.layout) for v in session.views if v.layout]

        for v in session.views:
            vid = v.view_id
            state[f"selected_src_id_{vid}"] = v.selected_src_id
            state[f"selected_array_{vid}"] = v.selected_array
            state[f"selected_label_{vid}"] = v.selected_label
            state[f"multi_selected_{vid}"] = list(v.multi_selected)
            state[f"multi_label_{vid}"] = v.multi_label
            state[f"locked_{vid}"] = v.locked
            state[f"crop_enabled_{vid}"] = v.crop_enabled
            state[f"crop_mode_{vid}"] = v.crop_mode
            state[f"crop_x_{vid}"] = list(v.crop_x)
            state[f"crop_y_{vid}"] = list(v.crop_y)
            state[f"crop_z_{vid}"] = list(v.crop_z)
            state[f"camera_{vid}"] = v.camera
            state.dirty(f"camera_{vid}")
            # Set active view LAST so its update handler sees the inputs in place.
            state[f"grid_view_{vid}"] = v.option

        for k, val in session.globals.items():
            state[k] = val

        state.grid_rebuild_key += 1
        state.dirty("grid_layout")
        state.has_data = True

    def activate_src():
        """Run the data-dependent setup once, when the first src exists.

        Invoked immediately when a file is supplied at startup, or by the file
        menu the first time a file is opened from the UI.
        """
        nonlocal activation_done
        if activation_done or not registry.has_src():
            return

        src = registry.default_src
        default_id = registry.default_src_id
        default_names = src.default_datasets()
        default_name = default_dataset_name(default_names)
        core_shape = src.core.core_shape
        ny, nx, nz = core_shape[0], core_shape[1], core_shape[2]

        # Global State
        # state.selected_layer : state for tracking which axial_plane is selected 
        # state.selected_i : state for tracking the x-index of the selected pin 
        # state.selected_j : state for tracking the y-index of the selected pin
        # state.selected_assembly_ij : state for tracking the row and col of the selected assembly in the core_map
        # state.max_time : state for tracking the maximum state number of all sources in the registry
        # state.selected_time : state for tracking the STATE_n being visualized, i.e. if state.selected_time == 2, STATE_0002 in the vera source is being visualized
        state.selected_layer = nz // 2
        state.max_layer = len(registry.global_axial_mesh) - 1
        state.selected_i = (nx // 2) - (1 if nx // 2 >= 1 else 0) # not a center pin
        state.selected_j = (ny // 2) - (1 if ny // 2 >= 1 else 0)
        center_assy = _center_assembly(src.core.reduced_core_map)
        assembly_i, assembly_j = src.core.reduced_core_map_ij(center_assy)
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
        place(x_axial_view,   0,  0, 3, 17, default_id)
        place(core_view,      6,  0, 4,  10, default_id)
        place(assembly_view,  9,  0, 3,  9, default_id)
        place(axial_plot,     6,  9, 3,  8, default_id)
        place(time_plot,      9,  9, 3,  8, default_id)
        place(volume_view,    3,  0, 3, 17, default_id)
        place(table_view,     0, 17, 6, 10, default_id)
        state.dirty("grid_layout")
        state.has_data = True
        activation_done = True
        # for view_id in all_view_ids:
        #     if state[f"grid_view_{view_id}"]["name"] == volume_view.option_for(view_id)["name"]:
        #         getattr(ctrl, f"reset_volume_{view_id}_camera")()

    ctrl.activate_src = activate_src
   
    build_layout(server, state, ctrl, registry)  # vue ui is built here

    if registry.has_src():
        # automatically run data dependent setup if source was provied from command line
        activate_src()



