import functools
import json
from pathlib import Path

import numpy as np
from trame.app.asynchronous import StateQueue
from trame.app.dev import remove_change_listeners
from trame_server.core import Server

from vera_core.data.analysis.color import finite_range, union_range
from vera_core.data.dtypes import LATERAL_SURFACES, VeraDataset, VeraDtype
from vera_core.data.model import VeraDataSource, nan_out_non_fuel
from vera_core.data.readers.h5 import open_vera_file_data_source
from vera_core.data.readers.rom_reciever import generate_stream_identifier
from vera_core.data.registry import VeraDataRegistry

from .color_ranges_cache import StateRangeCache
from .features import (
    DatasetPicker,
    DeriveMenu,
    DiffMenu,
    FileMenu,
    LocateMenu,
    SaveSession,
    StreamMenu,
    ThresholdMenu,
    VersionChecker,
)
from .features.appdata import validate_file_overrides
from .helpers import (
    decode_tokens,
    default_dataset_name,
    encode_tokens,
    format_label,
    get_next_y_from_layout,
    get_thresholds,
    get_time,
    is_view_locked,
)
from .layout import build_layout
from .session import Session, ViewSession, recipe_sources
from .views import (
    MAX_VIS_GROUPS,
    assembly_view,
    axial_plot,
    cips_view,
    core_axial_view,
    core_view,
    empty,
    histogram_view,
    surface_assembly_view,
    surface_core_view,
    table_view,
    time_plot,
    volume_view,
    x_axial_view,
    y_axial_view,
)

DEFAULT_NB_ROWS = 8
# Color bars span every state (True) or only the displayed state (False).
COLOR_ALL_STATES_DEFAULT = False
NUM_VIEW_SLOTS = 10

NO_DATASET_LABEL = "No dataset"
NO_SELECTION_LABEL = "Select datasets"

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
    histogram_view,
    cips_view,
]

# Default card arrangement applied when the first source is activated:
# (view module, x, y, w, h).
DEFAULT_ARRANGEMENT = [
    (x_axial_view, 0, 0, 3, 17),
    (core_view, 6, 0, 4, 10),
    (assembly_view, 6, 0, 4, 10),
    (axial_plot, 6, 9, 3, 8),
    (time_plot, 9, 9, 3, 8),
    (volume_view, 3, 0, 3, 17),
    (table_view, 0, 17, 6, 10),
]

SESSION_VIEW_FIELDS = (
    "selected_src_id",
    "selected_array",
    "selected_label",
    "selected_group",
    "multi_selected",
    "multi_label",
    "locked",
    "assembly_decimals",
    "crop_enabled",
    "crop_mode",
    "crop_x",
    "crop_y",
    "crop_z",
)


def _center_assembly(reduced_core_map):
    """0-based index of the loaded assembly nearest the core center."""
    positions = np.argwhere(reduced_core_map > 0)
    center = (np.array(reduced_core_map.shape) - 1) / 2.0
    i, j = positions[np.argmin(np.linalg.norm(positions - center, axis=1))]
    return int(reduced_core_map[i, j]) - 1


def _dedupe(pairs):
    """Order-preserving deduplicate"""
    return list(dict.fromkeys(pairs))


def _build_group_slice(ndim, group_axis, group, surface_axis: int | None = None):
    index = [slice(None)] * ndim
    index[group_axis] = group
    if surface_axis is not None:
        index[surface_axis] = LATERAL_SURFACES
    return tuple(index)


def _group_arrays(array: VeraDataset, group: int | None = None):
    dtype = array.dataset_type
    if not dtype.has_energy_group_dim():
        return [array]
    group_axis = dtype.energy_group_dim_idx
    surface_axis = dtype.surface_dim_idx if dtype.has_surface_dim() else None
    n = array.shape[group_axis]
    if group is None:
        idxs = range(min(n, MAX_VIS_GROUPS))
    else:
        idxs = [min(max(group - 1, 0), n - 1)]
    return [array[_build_group_slice(array.ndim, group_axis, i, surface_axis)] for i in idxs]


def _copy_value(value):
    """Copy sequences so restored session state is not shared with the session object."""
    return list(value) if isinstance(value, (list, tuple)) else value


def initialize(server: Server, registry: VeraDataRegistry, state_queue: StateQueue):
    state, ctrl = server.state, server.controller
    state.trame__title = "VERACore"

    all_view_ids = [f"{v + 1}" for v in range(NUM_VIEW_SLOTS)]
    available_view_ids = list(all_view_ids)
    activation_done = False

    def requires_src(func):
        """No-op the wrapped callback while the registry is empty."""

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not registry.has_src():
                return
            return func(*args, **kwargs)

        return wrapper

    def _src(src_id):
        """Source for an id, or None when the id is unset or unknown."""
        return registry.get(src_id) if src_id else None

    def _default_array_for_option(src_id, option):
        """Dataset on src_id allowed by `option`, preferring pin_powers.

        None when the source is unknown or cannot satisfy the option.
        """
        src = _src(src_id)
        if src is None:
            return None
        names = src.default_datasets()
        allowed = option.get("allowed_categories")
        candidates = set(names) if allowed is None else set(names) & set(allowed)
        if not candidates:
            return None
        pin = VeraDtype.PIN.title
        if pin in candidates:
            return "pin_powers" if "pin_powers" in names.values() else names[pin]
        return names[sorted(candidates)[0]]

    def _resolve_pair(option, src_id, array_name, group, view_id):
        allowed = option.get("allowed_categories")
        src = _src(src_id)
        dtype = (
            src.get_dataset_dtype(array_name, state_idx=get_time(state, view_id))
            if src is not None and array_name
            else VeraDtype.UNKNOWN
        )
        if dtype is not VeraDtype.UNKNOWN and (allowed is None or dtype.title in allowed):
            return (src_id, array_name, group)
        others = [s for s in registry.src_ids() if s and s != src_id]
        for candidate in [src_id, *others]:
            fallback = _default_array_for_option(candidate, option)
            if fallback is not None:
                return (candidate, fallback, None)
        return None

    def _set_multi_selection(view_id, triples):
        state[f"multi_selected_{view_id}"] = encode_tokens(triples)
        state[f"multi_label_{view_id}"] = (
            f"{len(triples)} selected" if triples else NO_SELECTION_LABEL
        )

    @ctrl.set("select_dataset")
    def select_dataset(view_id, src_id, array_name, group=None):
        """Point a card at a dataset. `group` is 1-based; None means all groups."""
        state[f"selected_src_id_{view_id}"] = src_id
        state[f"selected_array_{view_id}"] = array_name
        state[f"selected_group_{view_id}"] = group
        state[f"selected_label_{view_id}"] = format_label(src_id, array_name)

    @ctrl.set("toggle_multi_array")
    def toggle_multi_array(view_id, src_id, array_name, group=None):
        """Add or remove a (source, dataset, group) selection on a multi-picker card."""
        triples = decode_tokens(state[f"multi_selected_{view_id}"])
        entry = (src_id, array_name, group)
        if entry in triples:
            triples.remove(entry)
        else:
            triples.append(entry)
        _set_multi_selection(view_id, triples)

    def _clear_view_source(view_id):
        with state:
            state[f"selected_src_id_{view_id}"] = None
            state[f"selected_array_{view_id}"] = ""
            state[f"selected_group_{view_id}"] = None
            state[f"selected_label_{view_id}"] = NO_DATASET_LABEL
        _set_multi_selection(view_id, [])

    range_cache = StateRangeCache()

    def _state_group_ranges(src: VeraDataSource, state_idx, array_name, group, thresholds):
        try:
            dataset = src.get_dataset(array_name, state_idx=state_idx)
        except RuntimeError:
            return None
        dataset = nan_out_non_fuel(dataset, src.core.pin_volumes)
        return [finite_range(array, thresholds) for array in _group_arrays(dataset, group)]

    @requires_src
    def _recompute_card_range(view_id):
        """
        Rescale a card's shared color bar to its current data.
        Spans every state when color_all_states is set, else the card's
        displayed state. Per-state results are cached.
        """
        if state[f"grid_view_{view_id}"].get("owns_color_bar", False):
            return
        src_id = state[f"selected_src_id_{view_id}"]
        array_name = state[f"selected_array_{view_id}"]
        group = state[f"selected_group_{view_id}"]
        src = _src(src_id)
        if src is None or not array_name:
            return
        thresholds = get_thresholds(state, view_id)
        key = (array_name, group, json.dumps(thresholds, sort_keys=True, default=repr))
        if not src.states:
            return
        if state.color_all_states:
            state_idxs = range(len(src.states))
        else:
            time = get_time(state, view_id)
            time = src.active_state_index if time is None else time
            state_idxs = [max(0, min(time, len(src.states) - 1))]
        per_state = [
            range_cache.get(
                src.states[i],
                key,
                functools.partial(_state_group_ranges, src, i, array_name, group, thresholds),
            )
            for i in state_idxs
        ]
        per_state = [ranges for ranges in per_state if ranges is not None]
        for g, group_ranges in enumerate(zip(*per_state, strict=True)):
            found = [r for r in group_ranges if r is not None]
            state[f"color_range_{view_id}_{g}"] = union_range(found)

    def _init_global_state():
        # selected_time: STATE_n being visualized (2 -> STATE_0002).
        # max_time: highest state index across every source in the registry.
        # selected_layer: axial plane being visualized.
        # selected_i / selected_j: x and y index of the selected pin.
        # selected_assembly_ij: row and column of the selected assembly in the
        #     core map.
        state.setdefault("grid_item_dirty_key", 0)
        state.setdefault("grid_layout", [])
        state.setdefault("grid_rebuild_key", 0)
        state.setdefault("has_data", registry.has_src())
        state.setdefault("selected_time", 0)
        state.setdefault("max_time", 0)
        state.setdefault("selected_layer", 0)
        state.setdefault("max_layer", 0)
        state.setdefault("selected_i", 0)
        state.setdefault("selected_j", 0)
        state.setdefault("selected_assembly_ij", dict(i=0, j=0))
        state.setdefault("selected_surface", 0)
        state.setdefault("dark_mode", True)
        state.setdefault("recipes", [])
        state.setdefault("color_preset", "jet")
        state.setdefault("color_all_states", COLOR_ALL_STATES_DEFAULT)

    def _init_view_state(view_id):
        """Namespaced state for one card.

        grid_view: the active view option dict for the card.
        grid_options: view options the card can switch to.
        selected_src_id / selected_array: the single-picker selection.
        selected_label: formatted label of that selection.
        multi_selected / multi_label: selection set for multi-picker views.
        locked: when true the card ignores global changes to selected_i,
            selected_j, selected_layer and selected_assembly_ij.
        label_info: indices and exposure of the data currently shown.
        """
        state[f"grid_options_{view_id}"] = []
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state[f"selected_src_id_{view_id}"] = registry.default_src_id
        state[f"selected_array_{view_id}"] = ""
        state[f"selected_group_{view_id}"] = None
        state[f"selected_label_{view_id}"] = NO_DATASET_LABEL
        state[f"multi_selected_{view_id}"] = []
        state[f"multi_label_{view_id}"] = NO_SELECTION_LABEL
        state[f"locked_{view_id}"] = {}
        state[f"view_loading_{view_id}"] = True  # flag to indicate this view is loading
        state[f"color_units_{view_id}"] = "unitless"
        state[f"label_info_{view_id}"] = {
            "Exposure": "-",
            "Assembly": "-",
            "Layer": "-",
            "Pin_x": "-",
            "Pin_y": "-",
        }
        for g in range(MAX_VIS_GROUPS):
            state[f"color_range_{view_id}_{g}"] = (0.0, 1.0)
        for module in VIEW_MODULES:
            module.initialize(server, registry, view_id)

    def _register_view_watchers(view_id):
        @state.change(
            f"selected_src_id_{view_id}",
            f"selected_array_{view_id}",
            f"selected_group_{view_id}",
            f"multi_selected_{view_id}",
            f"locked_{view_id}",
        )
        def _on_selection_change(**kwargs):
            _recompute_card_range(view_id)
            src_id = state[f"selected_src_id_{view_id}"]
            array_name = state[f"selected_array_{view_id}"]
            src = _src(src_id)
            if src is not None and array_name:
                state[f"color_units_{view_id}"] = src.get_dataset_units(
                    array_name, state_idx=get_time(state, view_id)
                )

        @state.change("thresholds")
        def _on_thres_range_changed(**kwargs):
            if is_view_locked(state, view_id):
                return
            _recompute_card_range(view_id)

        @state.change(f"grid_view_{view_id}")
        @requires_src
        def _on_view_change(**kwargs):
            """Re-point the card at datasets the newly selected view accepts."""
            option = state[f"grid_view_{view_id}"]
            multi = option.get("multi_picker", False)
            pairs = decode_tokens(state[f"multi_selected_{view_id}"]) if multi else []
            if not pairs:
                pairs = [
                    (
                        state[f"selected_src_id_{view_id}"],
                        state[f"selected_array_{view_id}"],
                        state[f"selected_group_{view_id}"],
                    )
                ]
            resolved = _dedupe(
                pair
                for pair in (_resolve_pair(option, *pair, view_id=view_id) for pair in pairs)
                if pair is not None
            )
            if not resolved:
                _clear_view_source(view_id)
                return
            # Keep the single-picker state coherent either way: a later switch
            # back to a single-picker view reads it.
            select_dataset(view_id, *resolved[0])
            if multi:
                _set_multi_selection(view_id, resolved)

    _init_global_state()

    # The file and stream menus read dataset-picker state, so it goes first.
    DatasetPicker.register_dataset_picker_state(state, registry)
    DiffMenu.register_diff_state_ctrl(state, ctrl, registry)
    ThresholdMenu.register_threshold_state_ctrl(state, ctrl, registry)
    DeriveMenu.register_derived_state_ctrl(state, ctrl, registry)
    FileMenu.register_file_menu_state_ctrl(state, ctrl, registry, view_ids=all_view_ids)
    FileMenu.register_session_state_ctrl(state, ctrl, registry)
    SaveSession.register_session_menu_state_ctrl(state, ctrl, registry, all_view_ids)
    StreamMenu.register_stream_menu_state_ctrl(state, ctrl, registry, state_queue)
    LocateMenu.register_locate_state_ctrl(state, ctrl, registry)
    VersionChecker.register_version_check_ctrl(state, ctrl)

    for view_id in all_view_ids:
        _init_view_state(view_id)
    for view_id in all_view_ids:
        _register_view_watchers(view_id)

    @state.change("selected_time")
    @requires_src
    def selected_time_changed(selected_time, **kwargs):
        """Fan the selected state index out to every view."""
        selected_time = int(selected_time)
        registry.change_all_active_state(selected_time)
        ctrl.on_vera_out_active_state_index_changed(selected_time=selected_time, **kwargs)
        if state.color_all_states:
            return  # the range already spans every state
        for view_id in all_view_ids:
            if not is_view_locked(state, view_id):
                _recompute_card_range(view_id)

    @state.change("color_all_states")
    def color_all_states_changed(**kwargs):
        for view_id in all_view_ids:
            _recompute_card_range(view_id)

    @state.change("src_tree_meta")
    def refresh_max_state(**kwargs):
        state.max_time = max(state.max_time, registry.max_state)
        state.max_layer = max(len(registry.global_axial_mesh) - 1, 0)
        if state.color_all_states:
            # States may have been added or dropped; only new ones are read.
            for view_id in all_view_ids:
                _recompute_card_range(view_id)

    def _reset_view_pool(used_ids=()):
        nonlocal available_view_ids
        used = set(used_ids)
        available_view_ids = [vid for vid in all_view_ids if vid not in used]

    @ctrl.set("grid_add_view")
    def add_view():
        """Append an empty card to the grid."""
        if not available_view_ids:
            return  # every slot is in use
        view_id = available_view_ids.pop(0)
        state.grid_layout.append(
            dict(
                x=0,
                y=get_next_y_from_layout(state.grid_layout),
                w=12,
                h=DEFAULT_NB_ROWS,
                i=view_id,
            )
        )
        state.dirty("grid_layout")

    @ctrl.set("grid_remove_view")
    def remove_view(view_id):
        """Remove a card and return its slot to the pool."""
        if view_id not in available_view_ids:
            available_view_ids.append(view_id)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state.grid_layout = [item for item in state.grid_layout if item.get("i") != view_id]
        state.grid_rebuild_key += 1

    @ctrl.set("toggle_lock")
    def toggle_lock(view_id):
        SELECTION_KEYS = (
            "selected_time",
            "selected_layer",
            "selected_i",
            "selected_j",
            "selected_surface",
            "selected_assembly_ij",
        )

        lock_key = f"locked_{view_id}"

        if state[lock_key]:
            state[lock_key] = {}
        else:
            state[lock_key] = {
                selection_key: state[selection_key] for selection_key in SELECTION_KEYS
            }

    def _place(module, src_id, x, y, w, h):
        """Add a card running `module`, unless no dataset on src_id satisfies it."""
        if not available_view_ids:
            return
        view_id = available_view_ids.pop(0)
        option = module.option_for(view_id)
        array_name = _default_array_for_option(src_id, option)
        if array_name is None:
            available_view_ids.insert(0, view_id)
            return
        select_dataset(view_id, src_id, array_name)
        _set_multi_selection(view_id, [(src_id, array_name, None)])
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = option
        _recompute_card_range(view_id)

    @ctrl.set("remove_source")
    def remove_source(src_id: str):
        if src_id not in registry:
            return
        nonlocal activation_done
        registry.remove_src(src_id)
        stream_id = src_id if state.ports_opened.pop(src_id, None) is not None else None
        if stream_id is not None:
            stream_full_id = generate_stream_identifier(stream_id)
            state_count_key = f"{stream_full_id}_state_count"
            state[stream_full_id] = None
            state[state_count_key] = None
            remove_change_listeners(server, stream_full_id, state_count_key)
        state.recipes = [r for r in state.recipes if src_id not in recipe_sources(r)]

        fallback_id = registry.default_src_id
        for view_id in all_view_ids:
            pairs = decode_tokens(state[f"multi_selected_{view_id}"])
            kept = [pair for pair in pairs if pair[0] != src_id]
            if len(kept) != len(pairs):
                _set_multi_selection(view_id, kept)

            if state[f"selected_src_id_{view_id}"] != src_id:
                continue
            array_name = _default_array_for_option(fallback_id, state[f"grid_view_{view_id}"])
            if array_name is None:
                _clear_view_source(view_id)
            else:
                select_dataset(view_id, fallback_id, array_name)
        with state:
            if stream_id is not None:
                state.dirty("ports_opened")
            if not registry.has_src():
                state.has_data = False
                state.grid_layout = []
                _reset_view_pool()
                activation_done = False
            max_time = registry.max_state
            clamped_time = max(0, min(state.selected_time, max_time))
            state.max_time = max_time
            state.max_layer = max(len(registry.global_axial_mesh) - 1, 0)
            state.selected_time = clamped_time
            registry.change_all_active_state(clamped_time)
            DatasetPicker.refresh_src_tree(state, registry)
            state.grid_rebuild_key += 1

    def _replay_recipes(recipes, present_src_ids):
        """Rebuild derived datasets in creation order. Returns error strings."""
        state.recipes = []
        errors = []
        for recipe in recipes:
            missing = recipe_sources(recipe) - present_src_ids
            if missing:
                errors.append(f"{recipe['name']}: missing source(s) {', '.join(missing)}")
                continue
            try:
                registry.apply_recipe(recipe)
                state.recipes = state.recipes + [recipe]
            except Exception as e:
                errors.append(f"{recipe['name']}: {e}")
        return errors

    @ctrl.set("load_session")
    def load_session(in_path: str):
        nonlocal activation_done
        raw_data: object = json.loads(Path(in_path).read_text(encoding="utf-8"))
        if not isinstance(raw_data, dict):
            raise ValueError("Session file must contain a JSON object")

        raw_views = raw_data.get("views")
        if not isinstance(raw_views, list):
            raise ValueError("Session file is missing a valid views list")

        views = [ViewSession(**view) for view in raw_views if isinstance(view, dict)]
        if len(views) != len(raw_views):
            raise ValueError("Session file contains an invalid view entry")

        # Accept the old key for sessions saved before the rename.
        raw_file_overrides = raw_data.get("file_overrides", raw_data.get("core_overrides", {}))
        file_overrides = validate_file_overrides(raw_file_overrides)

        session = Session(
            version=int(raw_data.get("version", 1)),
            file_paths=dict(raw_data.get("file_paths", {})),
            stream_ports=dict(raw_data.get("stream_ports", {})),
            file_overrides=file_overrides,
            default_src_id=raw_data.get("default_src_id"),
            globals=dict(raw_data.get("globals", {})),
            views=views,
            recipes=list(raw_data.get("recipes", [])),
        )
        with state:
            for path in session.file_paths.values():
                if not Path(path).is_file():
                    state.file_error = f"Session file not found: {path}"
                    return

            registry.clear()
            file_overrides = session.file_overrides
            for src_id, path in session.file_paths.items():
                registry.add_src(
                    open_vera_file_data_source(path, core_overrides=file_overrides.get(path, {})),
                    src_id=src_id,
                )
            for src_id, port in session.stream_ports.items():
                print("connecting to stream", port)
                ctrl.connect_stream(port, src_id)
            if session.default_src_id in registry:
                registry.default_src_id = session.default_src_id

            errors = _replay_recipes(session.recipes, set(session.file_paths))
            if errors:
                state.file_error = "Some datasets failed to rebuild: " + "; ".join(errors)

            DatasetPicker.refresh_src_tree(state, registry)
            state.grid_layout = [dict(v.layout) for v in session.views if v.layout]

            for view in session.views:
                vid = view.view_id
                for field in SESSION_VIEW_FIELDS:
                    state[f"{field}_{vid}"] = _copy_value(getattr(view, field))
                state[f"camera_{vid}"] = view.camera
                state.dirty(f"camera_{vid}")
                state[f"grid_view_{vid}"] = view.option

            state.max_time = registry.max_state
            for key, value in session.globals.items():
                state[key] = value
            registry.change_all_active_state(session.globals.get("selected_time", 0))

            # Slots used by the session must not be handed out again, and the
            # default arrangement must not overwrite the restored one.
            _reset_view_pool(item["i"] for item in state.grid_layout)
            activation_done = True

            state.grid_rebuild_key += 1
            state.dirty("grid_layout")
            state.has_data = True
            # Mark all views as finished loading
        with state:
            for view_id in all_view_ids:
                state[f"view_loading_{view_id}"] = False
        print("loaded session")

    def activate_src():
        """Run the data-dependent setup once, when the first source exists.

        Called immediately when a file is supplied on the command line, or by
        the file menu the first time a file is opened from the UI.
        """
        nonlocal activation_done
        if activation_done or not registry.has_src():
            return

        src = registry.default_src
        default_id = registry.default_src_id
        default_name = default_dataset_name(src.default_datasets())
        ny, nx, nz = src.core.core_shape[:3]

        with state:
            state.selected_layer = nz // 2
            state.max_layer = max(len(registry.global_axial_mesh) - 1, 0)
            state.selected_i = max((nx // 2) - 1, 0)  # not a center pin
            state.selected_j = max((ny // 2) - 1, 0)
            center_assembly = _center_assembly(src.core.reduced_core_map)
            assembly_i, assembly_j = src.core.reduced_core_map_ij(center_assembly)
            state.selected_assembly_ij = {"i": assembly_i, "j": assembly_j}
            state.max_time = registry.max_state
            state.selected_time = 0

            for view_id in all_view_ids:
                select_dataset(view_id, default_id, default_name)
                _set_multi_selection(view_id, [(default_id, default_name, None)])
                _recompute_card_range(view_id)
            for module, x, y, w, h in DEFAULT_ARRANGEMENT:
                _place(module, default_id, x, y, w, h)

            state.dirty("grid_layout")
            state.has_data = True
            activation_done = True
        # Mark all views as finished loading
        with state:
            for view_id in all_view_ids:
                state[f"view_loading_{view_id}"] = False

    ctrl.activate_src = activate_src

    build_layout(server, state, ctrl, registry)  # vue ui is built here

    if registry.has_src():
        # A source came from the command line, so run the setup now.
        activate_src()
