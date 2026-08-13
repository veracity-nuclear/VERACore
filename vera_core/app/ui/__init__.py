import functools
import json
from pathlib import Path

import numpy as np
from trame.app.asynchronous import StateQueue
from trame_server.core import Server

from vera_core.data.analysis.color import array_range
from vera_core.data.dtypes import LATERAL_SURFACES, MAX_NUM_GROUPS, VeraDtype
from vera_core.data.readers.h5 import open_vera_file_data_source
from vera_core.data.registry import VeraDataRegistry

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
    default_dataset_name,
    format_label,
    get_next_y_from_layout,
    is_view_locked,
)
from .layout import build_layout
from .session import Session, ViewSession, recipe_sources
from .views import (
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
NUM_VIEW_SLOTS = 10
MULTI_SEP = "\x1f"

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


def _decode_tokens(tokens):
    """[(src_id, array_name)] from serialized multi-picker tokens.

    Multi selections are stored as separator-joined strings rather than tuples
    so trame can serialize them. Malformed entries are dropped.
    """
    return [tuple(token.split(MULTI_SEP, 1)) for token in tokens if MULTI_SEP in token]


def _encode_tokens(pairs):
    return [f"{src_id}{MULTI_SEP}{array_name}" for src_id, array_name in pairs]


def _dedupe(pairs):
    """Order-preserving deduplicate"""
    return list(dict.fromkeys(pairs))


def _group_arrays(array):
    dataset_type = array.dataset_type
    if dataset_type in (VeraDtype.COMP_ASSY_ENERGY, VeraDtype.COMP_NODAL_ENERGY):
        groups = [array[g] for g in range(array.shape[0])]
    elif dataset_type in (VeraDtype.COMP_ASSY_SURFACE, VeraDtype.COMP_NODAL_SURFACE):
        groups = [array[LATERAL_SURFACES, g] for g in range(array.shape[1])]
    else:
        groups = [array]
    return groups[:MAX_NUM_GROUPS]


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

    def _resolve_pair(option, src_id, array_name):
        """(src_id, array_name) satisfying `option`, or None if nothing can.

        Keeps the current selection when it is still valid, then falls back to a
        default dataset on the same source, then on any other loaded source.
        """
        allowed = option.get("allowed_categories")
        src = _src(src_id)
        # array_dtype returns None for a name that no longer exists, e.g. one
        # left behind by a removed source or recipe.
        dtype = src.get_dataset_dtype(array_name) if src is not None and array_name else None
        if dtype is not None and (allowed is None or dtype.title in allowed):
            return (src_id, array_name)
        others = [s for s in registry.src_ids() if s and s != src_id]
        for candidate in [src_id, *others]:
            fallback = _default_array_for_option(candidate, option)
            if fallback is not None:
                return (candidate, fallback)
        return None

    def _set_multi_selection(view_id, pairs):
        state[f"multi_selected_{view_id}"] = _encode_tokens(pairs)
        state[f"multi_label_{view_id}"] = f"{len(pairs)} selected" if pairs else NO_SELECTION_LABEL

    @ctrl.set("select_dataset")
    def select_dataset(view_id, src_id, array_name):
        """Point a card at a dataset.

        Args:
            view_id: card whose per-view state is updated.
            src_id: registry id of the owning source; names are not unique
                across sources.
            array_name: dataset to visualize.
        """
        state[f"selected_src_id_{view_id}"] = src_id
        state[f"selected_array_{view_id}"] = array_name
        state[f"selected_label_{view_id}"] = format_label(src_id, array_name)

    @ctrl.set("toggle_multi_array")
    def toggle_multi_array(view_id, src_id, array_name):
        """Add or remove a (source, dataset) pair in a multi-picker card."""
        pairs = _decode_tokens(state[f"multi_selected_{view_id}"])
        pair = (src_id, array_name)
        if pair in pairs:
            pairs.remove(pair)
        else:
            pairs.append(pair)
        _set_multi_selection(view_id, pairs)

    def _clear_view_source(view_id):
        state[f"selected_src_id_{view_id}"] = None
        state[f"selected_array_{view_id}"] = ""
        state[f"selected_label_{view_id}"] = NO_DATASET_LABEL
        _set_multi_selection(view_id, [])

    @requires_src
    def _recompute_card_range(view_id):
        """Rescale a card's shared color bar to its current data."""
        if state[f"grid_view_{view_id}"].get("owns_color_bar", False):
            return
        src_id = state[f"selected_src_id_{view_id}"]
        array_name = state[f"selected_array_{view_id}"]
        src = _src(src_id)
        if src is None or not array_name:
            return
        for g, group_array in enumerate(_group_arrays(src.get_dataset(array_name))):
            state[f"color_range_{view_id}_{g}"] = array_range(group_array)

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
        state[f"selected_label_{view_id}"] = NO_DATASET_LABEL
        state[f"multi_selected_{view_id}"] = []
        state[f"multi_label_{view_id}"] = NO_SELECTION_LABEL
        state[f"locked_{view_id}"] = False
        state[f"color_units_{view_id}"] = "unitless"
        state[f"label_info_{view_id}"] = {
            "Exposure": "-",
            "Assembly": "-",
            "Layer": "-",
            "Pin_x": "-",
            "Pin_y": "-",
        }
        for g in range(MAX_NUM_GROUPS):
            state[f"color_range_{view_id}_{g}"] = (0.0, 1.0)
        for module in VIEW_MODULES:
            module.initialize(server, registry, view_id)

    def _register_view_watchers(view_id):
        @state.change(
            f"selected_src_id_{view_id}",
            f"selected_array_{view_id}",
            f"multi_selected_{view_id}",
            f"locked_{view_id}",
        )
        def _on_selection_change(**kwargs):
            _recompute_card_range(view_id)
            src_id = state[f"selected_src_id_{view_id}"]
            array_name = state[f"selected_array_{view_id}"]
            src = _src(src_id)
            if src is not None and array_name:
                state[f"color_units_{view_id}"] = src.get_dataset_units(array_name)

        @state.change(f"grid_view_{view_id}")
        @requires_src
        def _on_view_change(**kwargs):
            """Re-point the card at datasets the newly selected view accepts."""
            option = state[f"grid_view_{view_id}"]
            multi = option.get("multi_picker", False)
            pairs = _decode_tokens(state[f"multi_selected_{view_id}"]) if multi else []
            if not pairs:
                # Single-picker view, or a multi view switched in from a single
                # one with nothing carried over.
                pairs = [
                    (
                        state[f"selected_src_id_{view_id}"],
                        state[f"selected_array_{view_id}"],
                    )
                ]
            resolved = _dedupe(
                pair
                for pair in (_resolve_pair(option, *pair) for pair in pairs)
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
    FileMenu.register_file_menu_state_ctrl(state, ctrl, registry)
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
        for view_id in all_view_ids:
            if not is_view_locked(state, view_id):
                _recompute_card_range(view_id)

    @state.change("src_tree_meta")
    def refresh_max_state(**kwargs):
        state.max_time = max(state.max_time, registry.max_state)

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
        _set_multi_selection(view_id, [(src_id, array_name)])
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = option
        _recompute_card_range(view_id)

    @ctrl.set("remove_source")
    def remove_source(src_id: str):
        if src_id not in registry:
            return
        nonlocal activation_done
        registry.remove_src(src_id)
        state.recipes = [r for r in state.recipes if src_id not in recipe_sources(r)]

        fallback_id = registry.default_src_id
        for view_id in all_view_ids:
            pairs = _decode_tokens(state[f"multi_selected_{view_id}"])
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

        if not registry.has_src():
            state.has_data = False
            state.grid_layout = []
            _reset_view_pool()
            activation_done = False

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
            file_overrides=file_overrides,
            default_src_id=raw_data.get("default_src_id"),
            globals=dict(raw_data.get("globals", {})),
            views=views,
            recipes=list(raw_data.get("recipes", [])),
        )
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

        for key, value in session.globals.items():
            state[key] = value

        # Slots used by the session must not be handed out again, and the
        # default arrangement must not overwrite the restored one.
        _reset_view_pool(item["i"] for item in state.grid_layout)
        activation_done = True

        state.grid_rebuild_key += 1
        state.dirty("grid_layout")
        state.has_data = True

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

        state.selected_layer = nz // 2
        state.max_layer = len(registry.global_axial_mesh) - 1
        state.selected_i = max((nx // 2) - 1, 0)  # not a center pin
        state.selected_j = max((ny // 2) - 1, 0)
        center_assembly = _center_assembly(src.core.reduced_core_map)
        assembly_i, assembly_j = src.core.reduced_core_map_ij(center_assembly)
        state.selected_assembly_ij = {"i": assembly_i, "j": assembly_j}
        state.max_time = registry.max_state
        state.selected_time = 0

        for view_id in all_view_ids:
            select_dataset(view_id, default_id, default_name)
            _set_multi_selection(view_id, [(default_id, default_name)])
            _recompute_card_range(view_id)
        for module, x, y, w, h in DEFAULT_ARRANGEMENT:
            _place(module, default_id, x, y, w, h)

        state.dirty("grid_layout")
        state.has_data = True
        activation_done = True

    ctrl.activate_src = activate_src

    build_layout(server, state, ctrl, registry)  # vue ui is built here

    if registry.has_src():
        # A source came from the command line, so run the setup now.
        activate_src()
