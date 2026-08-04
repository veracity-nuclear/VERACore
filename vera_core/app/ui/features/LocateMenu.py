import numpy as np
from trame.widgets import html, vuetify
from trame_server.core import Controller, State

from vera_core.app.core import (
    VeraDataRegistry,
    VeraDataset,
    VeraDataSource,
    VeraDtype,
    nan_out_reflected,
)

from ..helpers import format_label
from .DatasetPicker import build_dataset_picker


def register_locate_state_ctrl(state: State, ctrl: Controller, registry: VeraDataRegistry):
    state.show_locate_dialog = False
    state.locate_source_file = registry.default_src_id
    state.locate_source_array = "pin_powers"
    state.locate_source_label = format_label(registry.default_src_id, "pin_powers")
    state.locate_assembly_scope = "all"  # "all" | "current"
    state.locate_time_scope = "current"  # "all" | "current"
    state.locate_error = ""

    def _assembly_scope_data(data: VeraDataset, assembly_id: int):
        dtype = data.dataset_type
        match dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                return data[:, :, :, assembly_id]
            case VeraDtype.ASSEMBLY:
                return data[0, :, assembly_id]
            case VeraDtype.RADIAL:
                return data[:, :, assembly_id]
            case VeraDtype.RADIAL_ASSEMBLY:
                return data[assembly_id]

    def _nan_control_rods_pos(control_rod_positions, vera_array: VeraDataset):
        if vera_array.dataset_type in (VeraDtype.PIN, VeraDtype.RADIAL):
            vera_array = vera_array.copy()
            control_rod_positions = control_rod_positions
            rod_rows, rod_cols = control_rod_positions
            vera_array[rod_rows, rod_cols] = np.nan
            return vera_array
        return vera_array

    def _find_extremum(
        mode: str,
        time_scope: str,
        src: VeraDataSource,
        array_name: str,
        assembly_search: bool = False,
        assembly_id=0,
    ):
        extremum_arg_finder = np.nanargmax if mode == "max" else np.nanargmin
        extremum = -np.inf if mode == "max" else np.inf
        comparison = (lambda x, e: x >= e) if mode == "max" else (lambda x, e: x <= e)
        indices_of_extremum = None
        winning_state_idx = None
        reduced_core_map = src.core.reduced_core_map
        core_sym = src.core.core_sym
        controls_rod_pos = src.core.control_rod_positions

        if time_scope == "all":
            for state_idx, state in enumerate(src.states):
                vera_array = state.get(array_name, None)
                if vera_array is None:
                    continue
                vera_array = nan_out_reflected(reduced_core_map, core_sym, vera_array)
                vera_array = _nan_control_rods_pos(controls_rod_pos, vera_array)
                if np.all(np.isnan(vera_array)):
                    continue
                if assembly_search:
                    vera_array = _assembly_scope_data(vera_array, assembly_id)
                flat_idx = extremum_arg_finder(vera_array)
                idx = np.unravel_index(flat_idx, vera_array.shape)
                local_extremum = vera_array[idx]
                if comparison(local_extremum, extremum):
                    extremum = local_extremum
                    indices_of_extremum = idx
                    winning_state_idx = state_idx
        elif time_scope == "current":
            vera_array = src.get_dataset(array_name)
            if vera_array is None or np.all(np.isnan(vera_array)):
                return None
            if assembly_search:
                vera_array = _assembly_scope_data(vera_array, assembly_id)
            vera_array = _nan_control_rods_pos(controls_rod_pos, vera_array)
            flat_idx = extremum_arg_finder(vera_array)
            indices_of_extremum = np.unravel_index(flat_idx, vera_array.shape)
            winning_state_idx = src.active_state_index
        return (indices_of_extremum, winning_state_idx)

    @state.change("has_data")
    def _reset_locate_source(has_data, **kwargs):
        if has_data and state.locate_source_file is None:
            state.locate_source_file = registry.default_src_id
            state.locate_source_array = "pin_powers"
            state.locate_source_label = format_label(registry.default_src_id, "pin_powers")

    @ctrl.set("set_locate_source")
    def set_locate_source(file, array):
        state.locate_source_file = file
        state.locate_source_array = array
        state.locate_source_label = format_label(file, array)

    @ctrl.set("locate_extremum")
    def locate_extremum(mode):
        """Find the max or min of the selected dataset and move the views to it.

        mode: "max" or "min".
        """
        state.locate_error = ""
        try:
            src = registry.get(state["locate_source_file"])
            array_name = state["locate_source_array"]
            assembly_scope = state["locate_assembly_scope"]  # "all" | "current"
            time_scope = state["locate_time_scope"]  # "all" | "current"

            py = state["selected_j"]
            px = state["selected_i"]
            ax = state["selected_layer"]
            assy_id = state["selected_assembly"]

            is_assembly_scoped = assembly_scope == "current"

            if mode not in ("max", "min"):
                raise RuntimeError(f"Unknown mode: {mode}. Cannot find extremum for this mode.")

            vera_array = src.get_dataset(array_name)
            vera_dtype = vera_array.dataset_type

            if vera_array is None:
                raise RuntimeError(f"Cannot find {array_name} in VERA data source.")

            # FIXME vvvvv, implement max/min for other veradtypes
            if vera_dtype not in (VeraDtype.PIN, VeraDtype.CHANNEL, VeraDtype.ASSEMBLY):
                raise RuntimeError(f"Max/min not implemented for dataset type {vera_dtype}.")

            indices, state_idx = _find_extremum(
                mode, time_scope, src, array_name, is_assembly_scoped, assy_id
            )

            if indices is None or state_idx is None:
                state.locate_error = "No valid values found in the selected dataset."
                return

            match vera_dtype:
                case VeraDtype.PIN | VeraDtype.CHANNEL:
                    if is_assembly_scoped:
                        py, px, ax = indices
                    else:
                        py, px, ax, assy_id = indices
                case VeraDtype.ASSEMBLY:
                    if is_assembly_scoped:
                        ax = indices
                    else:
                        ax, assy_id = indices
            state["selected_j"] = int(py)
            state["selected_i"] = int(px)
            state["selected_layer"] = int(ax)
            state["selected_assembly"] = int(assy_id)
            if vera_dtype.is_computational():
                ctrl.sync_comp_assembly(int(assy_id))
            else:
                ctrl.sync_core_assembly(int(assy_id))
            state["selected_time"] = int(state_idx)
            state.show_locate_dialog = False
        except Exception as e:
            state.locate_error = str(e)


def build_locate_dialog(state, ctrl, registry):
    with vuetify.VDialog(v_model=("show_locate_dialog",), max_width=560, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Locate Extremum", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div(
                        "1. Select Dataset",
                        classes="text-caption font-weight-medium mb-2",
                    )
                    build_dataset_picker(
                        ctrl,
                        "locate_source_label",
                        "set_locate_source",
                        "[src, entry.value]",
                    )

                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div(
                        "2. Assembly Scope",
                        classes="text-caption font-weight-medium mb-2",
                    )
                    with vuetify.VRadioGroup(
                        v_model=("locate_assembly_scope",),
                        row=True,
                        hide_details=True,
                        dense=True,
                        mandatory=True,
                        classes="ma-0",
                    ):
                        vuetify.VRadio(label="All Assemblies", value="all")
                        vuetify.VRadio(label="Current Assembly", value="current")

                with vuetify.VCard(outlined=True, classes="pa-3"):
                    html.Div(
                        "3. State Point Scope",
                        classes="text-caption font-weight-medium mb-2",
                    )
                    with vuetify.VRadioGroup(
                        v_model=("locate_time_scope",),
                        row=True,
                        hide_details=True,
                        dense=True,
                        mandatory=True,
                        classes="ma-0",
                    ):
                        vuetify.VRadio(label="All State Points", value="all")
                        vuetify.VRadio(label="Current State Point", value="current")

                vuetify.VAlert(
                    "{{ locate_error }}",
                    v_show=("locate_error",),
                    type="error",
                    dense=True,
                    text=True,
                    classes="mt-3 mb-0",
                )

            vuetify.VDivider()
            with vuetify.VCardActions():
                with vuetify.VBtn(color="primary", click=(ctrl.locate_extremum, "['max']")):
                    vuetify.VIcon("mdi-arrow-up-bold", left=True)
                    html.Span("Find Maximum")
                with vuetify.VBtn(text=True, click=(ctrl.locate_extremum, "['min']")):
                    vuetify.VIcon("mdi-arrow-down-bold", left=True)
                    html.Span("Find Minimum")
                vuetify.VSpacer()
                vuetify.VBtn("Close", text=True, click="show_locate_dialog = false")
