from trame.widgets import html, vuetify
from trame_server.core import State, Controller

from vera_core.app.core import VeraDataRegistry
from ..helpers import format_label
from .FileMenu import build_file_dataset_picker


def register_threshold_state_ctrl(state : State, ctrl : Controller, registry: VeraDataRegistry):
    state.show_threshold_dialog = False
    state.thresholds = {}
    state.threshold_file = "pin_powers"
    state.threshold_file = registry.default
    state.threshold_file = format_label(registry.default, "pin_powers")
    state.threshold_value = None
    state.threshold_error = ""
    state.threshold_operator = ">"

    @state.change("has_data")
    def update_thres_label(has_data, **kwargs):
        if has_data and state.threshold_file is None:
            state.threshold_file = registry.default
            state.threshold_file = "pin_powers"
            state.threshold_file = format_label(registry.default, "pin_powers")

    @ctrl.set("set_threshold")
    def set_threshold(file : str, array : str):
        print("in set threshold")
        state.threshold_array = array
        state.threshold_file = file
        state.threshold_label = format_label(file, array)

    @ctrl.set("add_threshold")
    def add_threshold():
        print("in_add_threshold")
        name = format_label(state.threshold_file, state.threshold_array)
        print(name, float(state.threshold_value))
        entry = {"op": state.threshold_operator, "value": float(state.threshold_value)}
        existing = state.thresholds.get(name, [])
        state.thresholds = {**state.thresholds, name: [*existing, entry]}
        state.threshold_value = None
        state.threshold_error = ""

    @ctrl.set("remove_threshold")
    def remove_threshold(name : str, index : int):
        remaining = [c for i, c in enumerate(state.thresholds.get(name, [])) if i != index]
        if remaining:
            state.thresholds = {**state.thresholds, name: remaining}
        else:
            state.thresholds = {k: v for k, v in state.thresholds.items() if k != name}

def build_threshold_dialog(ctrl : Controller):
    with vuetify.VDialog(v_model=("show_threshold_dialog",), max_width=480, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Dataset Thresholds", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                with html.Div(classes="d-flex align-center", style="gap: 8px;"):
                    build_file_dataset_picker(ctrl, "threshold_label", "set_threshold", "[file, entry.value]")
                    with vuetify.VCol(cols="auto", classes="pl-2"):
                        vuetify.VSelect(
                            v_model=("threshold_operator",),
                            items=("threshold_operators", [">", ">=", "<", "<=", "==", "!="]),
                            hide_details=True,
                            dense=True,
                            style="width: 80px",
                        )
                        
                    with vuetify.VCol(cols="auto", classes="pl-2"):
                        vuetify.VTextField(
                            v_model=("threshold_value",),
                            label="Value",
                            type="number",
                            hide_details=True,
                            dense=True,
                            style="width: 110px",
                        )
                    with vuetify.VCol(cols="auto", classes="pl-2"):
                        with vuetify.VBtn(icon=True, click=ctrl.add_threshold):
                            vuetify.VIcon("mdi-plus")

                vuetify.VAlert(
                    "{{ threshold_error }}",
                    v_show=("threshold_error",),
                    type="error",
                    dense=True,
                    text=True,
                    classes="mt-3 mb-0",
                )

                vuetify.VDivider(classes="my-3")

                html.Div(
                    "No thresholds set.",
                    v_show=("Object.keys(thresholds).length === 0",),
                    classes="text-caption text--secondary",
                )
                with vuetify.VList(dense=True, v_show=("Object.keys(thresholds).length > 0",)):
                    with html.Template(v_for="(condition_list, name) in thresholds", key="name"):
                        vuetify.VSubheader("{{ name }}", classes="px-2", style="height: 24px;")
                        with vuetify.VListItem(
                            v_for="(condition, index) in condition_list",
                            key="index",
                        ):
                            with vuetify.VListItemContent():
                                vuetify.VListItemTitle("{{ condition.op }} {{ condition.value }}")
                            with vuetify.VListItemAction():
                                with vuetify.VBtn(
                                    icon=True, x_small=True,
                                    click=(ctrl.remove_threshold, "[name, index]"),
                                ):
                                    vuetify.VIcon("mdi-close", small=True)

                vuetify.VDivider()
                with vuetify.VCardActions():
                    vuetify.VSpacer()
                    vuetify.VBtn("Close", text=True, click="show_threshold_dialog = false")
