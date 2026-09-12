from trame.widgets import html, vuetify
from trame_server.core import Controller, State

from vera_core.app.core import VeraDataRegistry
from vera_core.app.core.thresholds import ThresholdCondition

from ..helpers import format_label
from .DatasetPicker import build_dataset_picker


def register_threshold_state_ctrl(state: State, ctrl: Controller, registry: VeraDataRegistry):
    """
    Register trame state for threshold menu

    state.thresholds : state that stores thresholds (an operator and value) for a specific dataset from a specific source
        views read this state and apply threshold if the view's visualized dataset has a threshold set
    state.threshold_dataset : state that stores name of selected dataset to apply threshold dataset for
    state.threshold_src_id : state that stores source id of the selected dataset
    state.threshold_label : formatted label of selected source and dataset to display
    state.threshold_value : numerical value of the threshold to be applieds
    state.threshold_operator : state that stores selected comparison operator to use in the threshold
    state.threshold_operators : state the stores predefined list of comparison operators that can be selected and used by user
    state.threshold_error : state the stores any error/exception messages generated in the threshold creation process.
    """
    state.show_threshold_dialog = False
    state.thresholds = {}
    state.threshold_dataset = ""
    state.threshold_src_id = None
    state.threshold_label = "Select dataset"
    state.threshold_value = None
    state.threshold_error = ""
    state.threshold_operator = ">"
    state.threshold_operators = [">", ">=", "<", "<=", "==", "!="]

    @ctrl.set("set_threshold")
    def set_threshold(file: str, array: str):
        state.threshold_dataset = array
        state.threshold_src_id = file
        state.threshold_label = format_label(file, array)
        state.threshold_error = ""

    @ctrl.set("add_threshold")
    def add_threshold():
        try:
            if registry.get(state.threshold_src_id) is None or state.threshold_dataset == "":
                state.threshold_error = "Please select a dataset to threshold first"
                return
            if state.threshold_value is None:
                state.threshold_error = "Please select a threshold value"
                return
            name = format_label(
                state.threshold_src_id, state.threshold_dataset
            )  # use src_id and dataset name as key for threshold
            entry: ThresholdCondition = {
                "op": state.threshold_operator,
                "value": float(state.threshold_value),
            }
            existing = state.thresholds.get(name, [])
            state.thresholds = {
                **state.thresholds,
                name: [*existing, entry],
            }  # reconstruct to trigger trame state change
            state.threshold_value = None
            state.threshold_src_id = None
            state.threshold_label = "Select dataset"
            state.threshold_dataset = ""
            state.threshold_error = ""
        except Exception as e:
            state.threshold_error = str(e)

    @ctrl.set("remove_threshold")
    def remove_threshold(name: str, index: int):
        remaining = [c for i, c in enumerate(state.thresholds.get(name, [])) if i != index]
        if remaining:
            state.thresholds = {**state.thresholds, name: remaining}
        else:
            state.thresholds = {k: v for k, v in state.thresholds.items() if k != name}


def build_threshold_dialog(ctrl: Controller):
    with vuetify.VDialog(v_model=("show_threshold_dialog",), max_width=580, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Dataset Thresholds", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                with html.Div(classes="d-flex align-center", style="gap: 8px;"):
                    with html.Div(style="flex: 1 1 auto; min-width: 0;"):
                        build_dataset_picker(
                            ctrl,
                            "threshold_label",
                            "set_threshold",
                            "[src, entry.value]",
                        )
                    vuetify.VSelect(
                        v_model=("threshold_operator",),
                        items=(
                            "threshold_operators",
                            [">", ">=", "<", "<=", "==", "!="],
                        ),
                        hide_details=True,
                        dense=True,
                        style="flex: 0 0 72px; width: 72px;",
                    )
                    vuetify.VTextField(
                        v_model=("threshold_value",),
                        label="Value",
                        type="number",
                        hide_details=True,
                        dense=True,
                        style="flex: 0 0 90px; width: 90px;",
                    )
                    with vuetify.VBtn(
                        icon=True,
                        small=True,
                        click=ctrl.add_threshold,
                        style="flex: 0 0 auto;",
                    ):
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
                                    icon=True,
                                    x_small=True,
                                    click=(ctrl.remove_threshold, "[name, index]"),
                                ):
                                    vuetify.VIcon("mdi-close", small=True)

                vuetify.VDivider()
                with vuetify.VCardActions():
                    vuetify.VSpacer()
                    vuetify.VBtn("Close", text=True, click="show_threshold_dialog = false")
