from trame.widgets import vuetify
from vera_core.app.core import VeraDataRegistry
from ..helpers import refresh_file_tree

def register_diff_state_ctrl(state, ctrl, registry: VeraDataRegistry):
    state.setdefault("show_diff_dialog", False)
    state.setdefault("diff_name", "")
    state.setdefault("diff_array_a", "pin_powers")
    state.setdefault("diff_operator", "-")
    state.setdefault("diff_array_b", "pin_powers")
    state.setdefault("diff_error", "")
    @ctrl.set("create_diff_dataset")
    def create_diff_dataset():
        state.show_diff_dialog = False
        registry.default_source.add_new_diff_dataset(state["diff_array_a"], state["diff_array_b"], state["diff_name"])
        refresh_file_tree(state, registry)

def build_diff_dialog(state, ctrl, registry):
    with vuetify.VDialog(v_model=("show_diff_dialog",), max_width=480, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Create diff Dataset", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                vuetify.VTextField(
                    v_model=("diff_name",),
                    label="Name",
                    hide_details=True,
                    dense=True,
                    classes="mb-3",
                )
                with vuetify.VRow(classes="ma-0", align="center"):
                    with vuetify.VCol(classes="pa-0"):
                        vuetify.VSelect(
                            v_model=("diff_array_a",),
                            items=("available_arrays",),
                            label="Array A",
                            hide_details=True,
                            dense=True,
                        )
                    with vuetify.VCol(cols="auto", classes="px-2"):
                        vuetify.VSelect(
                            v_model=("diff_operator",),
                            items=("diff_operators", ["+", "-", "*", "/"]),
                            hide_details=True,
                            dense=True,
                            style="width: 64px",
                        )
                    with vuetify.VCol(classes="pa-0"):
                        vuetify.VSelect(
                            v_model=("diff_array_b",),
                            items=("available_arrays",),
                            label="Array B",
                            hide_details=True,
                            dense=True,
                        )
                vuetify.VAlert(
                    "{{ diff_error }}",
                    v_show=("diff_error",),
                    type="error",
                    dense=True,
                    text=True,
                    classes="mt-3 mb-0",
                )
            vuetify.VDivider()
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn("Cancel", text=True, click="show_diff_dialog = false")
                vuetify.VBtn("Create", color="primary", click=ctrl.create_diff_dataset)

