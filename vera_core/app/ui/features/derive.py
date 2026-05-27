from trame.widgets import html, vuetify
from vera_core.app.core import VeraDataRegistry, VeraDerivation
from ..helpers import format_label, build_dataset_picker, refresh_file_tree

DERIVATION_PRESETS = [
    {"text": "Assembly", "value": "ASSEMBLY"},
    {"text": "Axial", "value": "AXIAL"},
    {"text": "Radial", "value": "RADIAL"},
    {"text": "Core", "value": "CORE"},
    {"text": "Node", "value": "NODE"},
    {"text": "Radial Assembly", "value": "RADIAL_ASSEMBLY"},
    {"text": "Radial Node", "value": "RADIAL_NODE"},
]
DERIVATION_METHODS = ["Average", "Sum", "Min", "Max"]  # placeholder

def register_derived_state_ctrl(state, ctrl, registry: VeraDataRegistry):
    state.show_derived_dialog = False
    state.derived_source_array = "pin_powers"
    state.derived_source_file = registry.default
    state.derived_source_label = format_label(registry.default, "pin_powers")
    state.derived_preset = "ASSEMBLY"
    state.derived_method = "Average"
    state.derived_use_factors = True
    state.derived_exclude_non_fuel = False
    state.derived_name = ""
    state.derived_error = ""
    state.derivation_presets = DERIVATION_PRESETS
    state.derivation_methods = DERIVATION_METHODS

    @ctrl.set("set_derived_source")
    def set_derived_source(file, array):
        print("in_set_derived_source")
        state.derived_source_file = file
        state.derived_source_array = array
        state.derived_source_label = format_label(file, array)

    @ctrl.set("create_derived_dataset")
    def create_derived_dataset():
        try:
            registry.get(state["derived_source_file"]).add_new_derived_dataset(state["derived_source_array"], state["derived_name"], VeraDerivation[state["derived_preset"]])
            refresh_file_tree(state, registry)
        except Exception as e:
            print(e)
            state.derived_error = "Something went wrong"

    @ctrl.set("create_derived_dataset_and_close")
    def create_derived_dataset_and_close():
        try:
            registry.get(state["derived_source_file"]).add_new_derived_dataset(state["derived_source_array"], state["derived_name"], VeraDerivation[state["derived_preset"]])
            refresh_file_tree(state, registry)
            state.show_derived_dialog = False
            state.derived_source_array = "pin_powers"
            state.derived_source_file = registry.default
            state.derived_source_label = format_label(registry.default, "pin_powers")
        except Exception as e:
            print(e)
            state.derived_error = "Something went wrong"

def build_derived_dialog(state, ctrl, registry):
    with vuetify.VDialog(v_model=("show_derived_dialog",), max_width=620, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Create Derived Dataset", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("1. Select Dataset", classes="text-caption font-weight-medium mb-2")
                    build_dataset_picker(ctrl, "derived_source_label", "set_derived_source", "[file, entry.value]")
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("2. Select Axes Over Which to Derive", classes="text-caption font-weight-medium mb-2")
                    vuetify.VSelect(
                        v_model=("derived_preset",),
                        items=("derivation_presets",),
                        hide_details=True,
                        dense=True,
                    )
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("3. Select Derivation Method", classes="text-caption font-weight-medium mb-2")
                    with vuetify.VRow(classes="ma-0", align="center"):
                        with vuetify.VCol(cols="4", classes="pa-0"):
                            vuetify.VSelect(
                                v_model=("derived_method",),
                                items=("derivation_methods",),
                                hide_details=True,
                                dense=True,
                            )
                        with vuetify.VCol(cols="auto", classes="pa-0 pl-4"):
                            vuetify.VCheckbox(
                                v_model=("derived_use_factors",),
                                label="Use Factors",
                                hide_details=True,
                                dense=True,
                            )
                        with vuetify.VCol(cols="auto", classes="pa-0 pl-4"):
                            vuetify.VCheckbox(
                                v_model=("derived_exclude_non_fuel",),
                                label="Exclude Non-Fuel Rods",
                                hide_details=True,
                                dense=True,
                            )
                with vuetify.VCard(outlined=True, classes="pa-3"):
                    html.Div("4. Enter New Dataset Name", classes="text-caption font-weight-medium mb-2")
                    vuetify.VTextField(
                        v_model=("derived_name",),
                        hide_details=True,
                        dense=True,
                    )

                vuetify.VAlert(
                    "{{ derived_error }}",
                    v_show=("derived_error",),
                    type="error",
                    dense=True,
                    text=True,
                    classes="mt-3 mb-0",
                )

            vuetify.VDivider()
            with vuetify.VCardActions():
                vuetify.VBtn("Create", text=True, click=ctrl.create_derived_dataset)
                vuetify.VBtn("Create and Close", color="primary", click=ctrl.create_derived_dataset_and_close)
                vuetify.VSpacer()
                vuetify.VBtn("Close", text=True, click="show_derived_dialog = false")
