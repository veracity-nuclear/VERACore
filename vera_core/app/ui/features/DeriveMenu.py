from dataclasses import asdict
from trame.widgets import html, vuetify
from trame_server.core import State, Controller
from vera_core.app.core import VeraDataRegistry, VeraAxes, DerivationMethod, derive_recipe
from .DatasetPicker import build_dataset_picker, refresh_src_tree
from ..helpers import format_label

"""
Commented out derivation presets and methods need to be implemented
"""

DERIVATION_PRESETS = [
    {"text": "Assembly", "value": "ASSEMBLY"},
    {"text": "Axial", "value": "AXIAL"},
    {"text": "Radial", "value": "RADIAL"},
    {"text": "Core", "value": "CORE"},
    # {"text": "Node", "value": "NODE"},
    {"text": "Radial Assembly", "value": "RADIAL_ASSEMBLY"},
    # {"text": "Radial Node", "value": "RADIAL_NODE"},
]

DERIVATION_METHODS = [method.value for method in DerivationMethod] # list of derivations methods in str

def register_derived_state_ctrl(state : State, ctrl : Controller, registry: VeraDataRegistry):
    """
    Register trame state for derived menu

    state.show_derived_dialog : show flag for derived menu UI
    state.derivation_src_dataset : name of h5 dataset to run derivation on
    state.derivation_src_id : the id of the source in registry to pull dataset from (two or more sources could have datasets with same name)
    state.axes_to_derive : the axes the derived dataset will have i.e. takes pin_powers -> assembly averaged power
    state.derivation_source_label : label to show on dataset picker after dataset is selected
    state.derivation_use_factors : FIXME state that will track whether to use factors in derivation or not, not 
        currently implemented, here so UI matches veraview
    state.derivation_exclude_non_fuel : FIXME similar to state.derivation_use_factors
    state.derived_name : state the stores what the name of the derived dataset will be
    state.derived_error : state the stores any error/exception messages generated in the derivation process
    state.derivation_axes_presets : state the stores preconfigured mapping from user selection to enum
    state.derivation_methods : state the stores preconfigured derivation method options i.e. "AVERAGE"
    """
    state.show_derived_dialog = False
    state.derivation_src_dataset = "pin_powers"
    state.derivation_src_id = registry.default_src_id
    state.derivation_source_label = format_label(registry.default_src_id, "pin_powers")
    state.axes_to_derive = "ASSEMBLY"
    state.derivation_method = "Average"
    state.derivation_use_factors = True
    state.derivation_exclude_non_fuel = False
    state.derived_name = ""
    state.derived_error = ""
    state.derivation_axes_presets = DERIVATION_PRESETS
    state.derivation_methods = DERIVATION_METHODS

    @state.change("has_data")
    def update_label(has_data, **kwargs):
        if has_data and state.derivation_src_id is None:
            state.derivation_src_id = registry.default_src_id
            state.derivation_src_dataset = "pin_powers"
            state.derivation_source_label = format_label(registry.default_src_id, "pin_powers")

    @ctrl.set("set_derived_source")
    def set_derived_source(file, array):
        print("in_set_derived_source")
        state.derivation_src_id = file
        state.derivation_src_dataset = array
        state.derivation_source_label = format_label(file, array)
    
    def _add_derived_dataset_to_source():
        """add a new derived dataset to selected source"""
        recipe = derive_recipe(
            src_id=state["derivation_src_id"],
            source_array=state["derivation_src_dataset"],
            name=state["derived_name"],
            method=state["derivation_method"],
            axes=state["axes_to_derive"],
        )
        registry.apply_recipe(recipe)
        state.recipes = state.recipes + [recipe]
        refresh_src_tree(state, registry)

    @ctrl.set("create_derived_dataset")
    def create_derived_dataset():
        try:
            _add_derived_dataset_to_source()
        except Exception as e:
            state.derived_error = str(e)

    @ctrl.set("create_derived_dataset_and_close")
    def create_derived_dataset_and_close():
        try:
            _add_derived_dataset_to_source()
            state.show_derived_dialog = False
            state.derivation_src_dataset = "pin_powers"
            state.derivation_src_id = registry.default_src_id
            state.derivation_source_label = format_label(registry.default_src_id, "pin_powers")
        except Exception as e:
            state.derived_error = str(e)


def build_derived_dialog(state, ctrl, registry):
    with vuetify.VDialog(v_model=("show_derived_dialog",), max_width=620, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Create Derived Dataset", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("1. Select Dataset", classes="text-caption font-weight-medium mb-2")
                    build_dataset_picker(ctrl, "derivation_source_label", "set_derived_source", "[src, entry.value]")
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("2. Select Axes Over Which to Derive", classes="text-caption font-weight-medium mb-2")
                    vuetify.VSelect(
                        v_model=("axes_to_derive",),
                        items=("derivation_axes_presets",),
                        hide_details=True,
                        dense=True,
                    )
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("3. Select Derivation Method", classes="text-caption font-weight-medium mb-2")
                    with vuetify.VRow(classes="ma-0", align="center"):
                        with vuetify.VCol(cols="4", classes="pa-0"):
                            vuetify.VSelect(
                                v_model=("derivation_method",),
                                items=("derivation_methods",),
                                hide_details=True,
                                dense=True,
                            )
                        with vuetify.VCol(cols="auto", classes="pa-0 pl-4"):
                            vuetify.VCheckbox(
                                v_model=("derivation_use_factors",),
                                label="Use Factors",
                                hide_details=True,
                                dense=True,
                            )
                        with vuetify.VCol(cols="auto", classes="pa-0 pl-4"):
                            vuetify.VCheckbox(
                                v_model=("derivation_exclude_non_fuel",),
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
