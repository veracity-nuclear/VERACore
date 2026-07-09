from trame.widgets import vuetify, html
from vera_core.app.core import VeraDataRegistry, VeraDataSource , VeraDtype, derive_recipe, diff_recipe
from .DatasetPicker import refresh_src_tree, build_dataset_picker
from ..helpers import format_label
from scipy.interpolate import make_interp_spline

DEGREE = {'Linear': 1, 'Quadratic': 2, 'Cubic': 3}
INTERPOLATION_OPTIONS = [key for key in DEGREE]

def register_diff_state_ctrl(state, ctrl, registry: VeraDataRegistry):
    """
    Register trame state for diff menu
    """
    state.show_diff_dialog = False
    state.diff_name = ""
    state.ref_dataset_name = "pin_powers"
    state.ref_src_id = registry.default_src_id
    state.ref_label = format_label(registry.default_src_id, "pin_powers")
    state.diff_operator = "-"
    state.comp_dataset_name = "pin_powers"
    state.comp_src_id = registry.default_src_id
    state.comp_label = format_label(registry.default_src_id, "pin_powers")
    state.diff_interp_kind = "Linear"
    state.diff_interp_kinds = INTERPOLATION_OPTIONS
    state.diff_error = ""

    @ctrl.set("set_ref_dataset")
    def set_ref_datset(src_id, dataset_name):
        state.ref_src_id = src_id
        state.ref_dataset_name = dataset_name
        state.ref_label = format_label(src_id, dataset_name)
    
    @ctrl.set("set_comp_dataset")
    def set_comp_datset(src_id, dataset_name):
        state.comp_src_id = src_id
        state.comp_dataset_name = dataset_name
        state.comp_label = format_label(src_id, dataset_name)

    @ctrl.set("create_diff_dataset")
    def create_diff_dataset():
        ref_src = registry.get(state.ref_src_id)
        comp_src = registry.get(state.comp_src_id)
        if ref_src is None or comp_src is None:
            state.diff_error = "Could not find source"
            return
        try:
            recipe = diff_recipe(
                ref_src_id=state.ref_src_id, ref_array=state.ref_dataset_name,
                comp_src_id=state.comp_src_id, comp_array=state.comp_dataset_name,
                name=state.diff_name, interp_degree=DEGREE[state.diff_interp_kind],
            )
            registry.apply_recipe(recipe)
            state.recipes = state.recipes + [recipe]
            refresh_src_tree(state, registry)
            state.show_diff_dialog = False
        except Exception as e:
            state.diff_error = str(e)

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
                        build_dataset_picker(ctrl, "ref_label", "set_ref_dataset", "[src, entry.value]")
                    with vuetify.VCol(cols="auto", classes="px-2"):
                        vuetify.VSelect(
                            v_model=("diff_operator",),
                            items=("diff_operators", ["+", "-", "*", "/"]),
                            hide_details=True,
                            dense=True,
                            style="width: 64px",
                        )
                    with vuetify.VCol(classes="pa-0"):
                        build_dataset_picker(ctrl, "comp_label", "set_comp_dataset", "[src, entry.value]")
                with vuetify.VRow(classes="ma-0 mt-3", align="center"):
                    with vuetify.VCol(cols="auto", classes="pa-0 pr-3"):
                        html.Div("Interpolation", classes="text-caption")
                    with vuetify.VCol(classes="pa-0"):
                        vuetify.VSelect(
                            v_model=("diff_interp_kind",),
                            items=("diff_interp_kinds",),
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

