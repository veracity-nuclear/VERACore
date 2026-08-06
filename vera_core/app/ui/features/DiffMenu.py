from trame.widgets import html, vuetify

from vera_core.data.dtypes import diff_recipe
from vera_core.data.registry import VeraDataRegistry

from ..helpers import format_label
from .DatasetPicker import build_dataset_picker, refresh_src_tree

DEGREE = {"Linear": 1, "Quadratic": 2, "Cubic": 3}
INTERPOLATION_OPTIONS = [key for key in DEGREE]


def register_diff_state_ctrl(state, ctrl, registry: VeraDataRegistry):
    """
    Register trame state for diff menu
    """
    state.show_diff_dialog = False
    state.diff_name = ""
    state.ref_dataset_name = ""
    state.ref_src_id = None
    state.ref_label = "Select dataset"
    state.ref_units = ""

    state.comp_dataset_name = ""
    state.comp_src_id = None
    state.comp_label = "Select dataset"
    state.comp_units = ""

    state.ref_shape = ""
    state.comp_shape = ""
    state.diff_compatible = True
    state.diff_shape_error = ""

    state.ref_scale = 1.0
    state.comp_scale = 1.0
    state.diff_units = ""

    state.diff_operator = "-"
    state.diff_interp_kind = "Linear"
    state.diff_interp_kinds = INTERPOLATION_OPTIONS
    state.diff_error = ""

    def _units_of(src_id, dataset_name):
        src = registry.get(src_id)
        if src is None or not dataset_name:
            return ""
        return src.get_dataset_units(dataset_name)

    def _ds_info(src_id, dataset_name):
        src = registry.get(src_id)
        if src is None or not dataset_name:
            return None
        return src.get_dataset_shape(dataset_name), src.get_dataset_dtype(dataset_name)

    def _drop_axis(shape, axis):
        if axis is None:
            return tuple(shape)
        return tuple(s for k, s in enumerate(shape) if k != axis)

    def _recompute_compat():
        ref = _ds_info(state.ref_src_id, state.ref_dataset_name)
        comp = _ds_info(state.comp_src_id, state.comp_dataset_name)
        if ref is None or comp is None:
            state.diff_compatible = True
            state.diff_shape_error = ""
            return
        ref_shape, ref_dtype = ref
        comp_shape, comp_dtype = comp
        state.ref_shape = "" if not ref_shape else str(ref_shape)
        state.comp_shape = "" if not comp_shape else str(comp_shape)

        if ref_dtype != comp_dtype:
            state.diff_compatible = False
            state.diff_shape_error = (
                f"Type mismatch: {ref.dataset_type} vs {comp.dataset_type}. "
                "Datasets must be the same type to diff."
            )
            return

        # allow axial to differ (interpolation handles it). Everything else must match
        ax = ref_dtype.axial_dim_idx if ref_dtype.has_axial_dim() else None
        r = _drop_axis(ref_shape, ax)
        c = _drop_axis(comp_shape, ax)
        if r != c:
            state.diff_compatible = False
            state.diff_shape_error = (
                f"Shape mismatch outside the axial dimension: {r} vs {c}. "
                "Only axial differences can be reconciled by interpolation."
            )
            return

        state.diff_compatible = True
        state.diff_shape_error = ""

    @ctrl.set("set_ref_dataset")
    def set_ref_datset(src_id, dataset_name):
        state.ref_src_id = src_id
        state.ref_dataset_name = dataset_name
        state.ref_label = format_label(src_id, dataset_name)
        state.ref_units = _units_of(src_id, dataset_name)
        state.ref_shape = str((_ds_info(src_id, dataset_name) or (tuple(),))[0])
        _recompute_compat()

    @ctrl.set("set_comp_dataset")
    def set_comp_datset(src_id, dataset_name):
        state.comp_src_id = src_id
        state.comp_dataset_name = dataset_name
        state.comp_label = format_label(src_id, dataset_name)
        state.comp_units = _units_of(src_id, dataset_name)
        state.ref_shape = str((_ds_info(src_id, dataset_name) or (tuple(),))[0])
        _recompute_compat()

    @ctrl.set("create_diff_dataset")
    def create_diff_dataset():
        ref_src = registry.get(state.ref_src_id)
        comp_src = registry.get(state.comp_src_id)
        if ref_src is None or state.ref_dataset_name == "":
            state.diff_error = "Please select a reference dataset"
            return
        if comp_src is None or state.comp_dataset_name == "":
            state.diff_error = "Please select a comparison dataset"
            return
        if state.diff_name == "":
            state.diff_error = "Please select a name for the new dataset"
            return
        if not state.diff_compatible:
            state.diff_error = state.diff_shape_error or "Datasets are not compatible."
            return
        try:
            ref_scale = float(state.ref_scale)
            comp_scale = float(state.comp_scale)
        except (TypeError, ValueError):
            state.diff_error = "Scale factors must be numbers"
            return
        try:
            recipe = diff_recipe(
                ref_src_id=state.ref_src_id,
                ref_array=state.ref_dataset_name,
                comp_src_id=state.comp_src_id,
                comp_array=state.comp_dataset_name,
                name=state.diff_name,
                interp_degree=DEGREE[state.diff_interp_kind],
                ref_scale=ref_scale,
                comp_scale=comp_scale,
                units=state.diff_units or "unitless",
            )
            registry.apply_recipe(recipe)
            state.recipes = state.recipes + [recipe]
            refresh_src_tree(state, registry)
            for k, v in (
                ("diff_name", ""),
                ("ref_dataset_name", ""),
                ("ref_src_id", None),
                ("ref_label", "Select dataset"),
                ("ref_units", ""),
                ("ref_scale", 1.0),
                ("comp_dataset_name", ""),
                ("comp_src_id", None),
                ("comp_label", "Select dataset"),
                ("comp_units", ""),
                ("comp_scale", 1.0),
                ("diff_units", ""),
                ("diff_error", ""),
                ("ref_shape", ""),
                ("comp_shape", ""),
            ):
                state[k] = v
        except Exception as e:
            state.diff_error = str(e)


def build_diff_dialog(state, ctrl, registry):
    with vuetify.VDialog(v_model=("show_diff_dialog",), max_width=680, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Create Diff Dataset", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                vuetify.VTextField(
                    v_model=("diff_name",),
                    label="Name",
                    hide_details=True,
                    dense=True,
                    classes="mb-4",
                )

                # Reference operand
                with vuetify.VCard(outlined=True, classes="pa-3 mb-2"):
                    html.Div("Reference (A)", classes="text-caption font-weight-medium mb-2")
                    with html.Div(classes="d-flex align-center", style="gap: 12px;"):
                        with html.Div(style="flex: 1 1 auto; min-width: 0;"):
                            build_dataset_picker(
                                ctrl,
                                "ref_label",
                                "set_ref_dataset",
                                "[src, entry.value]",
                            )
                        vuetify.VTextField(
                            v_model=("ref_scale",),
                            label="× scale",
                            type="number",
                            hide_details=True,
                            dense=True,
                            style="max-width: 110px; flex: 0 0 auto;",
                        )
                    html.Div(
                        "{{ 'Units: ' + ref_units }}",
                        classes="text-caption text--secondary mt-1",
                        v_show=("ref_units",),
                    )
                    html.Div(
                        "{{ 'Shape: ' + ref_shape }}",
                        classes="text-caption text--secondary",
                        v_show=("ref_shape",),
                    )

                # Operator joint
                with html.Div(classes="d-flex justify-center align-center my-1"):
                    vuetify.VSelect(
                        v_model=("diff_operator",),
                        items=("diff_operators", ["+", "-", "*", "/"]),
                        hide_details=True,
                        dense=True,
                        solo=True,
                        flat=True,
                        style="max-width: 80px;",
                        classes="text-h6",
                    )

                # Comparison operand
                with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                    html.Div("Comparison (B)", classes="text-caption font-weight-medium mb-2")
                    with html.Div(classes="d-flex align-center", style="gap: 12px;"):
                        with html.Div(style="flex: 1 1 auto; min-width: 0;"):
                            build_dataset_picker(
                                ctrl,
                                "comp_label",
                                "set_comp_dataset",
                                "[src, entry.value]",
                            )
                        vuetify.VTextField(
                            v_model=("comp_scale",),
                            label="× scale",
                            type="number",
                            hide_details=True,
                            dense=True,
                            style="max-width: 110px; flex: 0 0 auto;",
                        )
                    html.Div(
                        "{{ 'Units: ' + comp_units }}",
                        classes="text-caption text--secondary mt-1",
                        v_show=("comp_units",),
                    )
                    html.Div(
                        "{{ 'Shape: ' + comp_shape }}",
                        classes="text-caption text--secondary",
                        v_show=("comp_shape",),
                    )

                # Shape / type incompatibility (blocks submission)
                vuetify.VAlert(
                    "{{ diff_shape_error }}",
                    v_show=("diff_shape_error",),
                    type="warning",
                    dense=True,
                    text=True,
                    classes="mb-3",
                )

                vuetify.VDivider(classes="mb-4")

                # Output options (describe the result, not the operands)
                with html.Div(classes="d-flex align-center", style="gap: 16px;"):
                    vuetify.VSelect(
                        v_model=("diff_interp_kind",),
                        items=("diff_interp_kinds",),
                        label="Interpolation",
                        hide_details=True,
                        dense=True,
                        style="max-width: 200px;",
                    )
                    vuetify.VTextField(
                        v_model=("diff_units",),
                        label="Output units",
                        hide_details=True,
                        dense=True,
                        placeholder="e.g. Celsius",
                    )

                vuetify.VAlert(
                    "{{ diff_error }}",
                    v_show=("diff_error",),
                    type="error",
                    dense=True,
                    text=True,
                    classes="mt-4 mb-0",
                )
            vuetify.VDivider()
            with vuetify.VCardActions(classes="px-4 py-3"):
                vuetify.VSpacer()
                vuetify.VBtn("Close", text=True, click="show_diff_dialog = false")
                vuetify.VBtn(
                    "Create",
                    color="primary",
                    disabled=("!diff_compatible",),
                    click=ctrl.create_diff_dataset,
                )
