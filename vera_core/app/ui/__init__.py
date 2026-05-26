import numpy as np

from trame.ui.vuetify import SinglePageLayout
from trame.widgets import client, grid, html, vuetify

from vera_core.widgets import vera
from vera_core.app.core.vera_data import VeraDataRegistry, VeraDerivation
# from vera_core.app.core.thresholds import 

from . import (
    assembly_view,
    axial_plot,
    core_view,
    empty,
    table_view,
    time_plot,
    x_axial_view,
    y_axial_view,
    volume_view,
    assets,
)

DEFAULT_NB_ROWS = 8
NUM_VIEW_SLOTS = 10

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
]

def get_next_y_from_layout(layout):
    next_y = 0
    for item in layout:
        y, h = item.get("y", 0), item.get("h", 1)
        if y + h > next_y:
            next_y = y + h
    return next_y

def create_diff_menu(state, ctrl, registry: VeraDataRegistry):
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
        state.available_arrays = [dict(text=k.replace("_", " ").title(), value=k) for k in registry.default_source.active_state_full_core_keys]

def create_threshold_menu(state, ctrl, registry: VeraDataRegistry):
    state.setdefault("show_threshold_dialog", False)
    state.setdefault("thresholds", {})
    state.setdefault("threshold_array", "pin_powers")
    state.setdefault("threshold_value", None)
    state.setdefault("threshold_error", "")
    state.setdefault("threshold_operator", ">")

    @ctrl.set("add_threshold")
    def add_threshold():
        name = state.threshold_array
        entry = {"op": state.threshold_operator, "value": float(state.threshold_value)}
        existing = state.thresholds.get(name, [])
        state.thresholds = {**state.thresholds, name: [*existing, entry]}
        state.threshold_value = None
        state.threshold_error = ""

    @ctrl.set("remove_threshold")
    def remove_threshold(name, index):
        remaining = [c for i, c in enumerate(state.thresholds.get(name, [])) if i != index]
        if remaining:
            state.thresholds = {**state.thresholds, name: remaining}
        else:
            state.thresholds = {k: v for k, v in state.thresholds.items() if k != name}

DERIVATION_PRESETS = [
    {"text": "Assembly", "value": "ASSEMBLY"},
    {"text": "Axial", "value": "AXIAL"},
    {"text": "Radial", "value": "RADIAL"},
    {"text": "Core", "value": "CORE"},
    {"text": "Node", "value": "NODE"},
    {"text": "Radial Assembly", "value": "RADIAL_ASSEMBLY"},
    {"text": "Radial Node", "value": "RADIAL_NODE"},
]
DERIVATION_METHODS = ["Average", "Sum", "Min", "Max"]  # placeholder — set your real method list

def create_derived_menu(state, ctrl, registry: VeraDataRegistry):
    state.setdefault("show_derived_dialog", False)
    state.setdefault("derived_source", "pin_powers")
    state.setdefault("derived_preset", "ASSEMBLY")
    state.setdefault("derived_method", "Average")
    state.setdefault("derived_use_factors", True)
    state.setdefault("derived_exclude_non_fuel", False)
    state.setdefault("derived_name", "")
    state.setdefault("derived_error", "")
    state.setdefault("derivation_presets", DERIVATION_PRESETS)
    state.setdefault("derivation_methods", DERIVATION_METHODS)

    @ctrl.set("create_derived_dataset")
    def create_derived_dataset():
        try:
            registry.default_source.add_new_derived_dataset(state["derived_source"], state["derived_name"], VeraDerivation[state["derived_preset"]])
            state.available_arrays = [dict(text=k.replace("_", " ").title(), value=k) for k in registry.default_source.active_state_full_core_keys]
        except Exception as e:
            print(e)
            state.derived_error = "Something went wrong"

    @ctrl.set("create_derived_dataset_and_close")
    def create_derived_dataset_and_close():
        try:
            registry.default_source.add_new_derived_dataset(state["derived_source"], state["derived_name"], VeraDerivation[state["derived_preset"]])
            state.available_arrays = [dict(text=k.replace("_", " ").title(), value=k) for k in registry.default_source.active_state_full_core_keys]
            state.show_derived_dialog = False
        except Exception as e:
            print(e)
            state.derived_error = "Something went wrong"


def initialize(server, registry: VeraDataRegistry):
    state, ctrl = server.state, server.controller
    state.trame__title = "VERACore"

    state.setdefault("grid_item_dirty_key", 0)

    # FIXME: For our example, fix this to match VeraView.
    # Come up with a way to autogenerate it.
    state.color_range = (0.0273, 1.95)
    state.selected_layer = 24
    state.selected_assembly = 36
    state.selected_time = 0
    state.max_time = max(0, len(registry.default_source.states) - 1)
    # FIXME ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    create_diff_menu(state=state, ctrl=ctrl, registry=registry)
    create_threshold_menu(state=state, ctrl=ctrl, registry=registry)
    create_derived_menu(state=state, ctrl=ctrl, registry=registry)


    def _array_range(selected_file, selected_array):
        array = registry.get(selected_file).array(selected_array)
        lo = float(np.nanmin(array))
        hi = float(np.nanmax(array))
        if not np.isfinite(lo) or not np.isfinite(hi):
            return (0.0, 1.0)
        if lo == hi:
            eps = max(abs(hi) * 1e-9, 1e-12)
            return (lo, hi + eps)
        return (lo, hi)

    def _recompute_card_range(view_id):
        selected_array = state[f"selected_array_{view_id}"]
        selected_file = state[f"selected_file_{view_id}"]
        state[f"color_range_{view_id}"] = _array_range(selected_file, selected_array)
       
    @state.change("selected_time")
    def selected_time_changed(selected_time, **kwargs):
        selected_time = int(selected_time)
        registry.change_active_state(selected_time)

        ctrl.on_vera_out_active_state_index_changed(
            selected_time=selected_time, **kwargs
        )
        # Automatically normalize color scale to current state
        for view_id in all_view_ids:
            _recompute_card_range(view_id)
        # Keep the global range in sync with the toolbar selection (for volume view).
        state.color_range = _array_range(registry.default, state.selected_array)
    
    @state.change("selected_array")
    def toolbar_array_changed(selected_array, **kwargs):
        # Global color_range still drives volume view.
        state.color_range = _array_range(registry.default, selected_array)

    # Keep selected_assembly and selected_assembly_ij in sync
    @state.change("selected_assembly_ij")
    def selected_assembly_ij_changed(selected_assembly_ij, **kwargs):
        i, j = selected_assembly_ij["i"], selected_assembly_ij["j"]
        state.selected_assembly = registry.default_source.core.reduced_core_map_assembly(i, j)

    @state.change("selected_assembly")
    def selected_assembly_changed(selected_assembly, **kwargs):
        i, j = registry.default_source.core.reduced_core_map_ij(selected_assembly)
        state.selected_assembly_ij = dict(i=i, j=j)
    
    @ctrl.set("card_selected_array_changed")
    def card_selected_array_changed(view_id, selected_array):
        if not selected_array:
            return
        state[f"selected_array_{view_id}"] = selected_array

    # Initialize all visualizations
    state.setdefault("grid_layout", [])

    # Reserve the various views
    all_view_ids = [f"{v+1}" for v in range(NUM_VIEW_SLOTS)]

    for view_id in all_view_ids:
        state[f"grid_options_{view_id}"] = []
        state[f"selected_array_{view_id}"] = "pin_powers"
        state[f"selected_file_{view_id}"] = registry.default
        
        state[f"color_range_{view_id}"] = (0.0, 1.0)  
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)

        state[f"selected_label_{view_id}"] = (f"{registry.default} | {'pin_powers'.replace('_', ' ').upper()}")
        state[f"file_menu_{view_id}"] = False

        state.file_tree = {fid: [{"text": k.replace("_", " ").title(), "value": k} for k in keys]
                            for fid, keys in registry.full_core_keys().items()}
        for module in VIEW_MODULES:
            module.initialize(server, registry, view_id)

    def _make_array_watcher(view_id):
        @state.change(f"selected_array_{view_id}")
        def _on_card_array_change(**kwargs):
            _recompute_card_range(view_id)
        return _on_card_array_change

    for view_id in all_view_ids:
        _make_array_watcher(view_id)
        _recompute_card_range(view_id)  # initial population
    available_view_ids = list(all_view_ids)

    def place(module, x, y, w, h):
        view_id = available_view_ids.pop(0)
        state.grid_layout.append(dict(x=x, y=y, w=w, h=h, i=view_id))
        state[f"grid_view_{view_id}"] = module.option_for(view_id)

    place(x_axial_view,   0,  0, 3, 17)
    place(core_view,      6,  0, 3,  9)
    place(assembly_view,  9,  0, 3,  9)
    place(axial_plot,     6,  9, 3,  8)
    place(time_plot,      9,  9, 3,  8)
    place(volume_view,    3,  0, 3, 17)
    place(table_view,     0, 17, 6, 10)

    @ctrl.set("grid_add_view")
    def add_view():
        next_view_id = available_view_ids.pop()
        next_y = get_next_y_from_layout(state.grid_layout)
        state.grid_layout.append(
            dict(x=0, w=12, h=DEFAULT_NB_ROWS, y=next_y, i=next_view_id)
        )
        state.dirty("grid_layout")

    @ctrl.set("grid_remove_view")
    def remove_view(view_id):
        available_view_ids.append(view_id)
        state[f"grid_view_{view_id}"] = empty.option_for(view_id)
        state.grid_layout = list(
            filter(lambda item: item.get("i") != view_id, state.grid_layout)
        )

    @ctrl.set("pick_file_dataset")
    def pick_file_dataset(view_id, file, key):
        print("here", view_id, file, key)
        state[f"selected_file_{view_id}"] = file
        state[f"selected_array_{view_id}"] = key
        state[f"file_menu_{view_id}"] = False
        state[f"selected_label_{view_id}"] = f"{file} | {key.replace('_', ' ').upper()}"


    # Setup main layout
    with SinglePageLayout(server) as layout:
        layout.root.classes = ("{ busy: trame__busy }",)

        with layout.toolbar as toolbar:
            toolbar.clear()
            toolbar.height = 36

            html.Img(src=assets.LOGO, height=25)
            vuetify.VSpacer()

            vera.ColorMapEditor(v_model="color_range", color_preset="jet")

            vuetify.VSpacer()

            with html.Div(style="width: 25px", classes="mr-2"):
                vuetify.VProgressCircular(
                    indeterminate=True,
                    v_show=("trame__busy",),
                    style="background-color: lightgray; border-radius: 50%",
                    background_opacity=1,
                    bg_color="#01549b",
                    color="#04a94d",
                    size=16,
                    width=3,
                )

            vuetify.VSelect(
                v_model=("selected_array", "pin_powers"),
                items=(
                    "available_arrays",
                    [
                        dict(text=key.replace("_", " ").title(), value=key)
                        for key in registry.default_source.active_state_full_core_keys
                    ],
                ),
                hide_details=True,
                dense=True,
                style="max-width: 220px",
            )
            with vuetify.VBtn(icon=True, click="show_diff_dialog = true"):
                vuetify.VIcon("mdi-delta")
            
            with vuetify.VBtn(icon=True, click="show_derived_dialog = true"):
                vuetify.VIcon("mdi-calculator-variant")

            with vuetify.VBtn(icon=True, click="show_threshold_dialog = true"):
                vuetify.VIcon("mdi-table-filter")

            with vuetify.VBtn(icon=True, click=ctrl.grid_add_view):
                vuetify.VIcon("mdi-plus")

        with layout.content:
            layout.content.style = "overflow: auto; margin: 36px 0px 35px; padding: 0;"
            with vuetify.VDialog(v_model=("show_derived_dialog",), max_width=620, persistent=True):
                with vuetify.VCard():
                    vuetify.VCardTitle("Create Derived Dataset", classes="text-subtitle-1")
                    vuetify.VDivider()
                    with vuetify.VCardText(classes="pt-4"):
                        with vuetify.VCard(outlined=True, classes="pa-3 mb-3"):
                            html.Div("1. Select Dataset", classes="text-caption font-weight-medium mb-2")
                            vuetify.VSelect(
                                v_model=("derived_source",),
                                items=("available_arrays",),
                                hide_details=True,
                                dense=True,
                            )
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
            with vuetify.VDialog(v_model=("show_threshold_dialog",), max_width=480, persistent=True):
                with vuetify.VCard():
                    vuetify.VCardTitle("Dataset Thresholds", classes="text-subtitle-1")
                    vuetify.VDivider()
                    with vuetify.VCardText(classes="pt-4"):
                        with vuetify.VRow(classes="ma-0", align="center"):
                            with vuetify.VCol(classes="pa-0"):
                                vuetify.VSelect(
                                    v_model=("threshold_array",),
                                    items=("available_arrays",),
                                    label="Dataset",
                                    hide_details=True,
                                    dense=True,
                                )
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
            with vuetify.VContainer(
                fluid=True,
                classes="pa-0 fill-height",
                style="user-select: none;",
            ):
                with grid.GridLayout(
                    layout=("grid_layout", []),
                    row_height=30,
                    vertical_compact=True,
                    style="width: 100%; height: 100%;",
                ):
                    with grid.GridItem(
                        v_for="item in grid_layout",
                        key="item.i",
                        v_bind="item",
                        style="touch-action: none;",
                        drag_ignore_from=".drag_ignore",
                    ):
                        with vuetify.VCard(style="height: 100%;", key="grid_item_dirty_key"):
                            with vuetify.VCardTitle(classes="py-1 px-1"):
                                with vuetify.VMenu(offset_y=True):
                                    with vuetify.Template(v_slot_activator="{ on, attrs }"):
                                        with vuetify.VBtn(icon=True, small=True, v_bind="attrs", v_on="on"):
                                            vuetify.VIcon(v_text="get(`grid_view_${item.i}`).icon")
                                        html.Div(
                                            "{{ get(`grid_view_${item.i}`).label }}",
                                            classes="ml-1 text-subtitle-2",
                                        )
                                    with vuetify.VList(dense=True):
                                        with vuetify.VListItem(
                                            v_for="(option, index) in get(`grid_options_${item.i}`)",
                                            key="index",
                                            click="""
                                                set(`grid_view_${item.i}`, option);
                                                grid_item_dirty_key++;
                                            """,
                                        ):
                                            with vuetify.VListItemIcon():
                                                vuetify.VIcon(v_text="option.icon")
                                            vuetify.VListItemTitle("{{ option.label }}")
                                vuetify.VSpacer()
                                with vuetify.VMenu(
                                    offset_y=True,
                                    close_on_content_click=False,
                                ):
                                    with vuetify.Template(v_slot_activator="{ on, attrs }"):
                                        with vuetify.VBtn(small=True, text=True, v_bind="attrs", v_on="on"):
                                            html.Span("{{ get(`selected_label_${item.i}`) }}")
                                            vuetify.VIcon("mdi-menu-down", small=True)
                                    with vuetify.VList(dense=True):
                                        with vuetify.VMenu(
                                            v_for="(entries, file) in file_tree",
                                            key="file",
                                            offset_x=True,
                                            open_on_hover=True,
                                            close_on_content_click=False,
                                        ):
                                            with vuetify.Template(v_slot_activator="{ on, attrs }"):
                                                with vuetify.VListItem(v_bind="attrs", v_on="on"):
                                                    vuetify.VListItemTitle("{{ file }}")
                                                    with vuetify.VListItemIcon():
                                                        vuetify.VIcon("mdi-menu-right", small=True)
                                            with vuetify.VList(dense=True):
                                                with vuetify.VListItem(
                                                    v_for="entry in entries",
                                                    key="entry.value",
                                                    click=(ctrl.pick_file_dataset, "[item.i, file, entry.value]"),
                                                ):
                                                    vuetify.VListItemTitle("{{ entry.text }}")
                                vuetify.VSpacer()
                                with vuetify.VBtn(
                                    icon=True,
                                    x_small=True,
                                    click=(ctrl.grid_remove_view, "[item.i]"),
                                ):
                                    vuetify.VIcon("mdi-delete-forever-outline", small=True)

                            vuetify.VDivider()

                            style = "; ".join([
                                "position: relative",
                                "height: calc(100% - 37px)",
                                "overflow: auto",
                            ])
                            with vuetify.VCardText(style=style, classes="drag_ignore"):
                                client.ServerTemplate(name=("get(`grid_view_${item.i}`).name",))

        with layout.footer as footer:
            footer.clear()
            footer.height = 35
            with vuetify.VBtn(
                icon=True,
                small=True,
                disabled=("selected_time == 0",),
                click="selected_time--",
            ):
                vuetify.VIcon("mdi-minus")
            html.Div(
                "State {{ selected_time }} / {{ max_time }}",
                classes="text-center",
                style="width: 100px;",
            )
            with vuetify.VBtn(
                icon=True,
                small=True,
                disabled=("selected_time == max_time",),
                click="selected_time++",
            ):
                vuetify.VIcon("mdi-plus")
            vuetify.VDivider(vertical=True, classes="mx-2")
            vuetify.VSlider(
                v_model=("selected_time",),
                min=0,
                max=("max_time", 0),
                dense=True,
                hide_details=True,
                ticks="always",
                tick_size="4",
                height=35,
            )