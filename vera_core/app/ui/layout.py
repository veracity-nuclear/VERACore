from trame.ui.vuetify import SinglePageLayout
from trame.widgets import client, grid, html, vuetify
from trame_server.core import Server, Controller, State
from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry

from . import assets
from .features import DeriveMenu, DiffMenu, ThresholdMenu, FileMenu, StreamMenu, DatasetPicker, LocateMenu, SaveSession

def build_toolbar(tb, ctrl: Controller, registry):
    tb.clear()
    tb.height = 36
    html.Img(src=assets.LOGO, height=25)
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

    vuetify.VSpacer()

    vuetify.VSwitch(
        v_model=("dark_mode",),
        change="$vuetify.theme.dark = $event",
        hide_details=True,
        dense=True,
        inset=True,
        prepend_icon="mdi-white-balance-sunny",
        append_icon="mdi-moon-waning-crescent",
        classes="mt-0",
    )

    with vuetify.VBtn(icon=True, click=ctrl.open_file_dialog):
        vuetify.VIcon("mdi-folder-open")
    
    with vuetify.VBtn(icon=True, click="show_session_dialog = true", disabled=("!has_data",)):
        vuetify.VIcon("mdi-content-save")
    
    with vuetify.VBtn(icon=True, click=ctrl.open_stream_dialog):
        vuetify.VIcon("mdi-access-point")

    with vuetify.VBtn(icon=True, click="show_derived_dialog = true", disabled=("!has_data",)):
        vuetify.VIcon("mdi-calculator-variant")
    
    # with vuetify.VBtn(icon=True, click="show_locate_dialog = true", disabled=("!has_data",)):   # add
    #     vuetify.VIcon("mdi-crosshairs-gps")  

    with vuetify.VBtn(icon=True, click="show_threshold_dialog = true", disabled=("!has_data",)):
        vuetify.VIcon("mdi-table-filter")

    with vuetify.VBtn(icon=True, click="show_diff_dialog = true", disabled=("!has_data",)):
        vuetify.VIcon("mdi-delta")

    with vuetify.VBtn(icon=True, click=ctrl.grid_add_view, disabled=("!has_data",)):
        vuetify.VIcon("mdi-plus")


def build_grid_card(ctrl: Controller):
    with grid.GridItem(
        v_for="item in grid_layout",
        key="item.i",
        v_bind="item",
        style="touch-action: none;",
        drag_ignore_from=".drag_ignore",
    ):
        with vuetify.VCard(
            style="height: 100%; display: flex; flex-direction: column;",
            key=("`card_${item.i}_${get(`grid_view_${item.i}`).name}`",),
        ):
            with vuetify.VCardTitle(classes="py-1 px-1", style="flex: 0 0 auto;"):
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
                            """,
                        ):
                            with vuetify.VListItemIcon():
                                vuetify.VIcon(v_text="option.icon")
                            vuetify.VListItemTitle("{{ option.label }}")

                vuetify.VSpacer()

                with vuetify.Template(v_if=("!get(`grid_view_${item.i}`).multi_picker",)):
                    DatasetPicker.build_dataset_picker(
                        ctrl,
                        "selected_label_${item.i}",
                        "select_dataset",
                        "[item.i, src, entry.value]",
                        "get(`grid_view_${item.i}`).allowed_categories",
                    )
                with vuetify.Template(v_if=("get(`grid_view_${item.i}`).multi_picker",)):
                    DatasetPicker.build_dataset_multi_picker(
                        ctrl,
                        "multi_label_${item.i}",
                        "multi_selected_${item.i}",
                        "toggle_multi_array",
                        "[item.i, src, entry.value]",
                        "get(`grid_view_${item.i}`).allowed_categories",
                    )

                vuetify.VSpacer()

                with vuetify.VBtn(
                    icon=True,
                    x_small=True,
                    click="set(`locked_${item.i}`, !get(`locked_${item.i}`))",
                ):
                    vuetify.VIcon(
                        v_text="get(`locked_${item.i}`) ? 'mdi-lock' : 'mdi-lock-open-variant'",
                        small=True,
                    )

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
                "flex: 1 1 0",
                "min-height: 0",
                "overflow : auto",
            ])
            with vuetify.VCardText(style=style, classes="drag_ignore"):
                client.ServerTemplate(name=("get(`grid_view_${item.i}`).name",))

def build_content(layout, state : State, ctrl : Controller, registry: VeraDataRegistry):
    layout.content.style = "overflow: auto; margin: 36px 72px 35px 0px; padding: 0;"
    DeriveMenu.build_derived_dialog(state, ctrl, registry)
    DiffMenu.build_diff_dialog(state, ctrl, registry)
    ThresholdMenu.build_threshold_dialog(ctrl)
    FileMenu.build_file_menu_dialog(ctrl)
    FileMenu.build_core_prompt_dialog(ctrl)
    SaveSession.build_session_menu_dialog(ctrl)
    StreamMenu.build_stream_dialog(ctrl)
    LocateMenu.build_locate_dialog(state, ctrl, registry) 
    build_axial_slider()
    with vuetify.VContainer(fluid=True, classes="pa-0 fill-height", style="user-select: none;"):
        # Empty state: prompt the user to open a file.
        with html.Div(
            v_if=("!has_data",),
            classes="d-flex flex-column align-center justify-center",
            style="width: 100%; height: 100%; gap: 12px;",
        ):
            html.Div("No file loaded.", classes="text-h6 text--secondary")
            with vuetify.VBtn(color="primary", click=ctrl.open_file_dialog):
                vuetify.VIcon("mdi-folder-open", left=True)
                html.Span("Open File")

        with grid.GridLayout(
            v_if=("has_data",),
            key=("grid_rebuild_key",),
            layout=("grid_layout", []),
            row_height=30,
            vertical_compact=True,
            style="width: 100%; height: 100%;",
        ):
            build_grid_card(ctrl)

def build_footer(ft):
    ft.clear()
    ft.height = 35
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

def build_axial_slider():
    """Vertical slider on the right edge that controls the selected axial layer.

    Implementation: a native <input type="range"> rendered vertically. The
    Vuetify slider could not be sized vertically in this stack, and rotating a
    horizontal one breaks its click-position math. The native input sizes with
    a plain inline height, has correct browser-native pointer handling, and
    fires `change` on release (commit-on-release, no per-pixel server traffic).
    writing-mode + direction make it vertical with min (layer 0) at the bottom;
    the orient attribute covers Firefox, -webkit-appearance covers older
    Chromium/Safari.
    """
    strip = "; ".join([
        "position: fixed",
        "top: 36px",
        "right: 0",
        "bottom: 35px",
        "width: 56px",
        "z-index: 4",
        "border-left: 1px solid #e0e0e0",
    ])
    with html.Div(style=strip, 
        v_bind_style=(
            "dark_mode "
            "? { backgroundColor: '#1e1e1e', borderLeft: '1px solid #333' } "
            ": { backgroundColor: 'white', borderLeft: '1px solid #e0e0e0' }"
        ),
        classes="d-flex flex-column align-center py-1"):
        html.Div("Axial", classes="text-caption mb-1")
        with html.Div(
            style=(
                "flex: 1 1 0;"
                " min-height: 0;"
                " width: 100%;"
                " display: flex;"
                " justify-content: center;"
            )
        ):
            html.Input(
                type="range",
                min=0,
                max=("max_layer",),
                step=1,
                orient="vertical",
                value=("selected_layer",),
                change="selected_layer = Number($event.target.value)",
                style=(
                    "writing-mode: vertical-lr;"
                    " direction: rtl;"
                    " -webkit-appearance: slider-vertical;"
                    " width: 24px;"
                    " height: 100%;"
                ),
            )
        with vuetify.VBtn(
            icon=True,
            small=True,
            disabled=("selected_layer == max_layer",),
            click="selected_layer++",
        ):
            vuetify.VIcon("mdi-plus")
        with vuetify.VBtn(
            icon=True,
            small=True,
            disabled=("selected_layer == 0",),
            click="selected_layer--",
        ):
            vuetify.VIcon("mdi-minus")
        html.Div(
            "{{ selected_layer + 1 }} / {{ max_layer + 1 }}",
            classes="text-center text-caption mt-1",
            style="width: 100%;",
        )

def build_layout(server : Server, state : State, ctrl : Controller, registry : VeraDataRegistry):
    with SinglePageLayout(server) as layout:
        layout.root.classes = ("{ busy: trame__busy }")
        client.ClientTriggers(mounted="$vuetify.theme.dark = dark_mode")
        with layout.toolbar as tb:
            build_toolbar(tb, ctrl, registry)
        with layout.content:
            build_content(layout, state, ctrl, registry)
        with layout.footer as ft:
            build_footer(ft)