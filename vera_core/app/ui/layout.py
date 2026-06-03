from trame.ui.vuetify import SinglePageLayout
from trame.widgets import client, grid, html, vuetify
from trame_server.core import Server, Controller, State
from vera_core.widgets import vera
from vera_core.app.core import VeraDataRegistry

from . import assets
from .features import DeriveMenu, DiffMenu, ThresholdMenu, FileMenu, StreamMenu, DatasetPicker

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

    with vuetify.VBtn(icon=True, click=ctrl.open_file_dialog):
        vuetify.VIcon("mdi-folder-open")
    
    with vuetify.VBtn(icon=True, click=ctrl.open_stream_dialog):
        vuetify.VIcon("mdi-access-point")

    with vuetify.VBtn(icon=True, click="show_derived_dialog = true", disabled=("!has_data",)):
        vuetify.VIcon("mdi-calculator-variant")

    with vuetify.VBtn(icon=True, click="show_threshold_dialog = true", disabled=("!has_data",)):
        vuetify.VIcon("mdi-table-filter")
    
    # FIXME this is btn for opening up the menu for creating difference datasets. 
    # Interpolation between datasets with different axial meshes not handled yet so button is not displayed.
    # vvvvv

    # with vuetify.VBtn(icon=True, click="show_diff_dialog = true", disabled=("!has_data",)):
    #     vuetify.VIcon("mdi-delta")

    with vuetify.VBtn(icon=True, click=ctrl.grid_add_view, disabled=("!has_data",)):
        vuetify.VIcon("mdi-plus")


def build_grid_card(ctrl : Controller):
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
                with vuetify.Template(v_if=("!get(`grid_view_${item.i}`).multi_picker",)):
                    DatasetPicker.build_dataset_picker(ctrl, "selected_label_${item.i}", "select_dataset", "[item.i, src, entry.value]")
                with vuetify.Template(v_if=("get(`grid_view_${item.i}`).multi_picker",)):
                    DatasetPicker.build_dataset_multi_picker(
                        ctrl,
                        "multi_label_${item.i}",
                        "multi_selected_${item.i}",
                        "toggle_multi_array",
                        "[item.i, src, entry.value]",
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
                "height: calc(100% - 37px)",
                "overflow: auto",
            ])
            with vuetify.VCardText(style=style, classes="drag_ignore"):
                client.ServerTemplate(name=("get(`grid_view_${item.i}`).name",))

def build_content(layout, state : State, ctrl : Controller, registry: VeraDataRegistry):
    layout.content.style = "overflow: auto; margin: 36px 0px 35px; padding: 0;"
    DeriveMenu.build_derived_dialog(state, ctrl, registry)
    # FIXME vvvv
    # build_diff_dialog(state, ctrl, registry)
    ThresholdMenu.build_threshold_dialog(ctrl)
    FileMenu.build_file_menu_dialog(ctrl)
    StreamMenu.build_stream_dialog(ctrl)

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


def build_layout(server : Server, state : State, ctrl : Controller, registry : VeraDataRegistry):
    with SinglePageLayout(server) as layout:
        layout.root.classes = ("{ busy: trame__busy }",)
        with layout.toolbar as tb:
            build_toolbar(tb, ctrl, registry)
        with layout.content:
            build_content(layout, state, ctrl, registry)
        with layout.footer as ft:
            build_footer(ft)