import asyncio, sys
from pathlib import Path
from trame.widgets import vuetify, html
from trame_server.core import Controller, State
from vera_core.app.core import VeraDataRegistry, VeraOutFile, recipe_sources
from .DatasetPicker import refresh_src_tree
from .file_picker_entry import launch_picker

file_menu_state_initialized = False

def register_file_menu_state_ctrl(state : State, ctrl : Controller, registry: VeraDataRegistry):
    """
    Register trame state for file menu
    
    state.show_file_dialog : state show flag for UI file menu
    state.file_error : error state of file menu, string rendered on file menu as a warning
    state.file_path : state that stores path to h5 file to be opened, transient
    state.recent_file_paths : state the stores a list of recently opened files in the session
    """
    refresh_src_tree(state, registry)
    state.show_file_dialog = False
    state.file_error = ""
    state.file_path = ""
    state.recent_file_paths = []

    @ctrl.set("open_file_dialog")
    def open_file_dialog():
        state.file_error = ""
        state.show_file_dialog = True

    @ctrl.set("pick_file")
    async def pick_file():
        state.file_error = ""
        try:
            path = await launch_picker("--mode", "open", "--filter", "h5",
                                    "--prompt", "Open VERA Output file")
        except asyncio.TimeoutError:
            state.file_error = "File picker timed out."
            return
        except Exception as e:
            state.file_error = f"Picker error: {e}"
            return
        if not path:
            return
        state.file_path = path
        load_file()

    @ctrl.set("load_recent")
    def load_recent(path):
        state.file_path = path
        load_file()

    @ctrl.set("load_file")
    def load_file():
        raw_path = (state.file_path or "").strip()
        if not raw_path:
            state.file_error = "Enter a path."
            return
        pathobj = Path(raw_path)
        if not pathobj.is_file():
            state.file_error = f"File not found: {raw_path}"
            return
        try:
            was_empty = registry.default_src_id is None
            registry.add_src(VeraOutFile(raw_path), src_id=pathobj.stem)
            state.max_layer = len(registry.global_axial_mesh) - 1
            recent = [raw_path] + [p for p in state.recent_file_paths if p != raw_path]
            state.recent_file_paths = recent[:10]
            refresh_src_tree(state, registry)
            state.file_path = ""
            state.file_error = ""
            state.show_file_dialog = False
            if was_empty: # if this was the first source loaded render the initial UI
                ctrl.activate_src()
        except Exception as e:
            print(e)
            state.file_error = f"Could not load: {e}"

    @ctrl.set("close_file")
    def close_file(file_id):
        try:
            ctrl.remove_source(file_id)
        except Exception as e:
            print(e)
            state.file_error = f"Could not remove file: {file_id}"
    
    global file_menu_state_initialized
    file_menu_state_initialized = True

import json
from dataclasses import asdict
from pathlib import Path

def register_session_state_ctrl(state, ctrl : Controller, registry : VeraDataRegistry):
    state.session_error = ""

    @ctrl.set("pick_session")
    async def pick_session():
        state.session_error = ""
        try:
            path = await launch_picker("--mode", "open", "--filter", "json",
                                    "--prompt", "Open Session")
        except Exception as e:
            state.session_error = f"Picker error: {e}"
            return
        if not path:
            return
        if not Path(path).is_file():
            state.session_error = f"File not found: {path}"
            return
        try:
            ctrl.load_session(path)
            state.show_file_dialog = False
        except Exception as e:
            state.session_error = f"Could not load session: {e}"

def build_file_menu_dialog(ctrl: Controller):
    """Build the file menu UI"""
    global file_menu_state_initialized
    if not file_menu_state_initialized:
        raise RuntimeError("FileMenu.register_file_menu_state_ctrl() must be called before this function")
    with vuetify.VDialog(v_model=("show_file_dialog",), max_width=720, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("VERAOutput Files", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-3 pb-1"):
                html.Div("Loaded files:", classes="text-caption mb-2")
                with vuetify.VSimpleTable(dense=True):
                    with html.Thead():
                        with html.Tr():
                            html.Th("Name", classes="text-left")
                            html.Th("Shape", classes="text-left")
                            html.Th("States", classes="text-left")
                            html.Th("")
                    with html.Tbody():
                        with html.Tr(v_for="(info, file) in src_tree_meta", key="file"):
                            html.Td("{{ file }}")
                            html.Td("{{ info.shape }}")
                            html.Td("{{ info.states }}")
                            with html.Td():
                                with vuetify.VBtn(icon=True, x_small=True, click=(ctrl.close_file, "[file]")):
                                    vuetify.VIcon("mdi-close", small=True)

            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-3 pb-1"):
                html.Div("Open new file:", classes="text-caption mb-2")
                vuetify.VAlert(
                    "{{ file_error }}",
                    v_if="file_error",
                    type="error",
                    dense=True,
                    classes="mb-2",
                )

                html.Div("Recent:", classes="text-caption mt-1 mb-1",
                         v_if="recent_file_paths.length")
                with vuetify.VList(dense=True, v_if="recent_file_paths.length"):
                    with vuetify.VListItem(
                        v_for="(p, idx) in recent_file_paths", key="idx",
                        click=(ctrl.load_recent, "[p]"),
                    ):
                        vuetify.VListItemTitle("{{ p }}")

                vuetify.VAlert(
                    "{{ session_error }}",
                    v_if="session_error",
                    type="error",
                    dense=True,
                    classes="mb-2 mt-2",
                )

            vuetify.VDivider()
            with vuetify.VCardActions(classes="px-4 py-3"):
                with vuetify.VBtn(color="primary", click=ctrl.pick_file):
                    vuetify.VIcon("mdi-file-upload", left=True)
                    html.Span("Upload H5 Output File")
                with vuetify.VBtn(color="secondary", click=ctrl.pick_session, classes="ml-2"):
                    vuetify.VIcon("mdi-folder-open", left=True)
                    html.Span("Load Session")
                vuetify.VSpacer()
                with vuetify.VBtn(color="secondary", click="show_file_dialog = false"):
                    html.Span("Close")