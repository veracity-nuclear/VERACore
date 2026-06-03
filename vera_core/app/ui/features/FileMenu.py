from pathlib import Path
from trame.widgets import vuetify, html
from trame_server.core import Controller, State
from vera_core.app.core import VeraDataRegistry, VeraOutFile, VeraDataStream
from .DatasetPicker import refresh_src_tree

file_menu_state_initialized = False

def register_file_menu_state_ctrl(state : State, ctrl : Controller, registry: VeraDataRegistry):
    refresh_src_tree(state, registry)
    state.setdefault("show_file_dialog", False)
    state.setdefault("file_to_open", None)
    state.setdefault("file_error", "")
    state.setdefault("file_path", "")
    state.setdefault("recent_file_paths", [])

    @ctrl.set("open_file_dialog")
    def open_file_dialog():
        state.file_to_open = None
        state.file_error = ""
        state.show_file_dialog = True

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
            recent = [raw_path] + [p for p in state.recent_file_paths if p != raw_path]
            state.recent_file_paths = recent[:10]
            refresh_src_tree(state, registry)
            state.file_path = ""
            state.file_error = ""
            state.show_file_dialog = False
            if was_empty:
                ctrl.activate_src()
        except Exception as e:
            state.file_error = f"Could not load: {e}"

    @ctrl.set("close_file")
    def close_file(file_id):
        #FIXME remove_source not implemented yet
        registry.remove_source(file_id)
        refresh_src_tree(state, registry)
    
    global file_menu_state_initialized
    file_menu_state_initialized = True

def build_file_menu_dialog(ctrl: Controller):
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
                                    # FIXME close_file logic depends on unimplememted registry.remove_src()
                                    # with html.Td():
                                    #     with vuetify.VBtn(icon=True, x_small=True, click=(ctrl.close_file, "[file]")):
                                    #         vuetify.VIcon("mdi-close", small=True)

            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-3"):
                html.Div("Open new file:", classes="text-caption mb-2")
                vuetify.VCombobox(
                v_model=("file_path",),
                items=("recent_file_paths",),
                label="Path to .h5 file",
                placeholder="/path/to/file.h5",
                hint="Recent files appear in the dropdown; or paste a full path",
                persistent_hint=True,
                clearable=True,
                dense=True,
            )
                with vuetify.VBtn(color="primary", click=ctrl.load_file,
                                disabled=("!file_path",)):
                    html.Span("Open")
                with vuetify.VBtn(color="primary", click="show_file_dialog = false"):
                    html.Span("Close")

            vuetify.VDivider()
