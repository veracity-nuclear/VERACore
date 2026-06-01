from pathlib import Path
from trame.widgets import vuetify, html
from trame_server.core import Controller, State
from vera_core.app.core import VeraDataRegistry, VeraOutFile, VeraDataStream

file_menu_state_initialized = False

def refresh_file_tree(state: State, registry : VeraDataRegistry):
    state.file_tree = {
        fid: [{"text": k.replace("_", " ").title(), "value": k}
              for k in registry.get(fid).active_state_full_core_keys]
        for fid in registry.source_ids()
    }
    state.file_tree_meta = {
        fid: {
            "shape": str(registry.get(fid).active_state.pin_powers.shape),
            "states": len(registry.get(fid).states),
        }
        for fid in registry.source_ids()
    }

def register_file_menu_state_ctrl(state : State, ctrl : Controller, registry: VeraDataRegistry):
    refresh_file_tree(state, registry)
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

    @ctrl.set("load_selected_file")
    def load_selected_file():
        raw_path = (state.file_path or "").strip()
        if not raw_path:
            state.file_error = "Enter a path."
            return
        pathobj = Path(raw_path)
        if not pathobj.is_file():
            state.file_error = f"File not found: {raw_path}"
            return
        try:
            was_empty = registry.default is None
            registry.add_source(VeraOutFile(raw_path), source_id=pathobj.stem)
            recent = [raw_path] + [p for p in state.recent_file_paths if p != raw_path]
            state.recent_file_paths = recent[:10]
            refresh_file_tree(state, registry)
            state.file_path = ""
            state.file_error = ""
            state.show_file_dialog = False
            if was_empty:
                ctrl.activate_source()
        except Exception as e:
            state.file_error = f"Could not load: {e}"

    @ctrl.set("close_file")
    def close_file(file_id):
        registry.remove_source(file_id)
        refresh_file_tree(state, registry)
    
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
                                with html.Tr(v_for="(info, file) in file_tree_meta", key="file"):
                                    html.Td("{{ file }}")
                                    html.Td("{{ info.shape }}")
                                    html.Td("{{ info.states }}")
                                    with html.Td():
                                        with vuetify.VBtn(icon=True, x_small=True, click=(ctrl.close_file, "[file]")):
                                            vuetify.VIcon("mdi-close", small=True)

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
                with vuetify.VBtn(color="primary", click=ctrl.load_selected_file,
                                disabled=("!file_path",)):
                    html.Span("Open")
                with vuetify.VBtn(color="primary", click="show_file_dialog = false"):
                    html.Span("Close")

            vuetify.VDivider()

def _make_label(selected_label_arg: str):
    return html.Span(f"{{{{ get(`{selected_label_arg}`) }}}}")

def build_file_dataset_picker(ctrl : Controller, selected_label_arg : str, ctrl_func : str = "_noop", ctrl_func_args : str = "[]"):
    with vuetify.VMenu(offset_y=True,close_on_content_click=False):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with vuetify.VBtn(
                    small=True, 
                    text=True, 
                    v_bind="attrs", 
                    v_on="on", 
                    style=(
                    "border-bottom: 1px solid rgba(0,0,0,0.42);"
                    "border-radius: 0;"
                    "padding-bottom: 2px;")
            ):
                _make_label(selected_label_arg)
                vuetify.VIcon("mdi-menu-down", small=True)
        with vuetify.VList(dense=True):
            with vuetify.VMenu(
                v_for="(entries, file) in file_tree",
                key="file",
                offset_x=True,
                open_on_hover=True,
                close_on_content_click=True,
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
                        click=(ctrl[ctrl_func], ctrl_func_args),
                    ):
                        vuetify.VListItemTitle("{{ entry.text }}")

def build_file_dataset_multi_picker(
    ctrl: Controller,
    label_arg: str,
    selected_arg: str,
    ctrl_func: str = "_noop",
    ctrl_func_args: str = "[file, entry.value]",
):
    with vuetify.VMenu(offset_y=True, close_on_content_click=False):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with vuetify.VBtn(
                small=True, text=True, v_bind="attrs", v_on="on",
                style=(
                    "border-bottom: 1px solid rgba(0,0,0,0.42);"
                    "border-radius: 0;"
                    "padding-bottom: 2px;"
                ),
            ):
                _make_label(label_arg)
                vuetify.VIcon("mdi-menu-down", small=True)
        with vuetify.VList(dense=True):
            with vuetify.VMenu(
                v_for="(entries, file) in file_tree",
                key="file",
                offset_x=True,
                open_on_hover=True,
                close_on_content_click=False,   # was True: keep open while picking
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
                        click=(ctrl[ctrl_func], ctrl_func_args),
                    ):
                        with vuetify.VListItemIcon():
                            # checkmark reflects membership in the selected set
                            vuetify.VIcon(
                                "mdi-check",
                                small=True,
                                v_show=(
                                    f"get(`{selected_arg}`)"
                                    f".includes(file + '\x1f' + entry.value)",
                                ),
                            )
                        vuetify.VListItemTitle("{{ entry.text }}")