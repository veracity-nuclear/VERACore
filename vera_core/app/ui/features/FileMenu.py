import asyncio
from pathlib import Path

from trame.widgets import html, vuetify
from trame_server.core import Controller, State

from vera_core.app.core import CorePropMissing, VeraDataRegistry, VeraOutFile

from .appdata import load_prefs, save_prefs
from .DatasetPicker import refresh_src_tree
from .file_picker_entry import launch_picker

file_menu_state_initialized = False


def register_file_menu_state_ctrl(
    state: State, ctrl: Controller, registry: VeraDataRegistry
):
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
    state.core_prompt = {}
    state.core_answer_npin = None
    state.core_none_npin = False
    state.core_answer_nax = None
    state.show_core_dialog = False
    state.recent_file_paths, state.core_overrides = load_prefs()

    @ctrl.set("open_file_dialog")
    def open_file_dialog():
        state.file_error = ""
        state.show_file_dialog = True

    @ctrl.set("pick_file")
    async def pick_file():
        state.file_error = ""
        try:
            path = await launch_picker(
                "--mode", "open", "--filter", "h5", "--prompt", "Open VERA Output file"
            )
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
            try:
                src = VeraOutFile(
                    raw_path, core_overrides=state.core_overrides.get(raw_path, {})
                )
            except CorePropMissing as e:
                state.core_prompt = {
                    "path": raw_path,
                    "missing": e.missing,
                    "inferred": e.inferred,
                }
                state.core_answer_npin = None
                state.core_none_npin = False
                state.core_answer_nax = None
                state.show_core_dialog = True
                return
            if not src.default_datasets():
                had_override = raw_path in state.core_overrides
                if had_override:
                    state.core_overrides = {
                        k: v for k, v in state.core_overrides.items() if k != raw_path
                    }
                    save_prefs(state.recent_file_paths, state.core_overrides)
                    state.file_error = (
                        "No datasets match the core properties you entered. "
                        "Please check the values and try again."
                    )
                    state.file_path = raw_path
                    load_file()  # re-prompt
                    return
                state.file_error = (
                    "No datasets in this file match its core geometry. "
                    "The file may be malformed or use an unsupported layout."
                )
                return
            registry.add_src(src, src_id=pathobj.stem)
            state.max_layer = len(registry.global_axial_mesh) - 1
            recent = [raw_path] + [p for p in state.recent_file_paths if p != raw_path]
            state.recent_file_paths = recent[:10]
            save_prefs(state.recent_file_paths, state.core_overrides)
            refresh_src_tree(state, registry)
            state.file_path = ""
            state.file_error = ""
            state.show_file_dialog = False
            if was_empty:  # if this was the first source loaded render the initial UI
                ctrl.activate_src()
        except Exception as e:
            print(e)
            state.file_error = f"Could not load: {e}"
            raise e

    @ctrl.set("close_file")
    def close_file(file_id):
        try:
            ctrl.remove_source(file_id)
        except Exception as e:
            print(e)
            state.file_error = f"Could not remove file: {file_id}"

    @ctrl.set("cancel_core_props")
    def cancel_core_props():
        _reset_core_prompt()
        state.file_path = ""

    @ctrl.set("submit_core_props")
    def submit_core_props():
        path = state.core_prompt.get("path")
        missing = state.core_prompt.get("missing", {})
        if not path:
            return

        overrides = {}
        if "npin" in missing:
            if state.core_none_npin:
                overrides["npin"] = 0
            else:
                val = _parse_positive_int(
                    state.core_answer_npin, "Pins across an assembly", allow_zero=False
                )
                if val is None:
                    return
                overrides["npin"] = val
        if "nax" in missing:
            val = _parse_positive_int(
                state.core_answer_nax, "Number of axial layers", allow_zero=False
            )
            if val is None:
                return
            overrides["nax"] = val

        state.core_overrides = {**state.core_overrides, path: overrides}
        _reset_core_prompt()
        state.file_path = path
        load_file()

    def _parse_positive_int(raw, label, allow_zero):
        if raw is None or str(raw).strip() == "":
            state.file_error = f"Enter a value for {label}."
            return None
        try:
            val = int(raw)
        except (TypeError, ValueError):
            state.file_error = f"{label} must be a whole number."
            return None
        if val < 0 or (val == 0 and not allow_zero):
            state.file_error = f"{label} must be positive."
            return None
        return val

    def _reset_core_prompt():
        state.core_answer_npin = None
        state.core_none_npin = False
        state.core_answer_nax = None
        state.show_core_dialog = False
        state.core_prompt = {}
        state.file_error = ""

    @ctrl.set("clear_core_override")
    def clear_core_override(path):
        state.core_overrides = {
            k: v for k, v in state.core_overrides.items() if k != path
        }
        save_prefs(state.recent_file_paths, state.core_overrides)
        src_id = Path(path).stem
        if src_id in registry:
            ctrl.remove_source(src_id)
            state.file_path = path
            load_file()

    global file_menu_state_initialized
    file_menu_state_initialized = True


def register_session_state_ctrl(state, ctrl: Controller, registry: VeraDataRegistry):
    state.session_error = ""

    @ctrl.set("pick_session")
    async def pick_session():
        state.session_error = ""
        try:
            path = await launch_picker(
                "--mode", "open", "--filter", "json", "--prompt", "Open Session"
            )
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
        raise RuntimeError(
            "FileMenu.register_file_menu_state_ctrl() must be called before this function"
        )
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
                                with vuetify.VBtn(
                                    icon=True,
                                    x_small=True,
                                    click=(ctrl.close_file, "[file]"),
                                ):
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

                html.Div(
                    "Recent:",
                    classes="text-caption mt-1 mb-1",
                    v_if="recent_file_paths.length",
                )
                with vuetify.VList(dense=True, v_if="recent_file_paths.length"):
                    with vuetify.VListItem(
                        v_for="(p, idx) in recent_file_paths",
                        key="idx",
                        click=(ctrl.load_recent, "[p]"),
                    ):
                        vuetify.VListItemTitle("{{ p }}")
                html.Div(
                    "Saved core properties:",
                    classes="text-caption mt-3 mb-1",
                    v_if="Object.keys(core_overrides).length",
                )
                with vuetify.VList(
                    dense=True, v_if="Object.keys(core_overrides).length"
                ):
                    with vuetify.VListItem(
                        v_for="(vals, path) in core_overrides", key="path"
                    ):
                        with vuetify.VListItemContent():
                            vuetify.VListItemTitle("{{ path }}")
                            vuetify.VListItemSubtitle("{{ JSON.stringify(vals) }}")
                        with vuetify.VListItemAction():
                            with vuetify.VBtn(
                                icon=True,
                                x_small=True,
                                click=(ctrl.clear_core_override, "[path]"),
                            ):
                                vuetify.VIcon("mdi-close", small=True)

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
                with vuetify.VBtn(
                    color="secondary", click=ctrl.pick_session, classes="ml-2"
                ):
                    vuetify.VIcon("mdi-folder-open", left=True)
                    html.Span("Load Session")
                vuetify.VSpacer()
                with vuetify.VBtn(color="secondary", click="show_file_dialog = false"):
                    html.Span("Close")


def build_core_prompt_dialog(ctrl: Controller):
    with vuetify.VDialog(v_model=("show_core_dialog",), max_width=560, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Core Properties", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                html.Div(
                    "This file does not specify all core properties. "
                    "Confirm the values below and provide the missing ones.",
                    classes="text-caption text--secondary mb-3",
                )

                html.Div(
                    "Determined from the file:",
                    classes="text-caption mb-1",
                    v_if="Object.keys(core_prompt.inferred || {}).length",
                )
                with vuetify.VSimpleTable(
                    dense=True, v_if="Object.keys(core_prompt.inferred || {}).length"
                ):
                    with html.Tbody():
                        with html.Tr(
                            v_for="(item, key) in core_prompt.inferred", key="key"
                        ):
                            html.Td("{{ key }}")
                            html.Td("{{ item.value }}")
                            html.Td("{{ item.source }}", classes="text--secondary")

                vuetify.VDivider(classes="my-3")

                with html.Div(v_if="core_prompt.missing && core_prompt.missing.npin"):
                    html.Div(
                        "Pins across an assembly:", classes="text-caption mb-1 mt-2"
                    )
                    with html.Div(classes="d-flex align-center", style="gap: 16px;"):
                        vuetify.VTextField(
                            v_model=("core_answer_npin",),
                            type="number",
                            hide_details=True,
                            dense=True,
                            disabled=("core_none_npin",),
                            style="max-width: 120px;",
                        )
                        vuetify.VCheckbox(
                            v_model=("core_none_npin",),
                            label="No pins",
                            hide_details=True,
                            dense=True,
                            classes="mt-0 pt-0",
                        )

                with html.Div(v_if="core_prompt.missing && core_prompt.missing.nax"):
                    html.Div(
                        "Number of axial layers:", classes="text-caption mb-1 mt-2"
                    )
                    vuetify.VTextField(
                        v_model=("core_answer_nax",),
                        type="number",
                        hide_details=True,
                        dense=True,
                        style="max-width: 120px;",
                    )

                vuetify.VAlert(
                    "{{ file_error }}",
                    v_if="file_error",
                    type="error",
                    dense=True,
                    text=True,
                    classes="mt-3 mb-0",
                )
            vuetify.VDivider()
            with vuetify.VCardActions(classes="px-4 py-3"):
                vuetify.VSpacer()
                with vuetify.VBtn(text=True, click=ctrl.cancel_core_props):
                    html.Span("Cancel")
                with vuetify.VBtn(color="primary", click=ctrl.submit_core_props):
                    html.Span("Load File")
