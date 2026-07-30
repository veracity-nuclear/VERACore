# session_menu.py
import json
from dataclasses import asdict
from pathlib import Path

from trame.widgets import html, vuetify

from vera_core.app.core import build_session

from .file_picker_entry import launch_picker

session_menu_state_initialized = False


def register_session_menu_state_ctrl(state, ctrl, registry, all_view_ids: list):
    global session_menu_state_initialized

    state.setdefault("show_session_dialog", False)
    state.setdefault("session_error", "")
    state.setdefault("session_saved_path", "")

    @ctrl.set("save_session")
    async def save_session():
        state.session_error = ""
        state.session_saved_path = ""
        try:
            path = await launch_picker(
                "--mode",
                "save",
                "--filter",
                "json",
                "--prompt",
                "Save Session",
                "--name",
                "session.json",
            )
        except Exception as e:
            state.session_error = f"Could not open save dialog: {e}"
            return
        if not path:
            return
        if not path.lower().endswith(".json"):
            path += ".json"
        try:
            ctrl.snapshot_volume_cameras()
            session = build_session(state, registry, all_view_ids)
            Path(path).write_text(json.dumps(asdict(session), indent=2))
            state.session_saved_path = path
            state.show_session_dialog = False
        except Exception as e:
            state.session_error = f"Could not save session: {e}"

    session_menu_state_initialized = True


def build_session_menu_dialog(ctrl):
    if not session_menu_state_initialized:
        raise RuntimeError("register_session_menu_state_ctrl() must be called first")
    with vuetify.VDialog(
        v_model=("show_session_dialog",), max_width=480, persistent=True
    ):
        with vuetify.VCard():
            vuetify.VCardTitle("Save Session", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                html.Div(
                    "Save the current layout, selections, and loaded files to a JSON file.",
                    classes="text-caption text--secondary mb-3",
                )
                vuetify.VAlert(
                    "{{ session_error }}",
                    v_if="session_error",
                    type="error",
                    dense=True,
                    text=True,
                    classes="mb-2",
                )
                vuetify.VAlert(
                    "Saved to {{ session_saved_path }}",
                    v_if="session_saved_path",
                    type="success",
                    dense=True,
                    text=True,
                    classes="mb-2",
                )
            vuetify.VDivider()
            with vuetify.VCardActions(classes="px-4 pb-4"):
                with vuetify.VBtn(color="primary", click=ctrl.save_session):
                    vuetify.VIcon("mdi-content-save", left=True)
                    html.Span("Choose Location & Save")
                vuetify.VSpacer()
                with vuetify.VBtn(text=True, click="show_session_dialog = false"):
                    html.Span("Close")
