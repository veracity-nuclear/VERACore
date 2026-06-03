# StreamMenu.py
from trame.widgets import vuetify, html
from trame_server.core import State, Controller
from trame.app.asynchronous import StateQueue

from vera_core.app.core import VeraDataRegistry
from vera_core.app.core import VeraDataStream, generate_stream_identifier
from .DatasetPicker import refresh_src_tree

stream_menu_state_initialized = False


def register_stream_menu_state_ctrl(state: State, ctrl: Controller, registry: VeraDataRegistry, state_queue : StateQueue):
    state.show_stream_dialog = False
    state.stream_port = None
    state.stream_source_name = ""
    state.stream_error = ""
    state.stream_connecting = False

    def _make_stream_watcher(stream_source_name, stream_source):
        @state.change(generate_stream_identifier(stream_source_name))
        def _on_stream_data_ready(**kwargs):
            if stream_source_name in registry.src_ids():
                return
            was_empty = registry.default_src_id is None
            registry.add_src(stream_source , stream_source_name)
            refresh_src_tree(state, registry)
            if was_empty:
                ctrl.activate_src()
            state.stream_connecting = False
            state.show_stream_dialog = False
            state.stream_port = None
            state.stream_source_name = ""

        @state.change(f"{generate_stream_identifier(stream_source_name)}_state_count")
        def _on_state_recieved(**kwargs):
            if stream_source_name not in registry.src_ids():
                return
            if registry.max_state > state.max_time:
                state.max_time = registry.max_state
        return (_on_stream_data_ready, _on_state_recieved)

    @ctrl.set("open_stream_dialog")
    def open_stream_dialog():
        state.stream_error = ""
        state.show_stream_dialog = True

    @ctrl.set("connect_stream")
    def connect_stream(stream_port, stream_source_name):
        port = int(stream_port)
        stream_source = VeraDataStream(stream_source_name, port, state_queue)
        _make_stream_watcher(stream_source_name, stream_source)
        stream_source.start() 
        state.stream_connecting = True
        state.stream_port = None
        state.stream_source_name = ""
        state.stream_error = ""
        # state.show_stream_dialog = False
        
    global stream_menu_state_initialized
    stream_menu_state_initialized = True


def build_stream_dialog(ctrl: Controller):
    global stream_menu_state_initialized
    if not stream_menu_state_initialized:
        raise RuntimeError("register_stream_menu_state_ctrl() must be called first")
    with vuetify.VDialog(v_model=("show_stream_dialog",), max_width=480, persistent=True):
        with vuetify.VCard():
            vuetify.VCardTitle("Connect Data Stream", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                with html.Div(
                    v_if=("stream_connecting",),
                    classes="d-flex flex-column align-center justify-center py-6",
                    style="gap: 12px;",
                ):
                    vuetify.VProgressCircular(indeterminate=True, color="primary", size=40)
                    html.Div("Waiting for data on stream...", classes="text-caption text--secondary")

                # Idle form
                with html.Div(v_if=("!stream_connecting",)):
                    vuetify.VTextField(
                        v_model=("stream_source_name",),
                        label="Source name",
                        placeholder="my_stream",
                        hide_details=True,
                        dense=True,
                        classes="mb-3",
                    )
                    vuetify.VTextField(
                        v_model=("stream_port",),
                        label="Port",
                        placeholder="8000",
                        type="number",
                        hide_details=True,
                        dense=True,
                    )
                    vuetify.VAlert(
                        "{{ stream_error }}",
                        v_show=("stream_error",),
                        type="error",
                        dense=True,
                        text=True,
                        classes="mt-3 mb-0",
                    )

            vuetify.VDivider()
            with vuetify.VCardActions():
                vuetify.VSpacer()
                vuetify.VBtn("Cancel", text=True, click="show_stream_dialog = false")
                vuetify.VBtn(
                    "Connect",
                    color="primary",
                    click=(ctrl.connect_stream, "[stream_port, stream_source_name]"),
                    disabled=("!stream_port || !stream_source_name || stream_connecting",),
                )