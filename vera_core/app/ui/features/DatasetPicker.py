from trame.widgets import vuetify, html
from trame_server.core import Controller, State
from vera_core.app.core import VeraDataRegistry

def refresh_src_tree(state: State, registry : VeraDataRegistry):
    state.src_tree = {
        fid: [{"text": k.replace("_", " ").title(), "value": k}
              for k in registry.get(fid).active_state_full_core_keys]
        for fid in registry.src_ids()
    }
    state.src_tree_meta = {
        fid: {
            "shape": str(registry.get(fid).active_state.pin_powers.shape),
            "states": len(registry.get(fid).states),
        }
        for fid in registry.src_ids()
    }

def register_dataset_picker_state(state : State, registry : VeraDataRegistry):
    refresh_src_tree(state, registry)

def _make_label(selected_label_arg: str):
    return html.Span(f"{{{{ get(`{selected_label_arg}`) }}}}")

def build_dataset_picker(ctrl : Controller, selected_label_arg : str, ctrl_func : str = "_noop", ctrl_func_args : str = "[]"):
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
                v_for="(entries, src) in src_tree",
                key="src",
                offset_x=True,
                open_on_hover=True,
                close_on_content_click=True,
            ):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    with vuetify.VListItem(v_bind="attrs", v_on="on"):
                        vuetify.VListItemTitle("{{ src }}")
                        with vuetify.VListItemIcon():
                            vuetify.VIcon("mdi-menu-right", small=True)
                with vuetify.VList(dense=True):
                    with vuetify.VListItem(
                        v_for="entry in entries",
                        key="entry.value",
                        click=(ctrl[ctrl_func], ctrl_func_args),
                    ):
                        vuetify.VListItemTitle("{{ entry.text }}")

def build_dataset_multi_picker(
    ctrl: Controller,
    label_arg: str,
    selected_arg: str,
    ctrl_func: str = "_noop",
    ctrl_func_args: str = "[src, entry.value]",
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
                v_for="(entries, src) in src_tree",
                key="src",
                offset_x=True,
                open_on_hover=True,
                close_on_content_click=False,   # was True: keep open while picking
            ):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    with vuetify.VListItem(v_bind="attrs", v_on="on"):
                        vuetify.VListItemTitle("{{ src }}")
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
                                    f".includes(src + '\x1f' + entry.value)",
                                ),
                            )
                        vuetify.VListItemTitle("{{ entry.text }}")