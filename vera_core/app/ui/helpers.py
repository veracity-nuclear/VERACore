import numpy as np

from trame_server.core import State, Controller
from trame.widgets import html, vuetify
from vera_core.app.core import VeraDataRegistry

def format_label(file : str, key : str):
    return f"{file} | {key.replace('_', ' ').upper()}"

def refresh_file_tree(state: State, registry : VeraDataRegistry):
    state.file_tree = {
        fid: [{"text": k.replace("_", " ").title(), "value": k}
              for k in registry.get(fid).active_state_full_core_keys]
        for fid in registry.source_ids()
    }

def array_range(array):
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo == hi:
        eps = max(abs(hi) * 1e-9, 1e-12)
        return (lo, hi + eps)
    return (lo, hi)

def get_next_y_from_layout(layout):
    next_y = 0
    for item in layout:
        y, h = item.get("y", 0), item.get("h", 1)
        if y + h > next_y:
            next_y = y + h
    return next_y

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
