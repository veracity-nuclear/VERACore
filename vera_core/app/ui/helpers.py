from trame.widgets import html, vuetify

def format_label(file, key):
    return f"{file} | {key.replace('_', ' ').upper()}"

def refresh_file_tree(state, registry):
    state.file_tree = {
        fid: [{"text": k.replace("_", " ").title(), "value": k}
              for k in registry.get(fid).active_state_full_core_keys]
        for fid in registry.source_ids()
    }
    
def get_next_y_from_layout(layout):
    next_y = 0
    for item in layout:
        y, h = item.get("y", 0), item.get("h", 1)
        if y + h > next_y:
            next_y = y + h
    return next_y

def _make_label(selected_label_arg: str):
    return html.Span(f"{{{{ get(`{selected_label_arg}`) }}}}")

def build_dataset_picker(ctrl, selected_label_arg, ctrl_func : str = "_noop", ctrl_func_args : str = "[]"):
    with vuetify.VMenu(offset_y=True,close_on_content_click=False):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with vuetify.VBtn(small=True, text=True, v_bind="attrs", v_on="on"):
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
