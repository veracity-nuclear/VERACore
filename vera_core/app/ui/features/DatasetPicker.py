from trame.widgets import vuetify, html
from trame_server.core import Controller, State
from vera_core.app.core import VeraDataRegistry

MENU_MAX_HEIGHT = "60vh"

def refresh_src_tree(state: State, registry: VeraDataRegistry):
    """
    Defines and refreshes state for tracking what datasets are avaliable in each opened source
    
    state.src_tree : state that tracks each sources' (in the src registry) avaliable datasets by category
        each src has a list of dicts, each dict corresponds to a VeraDtype category
        each category dict contains a list of the names of datasets in this category, 
        the names are stores as key:value pairs to map user selection from dataset picker to raw h5 dataset name

    state.src_tree_meta : state that tracks the number of states and core shape of each source in the registry

    """
    state.src_tree = {
        fid: [
            {
                "key": cat,
                "category": cat.replace("_", " ").title(),
                "entries": [
                    {"text": k.replace("_", " ").title(), "value": k} for k in names
                ],
            }
            for cat, names in registry.get(fid).active_state_grouped_keys
        ]   
        for fid in registry.src_ids()
    }
    state.src_tree_meta = {
        fid: {
            "shape": str(registry.get(fid).core.core_shape),
            "states": len(registry.get(fid).states),
        }
        for fid in registry.src_ids()
    }

def register_dataset_picker_state(state : State, registry : VeraDataRegistry):
    """Init dataset picker state"""
    refresh_src_tree(state, registry)

def _make_label(selected_label_arg: str):
    """helper method for creating a JS span that shows selected_label_arg, truncating with an ellipsis when it's too long for the picker button"""
    return html.Span(
        f"{{{{ get(`{selected_label_arg}`) }}}}",
        style=(
            "max-width: 26ch;"
            "overflow: hidden;"
            "text-overflow: ellipsis;"
            "white-space: nowrap;"
        ),
    )

def build_dataset_picker(ctrl: Controller, selected_label_arg: str, ctrl_func: str = "_noop", ctrl_func_args: str = "[]", allowed_arg=None):
    """Create trame UI for selecting a single dataset to be visualized
    
    Args:
        ctrl : refrence to trame controller
        selected_label_arg: string to appear as default on the picker
        ctrl_func : str name of trame controller function to call when a user clicks a dataset
            defaults to a "_noop" which will trigger no ctrl action/functino.
        ctrl_func_args : str represnting which JS variables to pass into the ctrl_func, this is what determines what JS values are sent
        to the python function as parameters, for example ctrl_func_args="[item.i, src, entry.value]" will pass these JS vars to the python
        func as parameters.
        allowed_args : optional list that defines what categories can be displayed on this dataset picker, use to restrict
        the datasets that can be selected from this dataset picker
    """
    groups_expr = "groups"
    if allowed_arg:
        groups_expr = f"groups.filter(g => !{allowed_arg} || {allowed_arg}.includes(g.key))"    
    with vuetify.VMenu( offset_y=True, close_on_content_click=False, max_height=MENU_MAX_HEIGHT):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with vuetify.VBtn(
                small=True, text=True, v_bind="attrs", v_on="on",
                style=(
                    "border-bottom: 1px solid rgba(0,0,0,0.42);"
                    "border-radius: 0;"
                    "padding-bottom: 2px;"
                ),
            ):
                _make_label(selected_label_arg)
                vuetify.VIcon("mdi-menu-down", small=True)
        with vuetify.VList(dense=True):
            with vuetify.VMenu(
                v_for="(groups, src) in src_tree",
                key="src",
                offset_x=True,
                open_on_hover=False,
                close_on_content_click=True,
                max_height=MENU_MAX_HEIGHT,
            ):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    with vuetify.VListItem(v_bind="attrs", v_on="on"):
                        vuetify.VListItemTitle("{{ src }}")
                        with vuetify.VListItemIcon():
                            vuetify.VIcon("mdi-menu-right", small=True)
                with vuetify.VList(dense=True):
                    with vuetify.VMenu(
                        v_for=f"group in {groups_expr}",
                        key="group.category",
                        offset_x=True,
                        open_on_hover=False,
                        close_on_content_click=True,
                        max_height=MENU_MAX_HEIGHT,
                    ):
                        with vuetify.Template(v_slot_activator="{ on, attrs }"):
                            with vuetify.VListItem(v_bind="attrs", v_on="on"):
                                vuetify.VListItemTitle("{{ group.category }}")
                                with vuetify.VListItemIcon():
                                    vuetify.VIcon("mdi-menu-right", small=True)
                        with vuetify.VList(dense=True):
                            with vuetify.VListItem(
                                v_for="entry in group.entries",
                                key="entry.value",
                                click=(ctrl[ctrl_func], ctrl_func_args),
                            ):
                                vuetify.VListItemTitle("{{ entry.text }}")

def build_dataset_multi_picker(ctrl: Controller, label_arg: str, selected_arg: str, ctrl_func: str = "_noop", ctrl_func_args: str = "[src, entry.value]", allowed_arg=None):
    """Create trame UI for selecting multiple dataset to be visualized
    
    Args:
        ctrl : refrence to trame controller
        selected_label_arg: string to appear as default on the picker
        ctrl_func : str name of trame controller function to call when a user clicks a dataset
            defaults to a "_noop" which will trigger no ctrl action/functino.
        ctrl_func_args : str represnting which JS variables to pass into the ctrl_func, this is what determines what JS values are sent
        to the python function as parameters, for example ctrl_func_args="[item.i, src, entry.value]" will pass these JS vars to the python
        func as parameters.
        allowed_args : optional list that defines what categories can be displayed on this dataset picker, use to restrict
        the datasets that can be selected from this dataset picker
    """
    groups_expr = "groups"
    if allowed_arg:
        groups_expr = f"groups.filter(g => !{allowed_arg} || {allowed_arg}.includes(g.key))"  
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
                v_for="(groups, src) in src_tree",
                key="src",
                offset_x=True,
                open_on_hover=False,
                close_on_content_click=False,
                max_height=MENU_MAX_HEIGHT,
            ):
                with vuetify.Template(v_slot_activator="{ on, attrs }"):
                    with vuetify.VListItem(v_bind="attrs", v_on="on"):
                        vuetify.VListItemTitle("{{ src }}")
                        with vuetify.VListItemIcon():
                            vuetify.VIcon("mdi-menu-right", small=True)
                with vuetify.VList(dense=True):
                    with vuetify.VMenu(
                        v_for=f"group in {groups_expr}",
                        key="group.category",
                        offset_x=True,
                        open_on_hover=False,
                        close_on_content_click=False,
                        max_height=MENU_MAX_HEIGHT,
                    ):
                        with vuetify.Template(v_slot_activator="{ on, attrs }"):
                            with vuetify.VListItem(v_bind="attrs", v_on="on"):
                                vuetify.VListItemTitle("{{ group.category }}")
                                with vuetify.VListItemIcon():
                                    vuetify.VIcon("mdi-menu-right", small=True)
                        with vuetify.VList(dense=True):
                            with vuetify.VListItem(
                                v_for="entry in group.entries",
                                key="entry.value",
                                click=(ctrl[ctrl_func], ctrl_func_args),
                            ):
                                with vuetify.VListItemIcon():
                                    vuetify.VIcon(
                                        "mdi-check",
                                        small=True,
                                        v_show=(
                                            f"get(`{selected_arg}`)"
                                            f".includes(src + '\x1f' + entry.value)",
                                        ),
                                    )
                                vuetify.VListItemTitle("{{ entry.text }}")