from trame.widgets import html, vuetify
from trame_server.core import Controller, State

from vera_core.data.dtypes import VeraDtype
from vera_core.data.model import VeraOutState
from vera_core.data.registry import VeraDataRegistry

from ..helpers import MULTI_SEP

MENU_MAX_HEIGHT = "60vh"


def _group_count(vera_state: VeraOutState, name: str, dtype: VeraDtype) -> int:
    """Number of energy groups in `name`; 0 when the dtype has no group dim."""
    if not dtype.has_energy_group_dim():
        return 0
    shape = vera_state.shape(name)
    return int(shape[dtype.energy_group_dim_idx])


def refresh_src_tree(state: State, registry: VeraDataRegistry):
    src_tree = {}
    src_tree_meta = {}
    for fid in registry.src_ids():
        src = registry.get(fid)
        vera_state = src.active_state
        src_tree[fid] = [
            {
                "key": cat,
                "category": cat.replace("_", " ").title(),
                "entries": [
                    {
                        "text": name.replace("_", " ").title(),
                        "value": name,
                        "groups": _group_count(vera_state, name, VeraDtype[cat]),
                    }
                    for name in names
                ],
            }
            for cat, names in vera_state.grouped_full_core_keys
        ]
        src_tree_meta[fid] = {
            "shape": str(src.core.core_shape),
            "states": len(src.states),
        }
    state.src_tree = src_tree
    state.src_tree_meta = src_tree_meta


def register_dataset_picker_state(state: State, registry: VeraDataRegistry):
    """Init dataset picker state"""
    refresh_src_tree(state, registry)


def _make_label(selected_label_arg: str):
    """helper method for creating a JS span that shows selected_label_arg, truncating with an ellipsis when it's too long for the picker button"""
    return html.Span(
        f"{{{{ get(`{selected_label_arg}`) }}}}",
        style=("max-width: 100%;overflow: hidden;text-overflow: ellipsis;white-space: nowrap;"),
    )


def _with_group(args: str, group_expr: str) -> str:
    """Append the chosen group to the caller's JS arg list."""
    inner = args.strip()[1:-1].strip()
    return f"[{inner}, {group_expr}]" if inner else f"[{group_expr}]"


def _key_expr(group_expr: str | None) -> str:
    base = f"src + '{MULTI_SEP}' + entry.value"
    return base if group_expr is None else f"{base} + '{MULTI_SEP}' + {group_expr}"


def _check_icon(selected_arg: str | None, group_expr: str | None) -> None:
    if not selected_arg:
        return
    with vuetify.VListItemIcon():
        vuetify.VIcon(
            "mdi-check",
            small=True,
            v_show=(f"get(`{selected_arg}`).includes({_key_expr(group_expr)})",),
        )


def _build_entries(
    ctrl: Controller,
    ctrl_func: str,
    ctrl_func_args: str,
    close_on_content_click: bool,
    selected_arg: str | None = None,
    allow_groups: bool = False,
):
    """One item per dataset. With allow_groups, entries that have energy
    groups open a submenu; otherwise every entry selects the whole dataset."""
    all_args = _with_group(ctrl_func_args, "null")
    if not allow_groups:
        with vuetify.VListItem(
            v_for="entry in group.entries",
            key="entry.value",
            click=(ctrl[ctrl_func], all_args),
        ):
            _check_icon(selected_arg, None)
            vuetify.VListItemTitle("{{ entry.text }}")
        return

    with vuetify.Template(v_for="entry in group.entries"):
        with vuetify.VListItem(
            v_if="!entry.groups",
            key="entry.value",
            click=(ctrl[ctrl_func], all_args),
        ):
            _check_icon(selected_arg, None)
            vuetify.VListItemTitle("{{ entry.text }}")
        with vuetify.VMenu(
            v_else=True,
            key="entry.value",
            offset_x=True,
            open_on_hover=False,
            close_on_content_click=close_on_content_click,
            max_height=MENU_MAX_HEIGHT,
        ):
            with vuetify.Template(v_slot_activator="{ on, attrs }"):
                with vuetify.VListItem(v_bind="attrs", v_on="on"):
                    if selected_arg:
                        with vuetify.VListItemIcon():
                            vuetify.VIcon(
                                "mdi-check",
                                small=True,
                                v_show=(
                                    f"get(`{selected_arg}`).some(k => {{"
                                    f"const b = src + '{MULTI_SEP}' + entry.value;"
                                    f"return k === b || k.startsWith(b + '{MULTI_SEP}');}})",
                                ),
                            )
                    vuetify.VListItemTitle("{{ entry.text }}")
                    with vuetify.VListItemIcon():
                        vuetify.VIcon("mdi-menu-right", small=True)
            with vuetify.VList(dense=True):
                with vuetify.VListItem(click=(ctrl[ctrl_func], all_args)):
                    _check_icon(selected_arg, None)
                    vuetify.VListItemTitle("All")
                with vuetify.VListItem(
                    v_for="g in entry.groups",
                    key="g",
                    click=(ctrl[ctrl_func], _with_group(ctrl_func_args, "g")),
                ):
                    _check_icon(selected_arg, "g")
                    vuetify.VListItemTitle("Group {{ g }}")


def build_dataset_picker(
    ctrl: Controller,
    selected_label_arg: str,
    ctrl_func: str = "_noop",
    ctrl_func_args: str = "[]",
    allowed_arg=None,
    group_arg=None,
):
    """Create trame UI for selecting a single dataset to be visualized

    Args:
        ctrl : reference to trame controller
        selected_label_arg: string to appear as default on the picker
        ctrl_func : str name of trame controller function to call when a user clicks a dataset
            defaults to a "_noop" which will trigger no ctrl action/function.
        ctrl_func_args : str represnting which JS variables to pass into the ctrl_func, this is what determines what JS values are sent
        to the python function as parameters, for example ctrl_func_args="[item.i, src, entry.value]" will pass these JS vars to the python
        func as parameters.
        allowed_args : optional list that defines what categories can be displayed on this dataset picker, use to restrict
        the datasets that can be selected from this dataset picker
    """
    groups_expr = "groups"
    if allowed_arg:
        groups_expr = f"groups.filter(g => !{allowed_arg} || {allowed_arg}.includes(g.key))"
    with vuetify.VMenu(offset_y=True, close_on_content_click=False, max_height=MENU_MAX_HEIGHT):
        with vuetify.Template(v_slot_activator="{ on, attrs }"):
            with vuetify.VBtn(
                small=True,
                text=True,
                v_bind="attrs",
                v_on="on",
                style=(
                    "min-width: 0;"
                    "border-bottom: 1px solid rgba(0,0,0,0.42);"
                    "border-radius: 0;"
                    "padding-bottom: 2px;"
                ),
            ):
                _make_label(selected_label_arg)
                if group_arg:
                    vuetify.VChip(
                        f"G{{{{ get(`{group_arg}`) }}}}",
                        v_if=(f"get(`{group_arg}`) !== null",),
                        x_small=True,
                        classes="ml-1 flex-shrink-0",
                    )
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
                            _build_entries(
                                ctrl,
                                ctrl_func,
                                ctrl_func_args,
                                close_on_content_click=True,
                                allow_groups=bool(group_arg),
                            )


def build_dataset_multi_picker(
    ctrl: Controller,
    label_arg: str,
    selected_arg: str,
    ctrl_func: str = "_noop",
    ctrl_func_args: str = "[src, entry.value]",
    allowed_arg=None,
):
    """Create trame UI for selecting multiple dataset to be visualized

    Args:
        ctrl : reference to trame controller
        selected_label_arg: string to appear as default on the picker
        ctrl_func : str name of trame controller function to call when a user clicks a dataset
            defaults to a "_noop" which will trigger no ctrl action/function.
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
                small=True,
                text=True,
                v_bind="attrs",
                v_on="on",
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
                            _build_entries(
                                ctrl,
                                ctrl_func,
                                ctrl_func_args,
                                close_on_content_click=False,
                                selected_arg=selected_arg,
                                allow_groups=True,
                            )
