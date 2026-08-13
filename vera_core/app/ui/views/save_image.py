from functools import partial
from typing import Callable

from trame.widgets import vuetify
from trame_server.core import State

from vera_core.data.analysis.color import ColorScope
from vera_core.data.renders.layout import Selection
from vera_core.data.renders.styles import DARK, LIGHT, ViewStyle

from ..features.file_picker_entry import launch_picker


def register_photo_state(state: State, view_id: int, name: str):
    """Returns (`msg_key`, `msg_show_key`)"""
    msg_key = f"{name}_image_msg_{view_id}"
    msg_show_key = f"{name}_image_msg_show_{view_id}"
    state.setdefault(msg_key, "")
    state.setdefault(msg_show_key, False)
    return msg_key, msg_show_key


def take_photo[Request](
    *,
    state: State,
    saved_sel: Callable[[], Selection[Request]],
    show_labels_key,
    decimals_key,
    msg_key,
    msg_show_key,
    n_groups_key,
):
    with vuetify.VBtn(
        icon=True,
        small=True,
        disabled=(f"{n_groups_key} < 1",),
        click=partial(
            save_image,
            state=state,
            saved_sel=saved_sel,
            show_labels_key=show_labels_key,
            decimals_key=decimals_key,
            msg_key=msg_key,
            msg_show_key=msg_show_key,
        ),
    ):
        vuetify.VIcon("mdi-image")


def notification(msg_key, msg_show_key):
    vuetify.VSnackbar(
        "{{ " + msg_key + " }}",
        v_model=(msg_show_key, False),
        timeout=4000,
        bottom=True,
    )


async def save_image[Request](
    *,
    state: State,
    saved_sel: Callable[[], Selection[Request]],
    show_labels_key: str | bool,
    decimals_key,
    msg_key,
    msg_show_key,
):
    def notify(text: str):
        state[msg_key] = text
        state[msg_show_key] = True

    if not isinstance(saved_sel, Callable):
        notify("Nothing to save.")
        return
    saved_sel = saved_sel()
    if saved_sel is None or not isinstance(saved_sel, Selection):
        notify("Nothing to save.")
        return
    try:
        path = await launch_picker(
            "--mode",
            "save",
            "--filter",
            "image",
            "--prompt",
            "Save Image",
            "--name",
            "core_photo.png",
        )
    except Exception as e:
        notify(f"Could not open save dialog: {e}")
        return
    if not path:
        return
    if not path.lower().endswith((".png", ".pdf", ".svg", ".jpg", ".jpeg")):
        path += ".png"
    try:
        vs = ViewStyle(
            theme=DARK if state["dark_mode"] else LIGHT,
            show_values=show_labels_key
            if isinstance(show_labels_key, bool)
            else state[show_labels_key],
            decimals=state[decimals_key],
        )
        saved_sel.savefig(path=path, color_scope=ColorScope.GROUP, style=vs)
        notify(f"Saved to {path}")
    except Exception as e:
        print(str(e))
        notify(f"Could not save image: {e}")
        raise e
