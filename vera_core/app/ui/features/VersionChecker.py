# Version checker (notify-only). The app never downloads or installs anything.
import asyncio
import json
import time
import urllib.request
from pathlib import Path

from packaging.version import parse as vparse
from platformdirs import user_data_dir
from trame.widgets import html, vuetify
from trame_server.core import Controller, State

from vera_core.version import __version__

MANIFEST = "https://jsalem5.github.io/vera-core-updates-test/version.json"
STATE_FILE = Path(user_data_dir("VeraCore")) / "update.json"
TIMEOUT = 5

version_check_initialized = False


def _fetch():
    req = urllib.request.Request(MANIFEST, headers={"User-Agent": "VeraCore"})
    with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
        return json.load(r)


def _read_state():
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {}


def _write_state(data):
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(data))


def _is_newer(latest):
    if __version__.startswith("0.0.0"):
        return False
    return vparse(latest) > vparse(__version__)


def register_version_check_ctrl(state: State, ctrl: Controller) -> None:
    global version_check_initialized

    state.setdefault("show_version_dialog", False)
    state.setdefault("version_current", __version__)
    state.setdefault("version_latest", "")
    state.setdefault("version_behind", False)
    state.setdefault("version_badge", False)
    state.setdefault("version_url", "")
    state.setdefault("version_error", "")
    state.setdefault("version_checking", False)

    async def _do_check(force: bool):
        """Fetch off the loop, write state on the loop."""
        if state.version_checking:
            return
        persisted = _read_state()

        state.version_checking = True
        state.version_error = ""
        state.flush()
        try:
            loop = asyncio.get_running_loop()
            manifest = await loop.run_in_executor(None, _fetch)
            latest = manifest["version"]
            _write_state({**persisted, "last": time.time()})  # keep dismissed

            state.version_latest = latest
            state.version_url = manifest.get("url", "")
            newer = _is_newer(latest)
            state.version_behind = newer
            if force:
                state.version_badge = newer
            else:
                state.version_badge = newer and latest != persisted.get("dismissed")
        except Exception as e:
            state.version_error = str(e)
        finally:
            state.version_checking = False
            state.flush()

    @ctrl.set("open_version_dialog")
    def open_version_dialog():
        """Open the version panel and run a fresh check (manual path)."""
        state.show_version_dialog = True
        asyncio.create_task(_do_check(force=True))

    @ctrl.set("recheck_version")
    def recheck_version():
        """Re-run the manual check from inside the dialog."""
        asyncio.create_task(_do_check(force=True))

    @ctrl.set("dismiss_version_badge")
    def dismiss_version_badge():
        """Silence the toolbar notice for this version until a newer one ships."""
        persisted = _read_state()
        persisted["dismissed"] = state.version_latest
        _write_state(persisted)
        state.version_badge = False  # only the notice; version_behind stays true

    @ctrl.set("run_launch_version_check")
    def run_launch_version_check():
        """Silent check on launch. Sets the notice only; opens no dialog."""
        asyncio.create_task(_do_check(force=False))

    version_check_initialized = True


def build_version_notice(ctrl):
    """Toolbar chip shown next to the logo when a newer version exists."""
    with vuetify.VChip(
        v_if=("version_badge",),
        small=True,
        color="success",
        click=ctrl.open_version_dialog,
        classes="ml-3",
        style="cursor: pointer;",
    ):
        vuetify.VIcon("mdi-arrow-up-circle", left=True, x_small=True)
        html.Span("New version available")


def build_version_dialog(ctrl):
    if not version_check_initialized:
        raise RuntimeError("register_version_check_ctrl() must be called first")
    with vuetify.VDialog(v_model=("show_version_dialog",), max_width=520):
        with vuetify.VCard():
            vuetify.VCardTitle("VERACore Version", classes="text-subtitle-1")
            vuetify.VDivider()
            with vuetify.VCardText(classes="pt-4"):
                html.Div(
                    "Installed version: {{ version_current }}",
                    classes="text-body-2 mb-1",
                )
                html.Div(
                    "Latest available: {{ version_latest || '—' }}",
                    v_if="!version_checking",
                    classes="text-body-2 mb-3",
                )
                vuetify.VProgressCircular(
                    indeterminate=True,
                    size=20,
                    width=2,
                    v_if="version_checking",
                    classes="mb-3",
                )
                vuetify.VAlert(
                    "A newer version is available. Download the installer to update.",
                    v_if="version_behind && !version_checking",
                    type="info",
                    dense=True,
                    text=True,
                    classes="mb-2",
                )
                vuetify.VAlert(
                    "You have the latest version.",
                    v_if=(
                        "!version_behind && version_latest && !version_error && !version_checking"
                    ),
                    type="success",
                    dense=True,
                    text=True,
                    classes="mb-2",
                )
                vuetify.VAlert(
                    "Could not reach the version server.",
                    v_if="version_error && !version_checking",
                    type="warning",
                    dense=True,
                    text=True,
                    classes="mb-2",
                )
            vuetify.VDivider()
            with vuetify.VCardActions(classes="px-4 pb-4"):
                with vuetify.VBtn(
                    color="primary",
                    href=("version_url",),
                    target="_blank",
                    v_if="version_behind",
                ):
                    vuetify.VIcon("mdi-open-in-new", left=True)
                    html.Span("Downloads")
                with vuetify.VBtn(
                    text=True,
                    click=ctrl.dismiss_version_badge,
                    v_if="version_badge",
                ):
                    html.Span("Remind Me Later")
                vuetify.VSpacer()
                with vuetify.VBtn(text=True, click="show_version_dialog = false"):
                    html.Span("Close")
