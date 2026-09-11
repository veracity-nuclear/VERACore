import os
from functools import partial
from multiprocessing import Queue
from pathlib import Path

from trame.app import dev, get_server
from trame.app.asynchronous import StateQueue, create_state_queue_monitor_task
from trame_server.core import Server

from vera_core.version import __version__

from ..data import VeraDataRegistry
from ..data.readers.h5 import open_vera_file_data_source
from . import ui

# The user can set this via an environment variable
DATA_PATH_ENV_NAME = "VERA_CORE_DATA_PATH"


def _reload(registry: VeraDataRegistry, state_queue: StateQueue):
    server = get_server()
    if server is None:
        return
    dev.reload(ui)
    ui.initialize(server, registry, state_queue)


def main(server: Server | None | str = None, **kwargs):
    print("Launching VERACore", __version__)
    # Get or create server
    if server is None:
        server = get_server(client_type="vue2")

    if isinstance(server, str):
        server = get_server(server, client_type="vue2")

    if server is None:
        # if get_server returns a None
        return
    server.options["desktop_debug"] = False
    data_kwargs = {
        "help": "Data file to load",
        "dest": "data_file",
    }

    default = os.getenv(DATA_PATH_ENV_NAME)
    if default is not None:
        # If the environment variable has been provided, use that for the default
        data_kwargs["default"] = default
    # else:
    #     # Otherwise, the CLI argument is required
    #     data_kwargs["required"] = True

    server.cli.add_argument("--data", **data_kwargs, default=None)
    args, _ = server.cli.parse_known_args()
    data_file = args.data_file

    raw_queue = Queue()
    state_queue = StateQueue(raw_queue)
    registry: VeraDataRegistry = VeraDataRegistry()

    if data_file is not None:
        file_path = Path(data_file)
        if not file_path.is_file():
            raise FileNotFoundError(f"{data_file} must be an existing path to a file")
        vera_out_file = open_vera_file_data_source(data_file)
        registry.add_src(src=vera_out_file, src_id=file_path.stem)

    f = partial(_reload, registry=registry, state_queue=state_queue)

    # Make UI auto reload
    server.controller.on_server_reload.add(f)

    # Init application
    print("Initializing UI")
    ui.initialize(server, registry, state_queue)

    @server.controller.add("on_server_ready")
    def start_stream(**kwargs):
        create_state_queue_monitor_task(server, raw_queue, delay=0.1)
        server.controller.run_launch_version_check()

    # Start server
    kwargs.setdefault("disable_logging", False)

    server.start(**kwargs)


if __name__ == "__main__":
    main()
