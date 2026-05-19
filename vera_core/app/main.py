from functools import partial
import os

from multiprocessing import Queue
from trame.app import get_server, dev
from trame.app.asynchronous import StateQueue, create_state_queue_monitor_task
import time
from . import ui
from .core.vera_out_file import VeraOutFile
from .core.vera_data_stream import VirtualVeraDataStream, VeraDataStream

# The user can set this via an environment variable
DATA_PATH_ENV_NAME = "VERA_CORE_DATA_PATH"


def _reload(vera_out_file):
    server = get_server()
    dev.reload(ui)
    ui.initialize(server, vera_out_file)


def main(server=None, **kwargs):
    # Get or create server
    if server is None:
        server = get_server(client_type="vue2")

    if isinstance(server, str):
        server = get_server(server, client_type="vue2")

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
    server.cli.add_argument("--stream", type=int, default=None, help="port to listen on for streamed data", dest="stream_port")
    args, _ = server.cli.parse_known_args()
    data_file = args.data_file
    stream_port = args.stream_port

    raw_queue = Queue()
    state_queue = StateQueue(raw_queue)

    if stream_port is not None:
        vera_out_file = VeraDataStream(stream_port, state_queue)
        # this is a crude fix to prevent reads on empty states from the vera data stream
        # will fix change this
        while(len(vera_out_file.states) < 1):
            time.sleep(0.3)
    else:
        vera_out_file = VeraOutFile(data_file)

    f = partial(_reload, vera_out_file=vera_out_file)

    # Make UI auto reload
    server.controller.on_server_reload.add(f)

    # Init application
    ui.initialize(server, vera_out_file)
    @server.controller.add("on_server_ready")
    def start_stream(**kwargs):
        create_state_queue_monitor_task(server, raw_queue, delay=0.1)
    # Start server
    kwargs.setdefault("disable_logging", False)

    server.start(**kwargs)


if __name__ == "__main__":
    main()
