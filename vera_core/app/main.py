import os, time, sys
from functools import partial
from pathlib import Path
from multiprocessing import Queue

from trame.app import get_server, dev
from trame.app.asynchronous import StateQueue, create_state_queue_monitor_task
from trame_server.core import Server

from . import ui
from .core import VeraDataRegistry, VeraOutFile, VeraDataSource
from .core.vera_data_stream import VirtualVeraDataStream, VeraDataStream

# The user can set this via an environment variable
DATA_PATH_ENV_NAME = "VERA_CORE_DATA_PATH"
import faulthandler, signal
faulthandler.register(signal.SIGUSR1, all_threads=True)


def _reload(registry: VeraDataRegistry, state_queue : StateQueue):
    server = get_server()
    if server is None:
        return
    dev.reload(ui)
    ui.initialize(server, registry, state_queue)


def main(server : Server | None | str = None, **kwargs):
    # Get or create server
    if server is None:
        server = get_server(client_type="vue2")

    if isinstance(server, str):
        server = get_server(server, client_type="vue2")

    if server is None:
        # if get_server returns a None
        return
    
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
    registry : VeraDataRegistry = VeraDataRegistry()

    if stream_port is not None:
        vera_out_file = VeraDataStream(stream_port, state_queue)
        # this is a crude fix to prevent reads on empty states from the vera data stream
        # will fix change this
        while(len(vera_out_file.states) < 1):
            time.sleep(0.3)
    elif data_file is not None:
        file_path = Path(data_file)
        if not file_path.is_file():
            raise FileNotFoundError(f"{data_file} must be an exsisting path to a file")
        vera_out_file = VeraOutFile(data_file)
        registry.add_src(src=vera_out_file, src_id=file_path.stem)

    f = partial(_reload, registry=registry)

    # Make UI auto reload
    server.controller.on_server_reload.add(f)

    # Init application
    ui.initialize(server, registry, state_queue)
    @server.controller.add("on_server_ready")
    def start_stream(**kwargs):
        create_state_queue_monitor_task(server, raw_queue, delay=0.1)
    
    # Start server
    kwargs.setdefault("disable_logging", False)

    server.start(**kwargs)

if __name__ == "__main__":
    main()
