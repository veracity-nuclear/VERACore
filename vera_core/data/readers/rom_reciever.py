import threading

import msgpack
import numpy as np
import zmq

from vera_core.data.model import VeraDataSource, VeraOutCore, VeraOutState
from vera_core.data.readers.mock import DictDatasetSource


def generate_stream_identifier(stream_id: str | int | float) -> str:
    return f"vera_data_stream_{stream_id}"


def receive_rom_stream(port):
    context = zmq.Context()
    subscriber = context.socket(zmq.SUB)

    subscriber.setsockopt(zmq.SUBSCRIBE, b"")

    # We only care about the newest available state.
    subscriber.setsockopt(zmq.CONFLATE, 1)

    subscriber.connect(f"tcp://127.0.0.1:{port}")

    while True:
        data = subscriber.recv()

        state = msgpack.unpackb(
            data,
            raw=False,
        )

        yield state


DUMMY_STATE = VeraOutState(source=DictDatasetSource(arrays={}), idx=0, core=None)


def _data_reciever(port, vera_source: VeraDataSource, U_map, U_map_shape, stream_id, queue=None):
    has_recieved_state = False
    power_weight = 1.01
    for idx, message in enumerate(receive_rom_stream(port)):
        core = vera_source.core
        reduced_state = np.asarray(message["reduced_state"])
        arrays = {}
        for key, value in message.items():
            if np.shape(value) == tuple():
                arrays[key] = np.float64(value)
        power = message["DT_power"] * (power_weight**idx)
        flux = np.reshape(U_map @ reduced_state * power, shape=U_map_shape)
        arrays["power"] = power
        arrays["flux"] = flux[0]
        ds_source = DictDatasetSource(arrays=arrays, dataset_dtypes=core.shape_to_dtype)
        if vera_source.active_state_index == 0 and vera_source.active_state is DUMMY_STATE:
            vera_source.replace_state(
                idx=0, state=VeraOutState(source=ds_source, idx=idx + 1, core=core)
            )
        else:
            vera_source.add_state(VeraOutState(source=ds_source, idx=idx + 1, core=core))

        if queue is not None and not has_recieved_state:
            has_recieved_state = True
            queue.update({f"{stream_id}": has_recieved_state})
        if queue is not None and idx % 10 == 0:
            queue.update({f"{stream_id}_state_count": max(len(vera_source.states) - 1, 0)})


def open_rom_stream(port: int, rom_matrices_path: str, stream_id: str, queue=None):
    stop_signal = threading.Event()

    def close_stream():
        stop_signal.set()

    with np.load(rom_matrices_path) as f:
        U_map = f["U_map"]
        U_map_shape = f["shape"]
        core_map = f["core_map"]
        core_geometry = {
            "npin": f["npin"],
            "core_map": core_map,
            "axial_mesh": f["axial_mesh"],
            "core_sym": f["core_sym"],
            "aspect_ratio": f["aspect_ratio"],
        }
    stream_id = generate_stream_identifier(stream_id)
    core = VeraOutCore(dataset_source=DictDatasetSource(core_geometry))
    vera_source = VeraDataSource(
        core=core,
        states=[DUMMY_STATE],
        provenance=f"stream from port {port}",
        close_callback=close_stream,
    )
    thread = threading.Thread(
        target=_data_reciever,
        args=(port, vera_source, U_map, U_map_shape, stream_id, queue),
        daemon=True,
    )
    return vera_source, thread


if __name__ == "__main__":
    open_rom_stream(port=5555, rom_matrices_path="./dev/maps.npz", stream_id="test_stream")
