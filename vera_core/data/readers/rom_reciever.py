import threading

import msgpack
import msgpack_numpy
import numpy as np
import zmq

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.model import DatasetSource, VeraDataSource, VeraOutCore, VeraOutState
from vera_core.data.readers.mock import DictDatasetSource
from vera_core.data.readers.rom import RomBasis

DUMMY_STATE = VeraOutState(source=DictDatasetSource(arrays={}), idx=0, core=None)


def generate_stream_identifier(stream_id: str | int | float) -> str:
    return f"vera_data_stream_{stream_id}"


class LiveRomDatasetSource(DatasetSource):
    """Scalars in memory, ROM-backed fields reconstructed on load.

    A state keeps its reduced coordinates and the scale applied to them. Fields
    named by the basis are expanded on demand and served from the basis' shared
    cache, so retained memory does not grow with the field size.

    Scalars report shape (1,) and unknown names return None, matching
    H5DatasetSource. A scalar name shadows a basis name of the same name.
    """

    def __init__(
        self,
        scalars: dict[str, float],
        reduced_state: np.ndarray,
        basis: RomBasis,
        scale: float = 1.0,
        provenance: str = "<rom>",
        dataset_dtypes: dict[tuple, VeraDtype] | None = None,
        units: dict[str, str] | None = None,
    ):
        self._scalars = {k: np.float64(v) for k, v in scalars.items()}
        self._reduced_state = np.asarray(reduced_state, dtype=np.float64)
        self._basis = basis
        self._scale = float(scale)
        self._token = basis.new_token()
        self._provenance = provenance
        self._dataset_dtypes = dataset_dtypes or {}
        self._units = units or {}
        self.load_count: dict[str, int] = {}

    @property
    def provenance(self) -> str:
        return self._provenance

    def names(self) -> list[str]:
        return [*self._scalars, *self._basis.names()]

    def shape(self, name: str) -> tuple[int, ...] | None:
        if name in self._scalars:
            return (1,)
        return self._basis.shape(name)

    def sample(self, name: str, indices):
        """One element of a ROM field, or None when the caller must use load().

        Optional fast path for consumers that read a single point across many
        states. Returns None for scalars, unknown names and partial indices,
        which the caller should treat as "not applicable" rather than "missing".
        """
        if name in self._scalars:
            return None
        flat = self._basis.flat_index(name, indices)
        if flat is None:
            return None
        return self._basis.sample(name, flat, self._reduced_state, self._scale)

    def load(self, name: str) -> VeraDataset | None:
        # The dtype is resolved on the same shape that shape() advertises, so a
        # scalar categorized as SCALAR by VeraOutState carries that dtype here
        # too.
        dtype_key = self.shape(name)
        if dtype_key is None:
            return None
        self.load_count[name] = self.load_count.get(name, 0) + 1
        if name in self._scalars:
            array = np.array([self._scalars[name]])
        else:
            array = self._basis.expand(self._token, name, self._reduced_state, self._scale)
        dtype = self._dataset_dtypes.get(dtype_key, VeraDtype.UNKNOWN)
        return VeraDataset(array, dtype, name, self._units.get(name, "unitless"))


class RomStream:
    """Handle for one live ROM connection."""

    def __init__(
        self,
        port: int,
        stream_id: str,
        queue=None,
    ):
        self.port = port
        self.stream_id = generate_stream_identifier(stream_id)
        self.queue = queue

        self.source: VeraDataSource | None = None
        self.error: Exception | None = None

        self._stop_signal = threading.Event()

        self._thread = threading.Thread(
            target=_data_receiver,
            args=(self,),
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop_signal.set()

    @property
    def is_alive(self) -> bool:
        return self._thread.is_alive()


def _receive_config(
    socket: zmq.Socket,
    stream: RomStream,
):
    """Receive CONFIG and construct the static VERA/ROM objects."""
    while not stream._stop_signal.is_set():
        try:
            message_type, payload = socket.recv_multipart()
        except zmq.Again:
            continue

        if message_type != b"CONFIG":
            continue

        config = msgpack.unpackb(
            payload,
            object_hook=msgpack_numpy.decode,
            raw=False,
        )

        rom = config["rom"]
        derived_datasets = {
            name: {
                "full_state_idx": i,
                "full_state_axis": rom["dataset_axis"],
            }
            for i, name in enumerate(rom["full_state_datasets"])
        }
        basis = RomBasis(
            basis_matrix=rom["U_map"],
            full_state_shape=tuple(rom["full_state_shape"]),
            derived_datasets=derived_datasets,
        )
        core = VeraOutCore(dataset_source=DictDatasetSource(config["core"]))
        vera_source = VeraDataSource(
            core=core,
            states=[DUMMY_STATE],
            provenance=f"stream from port {stream.port}",
            close_callback=stream.close,
        )
        scale_name = rom["scale_name"]
        return vera_source, basis, scale_name
    return None


def _receive_states(
    socket: zmq.Socket,
    stream: RomStream,
    vera_source: VeraDataSource,
    basis: RomBasis,
    scale_name: str,
):
    """Receive and append live ROM states."""
    has_received_state = False
    idx = 0
    socket.send(b"READY")
    while not stream._stop_signal.is_set():
        try:
            message_type, payload = socket.recv_multipart()
        except zmq.Again:
            continue

        if message_type != b"STATE":
            continue

        message = msgpack.unpackb(
            payload,
            object_hook=msgpack_numpy.decode,
            raw=False,
        )

        core = vera_source.core

        scalars = {key: value for key, value in message.items() if np.shape(value) == ()}

        ds_source = LiveRomDatasetSource(
            scalars=scalars,
            reduced_state=message["reduced_state"],
            basis=basis,
            scale=scalars[scale_name],
            provenance=f"rom stream {stream.stream_id}",
            dataset_dtypes=core.shape_to_dtype,
        )

        state = VeraOutState(
            source=ds_source,
            idx=idx + 1,
            core=core,
        )

        if vera_source.active_state_index == 0 and vera_source.active_state is DUMMY_STATE:
            vera_source.replace_state(
                idx=0,
                state=state,
            )
        else:
            vera_source.add_state(state)

        if stream.queue is not None and not has_received_state:
            has_received_state = True
            stream.queue.update({stream.stream_id: True})

        if stream.queue is not None and idx % 10 == 0:
            stream.queue.update(
                {
                    f"{stream.stream_id}_state_count": max(
                        len(vera_source.states) - 1,
                        0,
                    )
                }
            )
        idx += 1
        socket.send(b"READY")


def _data_receiver(stream: RomStream):
    context = zmq.Context()
    socket = context.socket(zmq.DEALER)

    socket.setsockopt(zmq.RCVTIMEO, 200)
    socket.connect(f"tcp://127.0.0.1:{stream.port}")

    try:
        socket.send(b"HELLO")
        config_result = _receive_config(
            socket,
            stream,
        )
        if config_result is None:
            return
        vera_source, basis, scale_name = config_result
        stream.source = vera_source
        _receive_states(
            socket,
            stream,
            vera_source,
            basis,
            scale_name,
        )

    except Exception as exc:
        stream.error = exc
        if stream.queue is not None:
            stream.queue.update({f"{stream.stream_id}_error": str(exc)})
    finally:
        socket.close()
        context.term()
