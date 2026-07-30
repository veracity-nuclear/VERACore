"""FIXME major design decisions around the stream are tbd"""

import threading

import msgpack
import msgpack_numpy as m
import zmq
from trame.app.asynchronous import StateQueue

from .vera_data import (
    DerivationMethod,
    VeraAxes,
    VeraDataSource,
    VeraOutCore,
    VeraOutState,
)

m.patch()


def generate_stream_identifier(stream_id: str | int | float) -> str:
    return f"vera_data_stream_{stream_id}"


class VeraDataStream(VeraDataSource):
    def __init__(
        self, stream_name: str, port_to_listen_on, state_queue: StateQueue = None
    ):
        context = zmq.Context()
        self.subscriber = context.socket(zmq.SUB)
        self.subscriber.connect(f"tcp://127.0.0.1:{port_to_listen_on}")
        self.subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        self.stream_id = generate_stream_identifier(stream_name)
        self._core = None
        self._states = []
        self._active_state_index = 0
        self._queue = state_queue
        self._lock = threading.Lock()

    def start(self):
        self._thread = threading.Thread(target=self._data_reciever, daemon=True)
        self._thread.start()

    def _setup_core_data(self, core: VeraOutCore):
        self._core = core
        self._core_shape = core.pin_volumes.shape
        if len(self.core_shape) != 4:
            raise ValueError(
                "[ERROR] Core shape should have 4 dimensions. Unbale to determine core dimensions."
            )

    def _data_reciever(self):
        has_recieved_state = False
        while True:
            message = msgpack.unpackb(self.subscriber.recv(), raw=False)
            if self._core is None:
                self._setup_core_data(
                    VeraOutCore.from_data(**message["core"], aspect_ratio=1)
                )
            with self._lock:
                self._states.append(VeraOutState.from_data(message["datasets"]))
                if self._queue is not None:
                    if not has_recieved_state:
                        self._queue.update({self.stream_id: True})
                        has_recieved_state = True
                    self._queue.update(
                        {f"{self.stream_id}_state_count": max(len(self._states) - 1, 0)}
                    )

    @property
    def core(self):
        return self._core

    @property
    def core_shape(self):
        return self._core_shape

    def close(self):
        self.subscriber.close()

    @property
    def active_state(self):
        with self._lock:
            if len(self._states) == 0:
                return None
            return self._states[self._active_state_index]

    @property
    def states(self):
        return self._states

    @property
    def active_state_index(self):
        return self._active_state_index

    @property
    def active_state_full_core_keys(self):
        state = self.active_state
        return state.full_core_keys if state is not None else []

    @property
    def active_state_grouped_keys(self):
        return self.active_state.grouped_full_core_keys

    @active_state_index.setter
    def active_state_index(self, index):
        with self._lock:
            index = max(0, min(index, len(self._states) - 1))
            if self._active_state_index == index:
                return
            self._active_state_index = index

    def add_new_diff_dataset(
        self,
        ref_dataset_name: str,
        comp_src: VeraDataSource,
        comp_dataset_name: str,
        new_diff_name: str,
        interpolation_order: int = 1,
    ):
        pass

    def add_new_derived_dataset(
        self,
        source_array_name: str,
        new_dataset_name: str,
        der_method: DerivationMethod,
        axes: VeraAxes,
    ):
        pass
