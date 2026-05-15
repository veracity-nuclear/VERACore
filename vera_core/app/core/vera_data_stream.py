from .vera_data_source import VeraDataSource
from .vera_out_file import VeraOutCore, VeraOutFile, VeraOutState
import threading, time
from multiprocessing import Queue
from trame.app.asynchronous import StateQueue
import h5py

import zmq
import msgpack
import msgpack_numpy as m

m.patch()

context = zmq.Context()
subscriber = context.socket(zmq.SUB)

subscriber.connect("tcp://127.0.0.1:8000")

subscriber.setsockopt_string(zmq.SUBSCRIBE, "")

# while True:
#     message = msgpack.unpackb(subscriber.recv(), raw=False)
#     print(message.keys())


class VirtualVeraDataStream(VeraDataSource):
    def __init__(self, filename: str, state_queue: StateQueue = None):
        self.f = h5py.File(filename, "r")
        state_keys = [key for key in self.f if key.startswith("STATE_")]
        indices = [int(key.split("_")[1]) for key in state_keys]
        self._core = VeraOutCore(self.f)
        self._core._cache_all()
        self._states_to_add = [VeraOutState(self.f, idx) for idx in indices]
        self._states = [self._states_to_add[0]]
        self._active_state_index = 0
        self._queue = state_queue
        self._thread = threading.Thread(target=self._data_runner, daemon=True)
        self._thread.start()

    def _data_runner(self):
        state_counter = 1
        for state in self._states_to_add[1:]:
            self._states.append(state)
            state_counter += 1
            if hasattr(self, "_queue") and self._queue is not None:
                self._queue.update({"max_time" : max(len(self._states) - 1, 0)})
            time.sleep(1)

    @property
    def core(self):
        return self._core
    
    def close(self):
        self.f.close()

    @property
    def active_state(self):
        return self._states[self.active_state_index]
    
    @property
    def states(self):
        return self._states

    @property
    def active_state_index(self):
        return self._active_state_index

    @active_state_index.setter
    def active_state_index(self, index):
        if hasattr(self, "_active_state_index"):
            if self._active_state_index == index:
                return
            else:
                # Clear the cache from the active state
                self.active_state._uncache_all()

        self._active_state_index = index
        self.active_state._cache_all()
    
    def array(self, array_name):
        arrays_on_core = [
            "pin_volumes",
        ]
        if array_name in arrays_on_core:
            # This one is on the core
            return getattr(self.core, array_name)

        # If not on the core, assume it is on the active states.
        return getattr(self.active_state, array_name)

class VeraDataStream(VeraDataSource):
    def __init__(self, port_to_listen_on, state_queue = None):
        context = zmq.Context()
        subscriber = context.socket(zmq.SUB)
        subscriber.connect(f"tcp://127.0.0.1:{port_to_listen_on}")
        subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self._core = None
        self._states = []
        self._active_state_index = 0
        self._queue = state_queue
        self._thread = threading.Thread(target=self._data_reciever, daemon=True)
        self._thread.start()

    def _data_reciever(self):
        state_counter = 1
        while True:
            message = msgpack.unpackb(subscriber.recv(), raw=False)
            if self._core is None:
                self._core = VeraOutCore.from_data(**message["core"])
            self._states.append(VeraOutState.from_data(**message["data"]))
            if self._queue is not None:
                self._queue.update({"max_time" : max(len(self._states) - 1, 0)})



        for state in self._states_to_add[1:]:
            self._states.append(state)
            state_counter += 1
            if hasattr(self, "_queue") and self._queue is not None:
                self._queue.update({"max_time" : max(len(self._states) - 1, 0)})
            time.sleep(1)

    @property
    def core(self):
        return self._core
    
    def close(self):
        self.f.close()

    @property
    def active_state(self):
        return self._states[self.active_state_index]
    
    @property
    def states(self):
        return self._states

    @property
    def active_state_index(self):
        return self._active_state_index

    @active_state_index.setter
    def active_state_index(self, index):
        if hasattr(self, "_active_state_index"):
            if self._active_state_index == index:
                return
        self._active_state_index = index
    
    def array(self, array_name):
        arrays_on_core = [
            "pin_volumes",
        ]
        if array_name in arrays_on_core:
            # This one is on the core
            return getattr(self.core, array_name)

        # If not on the core, assume it is on the active states.
        return getattr(self.active_state, array_name)
