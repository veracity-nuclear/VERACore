import tempfile, os
from .vera_data import VeraDataSource
from .vera_out_file import VeraOutFile

class VeraDataRegistry:
    def __init__(self):
        self._srcs : dict[str, VeraDataSource] = {}
        self.default_src_id : str = None
    
    def add_src(self, src : VeraDataSource, src_id : str):
        if self.default_src_id is None:
            self.default_src_id = src_id
        if src_id in self._srcs:
            raise ValueError(f"{src_id} already exsists in the registry, skipped adding")
        self._srcs[src_id] = src
    
    def add_src_from_upload(self, name, content):
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.write(fd, content)
        os.close(fd)
        self.add_src(VeraOutFile(path), src_id=name)

    @property
    def default_src(self) -> VeraDataSource | None:
        if self.default_src_id is None:
            return None
        return self._srcs[self.default_src_id]
    
    @property
    def max_state(self) -> int:
        if not self._srcs:
            return 0
        return max(max(len(src.states) for src in self._srcs.values()) - 1, 0)

    def get(self, src_id: str) -> VeraDataSource:
        return self._srcs.get(src_id)
    
    def src_ids(self):
        return self._srcs.keys()
    
    def change_active_state(self, src_id : str, nstate : int):
        if src_id not in self._srcs:
            raise ValueError(f"Could not find {src_id} in registry")
        self._srcs[src_id].active_state_index = nstate

    def change_all_active_state(self, nstate: int):
        for src in self._srcs.values():
            src.active_state_index = nstate
    
    def full_core_keys(self):
        full_core_keys = {src_id : self._srcs[src_id].active_state_full_core_keys for src_id in self._srcs.keys()}
        return full_core_keys