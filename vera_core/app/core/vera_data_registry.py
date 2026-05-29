import tempfile, os
from .vera_data import VeraDataSource
from .vera_out_file import VeraOutFile

class VeraDataRegistry:
    def __init__(self):
        self._sources : dict[str, VeraDataSource] = {}
        self.default : str = None
    
    def add_source(self, source : VeraDataSource, source_id : str):
        if self.default is None:
            self.default = source_id
        if source_id in self._sources:
            raise ValueError(f"{source_id} already exsists in the registry, skipped adding")
        self._sources[source_id] = source
    
    def add_source_from_upload(self, name, content):
        fd, path = tempfile.mkstemp(suffix=".h5")
        os.write(fd, content)
        os.close(fd)
        self.add_source(VeraOutFile(path), source_id=name)

    @property
    def default_source(self) -> VeraDataSource | None:
        if self.default is None:
            return None
        return self._sources[self.default]
    
    def get(self, source_id: str) -> VeraDataSource:
        return self._sources.get(source_id)
    
    def source_ids(self):
        return self._sources.keys()
    
    def change_active_state(self, source_id : str, nstate : int):
        if source_id not in self._sources:
            raise ValueError(f"Could not find {source_id} in registry")
        self._sources[source_id].active_state_index = nstate

    def change_all_active_state(self, nstate: int):
        for source in self._sources.values():
            source.active_state_index = nstate
    
    def full_core_keys(self):
        full_core_keys = {source_id : self._sources[source_id].active_state_full_core_keys for source_id in self._sources.keys()}
        return full_core_keys