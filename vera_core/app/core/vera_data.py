from abc import ABC, abstractmethod
from enum import Enum 
import numpy as np
class VeraDatasetType(Enum):
    PIN = 1
    ASSEMBLY = 2
    AXIAL = 3
    NODE = 4
    RADIAL = 5
    CORE = 6
    SCALAR = 6 # SCALAR is an alias for CORE
    RADIAL_ASSEMBLY = 7

class VeraDerivation(Enum):
    ASSEMBLY = 1
    AXIAL = 2
    RADIAL = 3
    CORE = 4
    NODE = 5
    RADIAL_ASSEMBLY = 6
    RADIAL_NODE = 7

class VeraDataset(np.ndarray):
    def __new__(cls, data, dataset_type : VeraDatasetType = VeraDatasetType.PIN):
        obj = np.asarray(data).view(cls)
        obj.dataset_type = dataset_type
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dataset_type = getattr(obj, "dataset_type", None)

class VeraDataSource(ABC):

    @property
    @abstractmethod
    def core(self):
        pass

    @property
    @abstractmethod
    def states(self) -> list:
        pass

    @abstractmethod
    def close(self):
        pass

    @property
    @abstractmethod
    def active_state_full_core_keys(self) -> list:
        pass

    @property
    @abstractmethod
    def active_state(self):
        pass

    @property
    @abstractmethod
    def active_state_index(self) -> int:
        pass

    @active_state_index.setter
    @abstractmethod
    def active_state_index(self, index):
        pass
    
    @abstractmethod
    def array(self, array_name) -> VeraDataset:
        pass

    @abstractmethod
    def add_new_diff_dataset(self, ref_array_name, comp_array_name, new_diff_name):
        pass

    @abstractmethod
    def add_new_derived_dataset(self, source_array_name, new_dataset_name, derivation: VeraDerivation):
        pass


class VeraDataRegistry:
    def __init__(self):
        self._sources : dict[str, VeraDataSource] = {}
        self.default : str = None
    
    def add_source(self, source, source_id : str):
        if self.default is None:
            self.default = source_id
        self._sources[source_id] = source

    @property
    def default_source(self) -> VeraDataSource:
        return self._sources[self.default]
    
    def get(self, source_id: str) -> VeraDataSource:
        return self._sources.get(source_id)
    
    def source_ids(self):
        return self._sources.keys()
    
    def change_active_state(self, nstate: int):
        for source in self._sources.values():
            source.active_state_index = nstate
    
    def full_core_keys(self):
        full_core_keys = {source_id : self._sources[source_id].active_state_full_core_keys for source_id in self._sources.keys()}
        return full_core_keys