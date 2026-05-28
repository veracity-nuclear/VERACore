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
