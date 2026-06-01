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
    
    def array(self, array_name: str, mask_reflected: bool = True) -> VeraDataset:
        # Get the array with the name "array_name", either on the active state,
        # or on the core.

        # These are on the core
        arrays_on_core = [
            "pin_volumes",
        ]
        if array_name in arrays_on_core:
            # This one is on the core
            return getattr(self.core, array_name)

        # If not on the core, assume it is on the active states.
        ax, ay = self.core.reduced_core_map.shape        
        array = getattr(self.active_state, array_name).copy()
        has_reflected_pins = array.dataset_type == VeraDatasetType.PIN
        if mask_reflected and has_reflected_pins and self.core.core_sym == 4 and array.ndim == 4: 
            # this is a "lazy" approach to fixing qtr core sym, could switch to eager later if necessary
            hpy = array.shape[0] // 2
            hpx = array.shape[1] // 2
            array[:hpy, :, :, :ax] = np.nan
            array[:, :hpx, :, self.core.reduced_core_map[:, 0] - 1] = np.nan
        return array

    @abstractmethod
    def add_new_diff_dataset(self, ref_array_name, comp_array_name, new_diff_name):
        pass

    @abstractmethod
    def add_new_derived_dataset(self, source_array_name, new_dataset_name, derivation: VeraDerivation):
        pass
