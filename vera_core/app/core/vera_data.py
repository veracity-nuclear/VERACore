from abc import ABC, abstractmethod
from enum import Enum, StrEnum
import numpy as np
from enum import StrEnum
class VeraDtype(Enum):
    """Dataset Identifiers"""
    PIN = 1
    ASSEMBLY = 2
    AXIAL = 3
    NODE = 4
    RADIAL = 5
    SCALAR = 6 
    CORE = 6 # CORE is an alias for SCALAR
    RADIAL_ASSEMBLY = 7
    CHANNEL = 8
    CHANNEL_RADIAL = 9
    RADIAL_NODE = 10
    UNKNOWN = 11

    def __str__(self):
        return self.name
    
    def is_channel(self):
        return self in (VeraDtype.CHANNEL, VeraDtype.CHANNEL_RADIAL)

class VeraAxes(Enum):
    """Derivation Axes"""
    ASSEMBLY = 1
    AXIAL = 2
    RADIAL = 3
    CORE = 4
    NODE = 5
    RADIAL_ASSEMBLY = 6
    RADIAL_NODE = 7

class DerivationMethod(StrEnum):
    AVERAGE = "Average"
    STDDEV = "Standard Deviation"
    RMS = "Root Mean Square"
class VeraDataset(np.ndarray):
    """Numpy array tagged with a vera dtype"""
    def __new__(cls, data, dataset_type : VeraDtype = VeraDtype.UNKNOWN):
        obj = np.asarray(data).view(cls)
        obj.dataset_type = dataset_type
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dataset_type = getattr(obj, "dataset_type", None)

def nan_out_reflected(reduced_core_map, core_sym, array):
    """Nans out reflected region if dataset has quarter core symmetry"""
    ax, ay = reduced_core_map.shape        
    has_reflected_pins = array.dataset_type in (VeraDtype.PIN, VeraDtype.CHANNEL, VeraDtype.RADIAL)
    if has_reflected_pins and core_sym == 4:
        array = array.copy()
        # this is a "lazy" approach to fixing qtr core sym, could switch to eager later if necessary
        hpy = array.shape[0] // 2
        hpx = array.shape[1] // 2
        match array.dataset_type:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                array[:hpy, :, :, :ax] = np.nan
                array[:, :hpx, :, reduced_core_map[:, 0] - 1] = np.nan
            case VeraDtype.RADIAL:
                array[:hpy, :, :ax] = np.nan
                array[:, :hpx, reduced_core_map[:, 0] - 1] = np.nan
    return array


def dataset_shape_category_dict(core_shape : tuple[int, int, int, int]) -> dict[tuple[int, ...], VeraDtype]:
    """Creates and retuns a dict mapping dataset shapes to dataset identifier (enums)"""
    npiny, npinx, nax, nass = core_shape
    channel_shape = (npiny + 1, npinx + 1, nax, nass)
    assembly_shape = (nax, nass)
    axial_shape = (nax,)
    radial_shape = (npiny, npinx, nass)
    node_shape = (4, nax, nass)
    radial_node_shape = (4, nass)
    radial_assembly_shape = (nass,)
    chan_radial_shape = (npiny + 1, npinx + 1, nass)
    return {
        core_shape : VeraDtype.PIN,
        channel_shape : VeraDtype.CHANNEL,
        assembly_shape : VeraDtype.ASSEMBLY,
        axial_shape : VeraDtype.AXIAL,
        radial_shape : VeraDtype.RADIAL,
        node_shape : VeraDtype.NODE,
        radial_assembly_shape : VeraDtype.RADIAL_ASSEMBLY,
        radial_node_shape : VeraDtype.RADIAL_NODE,
        chan_radial_shape : VeraDtype.CHANNEL_RADIAL,
        (1,) : VeraDtype.SCALAR,
        () : VeraDtype.SCALAR
    }
class VeraDataSource(ABC):
    """Abstract class representing a valid data source for VeraCore to visualize datasets from"""

    @property
    @abstractmethod
    def core(self):
        pass

    @property
    @abstractmethod
    def core_shape(self):
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
    def active_state_grouped_keys(self) -> list:
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
        array = getattr(self.active_state, array_name)
        if mask_reflected:
            array = nan_out_reflected(self.core.reduced_core_map, self.core.core_sym, array)
        return array
    
    def array_dtype(self, array_name : str) -> VeraDtype:
        arrays_on_core = [
            "pin_volumes",
        ]
        if array_name in arrays_on_core:
            # This one is on the core
            return getattr(self.core, array_name)
        if hasattr(self.active_state, array_name) and isinstance(getattr(self.active_state, array_name), VeraDataset):
            return getattr(self.active_state, array_name).dataset_type
        else:
            return VeraDtype.UNKNOWN


    @abstractmethod
    def add_new_diff_dataset(self, ref_array_name, comp_array_name, new_diff_name):
        pass

    @abstractmethod
    def add_new_derived_dataset(self, source_array_name: str, new_dataset_name: str, der_method : DerivationMethod, axes: VeraAxes):
        pass
