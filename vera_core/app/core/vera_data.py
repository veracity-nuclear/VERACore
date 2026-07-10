import string
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Union
from enum import Enum, StrEnum
import numpy as np
import h5py

NUM_ENERGY_GROUPS = 2
MAX_NUM_GROUPS = 8
NUM_DF = 6
NUM_NODES = 4
LATERAL_SURACES = slice(0,4)

@dataclass(frozen=True)
class _Info:
    axial_idx: int | None = None   # None = no axial axis
    computational: bool = False
    nodal: bool = False
    assembly: bool = False
    surface: bool = False
    channel: bool = False

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
    # all below use computational core map for shape
    COMP_NODAL = 12
    COMP_NODAL_ENERGY = 13
    COMP_NODAL_SURFACE = 14
    COMP_ASSY = 15
    COMP_ASSY_ENERGY = 16
    COMP_ASSY_SURFACE = 17

    def __str__(self):
        return self.name
    
    @property
    def str(self):
        return self.name
    
    @property
    def title(self):
        return str(self).upper()
    
    @property
    def _info(self):
        return _INFO[self]

    @property
    def axial_dim_idx(self):
        idx = self._info.axial_idx
        if idx is None:
            raise ValueError(f"{self} has no axial dimension")
        return idx

    def is_computational(self): return self._info.computational
    def is_nodal(self):         return self._info.nodal
    def is_assembly(self):      return self._info.assembly
    def is_surface(self):       return self._info.surface
    def is_channel(self):       return self._info.channel

# the single place per-dtype facts are declared
_INFO = {
    VeraDtype.PIN:                _Info(axial_idx=2),
    VeraDtype.ASSEMBLY:           _Info(axial_idx=0, assembly=True),
    VeraDtype.AXIAL:              _Info(axial_idx=0),
    VeraDtype.NODE:               _Info(axial_idx=1, nodal=True),
    VeraDtype.RADIAL:             _Info(),
    VeraDtype.SCALAR:             _Info(),                       # == CORE
    VeraDtype.RADIAL_ASSEMBLY:    _Info(assembly=True),
    VeraDtype.CHANNEL:            _Info(axial_idx=2, channel=True),
    VeraDtype.CHANNEL_RADIAL:     _Info(channel=True),
    VeraDtype.RADIAL_NODE:        _Info(nodal=True),
    VeraDtype.UNKNOWN:            _Info(),
    VeraDtype.COMP_NODAL:         _Info(axial_idx=1, computational=True, nodal=True),
    VeraDtype.COMP_NODAL_ENERGY:  _Info(axial_idx=2, computational=True, nodal=True),
    VeraDtype.COMP_NODAL_SURFACE: _Info(axial_idx=3, computational=True, nodal=True, surface=True),
    VeraDtype.COMP_ASSY:          _Info(axial_idx=1, computational=True, assembly=True),
    VeraDtype.COMP_ASSY_ENERGY:   _Info(axial_idx=2, computational=True, assembly=True),
    VeraDtype.COMP_ASSY_SURFACE:  _Info(axial_idx=3, computational=True, assembly=True, surface=True),
}
class VeraAxes(Enum):
    """Derivation Axes"""
    ASSEMBLY = 1
    AXIAL = 2
    RADIAL = 3
    CORE = 4
    NODE = 5
    RADIAL_ASSEMBLY = 6
    RADIAL_NODE = 7

class Surface(Enum):
    WEST = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    TOP = 4
    BOTTOM = 5
    
    @property
    def str(self):
        return self.name
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
        self.dataset_type : VeraDtype = getattr(obj, "dataset_type", VeraDtype.UNKNOWN)
    
    def is_computational(self) -> bool:
        return self.dataset_type.is_computational()
    
    def is_assembly(self) -> bool:
        return self.dataset_type.is_assembly()

def nan_out_reflected(cm, core_sym, array):
    """Nans out reflected region if dataset has quarter core symmetry"""
    ax, ay = cm.shape
    dtype = array.dataset_type
    has_reflected_pins = dtype in (VeraDtype.PIN, VeraDtype.CHANNEL, VeraDtype.RADIAL)
    if has_reflected_pins and core_sym == 4:
        array = array.copy()
        hpy = array.shape[0] // 2
        hpx = array.shape[1] // 2
        match array.dataset_type:
            case VeraDtype.PIN | VeraDtype.CHANNEL:
                array[:hpy, :, :, :ax] = np.nan
                array[:, :hpx, :, cm[:, 0] - 1] = np.nan
            case VeraDtype.RADIAL:
                array[:hpy, :, :ax] = np.nan
                array[:, :hpx, cm[:, 0] - 1] = np.nan
    elif dtype == VeraDtype.COMP_NODAL and core_sym == 4:
        array[:int(NUM_NODES/2), :, :ax] = np.nan
        array[0, :, cm[:, 0] - 1] = np.nan
        array[2, :, cm[:, 0] - 1] = np.nan
    elif dtype == VeraDtype.COMP_NODAL_ENERGY and core_sym == 4:
        array[:, :int(NUM_NODES/2), :, :ax] = np.nan
        array[:, 0, :, cm[:, 0] - 1] = np.nan
        array[:, 2, :, cm[:, 0] - 1] = np.nan
    return array

def build_core_dtypes(npiny = None, 
                                npinx = None, 
                                nax = None, 
                                nass = None,
                                comp_nax = None,
                                comp_nass = None,
) -> dict[tuple[int, ...], VeraDtype]:
    """Creates and retuns a dict mapping dataset shapes to dataset identifier (enums)"""
    shape_to_dtype = {}
    if nax and nass:
        shape_to_dtype |= {
            (nax, nass) : VeraDtype.ASSEMBLY,
            (nax,) : VeraDtype.AXIAL,
            (nass,) : VeraDtype.RADIAL_ASSEMBLY,
            (1,) : VeraDtype.SCALAR,
            () : VeraDtype.SCALAR
        }
    if npiny and npinx and nax and nass:
        shape_to_dtype |= {
            (npiny, npinx, nax, nass) : VeraDtype.PIN,
            (npiny + 1, npinx + 1, nax, nass) : VeraDtype.CHANNEL,
            (npiny, npinx, nass) : VeraDtype.RADIAL,
            (NUM_NODES, nax, nass) : VeraDtype.NODE,
            (NUM_NODES, nass) : VeraDtype.RADIAL_NODE,
            (npiny + 1, npinx + 1, nass) : VeraDtype.CHANNEL_RADIAL,
        }
    if comp_nax and comp_nass:
        shape_to_dtype |= {
            (NUM_NODES, comp_nax, comp_nass) : VeraDtype.COMP_NODAL,
            (NUM_DF, NUM_ENERGY_GROUPS, NUM_NODES, comp_nax, comp_nass) : VeraDtype.COMP_NODAL_SURFACE,
            (NUM_DF, 8, NUM_NODES, comp_nax, comp_nass) : VeraDtype.COMP_NODAL_SURFACE,
            (NUM_ENERGY_GROUPS, NUM_NODES, comp_nax, comp_nass) : VeraDtype.COMP_NODAL_ENERGY,
            (8, NUM_NODES, comp_nax, comp_nass) : VeraDtype.COMP_NODAL_ENERGY,
            (1, comp_nax, comp_nass) : VeraDtype.COMP_ASSY,
            (NUM_DF, NUM_ENERGY_GROUPS, 1, comp_nax, comp_nass) : VeraDtype.COMP_ASSY_SURFACE,
            (NUM_DF, 8, 1, comp_nax, comp_nass) : VeraDtype.COMP_ASSY_SURFACE,
            (NUM_ENERGY_GROUPS, 1, comp_nax, comp_nass) : VeraDtype.COMP_ASSY_ENERGY,
            (8, 1, comp_nax, comp_nass) : VeraDtype.COMP_ASSY_ENERGY
        }
    return shape_to_dtype

H5_ARRAY_TYPE = Union[h5py.Dataset, np.ndarray]

class LazyHDF5Loader:
    """Lazily exposes HDF5 datasets as attributes, reading on access.

    Datasets in _dataset_names are not held in memory by default. Accessing one
    that isn't cached falls through to __getattr__, which reads it transiently.
    _cache stores a dataset as an instance attribute. _uncache deletes
    that attribute so access reverts to the lazy
    """
    def __init__(
        self,
        f: "h5py.File",
        path: str,
        dataset_names: list[str],
        dataset_dtypes: dict[tuple, "VeraDtype"] | None = None,
    ):
        """Store the file handle and dataset names, start lazy.

        args:
            f: an open h5py.File handle
            path: the group path the datasets live under (e.g. "/CORE")
            dataset_names: names of the datasets this loader manages, these are cache and uncached
            dataset_dtypes: optional, shape -> VeraDtype map for typing datasets
        """
        self._f = f
        self._path = path
        self._dataset_names = dataset_names
        self._dataset_dtypes = dataset_dtypes

        self._uncache_all()  # start lazy
    
    def _in_h5(self, name : str) -> bool:
        return name in self._f[f"{self._path}"]

    def _load_dataset(self, name : str) -> "h5py.Dataset":
        """Return the raw h5py dataset handle for a name (no read)."""
        return self._f[f"{self._path}/{name}"]

    def _make_dataset(self, name : str) -> VeraDataset:
        """Load a full dataset into memory and wrap it as a typed VeraDataset.

        Scalars are stored as (1,) arrays for uniform access. The type is looked up from
        dataset_shapes if available.
        """
        if not self._in_h5(name):
            return None
        raw = self._load_dataset(name)[()]
        shape = np.shape(raw)
        dtype = VeraDtype.UNKNOWN
        if self._dataset_dtypes and self._dataset_dtypes.get(shape) is not None:
            dtype = self._dataset_dtypes.get(shape)
        arr = raw if isinstance(raw, np.ndarray) else np.array([raw])
        return VeraDataset(arr, dtype)

    def _cache(self, name) -> None:
        """Read a dataset and store it as an instance attribute."""
        if self._f is None:
            return
        if name not in self._dataset_names:
            raise AttributeError(name)
        if not self._in_h5(name):
            return
        setattr(self, name, self._make_dataset(name))

    def _uncache(self, name) -> None:
        """Drop a dataset's resident attribute so access reverts to lazy."""
        if self._f is None:
            return
        if name not in self._dataset_names:
            raise AttributeError(name)
        self.__dict__.pop(name, None)

    def _cache_all(self) -> None:
        """Load every managed dataset in memory."""
        for name in self._dataset_names:
            self._cache(name)

    def _uncache_all(self) -> None:
        """Drop every managed dataset back to the lazy."""
        for name in self._dataset_names:
            self._uncache(name)
    
    def has_dataset(self, name):
        """True if name is a managed dataset or an attribute.

        This is here to avoids the disk read that hasattr
        triggers via __getattr__
        """
        return name in self._dataset_names or name in self.__dict__

    def __getattr__(self, name : str):
        """Read a managed but uncached dataset transiently on attribute miss

        Only fires when normal lookup fails. The result is returned, not stored.
        """
        # Runs only when normal lookup fails (name was uncached/deleted).
        names = self.__dict__.get("_dataset_names")  # avoid recursion
        if names and name in names:
            return self._make_dataset(name)  # transient: not stored
        raise AttributeError(name)


def _make_ji_safe(j : int, i : int, array : np.ndarray):
    """Return j,i clipped to array"""
    max_row, max_col = np.shape(array)
    j = np.clip(j, 0, max_row - 1)
    i = np.clip(i, 0, max_col - 1)
    return j, i

def _nearest_nonzero_ij(array, j, i):
    """Nearest non-zero cell to (j, i) by squared Euclidean distance."""
    # rows are [row, col] == [j, i]
    cells = np.argwhere(array > 0)          
    if cells.size == 0:
        return None
    d = (cells[:, 0] - j) ** 2 + (cells[:, 1] - i) ** 2
    nj, ni = cells[np.argmin(d)]
    return int(nj), int(ni)

class VeraOutCore(LazyHDF5Loader):
    """Holds the core-level data for a VERA output file (the /CORE group).

    The four annotated attributes below double as the manifest of datasets to
    read from /CORE.
    Core data is small and shared by every state, so it is cached eagerly rather
    than lazily.
    """
    axial_mesh: H5_ARRAY_TYPE = None
    core_map: H5_ARRAY_TYPE = None
    core_sym: H5_ARRAY_TYPE = None
    pin_volumes: H5_ARRAY_TYPE = None

    def __init__(self, f: "h5py.File", aspect_ratio: float | None = None,):
        """Build the core from an open h5 file.

        Pass f to read the /CORE datasets from the file

        args:
            f: an open h5py.File handle (not a path)
            aspect_ratio: dx / dy of a pin cell
        """
        self.f = f
        super().__init__(f, "/CORE", list(self.__annotations__))
        self.aspect_ratio = f["/CORE/aspect_ratio"][()] if "aspect_ratio" in f["/CORE/"] else 1 # dx / dy
        self._cache_all()
        if not hasattr(self, "core_map") or self.core_map is None:
            raise RuntimeError("[ERROR] core_map not found in h5 file, unable to visualize data")
        if not hasattr(self, "axial_mesh") or self.axial_mesh is None:
            raise RuntimeError("[ERROR} axial_mesh not found in h5 file, unable to visualize data")
        self._determine_core_shape()
        self._determine_computational_core_shape()
        self._shape_to_dtype = build_core_dtypes(npiny=self.npy, npinx=self.npx, nax=self.nax, nass=self.nass,
                                                 comp_nax=self.comp_nax, comp_nass=self.comp_nass)
        self.compute_reduced_core_map()
        self.compute_axial_mesh_pixels()
        self.compute_control_rod_positions()
        self.compute_axial_mesh_means()

    def _determine_core_shape(self):
        cm = self.core_map
        self.nass = np.count_nonzero(np.unique(cm[~np.isnan(cm)]))
        self.nax = len(self.axial_mesh) - 1
        self.npy = 0 # if the core only contains assembly and axial data, then npy and npx will be zero
        self.npx = 0
        if hasattr(self, "pin_volumes") and self.pin_volumes is not None:
            self.npy, self.npx, nax, nass = np.shape(self.pin_volumes)
            if nax != self.nax or nass != self.nass:
                raise RuntimeError("core shape mismatch between core_map, axial_mesh and pin_volumes")
        elif "STATE_0001/pin_powers" in self.f:
            # if no pin_volumes see if state contains pin_powers as a source for core_shape
            self.npy, self.npx, nax, nass = np.shape(self.f["STATE_0001/pin_powers"])
            if nax != self.nax or nass != self.nass:
                raise RuntimeError("core shape mismatch between core_map, axial_mesh and pin_powers")
        
    def _determine_computational_core_shape(self):
        self.comp_nass = None
        self.comp_nax = None
        if "computational_core_map" not in self.f["CORE"]:
            print("Could not find computational_core_map, unable to determine computational core shape")
            return
        # FIXME vvvv, looks for computational axial mesh under NODAL_XS, this will need to be changed
        # when computational axial mesh moved to core
        if "STATE_0001/NODAL_XS/AXIALMESH" not in self.f:
            print("Could not find computational axial_mesh, unable to determine computational core shape")
            return
        self.comp_core_map = self.f["CORE/computational_core_map"][()]
        self.comp_nass = np.count_nonzero(np.unique(self.comp_core_map[~np.isnan(self.comp_core_map)]))
        self.comp_axial_mesh = self.f["STATE_0001/NODAL_XS/AXIALMESH"][()]
        self.comp_nax = len(self.comp_axial_mesh) - 1
        self.comp_core_map[np.isnan(self.comp_core_map)] = 0
        self._is_comp_rolled = np.count_nonzero(self.comp_core_map) == np.count_nonzero(np.unique(self.comp_core_map))
    
    def has_comp_core(self) -> bool:
        return hasattr(self, "comp_core_map") and self.comp_core_map is not None
    
    def has_comp_axial_mesh(self) -> bool:
        return hasattr(self, "comp_axial_mesh") and self.comp_axial_mesh is not None

    @property
    def core_shape(self) -> tuple[int, ...]:
        """Shape of core (num_piny, num_pinx, num_axial_levels, num_assemblys)"""
        return (self.npy, self.npx, self.nax, self.nass)
    
    @property
    def comp_core_shape(self) -> tuple[int, ...]:
        """Shape of computational core (num_piny, num_pinx, num_computational_axial_levels, num_computational_assemblys)"""
        return (self.npy, self.npx, self.comp_nax, self.comp_nass)
    
    def core_dtypes(self, dataset_shape : tuple[int, ...]) -> VeraDtype:
        """Get VeraDtype associated with dataset_shape
        
        args:
            dataset_shape: the shape of the dataset

        Return VeraDtype.UKNOWN if shape is not known.
        """
        return self._shape_to_dtype.get(dataset_shape, VeraDtype.UNKNOWN)
    
    def compute_reduced_core_map(self) -> None:
        """Compute the reduced core map based upon the core_sym"""
        sym = self.core_sym[()] 
        has_comp_core = self.has_comp_core()
        if sym == 1:
            self.reduced_core_map = self.core_map[:].copy()
            self.reduced_core_map_start_index = 0   
            self.comp_map_start_index = 0             
        elif sym == 4:
            w, h = self.core_map[:].shape
            start_w = w // 2
            start_h = h // 2
            self.reduced_core_map = self.core_map[start_w:, start_h:]
            self.reduced_core_map_start_index = start_w
            if has_comp_core and not self._is_comp_rolled:
                print("comp not rolled")
                cw, ch = self.comp_core_map[:].shape
                cstart_w = cw // 2
                cstart_h = ch // 2
                self.comp_core_map = self.comp_core_map[cstart_w:, cstart_h:]
                self.comp_map_start_index = cstart_w
            elif self.has_comp_core:
                self.comp_map_start_index = start_w
        else:
            raise Exception(f"Unhandled symmetry: {sym}")

        num_cols = self.reduced_core_map.shape[1]
        alphabet = [*string.ascii_uppercase]
        self.reduced_core_map_column_labels = list(reversed(alphabet[:num_cols]))
        if self.has_comp_core():
            comp_num_cols = self.comp_core_map.shape[1]
            cols_dif = comp_num_cols - num_cols
            if cols_dif > 0:
                self.comp_core_map_column_labels = (
                    self.reduced_core_map_column_labels + list(reversed(alphabet[num_cols:comp_num_cols]))
                )
            else:
                self.comp_core_map_column_labels = list(reversed(alphabet[:comp_num_cols]))

    def compute_axial_mesh_pixels(self) -> None:
        """Compute the number of pixels that we will be displaying in
        the axial direction for each length in the axial mesh.
        """
        diff_array = np.diff(self.axial_mesh[:])

        # The min diff will be three pixels high. The rest will be computed based
        # upon the min diff.
        MIN_DIFF_PIXELS_HEIGHT = 3
        pixel_height = np.min(diff_array) / MIN_DIFF_PIXELS_HEIGHT
        pixel_height_array = diff_array / pixel_height
        self.axial_mesh_pixels = np.round(pixel_height_array).astype(np.int64)

        if self.has_comp_axial_mesh():
            comp_diff_array = np.diff(self.comp_axial_mesh)
            comp_pixel_height = np.min(comp_diff_array) / MIN_DIFF_PIXELS_HEIGHT
            comp_pixel_height_array = comp_diff_array / comp_pixel_height
            self.comp_axial_mesh_pixels = np.round(comp_pixel_height_array).astype(np.int64)

    def compute_control_rod_positions(self) ->None:
        """Locate control-rod positions as the pins with zero volume.
        Assumes the rod layout is identical in every axial volume.
        """
        if self.pin_volumes is not None:
            first_volume = self.pin_volumes[:, :, 0, 0]
            self.control_rod_positions = np.where(first_volume == 0)
        else:
            self.control_rod_positions = None

    def compute_axial_mesh_means(self):
        """Midpoint between each pair of neighboring mesh boundaries."""
        self.axial_mesh_means = self._midpoints(self.axial_mesh)
        if self.has_comp_axial_mesh():
            self.comp_axial_mesh_means = self._midpoints(self.comp_axial_mesh)
            self.gross_axial_mesh = np.union1d(self.axial_mesh_means, self.comp_axial_mesh_means)
        else:
            self.gross_axial_mesh = self.axial_mesh_means

    @staticmethod
    def _midpoints(mesh, decimals=4):
        mesh = np.asarray(mesh, dtype=np.float64)
        return np.round((mesh[:-1] + mesh[1:]) / 2.0, decimals)

    def row_assembly_indices(self, assembly_idx, is_comp=False) -> np.ndarray:
        """Get indices of all assemblies in the same row as this assembly"""
        # The core map and reduced core map use 1-based indexing
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        row = np.where(cm == assembly_idx + 1)[0][0]
        ids = cm[row]
        # Remove any zeros
        ids = ids[ids > 0]
        return ids - 1

    def col_assembly_indices(self, assembly_idx, is_comp=False) -> np.ndarray:
        """Get indices of all assemblies in the same column as this assembly"""
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        col = np.where(cm == assembly_idx + 1)[1][0]
        ids = cm[:, col]
        # Remove any zeros
        ids = ids[ids > 0]
        return ids - 1

    def reduced_core_map_assembly(self, i, j, is_comp=False) -> int:
        """Get the index of the assembly at reduced core map position i, j
        
        clamps to the nearest real assembly in the same map, so the result
        is always a valid assembly when the map has any. Returns -1 only if the
        chosen map has no assemblies at all.
        """
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        j, i = _make_ji_safe(j, i, cm)
        snapped = _nearest_nonzero_ij(cm, j, i)
        if snapped is None:
            return -1
        j, i = snapped
        return int(cm[j, i]) - 1    
    
    def reduced_core_map_ij(self, assembly_idx, is_comp=False) -> tuple[int, int]:
        """Return the (column, row) position of an assembly in the reduced map."""
        target = assembly_idx + 1
        cm = self.comp_core_map if is_comp and self.has_comp_core() else self.reduced_core_map
        rows, cols = np.where(cm == target)
        if len(rows) == 0:
            raise ValueError(
                f"Assembly index {assembly_idx} was not found in reduced_core_map. \nLooked for value {target}."
            )
        if len(rows) > 1:
            raise ValueError(
                f"Assembly index {assembly_idx} appears multiple times in reduced_core_map. \n Looked for value {target}; found {len(rows)} matches."
            )
        j = int(rows[0])
        i = int(cols[0])
        return i, j

    def reduced_core_map_label(self, assembly_idx, is_comp=False) -> str:
        """Return the combined column-row label for an assembly (e.g. C-9)."""
        row_label = self.reduced_core_map_row_label(assembly_idx, is_comp)
        col_label = self.reduced_core_map_column_label(assembly_idx, is_comp)
        return f"{col_label}-{row_label}"

    def reduced_core_map_row_label(self, assembly_idx, is_comp=False) -> str:
        """Return the row-number label for an assembly."""
        i, j = self.reduced_core_map_ij(assembly_idx, is_comp)
        start_index = self.comp_map_start_index if is_comp and self.has_comp_core() else self.reduced_core_map_start_index
        row_len = len(self.core_map) + 1 if not is_comp else len(self.comp_core_map) + start_index + 1
        rows = list(range(start_index + 1, row_len))
        return str(rows[j])

    def reduced_core_map_column_label(self, assembly_idx, is_comp=False) -> str:
        """Return the column-letter label for an assembly."""
        i, j = self.reduced_core_map_ij(assembly_idx, is_comp)
        labels = self.reduced_core_map_column_labels[i] if not is_comp else self.comp_core_map_column_labels[i]
        return labels
        

class VeraOutState(LazyHDF5Loader):
    """Stores the datasets for a single VERA STATE_NNNN point.

    Datasets are discovered and categorized by shape, then loaded lazily
    through LazyHDF5Loader. Diff and derived datasets added after
    construction live on the instance but are never written back to the file.
    """
    def __init__(self, f : "h5py.File", 
                 idx : int,
                 core : VeraOutCore  
    ):
        """Build a state either from an open h5 file or from raw Python data.

        Pass either (f, idx) to read from a file, or
        (data) to construct in memory.

        args:
            f: an open h5py.File handle (not a path), kept open by the owner
            idx: the state number, formatted into the /STATE_{idx:04} group
            core: reference to VeraOutCore object that contains core data for this state
        """
        # These are the attributes that will be read from the HDF5 file
        self.categorized_ds_names = {dataset_type : set() for dataset_type in VeraDtype}
        self.dataset_dtypes = {}
        self._core = core
        self._index = idx

        self.__annotations__ = dict()
        state = f[f"/STATE_{idx:04}"]
        self._search_for_datasets(state)
        self.all_datasets = [dataset for category in self.categorized_ds_names.values() for dataset in category]

        super().__init__(f, f"/STATE_{idx:04}", self.all_datasets, dataset_dtypes=self.dataset_dtypes)
        self.diff_datasets = dict()
        self.derived_datasets = dict()
    
    @property
    def core(self) -> VeraOutCore:
        """Reference to this state's core data"""
        return self._core

    @property
    def scalar_datasets(self) -> list[str]:
        """List of dataset names categorized as SCALAR."""
        return sorted(self.categorized_ds_names[VeraDtype.SCALAR])

    @property
    def grouped_full_core_keys(self) -> list[tuple[str, list[str]]]:
        """Group the available dataset names by category for grouped UI display.

        Builds in _CATEGORY_ORDER and, for each one that has any
        datasets, returns a (category_name, sorted_names) pair. Empty categories
        are skipped.
        """
        groups = []
        for dtype in VeraDtype:
            names = sorted(self.categorized_ds_names[dtype])
            if names:
                groups.append((str(dtype), names))
        return groups

    @property
    def full_core_keys(self) -> list[str]:
        """Flat, category-ordered list of all dataset names."""
        return [name for _, names in self.grouped_full_core_keys for name in names]
    
    def _search_for_datasets(self, data):
        """Find all datasets with known shape and categorize them by VeraDdtype for one state."""
        if "pin_powers" in data and np.shape(data["pin_powers"]) != self.core.core_shape:
            raise RuntimeError(f"Mismatch between the shape of STATE_{self._index:04}'s data and the core shape")
        
        def _loop_through_datasets(h5_group, group_name=""):
            for ds_name in h5_group.keys():
                ds = h5_group[ds_name]
                full_name = "/".join(part for part in (group_name, ds_name) if part)
                if isinstance(ds, h5py.Group): # recurse on group (subdir)
                    _loop_through_datasets(ds, group_name=full_name)
                    continue
                ds_shape = np.shape(ds)
                ds_dtype = self.core.core_dtypes(ds_shape)
                if ds_shape == VeraDtype.UNKNOWN:
                    continue
                self.categorized_ds_names[ds_dtype].add(full_name)
                self.dataset_dtypes[ds_shape] = ds_dtype
        _loop_through_datasets(data)
    
    def add_diff_dataset(self, dataset_name: str, dataset : VeraDataset)-> None:
        """Attach an in-memory diff dataset to this state.

        The dataset is set as an attribute and tracked in diff_datasets.
        It is not categorized and not written to the file.
        """
        setattr(self, dataset_name, dataset)
        self.diff_datasets.update({dataset_name: "H5_ARRAY_TYPE = NONE"})
        ds_dtype = dataset.dataset_type
        if ds_dtype == VeraDtype.UNKNOWN:
            return
        self.categorized_ds_names[ds_dtype].add(dataset_name)

    def add_derived_dataset(self, dataset_name: str, dataset : VeraDataset) -> None:
        """Attach an in-memory derived dataset to this state.

        Like add_diff_dataset, but also categorizes the dataset by shape so it
        shows up in the grouped key listings. Not written to the file.
        """
        setattr(self, dataset_name, dataset)
        self.derived_datasets.update({dataset_name: "H5_ARRAY_TYPE = NONE"})
        ds_dtype = dataset.dataset_type
        if ds_dtype == VeraDtype.UNKNOWN:
            return
        self.categorized_ds_names[ds_dtype].add(dataset_name)
class VeraDataSource(ABC):
    """Abstract class representing a valid data source for VeraCore to visualize datasets from
    
    Concrete sources expose a single core, an ordered list of states with one
    active at a time, and lookup of named arrays from either the active state
    or the core.
    """

    @property
    @abstractmethod
    def file_path(self) -> str:
        """Raw path to file on disk"""
        pass

    @property
    @abstractmethod
    def core(self) -> VeraOutCore:
        """The core-level data shared across all states."""
        pass

    @property
    @abstractmethod
    def states(self) -> list[VeraOutState]:
        """All state points in the source, in order."""
        pass

    @abstractmethod
    def close(self):
        """Release any open handles/ports/etc held by the source."""
        pass

    @property
    @abstractmethod
    def active_state_full_core_keys(self) -> list:
        """Flat list of dataset names available on the active state."""
        pass

    @property
    @abstractmethod
    def active_state_grouped_keys(self) -> list:
        """Active-state dataset names grouped by category for UI display."""
        pass

    @property
    @abstractmethod
    def active_state(self) -> VeraOutState:
        """The currently selected state."""
        pass

    @property
    @abstractmethod
    def active_state_index(self) -> int:
        """Index of the active state within states."""
        pass

    @active_state_index.setter
    @abstractmethod
    def active_state_index(self, index):
        """Set the active state, switching which state's data is exposed."""
        pass
    
    def array(self, array_name: str, mask_reflected: bool = True) -> VeraDataset:
        """Return a named array from the core or the active state.

        Core arrays (e.g. pin_volumes) come from the core. 
        Everything else comes from the active state, with reflected positions masked to NaN unless mask_reflected is False.
        """

        # These are on the core
        arrays_on_core = [
            "pin_volumes",
        ]
        if array_name in arrays_on_core:
            # This one is on the core
            return getattr(self.core, array_name)
        array : VeraDataset = getattr(self.active_state, array_name)
        cm = self.core.reduced_core_map if not array.dataset_type.is_computational() else self.core.comp_core_map
        if mask_reflected:
            array = nan_out_reflected(cm, self.core.core_sym, array)
        return array
    
    def array_dtype(self, array_name : str) -> VeraDtype:
        """Return the VeraDtype of a named array, or UNKNOWN if not found.
        Resolves against the core for core arrays and the active state otherwise.
        """
        arrays_on_core = [
            "pin_volumes",
        ]
        if array_name in arrays_on_core:
            # This one is on the core
            return getattr(self.core, array_name).dataset_type
        if self.active_state.has_dataset(array_name) and isinstance(getattr(self.active_state, array_name), VeraDataset):
            return getattr(self.active_state, array_name).dataset_type
        else:
            return VeraDtype.UNKNOWN

    @abstractmethod
    def add_new_diff_dataset(self, ref_dataset_name: str, comp_src : "VeraDataSource", comp_dataset_name: str, new_diff_name: str, interpolation_order : int = 1):
        """Create a difference dataset (ref minus comp) on each state."""
        pass

    @abstractmethod
    def add_new_derived_dataset(self, source_array_name: str, new_dataset_name: str, der_method : DerivationMethod, axes: VeraAxes):
        """Create a derived dataset from a source array using a reduction over axes."""
        pass
