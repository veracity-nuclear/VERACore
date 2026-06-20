import string
from abc import ABC, abstractmethod
from typing import Union
from enum import Enum, StrEnum
import numpy as np
import h5py
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
        dataset_shapes: dict[tuple, "VeraDtype"] | None = None,
    ):
        """Store the file handle and dataset names, start lazy.

        args:
            f: an open h5py.File handle
            path: the group path the datasets live under (e.g. "/CORE")
            dataset_names: names of the datasets this loader manages, these are cache and uncached
            dataset_shapes: optional, shape -> VeraDtype map for typing datasets
        """
        self._f = f
        self._path = path
        self._dataset_names = dataset_names
        self._dataset_shapes = dataset_shapes

        self._uncache_all()  # start lazy

    def _load_dataset(self, name : str) -> "h5py.Dataset":
        """Return the raw h5py dataset handle for a name (no read)."""
        return self._f[f"{self._path}/{name}"]

    def _make_dataset(self, name : str) -> VeraDataset:
        """Load a full dataset into memory and wrap it as a typed VeraDataset.

        Scalars are stored as (1,) arrays for uniform access. The type is looked up from
        dataset_shapes if available.
        """
        raw = self._load_dataset(name)[()]
        shape = np.shape(raw)
        dtype = VeraDtype.UNKNOWN
        if self._dataset_shapes and self._dataset_shapes.get(shape) is not None:
            dtype = self._dataset_shapes.get(shape)
        arr = raw if isinstance(raw, np.ndarray) else np.array([raw])
        return VeraDataset(arr, dtype)

    def _cache(self, name) -> None:
        """Read a dataset and store it as an instance attribute."""
        if self._f is None:
            return
        if name not in self._dataset_names:
            raise AttributeError(name)
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

    def __init__(
        self,
        f: "h5py.File | None" = None,
        axial_mesh: np.ndarray | None = None,
        core_map: np.ndarray | None = None,
        core_sym: np.ndarray | None = None,
        pin_volumes: np.ndarray | None = None,
        aspect_ratio: float | None = None,
    ):
        """Build the core from an open h5 file or from raw data.

        Pass f to read the /CORE datasets from the file, or pass all of
        axial_mesh, core_map, core_sym, pin_volumes, and aspect_ratio to
        construct in memory.

        args:
            f: an open h5py.File handle (not a path)
            axial_mesh: axial mesh boundaries
            core_map: assembly layout, 1-based ids with 0 for empty positions
            core_sym: core symmetry flag (1 = full, 4 = quarter)
            pin_volumes: per-pin volumes, used to locate control-rod positions
            aspect_ratio: dx / dy of a pin cell
        """
        if f is not None:
            super().__init__(f, "/CORE", list(self.__annotations__))
            self.aspect_ratio = f["/CORE/aspect_ratio"][()] if "aspect_ratio" in f["/CORE/"] else 1 # dx / dy
            self._cache_all()
        elif (
            axial_mesh is not None 
            and core_map is not None 
            and core_sym is not None 
            and pin_volumes is not None 
            and aspect_ratio is not None
        ):
            self.axial_mesh = axial_mesh
            self.core_map = core_map
            self.core_sym = core_sym
            self.pin_volumes = pin_volumes
            self.aspect_ratio = aspect_ratio
        else:
            raise ValueError("Either a filename or raw data must be provided")
        self.compute_reduced_core_map()
        self.compute_axial_mesh_pixels()
        self.compute_control_rod_positions()
        self.compute_axial_mesh_means()

    @classmethod
    def from_h5(cls, f: "h5py.File") -> "VeraOutCore":
        """Build a core from an open h5 file handle."""
        return cls(f=f)

    @classmethod
    def from_data(
        cls,
        axial_mesh: np.ndarray,
        core_map: np.ndarray,
        core_sym: np.ndarray,
        pin_volumes: np.ndarray,
        aspect_ratio: float,
    ) -> "VeraOutCore":
        """Build a core from in-memory arrays instead of a file."""
        return cls(axial_mesh=axial_mesh, core_map=core_map, core_sym=core_sym, pin_volumes=pin_volumes, aspect_ratio=aspect_ratio)

    def compute_reduced_core_map(self) -> None:
        """Compute the reduced core map based upon the core_sym"""
        sym = self.core_sym[()] 
        if sym == 1:
            self.reduced_core_map = self.core_map[:].copy()
            self.reduced_core_map_start_index = 0
        elif sym == 4:
            w, h = self.core_map[:].shape
            start_w = w // 2
            start_h = h // 2
            self.reduced_core_map = self.core_map[start_w:, start_h:]
            self.reduced_core_map_start_index = start_w
        else:
            raise Exception(f"Unhandled symmetry: {sym}")

        num_cols = self.reduced_core_map.shape[1]
        alphabet = [*string.ascii_uppercase]
        self.reduced_core_map_column_labels = list(reversed(alphabet[:num_cols]))

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

    def compute_control_rod_positions(self) ->None:
        """Locate control-rod positions as the pins with zero volume.
        Assumes the rod layout is identical in every axial volume.
        """
        first_volume = self.pin_volumes[:, :, 0, 0]
        self.control_rod_positions = np.where(first_volume == 0)

    def compute_axial_mesh_means(self):
        """Compute the mean between each neighbor"""
        repeats = [2] * len(self.axial_mesh)
        repeats[0] = 1
        repeats[-1] = 1

        repeated_mesh = np.repeat(self.axial_mesh, repeats)
        reshaped = repeated_mesh.reshape((repeated_mesh.shape[0] // 2, 2))

        self.axial_mesh_means = np.mean(reshaped, axis=1)

    def row_assembly_indices(self, assembly_idx) -> np.ndarray:
        """Get indices of all assemblies in the same row as this assembly"""
        # The core map and reduced core map use 1-based indexing
        row = np.where(self.reduced_core_map == assembly_idx + 1)[0][0]
        ids = self.reduced_core_map[row]
        # Remove any zeros
        ids = ids[ids > 0]
        return ids - 1

    def col_assembly_indices(self, assembly_idx) -> np.ndarray:
        """Get indices of all assemblies in the same column as this assembly"""
        col = np.where(self.reduced_core_map == assembly_idx + 1)[1][0]
        ids = self.reduced_core_map[:, col]
        # Remove any zeros
        ids = ids[ids > 0]
        return ids - 1

    def reduced_core_map_assembly(self, i, j) -> int:
        """Get the index of the assembly at reduced core map position i, j"""
        return int(self.reduced_core_map[j, i] - 1)

    def reduced_core_map_ij(self, assembly_idx) -> tuple[int, int]:
        """Return the (column, row) position of an assembly in the reduced map."""
        target = assembly_idx + 1
        rows, cols = np.where(self.reduced_core_map == target)
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

    def reduced_core_map_label(self, assembly_idx) -> str:
        """Return the combined column-row label for an assembly (e.g. C-9)."""
        row_label = self.reduced_core_map_row_label(assembly_idx)
        col_label = self.reduced_core_map_column_label(assembly_idx)
        return f"{col_label}-{row_label}"

    def reduced_core_map_row_label(self, assembly_idx) -> str:
        """Return the row-number label for an assembly."""
        i, j = self.reduced_core_map_ij(assembly_idx)
        start_index = self.reduced_core_map_start_index
        rows = list(range(start_index + 1, len(self.core_map) + 1))
        return str(rows[j])

    def reduced_core_map_column_label(self, assembly_idx) -> str:
        """Return the column-letter label for an assembly."""
        i, j = self.reduced_core_map_ij(assembly_idx)
        return self.reduced_core_map_column_labels[i]


class VeraOutState(LazyHDF5Loader):
    """Stores the datasets for a single VERA STATE_NNNN point.

    Datasets are discovered and categorized by shape, then loaded lazily
    through LazyHDF5Loader. Diff and derived datasets added after
    construction live on the instance but are never written back to the file.
    """
    def __init__(self, f : "h5py.File | None" = None, idx : int | None = None, data : dict[str, np.ndarray] | None = None):
        """Build a state either from an open h5 file or from raw Python data.

        Pass either (f, idx) to read from a file, or
        (data) to construct in memory.

        args:
            f: an open h5py.File handle (not a path), kept open by the owner
            idx: the state number, formatted into the /STATE_{idx:04} group
            data: name -> vera dataset array map
        """
        # These are the attributes that will be read from the HDF5 file
        self.dataset_shapes = dict()
        self.dataset_categories = { str(dataset_type) : set() for dataset_type in VeraDtype}
        if f is not None:
            self.__annotations__ = dict()
            self._search_for_datasets_in_file(f, idx)
            self.all_datasets = [dataset for category in self.dataset_categories.values() for dataset in category]
            super().__init__(f, f"/STATE_{idx:04}", self.all_datasets, dataset_shapes=self.dataset_shapes)
            self._index = idx
        elif data is not None:
            self._search_for_datasets(data, raw=True)
            self.all_datasets = [dataset for category in self.dataset_categories.values() for dataset in category]
            self._f = None
            self._path = None
            self._dataset_names = self.all_datasets
            self._dataset_shapes = self.dataset_shapes
        else:
            raise ValueError("Must pass in filename or data parameters")
        self.diff_datasets = dict()
        self.derived_datasets = dict()

    @classmethod
    def from_data(cls, data) -> "VeraOutState":
        """Construct a state from in-memory data instead of a file."""
        return cls(data=data)
    
    @property
    def scalar_datasets(self) -> set[str]:
        """The set of dataset names categorized as SCALAR."""
        return self.dataset_categories[str(VeraDtype.SCALAR)]
    
    _CATEGORY_ORDER = [
        VeraDtype.PIN,
        VeraDtype.CHANNEL,
        VeraDtype.CHANNEL_RADIAL,
        VeraDtype.ASSEMBLY,
        VeraDtype.RADIAL,
        VeraDtype.RADIAL_ASSEMBLY,
        VeraDtype.AXIAL,
        VeraDtype.NODE,
        VeraDtype.RADIAL_NODE,
        VeraDtype.SCALAR,
        VeraDtype.UNKNOWN,
    ]

    @property
    def grouped_full_core_keys(self) -> list[tuple[str, list[str]]]:
        """Group the available dataset names by category for grouped UI display.

        Builds in _CATEGORY_ORDER and, for each one that has any
        datasets, returns a (category_name, sorted_names) pair. Empty categories
        are skipped.
        """
        groups = []
        for dtype in self._CATEGORY_ORDER:
            names = sorted(self.dataset_categories[str(dtype)])
            if names:
                groups.append((str(dtype), names))
        return groups

    @property
    def full_core_keys(self) -> list[str]:
        """Flat, category-ordered list of all dataset names."""
        return [name for _, names in self.grouped_full_core_keys for name in names]
    
    def _search_for_datasets(self, data, raw = False):
        """Populate dataset_shapes and dataset_categories for one state.

        Uses pin_powers as the reference full-core shape, builds the
        shape -> category map from it, then bins every dataset in the state
        group whose shape matches a known category.
        """
        core_shape = np.shape(data["pin_powers"])
        self.dataset_shapes = dataset_shape_category_dict(core_shape)
        for dataset_name in data.keys():
            dataset = data[dataset_name]
            dataset_shape = np.shape(dataset)
            if dataset_shape not in self.dataset_shapes:
                continue
            dtype = self.dataset_shapes[dataset_shape]
            self.dataset_categories[str(dtype)].add(dataset_name)
            if raw:
                arr = dataset if isinstance(dataset, np.ndarray) else np.array([dataset])
                setattr(self, dataset_name, VeraDataset(arr, dtype))

    def _search_for_datasets_in_file(self, f, idx) -> None:
        """Populate dataset_shapes and dataset_categories for one state from a file handle."""
        state = f[f"/STATE_{idx:04}"]
        self._search_for_datasets(state)
    
    def add_diff_dataset(self, dataset_name: str, dataset : VeraDataset)-> None:
        """Attach an in-memory diff dataset to this state.

        The dataset is set as an attribute and tracked in diff_datasets.
        It is not categorized and not written to the file.
        """
        setattr(self, dataset_name, dataset)
        self.diff_datasets.update({dataset_name: "H5_ARRAY_TYPE = NONE"})
        dataset_shape = np.shape(dataset)
        if dataset_shape in self.dataset_shapes:
                dataset_type_str = str(self.dataset_shapes[dataset_shape])
                self.dataset_categories[dataset_type_str].add(dataset_name)

    def add_derived_dataset(self, dataset_name: str, dataset : VeraDataset) -> None:
        """Attach an in-memory derived dataset to this state.

        Like add_diff_dataset, but also categorizes the dataset by shape so it
        shows up in the grouped key listings. Not written to the file.
        """
        setattr(self, dataset_name, dataset)
        self.derived_datasets.update({dataset_name: "H5_ARRAY_TYPE = NONE"})
        dataset_shape = np.shape(dataset)
        if dataset_shape in self.dataset_shapes:
                dataset_type_str = str(self.dataset_shapes[dataset_shape])
                self.dataset_categories[dataset_type_str].add(dataset_name)
class VeraDataSource(ABC):
    """Abstract class representing a valid data source for VeraCore to visualize datasets from
    
    Concrete sources expose a single core, an ordered list of states with one
    active at a time, and lookup of named arrays from either the active state
    or the core.
    """

    @property
    @abstractmethod
    def core(self) -> VeraOutCore:
        """The core-level data shared across all states."""
        pass

    @property
    @abstractmethod
    def core_shape(self) -> tuple:
        """Full-core array shape, as (npin, npin, naxial, nassembly)."""
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
        array = getattr(self.active_state, array_name)
        if mask_reflected:
            array = nan_out_reflected(self.core.reduced_core_map, self.core.core_sym, array)
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
