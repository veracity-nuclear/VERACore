import string
from typing import Union

import h5py
import numpy as np
from .vera_tools.VERAout import VERAout

from .vera_data import VeraDataSource, VeraDataset, VeraDtype, VeraAxes, DerivationMethod, dataset_shape_category_dict

H5_ARRAY_TYPE = Union[h5py.Dataset, np.ndarray]
class VeraOutFile(VeraDataSource):
    def __init__(self, filename):
        # Keep this open for better performance
        self.f = h5py.File(filename, "r")
        self.vera_calculator = VERAout(filename=filename) # from pyvera, use this for calculating avgs
        self._core = VeraOutCore(self.f)
        self._core._cache_all()
        
        self._states = []
        self._create_states()

        self.active_state_index = 0
        num_pin = self.vera_calculator.num_pins
        naxx = self.vera_calculator.num_axials
        nass = self.vera_calculator.num_assys
        self._core_shape = (num_pin, num_pin, naxx, nass)
        self.dataset_shape_to_category_lookup = dataset_shape_category_dict(self.core_shape)
        if (
            hasattr(self.core, "pin_volumes") 
            and self.core.pin_volumes is not None 
            and self.core.pin_volumes.shape != self.core_shape
        ):
            raise ValueError("[ERROR] Core shape and pin volumes mismatch. Unable to determine core dimensions.")
        if (
            hasattr(self.active_state, "pin_powers") 
            and self.active_state.pin_powers is not None
            and self.active_state.pin_powers.shape != self.core_shape
        ):
            raise ValueError("[ERROR] Core shape and pin powers mismatch. Unable to determine core dimensions.")

    @property
    def core_shape(self):
        return self._core_shape

    @property
    def core(self):
        return self._core

    def close(self):
        self.f.close()

    def _create_states(self):
        state_keys = [key for key in self.f if key.startswith("STATE_")]
        indices = [int(key.split("_")[1]) for key in state_keys]
        self._states = [VeraOutState(self.f, idx) for idx in indices]

    @property
    def states(self):
        return self._states
        
    @property
    def active_state(self):
        return self.states[self.active_state_index]
    
    @property
    def active_state_full_core_keys(self):
        return self.active_state.full_core_keys

    @property
    def active_state_grouped_keys(self):
        return self.active_state.grouped_full_core_keys

    @property
    def active_state_index(self):
        return self._active_state_index

    @active_state_index.setter
    def active_state_index(self, index: int):
        index = max(0, min(index, len(self._states) - 1))
        if hasattr(self, "_active_state_index"):
            if self._active_state_index == index:
                return
            else:
                # Clear the cache from the active state
                self.active_state._uncache_all()

        self._active_state_index = index
        self.active_state._cache_all()
    
    def add_new_diff_dataset(self, ref_array_name: str, comp_array_name: str, new_diff_name: str):
        for state in self._states:
            if state.has_dataset(ref_array_name) and state.has_dataset(comp_array_name):
                ref = getattr(state, ref_array_name)
                comp = getattr(state, comp_array_name)
                if ref.shape == comp.shape:
                    diff = ref - comp
                    state.add_diff_dataset(new_diff_name, diff)
    
    def _run_avg_over_axes(self, data, axes: VeraAxes = VeraAxes.CORE):
        match axes:
            case VeraAxes.ASSEMBLY:
                der = VeraDataset(self.vera_calculator.Assembly(data), VeraDtype.ASSEMBLY)
            case VeraAxes.AXIAL:
                der = VeraDataset(self.vera_calculator.Axial(data), VeraDtype.AXIAL)
            case VeraAxes.CORE:
                der = VeraDataset(np.array([self.vera_calculator.Average(data)]), VeraDtype.SCALAR)
            case VeraAxes.NODE:
                der = VeraDataset(self.vera_calculator.Node(data), VeraDtype.NODE)
            case VeraAxes.RADIAL:
                der = VeraDataset(self.vera_calculator.Radial(data), VeraDtype.RADIAL)
            case VeraAxes.RADIAL_ASSEMBLY:
                der = VeraDataset(self.vera_calculator.Radial_Assembly(data), VeraDtype.RADIAL_ASSEMBLY)
            case _:
                raise ValueError(f"Derivation: {axes} not implemented")
        return der
    
    def add_new_derived_dataset(self, source_array_name: str, new_dataset_name: str, der_method : DerivationMethod, axes: VeraAxes):
        for state in self._states:
            if state.has_dataset(new_dataset_name):
                raise ValueError(f"A dataset named {new_dataset_name} already exists in this source, please pick a unique name.")
            if not state.has_dataset(source_array_name):
                continue
            data = getattr(state, source_array_name)
            match der_method:
                case DerivationMethod.AVERAGE:
                    der = self._run_avg_over_axes(data, axes)
                case DerivationMethod.STDDEV:
                    mean = self._run_avg_over_axes(data)
                    var = self._run_avg_over_axes((data - mean)**2, axes)
                    der = np.sqrt(var)
                case DerivationMethod.RMS:
                    der = np.sqrt(self._run_avg_over_axes(data**2, axes))
            state.add_derived_dataset(new_dataset_name, der)
        return True

class LazyHDF5Loader:
    def __init__(self, f, path, dataset_names, dataset_shapes = None):
        self._f = f
        self._path = path
        self._dataset_names = dataset_names
        self._dataset_shapes = dataset_shapes

        self._uncache_all()  # start lazy

    def _load_dataset(self, name):
        return self._f[f"{self._path}/{name}"]

    def _make_dataset(self, name) -> VeraDataset:
        raw = self._load_dataset(name)[()]
        shape = np.shape(raw)
        dtype = VeraDtype.UNKNOWN
        if self._dataset_shapes and self._dataset_shapes.get(shape) is not None:
            dtype = self._dataset_shapes.get(shape)
        arr = raw if isinstance(raw, np.ndarray) else np.array([raw])
        return VeraDataset(arr, dtype)

    def _cache(self, name):
        if name not in self._dataset_names:
            raise AttributeError(name)
        setattr(self, name, self._make_dataset(name))

    def _uncache(self, name):
        if name not in self._dataset_names:
            raise AttributeError(name)
        self.__dict__.pop(name, None)

    def _cache_all(self):
        for name in self._dataset_names:
            self._cache(name)

    def _uncache_all(self):
        for name in self._dataset_names:
            self._uncache(name)
    
    def has_dataset(self, name):
        return name in self._dataset_names or name in self.__dict__

    def __getattr__(self, name):
        # Runs only when normal lookup fails (name was uncached/deleted).
        names = self.__dict__.get("_dataset_names")  # avoid recursion
        if names and name in names:
            return self._make_dataset(name)  # transient: not stored
        raise AttributeError(name)
class VeraOutCore(LazyHDF5Loader):
    # These are the attributes that will be read from the HDF5 file
    axial_mesh: H5_ARRAY_TYPE = None
    core_map: H5_ARRAY_TYPE = None
    core_sym: H5_ARRAY_TYPE = None
    pin_volumes: H5_ARRAY_TYPE = None

    def __init__(self, f = None, axial_mesh = None, core_map = None, core_sym = None, pin_volumes = None, aspect_ratio = None):
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
    def from_h5(cls, f):
        return cls(f=f)

    @classmethod
    def from_data(cls, axial_mesh, core_map, core_sym, pin_volumes, aspect_ratio):
        return cls(axial_mesh, core_map, core_sym, pin_volumes, aspect_ratio)

    def compute_reduced_core_map(self):
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

    def compute_axial_mesh_pixels(self):
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

    def compute_control_rod_positions(self):
        # Assume they are the same in every volume
        first_volume = self.pin_volumes[:, :, 0, 0]
        self.control_rod_positions = np.where(first_volume == 0)

    def compute_axial_mesh_means(self):
        # Compute the mean between each neighbor
        repeats = [2] * len(self.axial_mesh)
        repeats[0] = 1
        repeats[-1] = 1

        repeated_mesh = np.repeat(self.axial_mesh, repeats)
        reshaped = repeated_mesh.reshape((repeated_mesh.shape[0] // 2, 2))

        self.axial_mesh_means = np.mean(reshaped, axis=1)

    def row_assembly_indices(self, assembly_idx):
        """Get indices of all assemblies in the same row as this assembly"""
        # The core map and reduced core map use 1-based indexing
        row = np.where(self.reduced_core_map == assembly_idx + 1)[0][0]
        ids = self.reduced_core_map[row]
        # Remove any zeros
        ids = ids[ids > 0]
        return ids - 1

    def col_assembly_indices(self, assembly_idx):
        """Get indices of all assemblies in the same column as this assembly"""
        col = np.where(self.reduced_core_map == assembly_idx + 1)[1][0]
        ids = self.reduced_core_map[:, col]
        # Remove any zeros
        ids = ids[ids > 0]
        return ids - 1

    def reduced_core_map_assembly(self, i, j):
        # Get the index of the assembly at reduced core map position i, j
        return int(self.reduced_core_map[j, i] - 1)

    def reduced_core_map_ij(self, assembly_idx):
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

    def reduced_core_map_label(self, assembly_idx):
        row_label = self.reduced_core_map_row_label(assembly_idx)
        col_label = self.reduced_core_map_column_label(assembly_idx)
        return f"{col_label}-{row_label}"

    def reduced_core_map_row_label(self, assembly_idx):
        i, j = self.reduced_core_map_ij(assembly_idx)
        start_index = self.reduced_core_map_start_index
        rows = list(range(start_index + 1, len(self.core_map) + 1))
        return str(rows[j])

    def reduced_core_map_column_label(self, assembly_idx):
        i, j = self.reduced_core_map_ij(assembly_idx)
        return self.reduced_core_map_column_labels[i]


class VeraOutState(LazyHDF5Loader):
    def __init__(self, f=None, idx=None, full_core_datasets = None, scalar_datasets = None):
        # These are the attributes that will be read from the HDF5 file
        if f is not None:
            self.__annotations__ = dict()
            self.dataset_shapes = dict()
            self.dataset_categories = { str(dataset_type) : set() for dataset_type in VeraDtype}
            self._search_for_datasets(f, idx)
            self.all_datasets = [dataset for category in self.dataset_categories.values() for dataset in category]
            super().__init__(f, f"/STATE_{idx:04}", self.all_datasets, dataset_shapes=self.dataset_shapes)
            self._index = idx
        elif full_core_datasets is not None and scalar_datasets is not None:
            self.full_core_datasets = full_core_datasets
            self.scalar_datasets = scalar_datasets
            for key in self.full_core_datasets.keys():
                setattr(self, key, VeraDataset(self.full_core_datasets[key], VeraDtype.PIN))
            for key in self.scalar_datasets.keys():
                setattr(self, key, VeraDataset(self.scalar_datasets[key], VeraDtype.SCALAR))
        else:
            raise ValueError("Must pass in filename or data parameters")
        self.diff_datasets = dict()
        self.derived_datasets = dict()

    @classmethod
    def from_data(cls, full_core_datasets, scalar_datasets):
        return cls(full_core_datasets=full_core_datasets, scalar_datasets=scalar_datasets)
    
    @property
    def scalar_datasets(self):
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
    def grouped_full_core_keys(self):
        groups = []
        for dtype in self._CATEGORY_ORDER:
            names = sorted(self.dataset_categories[str(dtype)])
            if names:
                groups.append((str(dtype), names))
        # if self.derived_datasets:
        #     groups.append(("DERIVED", sorted(self.derived_datasets)))
        # if self.diff_datasets:
        #     groups.append(("DIFF", sorted(self.diff_datasets)))
        return groups

    @property
    def full_core_keys(self):
        return [name for _, names in self.grouped_full_core_keys for name in names]

    def _search_for_datasets(self, f, idx):
        # Search for available full core/scalar datasets at each state point
        state = f[f"/STATE_{idx:04}"]
        pin_powers = state["pin_powers"]
        core_shape = np.shape(pin_powers)
        self.dataset_shapes = dataset_shape_category_dict(core_shape)
        
        for dataset_name in state.keys():
            dataset_shape = np.shape(state[dataset_name])
            if dataset_shape in self.dataset_shapes:
                dataset_type_str = str(self.dataset_shapes[dataset_shape])
                self.dataset_categories[dataset_type_str].add(dataset_name) 
    
    def add_diff_dataset(self, dataset_name: str, dataset : VeraDataset):
        setattr(self, dataset_name, dataset)
        self.diff_datasets.update({dataset_name: "H5_ARRAY_TYPE = NONE"})

    def add_derived_dataset(self, dataset_name: str, dataset : VeraDataset):
        setattr(self, dataset_name, dataset)
        self.derived_datasets.update({dataset_name: "H5_ARRAY_TYPE = NONE"})
        dataset_shape = np.shape(dataset)
        if dataset_shape in self.dataset_shapes:
                dataset_type_str = str(self.dataset_shapes[dataset_shape])
                self.dataset_categories[dataset_type_str].add(dataset_name) 
