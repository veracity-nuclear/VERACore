import h5py
import numpy as np
import os
from scipy.interpolate import make_interp_spline
from .vera_tools.VERAout import VERAout
from .vera_data import VeraDataSource, VeraDataset, VeraDtype, VeraAxes, DerivationMethod, dataset_shape_category_dict, VeraOutCore, VeraOutState
class VeraOutFile(VeraDataSource):
    
    def __init__(self, filename):
        """Open a VERA output file and build its core and state objects.

        Opens two handles on the file (a direct h5py.File and a VERAout
        for averaging). 
        It eagerly caches the core, and validates that the core shape agrees with pin_volumes and pin_powers.
        """
        self.f = h5py.File(filename, "r", locking=False)
        try:
            self.vera_calculator = VERAout(filename=filename) # from pyvera, use this for calculating avgs
        except Exception as e:
            print(str(e))
            self.vera_calculator = None
        self._core = VeraOutCore(self.f)
        self._core._cache_all()
        self._states = []
        self._determine_core_shape()
        self._create_states()
        self.active_state_index = 0
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
        
        
    def _determine_core_shape(self):
        if self.vera_calculator is not None:
            num_pin = self.vera_calculator.num_pins
            nax = self.vera_calculator.num_axials
            nass = self.vera_calculator.num_assys
            self._core_shape = {"npiny" : num_pin, "npinx" : num_pin, "nax" : nax, "nass" : nass}
        else:
            cm = self._core.core_map
            nass = np.count_nonzero(np.unique(cm[~np.isnan(cm)]))
            nax = len(self._core.axial_mesh) - 1
            print(nass, nax)
            self._core_shape = {"npiny" : None, "npinx" : None, "nax" : nax, "nass" : nass}
        self.dataset_shape_to_category_lookup = dataset_shape_category_dict(**self._core_shape)

    @property
    def core_shape(self):
        if (self._core_shape["npiny"] is not None 
            and self._core_shape["npinx"] is not None
            and self._core_shape["nax"] is not None
            and self._core_shape["nass"] is not None):

            return (self._core_shape["npiny"], self._core_shape["npinx"], self._core_shape["nax"], self._core_shape["nass"])
        elif self._core_shape["nax"] is not None and self._core_shape["nass"] is not None:
            return (self._core_shape["nax"], self._core_shape["nass"])
        else:
            raise RuntimeError("Could not determine core shape")

    @property
    def core(self):
        return self._core

    def close(self):
        self.f.close()
        self.vera_calculator.h5f.close()

    def _create_states(self):
        """Build a VeraOutState for every STATE_ group found in the file."""
        state_keys = [key for key in self.f if key.startswith("STATE_")]
        indices = [int(key.split("_")[1]) for key in state_keys]
        self._states = [VeraOutState(self.f, idx, core_shape=self._core_shape) for idx in indices]

    def default_datasets(self):
        dataset_categories = self.active_state.dataset_categories
        default_names = {dtype : next(iter(dataset_categories[dtype])) for dtype in dataset_categories if len(dataset_categories[dtype]) > 0}
        if not default_names:
            return None
        if default_names and str(VeraDtype.PIN) in dataset_categories and "pin_powers" in dataset_categories[str(VeraDtype.PIN)]:
            default_names[str(VeraDtype.PIN)] = "pin_powers"
        return default_names

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
        """Set the active state, clamping to range and no-opping if unchanged.
       
        Switching states uncaches the previous active state and caches the new
        one
        """
        index = max(0, min(index, len(self._states) - 1))
        if hasattr(self, "_active_state_index"):
            if self._active_state_index == index:
                return
            else:
                # Clear the cache from the active state
                self.active_state._uncache_all()

        self._active_state_index = index
        self.active_state._cache_all()
    
    def add_new_diff_dataset(self, ref_dataset_name: str, comp_src : VeraDataSource, comp_dataset_name: str, new_diff_name: str, interpolation_order : int = 1):
        for idx, state in enumerate(self._states):
            if idx >= len(comp_src.states):
                return
            if not state.has_dataset(ref_dataset_name) or not state.has_dataset(comp_dataset_name):
                continue
            ref_data = getattr(state, ref_dataset_name)
            comp_data = getattr(comp_src.states[idx], comp_dataset_name)
            ref_axial_mesh_means = self.core.axial_mesh_means
            comp_axial_mesh_means = comp_src.core.axial_mesh_means
            if ref_data.dataset_type != comp_data.dataset_type:
                continue
            if np.allclose(ref_axial_mesh_means, comp_axial_mesh_means):
                diff = ref_data - comp_data
            else:
                # data is (py, px, nax, nass) shape. py, px, and nass must match between the two dataset
                spl = make_interp_spline(comp_axial_mesh_means, comp_data, k=interpolation_order, axis=2)
                comp_data_on_ref_mesh = spl(ref_axial_mesh_means, extrapolate=False)
                diff = ref_data - comp_data_on_ref_mesh
            state.add_diff_dataset(new_diff_name, diff)
    
    def _run_avg_over_axes(self, data, axes: VeraAxes = VeraAxes.CORE):
        """Reduce data over the given axes using the VERAout calculator.

        Dispatches to the matching VERAout averaging routine and wraps the result
        as a VeraDataset of the corresponding type; raises ValueError for an
        unsupported axes value.
        """
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
        if not self.vera_calculator:
            return
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
