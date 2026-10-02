import h5py
import numpy as np
from scipy.interpolate import make_interp_spline

from .types import CoreOverride
from .vera_data import (
    DerivationMethod,
    VeraAxes,
    VeraDataset,
    VeraDataSource,
    VeraDtype,
    VeraOutCore,
    VeraOutState,
)
from .vera_tools.VERAout import VERAout


class VeraOutFile(VeraDataSource):
    def __init__(self, filename: str, core_overrides: CoreOverride | None = None):
        """Open a VERA output file and build its core and state objects.

        Opens two handles on the file (a direct h5py.File and a VERAout
        for averaging).
        It eagerly caches the core, and validates that the core shape agrees with pin_volumes and pin_powers.
        """
        if core_overrides is None:
            core_overrides = {}
        self._file_path = filename
        self.f = h5py.File(filename, "r", locking=False)
        try:
            try:
                self.vera_calculator = VERAout(
                    filename=filename
                )  # from pyvera, use this for calculating avgs
            except Exception:
                self.vera_calculator = None
            self._states = []
            self._core = VeraOutCore(self.f, overrides=core_overrides)
            self._create_states()
            self.active_state_index = 0
            self._determine_time_axes()
        except Exception as e:
            self.f.close()
            raise e

    def _determine_time_axes(self):
        self._time_axes = {}
        for time_data_point in ("exposure", "core_exposure", "exposure_efpd"):
            time_axis = [
                getattr(state, time_data_point).item()
                for state in self.states
                if state.has_dataset(time_data_point)
            ]
            if (
                np.shape(time_axis) != np.shape(self.states)
                or not np.all(np.asarray(time_axis) >= 0)
                or not np.all(np.diff(time_axis) >= 0)
            ):
                continue
            self._time_axes[time_data_point] = time_axis
        self._time_axes["state_count"] = [state_num for state_num in range(len(self.states))]

    @property
    def file_path(self) -> str:
        return self._file_path

    @property
    def core(self) -> VeraOutCore:
        """Reference to this h5 files core data"""
        return self._core

    def close(self):
        self.f.close()
        if self.vera_calculator is not None:
            self.vera_calculator.h5f.close()

    def _create_states(self):
        """Build a VeraOutState for every STATE_ group found in the file."""
        state_keys = [key for key in self.f if key.startswith("STATE_")]
        indices = [int(key.split("_")[1]) for key in state_keys]
        self._states = [VeraOutState(self.f, idx, self.core) for idx in indices]

    def default_datasets(self) -> dict[str, str]:
        categorized_ds_names = self.active_state.categorized_ds_names
        default_names = {
            category.title: sorted(names)[0]
            for category, names in categorized_ds_names.items()
            if category != VeraDtype.UNKNOWN and names
        }
        if not default_names:
            return {}
        if "pin_powers" in categorized_ds_names.get(VeraDtype.PIN, ()):
            default_names[VeraDtype.PIN.title] = "pin_powers"
        return default_names

    def time_axes(self):
        return self._time_axes

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

    def add_new_diff_dataset(
        self,
        ref_dataset_name: str,
        comp_src: "VeraDataSource",
        comp_dataset_name: str,
        new_diff_name: str,
        interpolation_order: int = 1,
        ref_scale: float = 1.0,
        comp_scale: float = 1.0,
        units: str = "unitless",
    ):
        produced = 0
        for idx, state in enumerate(self._states):
            if idx >= len(comp_src.states):
                return
            comp_state = comp_src.states[idx]
            if not state.has_dataset(ref_dataset_name) or not comp_state.has_dataset(
                comp_dataset_name
            ):
                continue
            ref_data: VeraDataset = getattr(state, ref_dataset_name) * ref_scale
            comp_data: VeraDataset = getattr(comp_state, comp_dataset_name) * comp_scale
            ref_axial_mesh_means = self.core.get_axial_mesh_means(dataset=ref_data)
            comp_axial_mesh_means = comp_src.core.get_axial_mesh_means(dataset=comp_data)
            if ref_data.dataset_type != comp_data.dataset_type:
                continue
            if np.allclose(ref_axial_mesh_means, comp_axial_mesh_means):
                diff = ref_data - comp_data
            elif ref_data.dataset_type.has_axial_dim():
                # all data dimensions that are not the axial dim must match between the two dataset
                spl = make_interp_spline(
                    comp_axial_mesh_means,
                    comp_data,
                    k=interpolation_order,
                    axis=ref_data.dataset_type.axial_dim_idx,
                )
                comp_data_on_ref_mesh = spl(ref_axial_mesh_means, extrapolate=False)
                diff: VeraDataset = ref_data - comp_data_on_ref_mesh
            else:
                continue
            diff.physical_units = units
            state.add_diff_dataset(new_diff_name, diff)
            produced += 1
        if produced == 0:
            raise ValueError(f"No overlapping/compatible states to diff for '{new_diff_name}'")

    def _run_avg_over_axes(self, data, axes: VeraAxes = VeraAxes.CORE):
        """Reduce data over the given axes using the VERAout calculator.

        Dispatches to the matching VERAout averaging routine and wraps the result
        as a VeraDataset of the corresponding type; raises ValueError for an
        unsupported axes value.
        """
        match axes:
            case VeraAxes.ASSEMBLY:
                der = VeraDataset(
                    self.vera_calculator.Assembly(data)[np.newaxis, ...],
                    VeraDtype.ASSEMBLY,
                )
            case VeraAxes.AXIAL:
                der = VeraDataset(self.vera_calculator.Axial(data), VeraDtype.AXIAL)
            case VeraAxes.CORE:
                der = VeraDataset(np.array([self.vera_calculator.Average(data)]), VeraDtype.SCALAR)
            case VeraAxes.NODE:
                der = VeraDataset(self.vera_calculator.Node(data), VeraDtype.NODE)
            case VeraAxes.RADIAL:
                der = VeraDataset(self.vera_calculator.Radial(data), VeraDtype.RADIAL)
            case VeraAxes.RADIAL_ASSEMBLY:
                der = VeraDataset(
                    self.vera_calculator.Radial_Assembly(data),
                    VeraDtype.RADIAL_ASSEMBLY,
                )
            case _:
                raise ValueError(f"Derivation: {axes} not implemented")
        return der

    def add_new_derived_dataset(
        self,
        source_array_name: str,
        new_dataset_name: str,
        der_method: DerivationMethod,
        axes: VeraAxes,
        use_factors: bool = True,
    ):
        if not self.vera_calculator:
            return
        for state in self._states:
            if state.has_dataset(new_dataset_name):
                raise ValueError(
                    f"A dataset named {new_dataset_name} already exists in this source, please pick a unique name."
                )
            if not state.has_dataset(source_array_name):
                continue
            data = getattr(state, source_array_name)
            match der_method:
                case DerivationMethod.AVERAGE:
                    der = self._run_avg_over_axes(data, axes)
                case DerivationMethod.STDDEV:
                    mean = self._run_avg_over_axes(data)
                    var = self._run_avg_over_axes((data - mean) ** 2, axes)
                    der = np.sqrt(var)
                case DerivationMethod.RMS:
                    der = np.sqrt(self._run_avg_over_axes(data**2, axes))
            state.add_derived_dataset(new_dataset_name, der)
