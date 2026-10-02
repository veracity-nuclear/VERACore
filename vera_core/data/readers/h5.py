from functools import partial

import h5py
import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import DatasetSource, VeraDataSource, VeraOutCore, VeraOutState
from .rom import RomBasis


def _get_units(h5_ref: h5py.Dataset):
    units = "Unitless"
    if "physical_units" in h5_ref.attrs:
        units = h5_ref.attrs["physical_units"]
    elif "units" in h5_ref.attrs:
        units = h5_ref.attrs["units"]
    if isinstance(units, np.ndarray):
        units = units.item() if units.size == 1 else units.tolist()[0]
    if isinstance(units, bytes):
        units = units.decode()
    return str(units)


def _search_for_datasets(group: h5py.Group) -> dict[str, tuple[int, ...]]:
    datasets = {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            datasets[name] = (1,) if obj.shape == () else tuple(obj.shape)

    group.visititems(visit)
    return datasets


class H5DatasetSource(DatasetSource):
    def __init__(
        self, file: h5py.File, dir: str, dataset_dtyes: dict[tuple, VeraDtype] | None = None
    ):
        self._f = file
        self._dir = dir
        if dir not in self._f:
            raise ValueError(f"'{dir}' is not a group in the file:", file.filename)
        group = self._f[dir]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"The directory {dir} is not a group in h5 file {file.filename}")
        datasets = _search_for_datasets(group)
        self._names = list(datasets)
        self._shapes = datasets
        self._dataset_dtypes = dataset_dtyes or {}

    @property
    def provenance(self):
        return self._f.filename

    def names(self):
        return self._names

    def _get_h5dataset_ref(self, name: str) -> h5py.Dataset | None:
        p = f"{self._dir}/{name}"
        if p in self._f:
            h5_ref = self._f[p]
        elif name in self._f:
            h5_ref = self._f[name]
        else:
            return None
        if not isinstance(h5_ref, h5py.Dataset):
            return None
        return h5_ref

    def sample(self, name: str, indices):
        h5_ref = self._get_h5dataset_ref(name)
        if h5_ref is None:
            return None
        if h5_ref.shape == ():
            return h5_ref[()]
        return h5_ref[indices]

    def shape(self, name):
        if name in self._shapes:
            return self._shapes[name]
        h5_ref = self._get_h5dataset_ref(name)
        if h5_ref is None:
            return None
        if h5_ref.shape == ():
            return (1,)
        return tuple(h5_ref.shape)

    def load(self, name):
        h5_ref = self._get_h5dataset_ref(name)

        if h5_ref is None:
            return None
        units = _get_units(h5_ref)
        raw = h5_ref[()]
        shape = np.shape(raw)
        dtype = VeraDtype.UNKNOWN
        if self._dataset_dtypes.get(shape) is not None:
            dtype = self._dataset_dtypes.get(shape)
        arr = raw if isinstance(raw, np.ndarray) else np.array([raw])
        ds = VeraDataset(arr, dtype, name, units)
        return ds


class RomH5DatasetSource(H5DatasetSource):
    def __init__(
        self,
        file: h5py.File,
        dir: str,
        basis: RomBasis,
        scale_name: str,
        dataset_dtyes: dict[tuple, VeraDtype] | None = None,
    ):
        super().__init__(file, dir, dataset_dtyes)
        self._rom_names = basis.names()
        self._basis = basis
        self._scale = super().load(scale_name)[0]
        self._reduced_state = self._f[f"{self._dir}/reduced_state"][()]
        self._token = basis.new_token()
        self.load_count: dict[str, int] = {}

    def names(self):
        return self._rom_names + super().names()

    def shape(self, name: str) -> tuple[int, ...] | None:
        if name in self._rom_names:
            return self._basis.shape(name)
        else:
            return super().shape(name)

    def sample(self, name: str, indices):
        """One element of a ROM field, or None when the caller must use load()."""
        if name not in self._rom_names:
            return super().sample(name, indices)
        flat = self._basis.flat_index(name, indices)
        if flat is None:
            return None
        return self._basis.sample(name, flat, self._reduced_state, self._scale)

    def load(self, name: str) -> VeraDataset | None:
        if name not in self._rom_names:
            return super().load(name)
        dtype_key = self.shape(name)
        if dtype_key is None:
            return None
        self.load_count[name] = self.load_count.get(name, 0) + 1
        array = self._basis.expand(self._token, name, self._reduced_state, self._scale)
        dtype = self._dataset_dtypes.get(dtype_key, VeraDtype.UNKNOWN)
        return VeraDataset(array, dtype, name, "unitless")


def open_vera_file_data_source(
    file_path: str, core_overrides: dict | None = None, active_state_idx: int = 0
) -> VeraDataSource:
    file = h5py.File(file_path, "r", locking=False)
    core_dataset_src = H5DatasetSource(file, "CORE")
    core = VeraOutCore(core_dataset_src, overrides=core_overrides)
    state_keys = [key for key in file if key.startswith("STATE_")]
    indices = [(key, int(key.split("_")[1])) for key in state_keys]
    sorted_indices = sorted(indices, key=lambda i: i[1])

    dataset_source = H5DatasetSource
    if "ROM" in file:
        rom_group = file["ROM"]
        basis = rom_group["U_map"]
        basis_matrix = basis[()]
        shape = tuple(basis.attrs["full_state_shape"])
        dataset_axis = int(basis.attrs["dataset_axis"]) if "dataset_axis" in basis.attrs else None
        dataset_names = basis.attrs["full_state_datasets"]
        derived_datasets = {}
        for i, name in enumerate(dataset_names):
            name = name.decode() if isinstance(name, bytes) else name
            derived_datasets[name] = {"full_state_idx": i, "full_state_axis": dataset_axis}
        basis = RomBasis(
            basis_matrix=basis_matrix,
            full_state_shape=shape,
            derived_datasets=derived_datasets,
        )
        dataset_source = partial(
            RomH5DatasetSource,
            basis=basis,
            scale_name="DT_power",
        )
    states = [
        VeraOutState(
            dataset_source(file=file, dir=state_dir, dataset_dtyes=core.shape_to_dtype), idx, core
        )
        for state_dir, idx in sorted_indices
    ]
    close_callback = file.close
    vs = VeraDataSource(
        core=core,
        states=states,
        active_state_idx=active_state_idx,
        provenance=file_path,
        filename=file_path,
        close_callback=close_callback,
    )
    return vs
