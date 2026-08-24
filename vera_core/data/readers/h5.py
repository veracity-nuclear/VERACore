import h5py
import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import DatasetSource, VeraDataSource, VeraOutCore, VeraOutState


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


def _search_for_datasets(data: h5py.Group):
    dataset_names = []

    def _loop_through_datasets(h5_group: h5py.Group, group_name=""):
        for ds_name in h5_group.keys():
            ds: h5py.Group | h5py.Dataset = h5_group[ds_name]
            full_name = "/".join(part for part in (group_name, ds_name) if part)
            if isinstance(ds, h5py.Group):  # recurse on group (subdir)
                _loop_through_datasets(ds, group_name=full_name)
                continue
            ds_shape = ds.shape
            if ds_shape is None:
                continue
            dataset_names.append(full_name)

    _loop_through_datasets(data)
    return dataset_names


class H5DatasetSource(DatasetSource):
    def __init__(
        self, file: h5py.File, dir: str, dataset_dtyes: dict[tuple, VeraDtype] | None = None
    ):
        self._f = file
        self._dir = dir
        group = self._f[dir]
        if not isinstance(group, h5py.Group):
            raise ValueError(f"The directory {dir} is not a group in h5 file {file.filename}")
        self._names = _search_for_datasets(group)
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

    def shape(self, name):
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


def open_vera_file_data_source(
    file_path: str, core_overrides: dict | None = None, active_state_idx: int = 0
) -> VeraDataSource:
    file = h5py.File(file_path, "r", locking=False)

    core_dataset_src = H5DatasetSource(file, "CORE")
    core = VeraOutCore(core_dataset_src, overrides=core_overrides)

    state_keys = [key for key in file if key.startswith("STATE_")]
    indices = [(key, int(key.split("_")[1])) for key in state_keys]
    sorted_indices = sorted(indices, key=lambda i: i[1])
    states = [
        VeraOutState(H5DatasetSource(file, state_dir, core.shape_to_dtype), idx, core)
        for state_dir, idx in sorted_indices
    ]
    close_callback = file.close
    return VeraDataSource(
        core=core,
        states=states,
        active_state_idx=active_state_idx,
        provenance=file_path,
        filename=file_path,
        close_callback=close_callback,
    )
