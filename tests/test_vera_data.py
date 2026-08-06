import h5py
import numpy as np
import pytest
from vera_core.app.core.vera_data import (
    LazyHDF5Loader,
    VeraDataset,
    VeraDtype,
    _make_ji_safe,
    _nearest_nonzero_ij,
    build_core_dtypes,
)


def test_vera_dataset_preserves_metadata_through_slicing():
    array = VeraDataset(np.arange(6).reshape(2, 3), VeraDtype.PIN, "W/cm3")

    sliced = array[:, 1:]

    assert isinstance(sliced, VeraDataset)
    assert sliced.dataset_type is VeraDtype.PIN
    assert sliced.physical_units == "W/cm3"


def test_vera_dtype_properties():
    assert VeraDtype.PIN.has_axial_dim()
    assert VeraDtype.PIN.has_fuel_pins()
    assert VeraDtype.COMP_NODAL.is_computational()
    assert VeraDtype.COMP_NODAL.is_nodal()
    assert VeraDtype.CHANNEL.is_channel()
    assert not VeraDtype.SCALAR.has_axial_dim()
    with pytest.raises(ValueError):
        _ = VeraDtype.SCALAR.axial_dim_idx


def test_build_core_dtypes_maps_representative_shapes():
    mapping = build_core_dtypes(
        npiny=2,
        npinx=3,
        nax=4,
        nass=5,
        comp_nax=6,
        comp_nass=7,
    )

    assert mapping[(2, 3, 4, 5)] is VeraDtype.PIN
    assert mapping[(3, 4, 4, 5)] is VeraDtype.CHANNEL
    assert mapping[(4,)] is VeraDtype.AXIAL
    assert mapping[(4, 6, 7)] is VeraDtype.COMP_NODAL
    assert mapping[(1, 6, 7)] is VeraDtype.COMP_ASSY


def test_make_ji_safe_clips_indices():
    assert _make_ji_safe(-2, 99, np.zeros((3, 4))) == (0, 3)


def test_nearest_nonzero_ij():
    array = np.array([[0, 0, 2], [0, 0, 0], [1, 0, 0]])
    assert _nearest_nonzero_ij(array, 0, 1) == (0, 2)
    assert _nearest_nonzero_ij(np.zeros((2, 2)), 0, 0) is None


def test_lazy_hdf5_loader_reads_units_and_remains_lazy(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as file:
        group = file.create_group("DATA")
        dataset = group.create_dataset("temperature", data=np.array([1.0, 2.0]))
        dataset.attrs["physical_units"] = "K"

    with h5py.File(path, "r") as file:
        loader = LazyHDF5Loader(
            file,
            "/DATA",
            ["temperature"],
            {(2,): VeraDtype.AXIAL},
        )

        assert "temperature" not in loader.__dict__
        first = loader.temperature
        assert "temperature" not in loader.__dict__
        np.testing.assert_array_equal(first, [1.0, 2.0])
        assert first.dataset_type is VeraDtype.AXIAL
        assert first.physical_units == "K"

        loader._cache("temperature")
        assert "temperature" in loader.__dict__
        loader._uncache("temperature")
        assert "temperature" not in loader.__dict__


def test_lazy_hdf5_loader_rejects_unknown_managed_name(tmp_path):
    path = tmp_path / "sample.h5"
    with h5py.File(path, "w") as file:
        file.create_group("DATA")

    with h5py.File(path, "r") as file:
        loader = LazyHDF5Loader(file, "/DATA", [])
        with pytest.raises(AttributeError):
            loader._cache("missing")
