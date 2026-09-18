import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.model import (
    DEFAULT_AXIAL_MESH_STEP,
    DEFAULT_PIN_PITCH,
    CorePropMissing,
    VeraOutCore,
    _nearest_nonzero_ij,
    col_label,
    nan_out_reflected,
)
from vera_core.data.readers.mock import DictDatasetSource, build_core, core_arrays

# Default mock core: 5x5 quarter-symmetric map, reduced map
# [[1, 2, 3], [4, 5, 6], [7, 8, 9]], 3x3 pins, 4 levels over 0..100.


@pytest.fixture
def core():
    return build_core()


def core_from(arrays, **kwargs):
    return VeraOutCore(DictDatasetSource(arrays), **kwargs)


# ---------------------------------------------------------------- construction


def test_default_core_shape(core):
    assert core.core_shape == (3, 3, 4, 9)
    assert core.core_sym == 4


def test_missing_core_map_raises():
    with pytest.raises(RuntimeError, match="core_map"):
        core_from({})


@pytest.mark.parametrize("map_sym", [1, 4])
def test_symmetry_inferred_when_not_declared(map_sym):
    assert build_core(core_sym=None, map_sym=map_sym).core_sym == map_sym


def test_missing_npin_and_nax_reported_together():
    with pytest.raises(CorePropMissing) as info:
        build_core(npin=None, with_axial_mesh=False)
    assert set(info.value.missing) == {"npin", "nax"}
    assert info.value.inferred["nass"]["value"] == 9


def test_missing_npin_reports_inferred_nax():
    with pytest.raises(CorePropMissing) as info:
        build_core(npin=None)
    assert set(info.value.missing) == {"npin"}
    assert info.value.inferred["nax"] == {"value": 4, "source": "/CORE/axial_mesh"}


def test_pin_count_inferred_from_pin_volumes_shape():
    arrays = core_arrays()
    del arrays["npin"]
    assert core_from(arrays).core_shape == (3, 3, 4, 9)


def test_nax_inferred_from_pin_volumes_when_no_mesh():
    core = build_core(with_axial_mesh=False)
    assert core.nax == 4
    assert len(core.axial_mesh) == 5


def test_nax_mismatch_between_mesh_and_geometry_raises():
    arrays = core_arrays()
    arrays["pin_volumes"] = np.ones((3, 3, 5, 9))
    with pytest.raises(RuntimeError, match="Mismatch"):
        core_from(arrays)


def test_npin_override_fills_missing_value():
    core = build_core(npin=None, overrides={"npin": 17})
    assert core.core_shape[:2] == (17, 17)


def test_nax_override_sets_nax():
    core = build_core(with_axial_mesh=False, with_pin_volumes=False, overrides={"nax": 6})
    assert core.nax == 6
    assert len(core.axial_mesh) == 7


def test_nax_override_mesh_uses_default_step():
    core = build_core(with_axial_mesh=False, with_pin_volumes=False, overrides={"nax": 4})
    np.testing.assert_allclose(np.diff(core.axial_mesh), DEFAULT_AXIAL_MESH_STEP)


def test_pin_pitch_defaults_and_apitch_override():
    assert build_core().pin_pitch == DEFAULT_PIN_PITCH
    core = build_core(apitch=21.0)
    assert core.apitch == 21.0
    assert core.pin_pitch == pytest.approx(7.0)


def test_aspect_ratio_default_and_declared():
    assert build_core().aspect_ratio == 1
    assert build_core(aspect_ratio=2.5).aspect_ratio == 2.5


# ---------------------------------------------------------------- maps and labels


def test_quarter_core_reduces_to_quadrant(core):
    np.testing.assert_array_equal(core.reduced_core_map, np.arange(1, 10).reshape(3, 3))
    assert core.reduced_core_map_start_index == 2


def test_full_core_keeps_whole_map():
    core = build_core(core_sym=1)
    assert core.reduced_core_map.shape == (5, 5)
    assert core.reduced_core_map_start_index == 0


def test_is_even():
    assert not build_core(nass_side=5).is_even()


def test_default_labels(core):
    assert core.reduced_core_map_column_labels == ["C", "B", "A"]
    assert core.reduced_core_map_row_labels == ['3', '4', '5']


def test_file_labels_sliced_to_reduced_map():
    core = build_core(labels=True)
    assert core.reduced_core_map_column_labels == ["C", "D", "E"]
    assert core.reduced_core_map_row_labels == ["3", "4", "5"]


def test_combined_label(core):
    assert core.reduced_core_map_label(0) == "C-3"
    assert core.reduced_core_map_label(8) == "A-5"


def test_row_label_is_str_without_file_labels(core):
    assert isinstance(core.reduced_core_map_row_label(0), str)


@pytest.mark.parametrize(
    "index, label", [(0, "A"), (25, "Z"), (26, "AA"), (27, "AB"), (701, "ZZ"), (702, "AAA")]
)
def test_col_label(index, label):
    assert col_label(index) == label


# ---------------------------------------------------------------- axial mesh


def test_axial_mesh_means_are_midpoints(core):
    np.testing.assert_array_equal(core.axial_mesh_means, [12.5, 37.5, 62.5, 87.5])


def test_gross_mesh_equals_means_without_extra_meshes(core):
    np.testing.assert_array_equal(core.gross_axial_mesh, core.axial_mesh_means)


def test_midpoints_round_to_four_decimals():
    np.testing.assert_array_equal(VeraOutCore._midpoints([0.0, 1.0 / 3.0]), [0.1667])


def test_uniform_mesh_gives_three_pixels_per_level(core):
    np.testing.assert_array_equal(core.axial_mesh_pixels, [3, 3, 3, 3])


def test_pixels_scale_with_level_height():
    arrays = core_arrays(with_pin_volumes=False)
    arrays["axial_mesh"] = np.array([0.0, 10.0, 30.0, 40.0, 70.0])
    np.testing.assert_array_equal(core_from(arrays).axial_mesh_pixels, [3, 6, 3, 9])


# ---------------------------------------------------------------- non-fuel


def test_non_fuel_locs_from_zero_volume(core):
    pin_y, pin_x, _, _ = core.non_fuel_locs
    assert set(pin_y) == {0} and set(pin_x) == {0}
    assert len(pin_y) == 4 * 9


def test_non_fuel_locs_none_without_pin_volumes():
    assert build_core(with_pin_volumes=False).non_fuel_locs is None


# ---------------------------------------------------------------- computational core


def test_no_comp_core_by_default(core):
    assert not core.has_comp_core()
    assert core.comp_nass is None


def test_comp_core_shape_and_map():
    core = build_core(with_comp_core=True)
    assert core.comp_core_shape == (3, 3, 4, 9)
    assert core.get_map(dataset_type=VeraDtype.COMP_PIN) is core.comp_core_map
    assert core.get_core_shape(dataset_type=VeraDtype.COMP_PIN) == core.comp_core_shape


def test_comp_mesh_used_for_computational_dtypes():
    core = build_core(with_comp_core=True)
    assert core.get_axial_mesh(dataset_type=VeraDtype.COMP_PIN) is core.comp_axial_mesh
    assert core.get_axial_mesh(dataset_type=VeraDtype.PIN) is core.axial_mesh


# ---------------------------------------------------------------- detectors


def test_detectors_absent_by_default(core):
    assert core.detector_map is None
    assert core.ndet is None
    np.testing.assert_array_equal(core.det_axial_mesh_means, core.axial_mesh_means)


def test_detector_map_and_mesh():
    core = build_core(with_detectors=True)
    assert core.ndet == 9
    assert core.ndax == 4
    assert core.detector_map.shape == (3, 3)
    assert core.get_map(dataset_type=VeraDtype.POINT_DETECTOR) is core.detector_map
    np.testing.assert_array_equal(
        core.get_axial_mesh_means(dataset_type=VeraDtype.POINT_DETECTOR), [5, 35, 65, 95]
    )


def test_gross_mesh_includes_detector_levels():
    core = build_core(with_detectors=True)
    np.testing.assert_array_equal(
        core.gross_axial_mesh, [5, 12.5, 35, 37.5, 62.5, 65, 87.5, 95]
    )


# ---------------------------------------------------------------- dtype dispatch


@pytest.mark.parametrize(
    "method",
    ["get_core_shape", "get_map", "get_axial_mesh", "get_axial_mesh_means", "get_axial_mesh_pixels"],
)
def test_dispatch_rejects_unknown(core, method):
    with pytest.raises(RuntimeError):
        getattr(core, method)(dataset_type=VeraDtype.UNKNOWN)


def test_dispatch_accepts_dataset_instead_of_dtype(core):
    ds = VeraDataset(np.zeros(4), VeraDtype.AXIAL)
    assert core.get_axial_mesh_means(dataset=ds) is core.axial_mesh_means


def test_core_dtypes_lookup(core):
    assert core.core_dtypes((3, 3, 4, 9)) == VeraDtype.PIN
    assert core.core_dtypes((99,)) == VeraDtype.UNKNOWN


# ---------------------------------------------------------------- assembly lookup


def test_row_and_col_assembly_indices(core):
    np.testing.assert_array_equal(core.row_assembly_indices(4), [3, 4, 5])
    np.testing.assert_array_equal(core.col_assembly_indices(4), [1, 4, 7])


def test_comp_and_detector_lookup_together_rejected(core):
    for method in (core.row_assembly_indices, core.col_assembly_indices):
        with pytest.raises(RuntimeError):
            method(0, is_comp=True, is_detector=True)


def test_reduced_core_map_ij_returns_column_row(core):
    assert core.reduced_core_map_ij(5) == (2, 1)


def test_reduced_core_map_ij_missing_raises(core):
    with pytest.raises(ValueError, match="not found"):
        core.reduced_core_map_ij(99)


def test_reduced_core_map_ij_duplicate_raises():
    core = build_core(core_sym=1, map_sym=4)  # full map, off-centre ids mirrored
    with pytest.raises(ValueError, match="multiple"):
        core.reduced_core_map_ij(4)


def test_reduced_core_map_assembly_in_range(core):
    assert core.reduced_core_map_assembly(1, 2) == 7  # i=col 1, j=row 2 -> id 8


def test_reduced_core_map_assembly_clamps_out_of_range(core):
    assert core.reduced_core_map_assembly(10, 10) == 8
    assert core.reduced_core_map_assembly(-3, -3) == 0


def test_nearest_nonzero_snaps_and_handles_empty():
    grid = np.array([[0, 0, 0], [0, 0, 7], [0, 0, 0]])
    assert _nearest_nonzero_ij(grid, 0, 0) == (1, 2)
    assert _nearest_nonzero_ij(np.zeros((2, 2)), 0, 0) is None


# ---------------------------------------------------------------- nan_out_reflected


def _pin_ones(nax=4, nass=9):
    return VeraDataset(np.ones((3, 3, nax, nass)), VeraDtype.PIN)


def test_reflected_pins_masked_on_quarter_core(core):
    out = nan_out_reflected(core.reduced_core_map, 4, _pin_ones())
    assert np.isnan(out[0, :, :, 0]).all()  # top row assembly: top pin row
    assert np.isnan(out[:, 0, :, 3]).all()  # left column assembly: left pin column
    assert np.isfinite(out[1:, 1:, :, 3]).all()
    assert np.isfinite(out[:, :, :, 4]).all()  # interior assembly untouched


def test_reflected_radial_masked(core):
    ds = VeraDataset(np.ones((3, 3, 9)), VeraDtype.RADIAL)
    out = nan_out_reflected(core.reduced_core_map, 4, ds)
    assert np.isnan(out[0, :, 1]).all()
    assert np.isfinite(out[:, :, 8]).all()


def test_reflected_nodes_masked(core):
    ds = VeraDataset(np.ones((4, 4, 9)), VeraDtype.NODAL)
    out = nan_out_reflected(core.reduced_core_map, 4, ds)
    assert np.isnan(out[[0, 1, 2], :, 0]).all()
    assert np.isfinite(out[3, :, 0]).all()
    assert np.isfinite(out[:, :, 4]).all()


def test_no_masking_for_full_core(core):
    out = nan_out_reflected(core.reduced_core_map, 1, _pin_ones())
    assert np.isfinite(out).all()


def test_no_masking_for_dtypes_without_reflection(core):
    ds = VeraDataset(np.ones((1, 4, 9)), VeraDtype.ASSEMBLY)
    assert np.isfinite(nan_out_reflected(core.reduced_core_map, 4, ds)).all()
