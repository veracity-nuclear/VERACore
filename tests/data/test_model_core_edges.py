"""VeraOutCore branches not covered by test_model_core.py: alternate
dataset names, override precedence, computational and detector geometry."""

import numpy as np
import pytest

from vera_core.data.dtypes import VeraDtype
from vera_core.data.model import VeraOutCore

from .slice_support import quarter_map
from .source_support import NASS, NAX, TypedSource, core_arrays, make_core

T = VeraDtype
COMP_MESH = np.linspace(0.0, 100.0, 4)  # 3 computational levels


def comp_core(**kwargs):
    return make_core(computational_core_map=quarter_map(), computational_axial_mesh=COMP_MESH, **kwargs)


def detector_core(mesh=np.array([5.0, 15.0])):
    return make_core(detector_map=quarter_map(), detector_axial_mesh=mesh)


# ---------------------------------------------------------------- lattice sources


def test_num_pins_is_accepted_in_place_of_npin():
    assert make_core(npin=None, num_pins=np.array([7])).core_shape[:2] == (7, 7)


def test_pin_volumes_shape_silently_overrides_declared_npin():
    """Documents a hazard: every geometry dataset found overwrites npin, with
    no mismatch check (unlike nax, which raises)."""
    core = make_core(npin=5, pin_volumes=np.ones((3, 3, NAX, NASS)))
    assert core.core_shape[:2] == (3, 3)


def test_nax_override_replaces_file_axial_mesh():
    """Documents a hazard: an nax override discards the real mesh and
    rebuilds a uniform one with DEFAULT_AXIAL_MESH_STEP."""
    core = VeraOutCore(TypedSource(core_arrays()), overrides={"nax": 7})
    assert core.nax == 7
    assert core.axial_mesh[-1] == 70.0


def test_npin_override_beats_geometry_dataset():
    core = VeraOutCore(
        TypedSource(core_arrays(pin_volumes=np.ones((3, 3, NAX, NASS)))), overrides={"npin": 17}
    )
    assert core.core_shape[:2] == (17, 17)


def test_apitch_uses_pin_count_from_geometry():
    core = make_core(npin=None, apitch=np.array([30.0]), pin_volumes=np.ones((3, 3, NAX, NASS)))
    assert core.pin_pitch == pytest.approx(10.0)


# ---------------------------------------------------------------- symmetry and maps


def test_unsupported_symmetry_raises():
    with pytest.raises(Exception, match="Unhandled symmetry: 2"):
        make_core(core_sym=np.array([2]))


def test_full_core_symmetry_inferred_from_unique_ids():
    core = make_core(core_map=np.arange(1, 26).reshape(5, 5), core_sym=None)
    assert core.core_sym == 1


def test_nan_cells_in_core_map_are_ignored_when_counting():
    cm = quarter_map().astype(float)
    cm[0, 0] = np.nan
    assert make_core(core_map=cm).nass == NASS


def test_even_quarter_core_reduces_to_quadrant_with_labels():
    q = np.array([[1, 2], [3, 4]])
    cm = np.block([[q[::-1, ::-1], q[::-1, :]], [q[:, ::-1], q]])
    core = make_core(core_map=cm)
    assert core.is_even()
    np.testing.assert_array_equal(core.reduced_core_map, q)
    assert core.reduced_core_map_row_labels == ["3", "4"]
    assert core.reduced_core_map_column_labels == ["B", "A"]


# ---------------------------------------------------------------- computational core


def test_comp_map_without_comp_mesh_is_ignored():
    core = make_core(computational_core_map=quarter_map())
    assert not core.has_comp_core()
    assert core.comp_nass is None


def test_comp_mesh_falls_back_to_nodal_xs_axial_mesh():
    core = make_core(
        **{"computational_core_map": quarter_map(), "STATE_0001/NODAL_XS/AXIALMESH": COMP_MESH}
    )
    assert core.comp_nax == 3


def test_full_comp_map_is_reduced_on_quarter_core():
    assert comp_core().comp_core_map.shape == (3, 3)


def test_already_reduced_comp_map_is_kept():
    core = make_core(
        computational_core_map=np.arange(1, 10).reshape(3, 3), computational_axial_mesh=COMP_MESH
    )
    np.testing.assert_array_equal(core.comp_core_map, np.arange(1, 10).reshape(3, 3))


def test_comp_accessors_use_comp_geometry():
    core = comp_core()
    np.testing.assert_array_equal(core.get_axial_mesh_pixels(dataset_type=T.COMP_PIN), [3, 3, 3])
    np.testing.assert_allclose(
        core.get_axial_mesh_means(dataset_type=T.COMP_NODAL), [16.6667, 50.0, 83.3333]
    )
    assert core.comp_core_map_column_labels == ["C", "B", "A"]
    assert core.reduced_core_map_label(0, is_comp=True) == "C-3"


def test_gross_mesh_merges_comp_levels():
    core = comp_core()
    assert set(core.comp_axial_mesh_means) <= set(core.gross_axial_mesh)
    assert set(core.axial_mesh_means) <= set(core.gross_axial_mesh)


def test_geometry_on_comp_grid_switches_nass_to_comp_nass():
    comp_map = np.arange(1, 13).reshape(3, 4)
    core = make_core(
        pin_volumes=np.ones((3, 3, NAX, 12)),
        computational_core_map=comp_map,
        computational_axial_mesh=COMP_MESH,
    )
    assert core.nass == 12
    assert core.core_map is core.comp_core_map


def test_comp_assembly_lookup():
    assert comp_core().reduced_core_map_assembly(1, 1, is_comp=True) == 4


def test_comp_label_falls_back_to_reduced_map_without_comp_core():
    assert make_core().reduced_core_map_label(0, is_comp=True) == "C-3"


# ---------------------------------------------------------------- detectors


def test_detector_shape_is_four_dimensional():
    """The docstring says (ndax, ndet); callers such as get_safe_idxs
    unpack four values, so the 4-tuple is the contract."""
    assert detector_core().detector_shape == (3, 3, 2, NASS)
    assert detector_core().get_core_shape(dataset_type=T.POINT_DETECTOR) == (3, 3, 2, NASS)


def test_detector_row_and_column_lookup():
    core = detector_core()
    np.testing.assert_array_equal(core.row_assembly_indices(4, is_detector=True), [3, 4, 5])
    np.testing.assert_array_equal(core.col_assembly_indices(4, is_detector=True), [1, 4, 7])


def test_empty_detector_map_lookup_returns_minus_one():
    assert make_core(detector_map=np.zeros((5, 5))).reduced_core_map_assembly(
        0, 0, is_detector=True
    ) == -1


def test_two_dimensional_detector_mesh_is_continuous():
    core = detector_core(np.array([[0.0, 10.0], [20.0, 30.0]]))
    assert core.is_continous_detector
    assert core.core_dtypes((2, NASS)) == T.CONTINOUS_DETECTOR
    assert {0.0, 10.0, 20.0, 30.0} <= set(core.gross_axial_mesh)


def test_detector_pixels_follow_fuel_mesh_not_detector_levels():
    """Documents a hazard: get_axial_mesh_pixels has no detector branch, so a
    detector with ndax != nax gets a pixel array of the wrong length."""
    core = detector_core()
    assert len(core.get_axial_mesh_pixels(dataset_type=T.POINT_DETECTOR)) == NAX != core.ndax
