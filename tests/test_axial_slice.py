"""Tests for the analysis layer.

The slice builders are where geometry decisions live, so these focus on the
things that go wrong silently: axis order, NaN placement, non-uniform mesh
edges, and writing into a caller's array.
"""

import numpy as np
import pytest
from conftest import (
    AXIAL_MESH,
    N_ASSEMBLIES,
    N_LAYERS,
    PIN_SIDE,
    FakeCore,
    FakeSource,
    make_axial_slice,
    make_core_slice,
)

from vera_core.data.analysis.axial_slice import (
    X_AXIS,
    Y_AXIS,
    AxialRequest,
    axial_slices,
    node_pair,
)
from vera_core.data.analysis.surface_slice import (
    FACES,
    SurfaceRequest,
    lateral_faces,
    surface_slices,
)
from vera_core.data.dtypes import VeraDtype

RNG = np.random.default_rng(11)


# -- CoreSlice -------------------------------------------------------------


class TestCoreSlice:
    def test_as_image_keeps_pins_adjacent(self):
        """A pin's neighbour in the flattened image must be its neighbour in
        the core, not the same pin from the next assembly."""
        slice_ = make_core_slice(cell=(2, 2), rows=2, cols=2)
        image = slice_.as_image()
        assert image.shape == (4, 4)
        assert image[0, 0] == slice_.data[0, 0, 0, 0] or np.isnan(image[0, 0])
        assert image[0, 2] == slice_.data[0, 1, 0, 0]
        assert image[2, 0] == slice_.data[1, 0, 0, 0]

    def test_finite_drops_nan(self):
        slice_ = make_core_slice()
        assert not np.isnan(slice_.finite()).any()
        assert len(slice_.finite()) < slice_.data.size

    def test_empty_position_is_all_nan(self):
        slice_ = make_core_slice()
        assert np.isnan(slice_.assembly(3, 3)).all()

    def test_validate_accepts_a_well_formed_slice(self):
        assert make_core_slice().validate() == []

    def test_validate_catches_label_count_mismatch(self):
        slice_ = make_core_slice()
        bad = type(slice_)(**{**slice_.__dict__, "x_labels": ["only", "two"]})
        assert any("x_labels" in p for p in bad.validate())

    def test_validate_catches_bad_value_range(self):
        slice_ = make_core_slice()
        bad = type(slice_)(**{**slice_.__dict__, "value_range": (5.0, 1.0)})
        assert any("value_range" in p for p in bad.validate())

    def test_validate_catches_group_outside_n_groups(self):
        slice_ = make_core_slice()
        bad = type(slice_)(**{**slice_.__dict__, "group": 4, "n_groups": 2})
        assert any("group" in p for p in bad.validate())


# -- SurfaceSlice ----------------------------------------------------------


def surface_source(n_nodes=4, n_groups=2):
    """(face, group, node, layer, assembly), the reader's layout."""
    data = RNG.uniform(0.8, 1.3, (6, n_groups, n_nodes, N_LAYERS, N_ASSEMBLIES))
    return FakeSource({"adf": (VeraDtype.COMP_NODAL_SURFACE, data)}), data


class TestSurfaceSlices:
    def test_one_slice_per_energy_group(self):
        source, _ = surface_source(n_groups=3)
        slices = surface_slices(source, SurfaceRequest(array="adf", state=0, z=2))
        assert len(slices) == 3
        assert [s.group for s in slices] == [0, 1, 2]
        assert all(s.n_groups == 3 for s in slices)

    def test_single_group_leaves_group_none(self):
        source, _ = surface_source(n_groups=1)
        (slice_,) = surface_slices(source, SurfaceRequest(array="adf", state=0, z=2))
        assert slice_.group is None, "downstream keys color specs off this"

    def test_shape_is_grid_by_node_square_by_face(self):
        source, _ = surface_source(n_nodes=4)
        (slice_, _) = surface_slices(source, SurfaceRequest(array="adf", state=0, z=1))
        assert slice_.data.shape == (4, 4, 2, 2, 4)
        assert slice_.node_side == 2 and slice_.n_nodes == 4

    def test_assembly_level_surface_is_one_node(self):
        source, _ = surface_source(n_nodes=1)
        (slice_, _) = surface_slices(source, SurfaceRequest(array="adf", state=0, z=1))
        assert slice_.node_side == 1

    def test_only_lateral_faces_are_kept(self):
        """The dataset holds top and bottom too; a radial map has no place for
        them, and including them would silently skew the color range."""
        source, data = surface_source()
        data[4:] = 99.0  # top and bottom
        (slice_, _) = surface_slices(source, SurfaceRequest(array="adf", state=0, z=0))
        assert slice_.finite().max() < 99.0

    def test_face_lookup_matches_the_reader_order(self):
        source, data = surface_source()
        request = SurfaceRequest(array="adf", state=0, z=3)
        (slice_, _) = surface_slices(source, request)
        for index, name in enumerate(FACES):
            expected = lateral_faces(data, 0, 3)[0][..., index]
            assert np.allclose(slice_.face(name)[0, 0], expected)

    def test_empty_map_position_is_nan(self):
        source, _ = surface_source()
        (slice_, _) = surface_slices(source, SurfaceRequest(array="adf", state=0, z=0))
        assert np.isnan(slice_.assembly(3, 3)).all()

    def test_rejects_unsupported_dtype(self):
        source = FakeSource({"pin_powers": (VeraDtype.PIN, np.zeros((1, 1)))})
        request = SurfaceRequest(array="pin_powers", state=0, z=0)
        assert surface_slices(source, request) == []

    def test_negative_z_raises(self):
        source, _ = surface_source()
        with pytest.raises(ValueError, match="z must be"):
            surface_slices(source, SurfaceRequest(array="adf", state=0, z=-1))

    def test_validate_accepts_a_built_slice(self):
        source, _ = surface_source()
        for slice_ in surface_slices(source, SurfaceRequest(array="adf", state=0, z=2)):
            assert slice_.validate() == []


# -- AxialSlice ------------------------------------------------------------


def axial_source(dtype, shape):
    return FakeSource({"q": (dtype, RNG.uniform(0.5, 1.5, shape))})


def _writable(source):
    """FakeSource hands back the same array, so tests can mutate it."""
    return source.datasets["q"][1]


class TestAxialSlices:
    @pytest.mark.parametrize(
        "dtype, shape, cell_width",
        [
            (VeraDtype.PIN, (PIN_SIDE, PIN_SIDE, N_LAYERS, N_ASSEMBLIES), PIN_SIDE),
            (VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES), 1),
            (VeraDtype.NODAL, (4, N_LAYERS, N_ASSEMBLIES), 2),
            (VeraDtype.NODAL, (1, N_LAYERS, N_ASSEMBLIES), 1),
        ],
    )
    def test_cell_width_per_dtype(self, dtype, shape, cell_width):
        """A single-node dataset holds one value per assembly. The trame view
        computes a full pin width here and slices past the array."""
        source = axial_source(dtype, shape)
        (slice_,) = axial_slices(source, AxialRequest(array="q", state=0, pin=1))
        assert slice_.cell_width == cell_width

    def test_row_zero_is_the_lowest_elevation(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        (slice_,) = axial_slices(source, AxialRequest(array="q", state=0))
        assert slice_.y_edges[0] < slice_.y_edges[-1]
        assert slice_.elevations[0] < slice_.elevations[-1]

    def test_edges_follow_the_non_uniform_mesh(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        (slice_,) = axial_slices(source, AxialRequest(array="q", state=0))
        assert np.allclose(slice_.y_edges, AXIAL_MESH)
        heights = np.diff(slice_.y_edges)
        assert heights.min() != heights.max(), "test mesh must be non-uniform"

    def test_edge_counts_match_the_data(self):
        source = axial_source(VeraDtype.NODAL, (4, N_LAYERS, N_ASSEMBLIES))
        (slice_,) = axial_slices(source, AxialRequest(array="q", state=0))
        assert len(slice_.y_edges) == slice_.n_layers + 1
        assert len(slice_.x_edges) == slice_.n_cols * slice_.cell_width + 1

    def test_empty_column_along_the_cut_is_nan(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        (slice_,) = axial_slices(source, AxialRequest(array="q", state=0))
        assert np.isnan(slice_.column(0)).all(), "row_assembly_indices returns -1 there"
        assert not np.isnan(slice_.column(1)).all()

    def test_does_not_write_into_the_source_dataset(self):
        """The trame view assigns NaN into the dataset in place, which
        persists into every later read of it."""
        locs = (np.array([0]), np.array([0]), np.array([0]), np.array([0]))
        source = axial_source(VeraDtype.PIN, (PIN_SIDE, PIN_SIDE, N_LAYERS, N_ASSEMBLIES))
        source.core = FakeCore(non_fuel_locs=locs)
        before = np.array(source.datasets["q"][1], copy=True)
        axial_slices(source, AxialRequest(array="q", state=0))
        assert np.array_equal(source.datasets["q"][1], before)

    def test_energy_groups_split(self):
        source = axial_source(VeraDtype.COMP_NODAL_ENERGY, (3, 4, N_LAYERS, N_ASSEMBLIES))
        slices = axial_slices(source, AxialRequest(array="q", state=0))
        assert [s.group for s in slices] == [0, 1, 2]

    def test_x_and_y_cuts_use_different_labels(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        x_cut = axial_slices(source, AxialRequest(array="q", state=0, axis=X_AXIS))[0]
        y_cut = axial_slices(source, AxialRequest(array="q", state=0, axis=Y_AXIS))[0]
        assert x_cut.x_labels != y_cut.x_labels

    def test_unknown_axis_raises(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        with pytest.raises(ValueError, match="axis must be"):
            axial_slices(source, AxialRequest(array="q", state=0, axis="diagonal"))

    def test_rejects_unsupported_dtype(self):
        source = FakeSource({"adf": (VeraDtype.COMP_NODAL_SURFACE, np.zeros((1, 1)))})
        assert axial_slices(source, AxialRequest(array="adf", state=0)) == []

    def test_labelable_tracks_cell_width(self):
        assert make_axial_slice(cell_width=2).labelable
        assert not make_axial_slice(cell_width=PIN_SIDE).labelable

    def test_validate_catches_descending_edges(self):
        slice_ = make_axial_slice()
        bad = type(slice_)(**{**slice_.__dict__, "y_edges": AXIAL_MESH[::-1]})
        assert any("ascend" in p for p in bad.validate())

    def test_validate_catches_edge_count_mismatch(self):
        slice_ = make_axial_slice()
        bad = type(slice_)(**{**slice_.__dict__, "y_edges": AXIAL_MESH[:-1]})
        assert any("y_edges" in p for p in bad.validate())


class TestNodePair:
    def test_x_cut_splits_on_the_upper_and_lower_halves(self):
        assert node_pair(0, is_x=True) == (0, 1)
        assert node_pair(1, is_x=True) == (2, 3)

    def test_y_cut_splits_on_the_left_and_right_halves(self):
        assert node_pair(0, is_x=False) == (0, 2)
        assert node_pair(1, is_x=False) == (1, 3)

    def test_out_of_range_pin_clamps(self):
        assert node_pair(99, is_x=True) == (2, 3)
        assert node_pair(99, is_x=False) == (1, 3)


class TestAxialEmptyCut:
    """An empty cut used to render as a blank panel with a 0-to-1 colorbar,
    which reads as a plot rather than a failure."""

    def test_no_assemblies_on_the_cut_raises(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        source.core.row_assembly_indices = lambda *a, **k: np.full(9, -1)
        with pytest.raises(ValueError, match="no assemblies"):
            axial_slices(source, AxialRequest(array="q", state=0))

    def test_all_nan_values_raise(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        source.datasets["q"][1][:] = np.nan
        with pytest.raises(ValueError, match="no finite values"):
            axial_slices(source, AxialRequest(array="q", state=0))

    def test_the_message_reports_the_shapes(self):
        source = axial_source(VeraDtype.ASSEMBLY, (1, N_LAYERS, N_ASSEMBLIES))
        source.datasets["q"][1][:] = np.nan
        with pytest.raises(ValueError, match=r"cut \(\d+, \d+, \d+\)"):
            axial_slices(source, AxialRequest(array="q", state=0))
