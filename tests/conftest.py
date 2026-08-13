import numpy as np
import pytest

from vera_core.data.analysis.axial_slice import AxialSlice
from vera_core.data.analysis.color import ColorSpec
from vera_core.data.analysis.core_slice import CoreSlice
from vera_core.data.analysis.surface_slice import SurfaceSlice
from vera_core.data.dtypes import VeraDtype

N_ASSEMBLIES = 12
PIN_SIDE = 17
N_LAYERS = 8
PIN_PITCH = 1.26

AXIAL_MESH = np.array([0.0, 12.0, 30.0, 55.0, 80.0, 110.0, 140.0, 175.0, 200.0])
"""Deliberately non-uniform: uniform layers would hide edge-placement bugs."""

COLUMN_LABELS = list("DCBA")
ROW_LABELS = ["4", "5", "6", "7"]


def core_map(n_rows=4, n_cols=4, n_entities=N_ASSEMBLIES):
    """1-based entity ids with 0 for empty positions.

    The last position is left empty on purpose, so every test exercises the
    NaN path for a position the map does not fill.
    """
    ids = np.zeros((n_rows, n_cols), dtype=int)
    ids.flat[:n_entities] = np.arange(1, n_entities + 1)
    return ids


class FakeCore:
    """Minimal VeraOutCore."""

    def __init__(self, non_fuel_locs=None, aspect_ratio=1.0):
        self.aspect_ratio = aspect_ratio
        self.non_fuel_locs = non_fuel_locs
        self.pin_pitch = PIN_PITCH
        self.core_shape = (PIN_SIDE, PIN_SIDE, N_LAYERS, N_ASSEMBLIES)
        self.reduced_core_map = core_map()
        self.comp_core_map = core_map()
        self.reduced_core_map_column_labels = COLUMN_LABELS
        self.reduced_core_map_row_labels = ROW_LABELS
        self.comp_core_map_column_labels = COLUMN_LABELS
        self.comp_core_map_row_labels = ROW_LABELS
        self.reduced_core_map_start_index = 3
        self.comp_map_start_index = 3

    def get_map(self, dataset_type=None):
        return self.comp_core_map if dataset_type.is_computational() else self.reduced_core_map

    def has_comp_core(self):
        return True

    def get_axial_mesh(self, dataset_type=None):
        return AXIAL_MESH

    def get_axial_mesh_means(self, dataset_type=None):
        return (AXIAL_MESH[:-1] + AXIAL_MESH[1:]) / 2

    def row_assembly_indices(self, assembly, is_comp=False, is_detector=False):
        """A cut across the map's second row, with one empty position."""
        return np.array([-1, 0, 1, 2, 3, 4, 5])

    col_assembly_indices = row_assembly_indices

    def reduced_core_map_assembly(self, i, j, is_comp=False, is_detector=False):
        return int(self.reduced_core_map[j, i]) - 1

    def reduced_core_map_label(self, assembly, is_comp=False):
        row, col = divmod(assembly, 4)
        return f"{COLUMN_LABELS[col]}-{ROW_LABELS[row]}"


class FakeSource:
    """Minimal VeraDataSource over a dict of datasets.

    Records how many times get_dataset() ran, so a test can prove the analysis
    layer copies rather than writing into what it was handed.
    """

    def __init__(self, datasets, core=None):
        self.datasets = datasets
        self.core = core or FakeCore()
        self.states = [{"exposure": np.array([0.0])}, {"exposure": np.array([52.413])}]
        self.active_state = self.states[0]
        self.reads = 0

    def get_dataset_dtype(self, name, state=0):
        return self.datasets[name][0]

    def get_dataset(self, name, mask_reflected=True, state_idx=0):
        self.reads += 1
        return self.datasets[name][1]

    def get_dataset_units(self, name, state=0):
        return "unitless"


# -- slice fixtures, for render tests that need no source -----------------


def _empty_at(rows, cols):
    """The grid position left empty, or None when the grid is too small to
    spare one."""
    return (rows - 1, cols - 1) if rows * cols > 1 else None


def make_core_slice(cell=(2, 2), rows=4, cols=4, scale=1.0, **kwargs):
    """A CoreSlice with one empty grid position and one NaN pin."""
    rng = np.random.default_rng(0)
    empty = _empty_at(rows, cols)
    data = np.full((rows, cols, *cell), np.nan) if cell else np.full((rows, cols), np.nan)
    for row in range(rows):
        for col in range(cols):
            if (row, col) == empty:
                continue
            data[row, col] = scale * rng.uniform(0.5, 1.5, cell or ())
    if cell == (2, 2):
        data[0, 0, 0, 0] = np.nan
    finite = data[~np.isnan(data)]
    kwargs.setdefault("slice_value_range", (float(finite.min()), float(finite.max())))
    return CoreSlice(
        data=data,
        x_labels=COLUMN_LABELS,
        y_labels=ROW_LABELS,
        units="unitless",
        **kwargs,
    )


def make_surface_slice(side=2, rows=4, cols=4, scale=1.0, **kwargs):
    rng = np.random.default_rng(1)
    empty = _empty_at(rows, cols)
    data = np.full((rows, cols, side, side, 4), np.nan)
    for row in range(rows):
        for col in range(cols):
            if (row, col) == empty:
                continue
            data[row, col] = scale * rng.uniform(0.85, 1.25, (side, side, 4))
    finite = data[~np.isnan(data)]
    kwargs.setdefault("slice_value_range", (float(finite.min()), float(finite.max())))
    return SurfaceSlice(
        data=data,
        x_labels=COLUMN_LABELS,
        y_labels=ROW_LABELS,
        units="unitless",
        **kwargs,
    )


def make_axial_slice(cell_width=2, n_cols=7, scale=1.0, **kwargs):
    rng = np.random.default_rng(2)
    data = scale * rng.uniform(0.4, 1.3, (N_LAYERS, n_cols, cell_width))
    data[:, 0] = np.nan  # empty position along the cut
    finite = data[~np.isnan(data)]
    kwargs.setdefault("slice_value_range", (float(finite.min()), float(finite.max())))
    return AxialSlice(
        data=data,
        x_edges=np.linspace(0, n_cols * PIN_SIDE * PIN_PITCH, n_cols * cell_width + 1),
        y_edges=AXIAL_MESH,
        x_labels=[f"C{i}" for i in range(n_cols)],
        dtype=VeraDtype.NODAL,
        units="unitless",
        **kwargs,
    )


@pytest.fixture
def core_slice():
    return make_core_slice()


@pytest.fixture
def surface_slice():
    return make_surface_slice()


@pytest.fixture
def axial_slice():
    return make_axial_slice()


@pytest.fixture
def two_groups():
    """Two core slices an order of magnitude apart, for color-range tests."""
    return [
        make_core_slice(scale=1.0, group=0, n_groups=2),
        make_core_slice(scale=0.02, group=1, n_groups=2),
    ]


@pytest.fixture
def spec():
    return ColorSpec(0.0, 10.0)

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
    make_surface_slice,
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
