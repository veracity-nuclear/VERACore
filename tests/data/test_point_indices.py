"""point_indices against the table view's original hand-written index tuples."""

import numpy as np
import pytest

from vera_core.data.dtypes import NUM_DF, NUM_NODES, VeraDataset, VeraDim, VeraDtype, point_indices

T = VeraDtype
NPIN, NAX, NASS, NG = 3, 5, 9, 2
J, I, LAYER, ASSY, SURF = 2, 1, 3, 7, 4
NODE = int(np.clip(I + J * (NUM_NODES // 2), 0, NUM_NODES - 1))  # helpers.convert_ji_to_node
SEL = {
    VeraDim.PIN_Y: J,
    VeraDim.PIN_X: I,
    VeraDim.NODE: NODE,
    VeraDim.AXIAL: LAYER,
    VeraDim.ASSEMBLY: ASSY,
    VeraDim.SURFACE: SURF,
}
SHAPES = {
    T.PIN: (NPIN, NPIN, NAX, NASS),
    T.COMP_PIN: (NPIN, NPIN, NAX, NASS),
    T.CHANNEL: (NPIN + 1, NPIN + 1, NAX, NASS),
    T.RADIAL: (NPIN, NPIN, NASS),
    T.CHANNEL_RADIAL: (NPIN + 1, NPIN + 1, NASS),
    T.ASSEMBLY: (1, NAX, NASS),
    T.COMP_ASSY: (1, NAX, NASS),
    T.ASSY_ENERGY: (NG, 1, NAX, NASS),
    T.COMP_ASSY_ENERGY: (NG, 1, NAX, NASS),
    T.ASSY_SURFACE: (NUM_DF, NG, 1, NAX, NASS),
    T.COMP_ASSY_SURFACE: (NUM_DF, NG, 1, NAX, NASS),
    T.RADIAL_ASSEMBLY: (NASS,),
    T.NODAL: (NUM_NODES, NAX, NASS),
    T.COMP_NODAL: (NUM_NODES, NAX, NASS),
    T.NODAL_ENERGY: (NG, NUM_NODES, NAX, NASS),
    T.COMP_NODAL_ENERGY: (NG, NUM_NODES, NAX, NASS),
    T.NODAL_SURFACE: (NUM_DF, NG, NUM_NODES, NAX, NASS),
    T.COMP_NODAL_SURFACE: (NUM_DF, NG, NUM_NODES, NAX, NASS),
    T.RADIAL_NODE: (NUM_NODES, NASS),
    T.AXIAL: (NAX,),
    T.POINT_DETECTOR: (NAX, NASS),
    T.CONTINOUS_DETECTOR: (NAX, NASS),
    T.RADIAL_POINT_DETECTOR: (NASS,),
    T.SCALAR: (1,),
}


def old_table_indices(dtype, shape):
    """The table view's match statement, bare ints wrapped as 1-tuples."""
    match dtype:
        case T.PIN | T.CHANNEL:
            return [(J, I, LAYER, ASSY)]
        case T.POINT_DETECTOR:
            return [(LAYER, ASSY)]
        case T.NODAL | T.COMP_NODAL:
            return [(NODE, LAYER, ASSY)]
        case T.RADIAL_NODE:
            return [(NODE, ASSY)]
        case T.COMP_ASSY | T.ASSEMBLY:
            return [(0, LAYER, ASSY)]
        case T.AXIAL:
            return [(LAYER,)]
        case T.RADIAL | T.CHANNEL_RADIAL:
            return [(J, I, ASSY)]
        case T.RADIAL_ASSEMBLY | T.RADIAL_POINT_DETECTOR:
            return [(ASSY,)]
        case T.SCALAR:
            return [(0,)]
        case T.COMP_NODAL_ENERGY | T.COMP_ASSY_ENERGY | T.ASSY_ENERGY | T.NODAL_ENERGY:
            idx = NODE if dtype in (T.COMP_NODAL_ENERGY, T.NODAL_ENERGY) else 0
            return [(g, idx, LAYER, ASSY) for g in range(shape[0])]
        case T.COMP_ASSY_SURFACE | T.COMP_NODAL_SURFACE:
            idx = 0 if dtype == T.COMP_ASSY_SURFACE else NODE
            return [(SURF, g, idx, LAYER, ASSY) for g in range(shape[1])]
    return None


HANDLED = [t for t in SHAPES if old_table_indices(t, SHAPES[t]) is not None]
NEW = [t for t in SHAPES if t not in HANDLED]


@pytest.mark.parametrize("dtype", HANDLED, ids=str)
def test_matches_original_tuples(dtype):
    assert point_indices(dtype, SHAPES[dtype], SEL) == old_table_indices(dtype, SHAPES[dtype])


def test_newly_handled_are_the_ones_the_views_missed():
    assert set(NEW) == {T.ASSY_SURFACE, T.COMP_PIN, T.CONTINOUS_DETECTOR, T.NODAL_SURFACE}


@pytest.mark.parametrize("dtype", NEW, ids=str)
def test_newly_handled_pick_the_selected_element(dtype):
    shape = SHAPES[dtype]
    data = VeraDataset(np.arange(np.prod(shape)).reshape(shape), dtype)
    points = point_indices(dtype, shape, SEL)
    groups = data.arrange(
        order=(),
        split=(VeraDim.GROUP,),
        pin=(J, I),
        node=NODE,
        axial=LAYER,
        assembly=ASSY,
        surface=SURF,
    )
    assert [data[p] for p in points] == [g.item() for g in groups]


def test_every_tuple_is_full_length_and_indexes_a_scalar():
    for dtype, shape in SHAPES.items():
        data = np.zeros(shape)
        for p in point_indices(dtype, shape, SEL):
            assert len(p) == len(shape) and np.ndim(data[p]) == 0, dtype


def test_split_none_requires_group_selection():
    with pytest.raises(ValueError, match="unindexed: group"):
        point_indices(T.NODAL_ENERGY, SHAPES[T.NODAL_ENERGY], SEL, split=())
    got = point_indices(T.NODAL_ENERGY, SHAPES[T.NODAL_ENERGY], {**SEL, VeraDim.GROUP: 1}, split=())
    assert got == [(1, NODE, LAYER, ASSY)]


def test_missing_dim_rejected():
    with pytest.raises(ValueError, match="unindexed: assembly"):
        point_indices(T.PIN, SHAPES[T.PIN], {VeraDim.PIN_Y: 0, VeraDim.PIN_X: 0, VeraDim.AXIAL: 0})


def test_unknown_with_data_rejected():
    with pytest.raises(ValueError, match="no semantic dims"):
        point_indices(T.UNKNOWN, (3, 4), SEL)
