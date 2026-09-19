"""Semantic layout of every VeraDtype.

LAYOUT and FLAGS are written out by hand, independent of _INFO, so an
accidental edit to _INFO fails here. Everything else in this file and in
test_arrange.py can then rely on dim_axes.
"""

import dataclasses

import numpy as np
import pytest

from vera_core.data.dtypes import (
    LATERAL_SURFACES,
    MAX_NUM_GROUPS,
    MIN_NUM_GROUPS,
    NUM_DF,
    NUM_NODES,
    Surface,
    VeraDim,
    VeraDtype,
    _Info,
    build_core_dtypes,
    make_slice,
)

T = VeraDtype
FIXED = "fixed"  # an axis with no meaning, always indexed at 0
PY, PX, AX, AS, ND, GR, SF = (
    VeraDim.PIN_Y,
    VeraDim.PIN_X,
    VeraDim.AXIAL,
    VeraDim.ASSEMBLY,
    VeraDim.NODE,
    VeraDim.GROUP,
    VeraDim.SURFACE,
)

# Physical axis order of every dtype.
LAYOUT = {
    T.PIN: (PY, PX, AX, AS),
    T.COMP_PIN: (PY, PX, AX, AS),
    T.RADIAL: (PY, PX, AS),
    T.CHANNEL: (PY, PX, AX, AS),
    T.CHANNEL_RADIAL: (PY, PX, AS),
    T.ASSEMBLY: (FIXED, AX, AS),
    T.COMP_ASSY: (FIXED, AX, AS),
    T.ASSY_ENERGY: (GR, FIXED, AX, AS),
    T.COMP_ASSY_ENERGY: (GR, FIXED, AX, AS),
    T.ASSY_SURFACE: (SF, GR, FIXED, AX, AS),
    T.COMP_ASSY_SURFACE: (SF, GR, FIXED, AX, AS),
    T.RADIAL_ASSEMBLY: (AS,),
    T.NODAL: (ND, AX, AS),
    T.COMP_NODAL: (ND, AX, AS),
    T.NODAL_ENERGY: (GR, ND, AX, AS),
    T.COMP_NODAL_ENERGY: (GR, ND, AX, AS),
    T.NODAL_SURFACE: (SF, GR, ND, AX, AS),
    T.COMP_NODAL_SURFACE: (SF, GR, ND, AX, AS),
    T.RADIAL_NODE: (ND, AS),
    T.AXIAL: (AX,),
    T.POINT_DETECTOR: (AX, AS),
    T.CONTINOUS_DETECTOR: (AX, AS),
    T.RADIAL_POINT_DETECTOR: (AS,),
    T.SCALAR: (),
    T.UNKNOWN: (),
}

# Which flags each dtype carries. "surface" has no public predicate, so it
# is read from _info (see test_surface_flag_has_no_public_predicate).
FLAGS = {
    T.PIN: {"fuel"},
    T.COMP_PIN: {"fuel", "comp"},
    T.RADIAL: {"fuel"},
    T.CHANNEL: {"channel"},
    T.CHANNEL_RADIAL: {"channel"},
    T.ASSEMBLY: {"assembly"},
    T.COMP_ASSY: {"assembly", "comp"},
    T.ASSY_ENERGY: {"assembly"},
    T.COMP_ASSY_ENERGY: {"assembly", "comp"},
    T.ASSY_SURFACE: {"assembly", "surface"},
    T.COMP_ASSY_SURFACE: {"assembly", "surface", "comp"},
    T.RADIAL_ASSEMBLY: {"assembly"},
    T.NODAL: {"nodal"},
    T.COMP_NODAL: {"nodal", "comp"},
    T.NODAL_ENERGY: {"nodal"},
    T.COMP_NODAL_ENERGY: {"nodal", "comp"},
    T.NODAL_SURFACE: {"nodal", "surface"},
    T.COMP_NODAL_SURFACE: {"nodal", "surface", "comp"},
    T.RADIAL_NODE: {"nodal"},
    T.AXIAL: set(),
    T.POINT_DETECTOR: {"assembly", "detector"},
    T.CONTINOUS_DETECTOR: {"assembly", "detector"},
    T.RADIAL_POINT_DETECTOR: {"assembly", "detector"},
    T.SCALAR: set(),
    T.UNKNOWN: set(),
}

PREDICATES = {
    "fuel": lambda d: d.has_fuel_pins(),
    "comp": lambda d: d.is_computational(),
    "nodal": lambda d: d.is_nodal(),
    "assembly": lambda d: d.is_assembly(),
    "channel": lambda d: d.is_channel(),
    "detector": lambda d: d.is_detector(),
    "surface": lambda d: d._info.surface,
}

ALL = list(VeraDtype)
SEMANTIC_DIMS = list(VeraDim)


def physical_order(dtype):
    order = [FIXED] * dtype.ndim
    for dim, axis in dtype.dim_axes.items():
        order[axis] = dim
    return tuple(order)


def without(layout, *drop):
    return tuple(axis for axis in layout if axis not in drop)


# ---------------------------------------------------------------- spec tables


def test_spec_tables_cover_every_dtype():
    assert set(LAYOUT) == set(ALL) == set(FLAGS)


def test_there_are_25_distinct_dtypes():
    assert len(ALL) == 25


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_layout_matches_spec(dtype):
    assert physical_order(dtype) == LAYOUT[dtype]


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_ndim_matches_spec(dtype):
    assert dtype.ndim == len(LAYOUT[dtype])


@pytest.mark.parametrize("dtype", ALL, ids=str)
@pytest.mark.parametrize("flag", sorted(PREDICATES))
def test_flags_match_spec(dtype, flag):
    assert PREDICATES[flag](dtype) == (flag in FLAGS[dtype])


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_fixed_axes_are_indexed_at_zero(dtype):
    fixed = {i for i, axis in enumerate(LAYOUT[dtype]) if axis == FIXED}
    assert dtype._info.fixed_idxs == dict.fromkeys(fixed, 0)


def test_surface_flag_has_no_public_predicate():
    """Every other flag has an is_/has_ method; surface is only on _info.
    has_surface_dim() is equivalent today (see the invariant tests)."""
    assert not hasattr(T.NODAL_SURFACE, "is_surface")


# ---------------------------------------------------------------- accessors


ACCESSORS = {
    AX: ("axial_dim_idx", "has_axial_dim"),
    GR: ("energy_group_dim_idx", "has_energy_group_dim"),
    ND: ("node_dim_idx", "has_node_dim"),
    AS: ("assembly_id_dim_idx", "has_assembly_id_dim"),
    SF: ("surface_dim_idx", "has_surface_dim"),
}


@pytest.mark.parametrize("dtype", ALL, ids=str)
@pytest.mark.parametrize("dim", list(ACCESSORS), ids=str)
def test_single_axis_accessors(dtype, dim):
    index_attr, has_attr = ACCESSORS[dim]
    layout = LAYOUT[dtype]
    assert getattr(dtype, has_attr)() == (dim in layout)
    if dim in layout:
        assert getattr(dtype, index_attr) == layout.index(dim)
    else:
        with pytest.raises(ValueError):
            getattr(dtype, index_attr)


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_pin_accessors(dtype):
    layout = LAYOUT[dtype]
    assert dtype.has_pin_level_dim() == (PY in layout)
    if PY in layout:
        assert dtype.pin_dim_idxs == (layout.index(PY), layout.index(PX))
    else:
        with pytest.raises(ValueError):
            dtype.pin_dim_idxs


# ---------------------------------------------------------------- invariants


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_pin_axes_come_as_a_pair_in_y_x_order(dtype):
    layout = LAYOUT[dtype]
    assert (PY in layout) == (PX in layout)
    if PY in layout:
        assert layout.index(PX) == layout.index(PY) + 1


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_assembly_axis_is_last_when_present(dtype):
    if dtype.has_assembly_id_dim():
        assert dtype.assembly_id_dim_idx == dtype.ndim - 1


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_axial_immediately_precedes_assembly(dtype):
    if dtype.has_axial_dim() and dtype.has_assembly_id_dim():
        assert dtype.axial_dim_idx == dtype.assembly_id_dim_idx - 1


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_surface_then_group_lead_when_present(dtype):
    layout = LAYOUT[dtype]
    if SF in layout:
        assert layout[:2] == (SF, GR)
    elif GR in layout:
        assert layout[0] == GR


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_flag_dimension_invariants(dtype):
    assert dtype._info.surface == dtype.has_surface_dim()
    assert dtype.is_nodal() == dtype.has_node_dim()
    if dtype.has_fuel_pins() or dtype.is_channel():
        assert dtype.has_pin_level_dim()
    if dtype.has_surface_dim():
        assert dtype.has_energy_group_dim()
    if dtype.has_energy_group_dim():
        assert dtype.is_nodal() or dtype.is_assembly()
    if dtype.is_detector():
        assert dtype.is_assembly() and not dtype.is_computational()


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_fuel_and_channel_are_exclusive(dtype):
    assert not (dtype.has_fuel_pins() and dtype.is_channel())


def test_every_semantic_dim_is_used():
    used = {axis for layout in LAYOUT.values() for axis in layout}
    assert set(SEMANTIC_DIMS) <= used


def test_dim_values_are_lowercase_names():
    assert all(dim.value == dim.name.lower() for dim in VeraDim)


# ---------------------------------------------------------------- dtype families


COMP_PAIRS = [
    (T.PIN, T.COMP_PIN),
    (T.NODAL, T.COMP_NODAL),
    (T.ASSEMBLY, T.COMP_ASSY),
    (T.NODAL_ENERGY, T.COMP_NODAL_ENERGY),
    (T.NODAL_SURFACE, T.COMP_NODAL_SURFACE),
    (T.ASSY_ENERGY, T.COMP_ASSY_ENERGY),
    (T.ASSY_SURFACE, T.COMP_ASSY_SURFACE),
]


@pytest.mark.parametrize("plain, comp", COMP_PAIRS, ids=lambda d: str(d))
def test_comp_variant_differs_only_by_computational_flag(plain, comp):
    assert dataclasses.replace(plain._info, computational=True) == comp._info


def test_every_comp_dtype_has_a_plain_variant():
    comps = {d for d in ALL if d.name.startswith("COMP_")}
    assert comps == {comp for _, comp in COMP_PAIRS}


@pytest.mark.parametrize(
    "base, energy",
    [(T.NODAL, T.NODAL_ENERGY), (T.ASSEMBLY, T.ASSY_ENERGY)],
    ids=str,
)
def test_energy_variant_prepends_group(base, energy):
    assert LAYOUT[energy] == (GR,) + LAYOUT[base]


@pytest.mark.parametrize(
    "energy, surface",
    [(T.NODAL_ENERGY, T.NODAL_SURFACE), (T.ASSY_ENERGY, T.ASSY_SURFACE)],
    ids=str,
)
def test_surface_variant_prepends_surface(energy, surface):
    assert LAYOUT[surface] == (SF,) + LAYOUT[energy]


@pytest.mark.parametrize(
    "full, radial",
    [
        (T.PIN, T.RADIAL),
        (T.CHANNEL, T.CHANNEL_RADIAL),
        (T.NODAL, T.RADIAL_NODE),
        (T.POINT_DETECTOR, T.RADIAL_POINT_DETECTOR),
    ],
    ids=str,
)
def test_radial_variant_drops_axial(full, radial):
    assert LAYOUT[radial] == without(LAYOUT[full], AX)


def test_radial_assembly_drops_fixed_and_axial():
    assert LAYOUT[T.RADIAL_ASSEMBLY] == without(LAYOUT[T.ASSEMBLY], FIXED, AX)


def test_point_and_continuous_detectors_share_layout():
    assert LAYOUT[T.POINT_DETECTOR] == LAYOUT[T.CONTINOUS_DETECTOR]


# ---------------------------------------------------------------- aliases and names


def test_core_is_scalar():
    assert T.CORE is T.SCALAR
    assert T["CORE"] is T.SCALAR


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_name_round_trips(dtype):
    assert T[str(dtype)] is dtype
    assert dtype.title == dtype.str == dtype.name


# ---------------------------------------------------------------- make_slice


FULL = slice(None)
SLICE_KW = {SF: "surface_idx", GR: "group_idx", ND: "node_idx", AX: "axial_idx", AS: "assembly_id"}


def expected_slice(dtype, picks):
    return tuple(
        0 if axis == FIXED else picks.get(axis, FULL) for axis in LAYOUT[dtype]
    )


def slice_kwargs(picks):
    kw = {SLICE_KW[d]: v for d, v in picks.items() if d in SLICE_KW}
    if PY in picks:
        kw["pin_idxs"] = (picks[PY], picks[PX])
    return kw


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_make_slice_without_selectors_only_fixes_fixed_axes(dtype):
    assert make_slice(dtype) == expected_slice(dtype, {})


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_make_slice_with_every_selector(dtype):
    picks = {PY: 11, PX: 12, AX: 13, AS: 14, ND: 15, GR: 16, SF: 17}
    assert make_slice(dtype, **slice_kwargs(picks)) == expected_slice(dtype, picks)


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_make_slice_one_selector_at_a_time(dtype):
    for dim in (AX, AS, ND, GR, SF):
        picks = {dim: 7}
        assert make_slice(dtype, **slice_kwargs(picks)) == expected_slice(dtype, picks), dim


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_make_slice_method_matches_function(dtype):
    kw = {"axial_idx": 1, "group_idx": 2, "pin_idxs": (3, 4)}
    assert dtype.make_slice(**kw) == make_slice(dtype, **kw)


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_make_slice_fully_indexes_to_a_scalar(dtype):
    data = np.zeros([2] * dtype.ndim)
    picks = {PY: 1, PX: 1, AX: 1, AS: 1, ND: 1, GR: 1, SF: 1}
    assert data[make_slice(dtype, **slice_kwargs(picks))].shape == ()


def test_make_slice_rejects_pin_pair_of_wrong_length():
    with pytest.raises(ValueError):
        make_slice(T.PIN, pin_idxs=(1,))


def test_make_slice_pin_pair_ignored_for_dtype_without_pins():
    assert make_slice(T.AXIAL, pin_idxs=(1,)) == (FULL,)


# ---------------------------------------------------------------- _Info validation


def test_info_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        T.PIN._info.axial_idx = 0


@pytest.mark.parametrize(
    "kwargs, error",
    [
        ({"axial_idx": 0, "fixed_idxs": {0: 0}}, ValueError),  # fixed collides
        ({"fixed_idxs": {1: 0}}, ValueError),  # not contiguous from 0
        ({"fixed_idxs": {-1: 0}}, ValueError),
        ({"fixed_idxs": {True: 0}}, TypeError),
        ({"fixed_idxs": {0: "0"}}, TypeError),
        ({"pin_idxs": (0, 1, 2)}, ValueError),
        ({"pin_idxs": (0, 0)}, ValueError),
        ({"pin_idxs": (1, 2)}, ValueError),
        ({"group_idx": 1}, ValueError),
        ({"surface_idx": 0, "group_idx": 0}, ValueError),
        ({"node_dim_idx": 0.0}, TypeError),
    ],
)
def test_info_rejects(kwargs, error):
    with pytest.raises(error):
        _Info(**kwargs)


def test_info_accepts_any_axis_order():
    info = _Info(assembly_id_idx=0, axial_idx=2, pin_idxs=(3, 1))
    assert info.ndim == 4


def test_info_ndim_is_not_an_init_argument():
    with pytest.raises(TypeError):
        _Info(ndim=3)


# ---------------------------------------------------------------- build_core_dtypes


# Realistic, pairwise-distinct sizes so no two layouts can collide.
CORE_SIZES = dict(npin=17, nax=49, nass=193, comp_nax=30, comp_nass=200, ndet=58, ndax=61)


def canonical_shape(dtype, n_groups):
    s = CORE_SIZES
    comp, det = dtype.is_computational(), dtype.is_detector()
    size = {
        PY: s["npin"] + dtype.is_channel(),
        PX: s["npin"] + dtype.is_channel(),
        AX: s["ndax"] if det else s["comp_nax"] if comp else s["nax"],
        AS: s["ndet"] if det else s["comp_nass"] if comp else s["nass"],
        ND: NUM_NODES,
        GR: n_groups,
        SF: NUM_DF,
        FIXED: 1,
    }
    return tuple(size[axis] for axis in LAYOUT[dtype])


@pytest.fixture(scope="module")
def realistic_map():
    s = CORE_SIZES
    return build_core_dtypes(
        s["npin"], s["npin"], s["nax"], s["nass"], s["comp_nax"], s["comp_nass"], s["ndet"], s["ndax"]
    )


MAPPED = [d for d in ALL if d not in (T.UNKNOWN, T.CONTINOUS_DETECTOR)]


@pytest.mark.parametrize("dtype", MAPPED, ids=str)
@pytest.mark.parametrize("n_groups", [MIN_NUM_GROUPS, 8, MAX_NUM_GROUPS])
def test_canonical_shape_maps_back_to_dtype(realistic_map, dtype, n_groups):
    assert realistic_map[canonical_shape(dtype, n_groups)] == dtype


def test_continuous_detector_replaces_point_detector():
    s = CORE_SIZES
    m = build_core_dtypes(nax=s["nax"], nass=s["nass"], ndet=s["ndet"], ndax=s["ndax"], continous_det=1)
    assert m[canonical_shape(T.CONTINOUS_DETECTOR, 0)] == T.CONTINOUS_DETECTOR


def test_unknown_is_never_produced(realistic_map):
    assert T.UNKNOWN not in realistic_map.values()


@pytest.mark.parametrize("n_groups", [MIN_NUM_GROUPS - 1, MAX_NUM_GROUPS + 1])
@pytest.mark.parametrize(
    "dtype", [T.ASSY_ENERGY, T.NODAL_ENERGY, T.ASSY_SURFACE, T.NODAL_SURFACE], ids=str
)
def test_group_counts_outside_range_are_unmapped(realistic_map, dtype, n_groups):
    assert canonical_shape(dtype, n_groups) not in realistic_map


def test_energy_dtypes_need_axial_and_assembly_counts():
    assert T.NODAL_ENERGY not in build_core_dtypes(nax=4).values()


def test_channel_overwrites_four_group_nodal_energy_when_npin_is_3():
    """Documents a hazard, not a contract. CHANNEL is (npin+1, npin+1, nax, nass)
    and NODAL_ENERGY is (n_groups, NUM_NODES, nax, nass); with npin = 3 they
    coincide for 4 groups and CHANNEL wins. The mock core uses npin = 3."""
    assert build_core_dtypes(3, 3, 10, 20)[(4, NUM_NODES, 10, 20)] == T.CHANNEL


def test_pin_overwrites_nodal_energy_when_npinx_equals_num_nodes():
    """Documents a hazard: any core with npinx == NUM_NODES maps
    (npiny, 4, nax, nass) to PIN, hiding NODAL_ENERGY with npiny groups."""
    assert build_core_dtypes(8, NUM_NODES, 10, 20)[(8, NUM_NODES, 10, 20)] == T.PIN


# ---------------------------------------------------------------- surfaces


def test_surface_count_matches_num_df():
    assert len(Surface) == NUM_DF


def test_surface_values_are_contiguous_axis_indices():
    assert [s.value for s in Surface] == list(range(NUM_DF))


def test_lateral_surfaces_are_the_four_sides():
    lateral = list(Surface)[LATERAL_SURFACES]
    assert lateral == [Surface.WEST, Surface.NORTH, Surface.EAST, Surface.SOUTH]


def test_surface_str():
    assert Surface.TOP.str == "TOP"


def test_lateral_slice_on_surface_axis():
    data = np.arange(NUM_DF * 3).reshape(NUM_DF, 3)
    assert data[LATERAL_SURFACES].shape == (4, 3)
    assert data[LATERAL_SURFACES][:, 0].tolist() == [0, 3, 6, 9]


# ---------------------------------------------------------------- group constants


def test_group_range_is_inclusive_and_ordered():
    assert 1 < MIN_NUM_GROUPS <= MAX_NUM_GROUPS
    assert len(range(MIN_NUM_GROUPS, MAX_NUM_GROUPS + 1)) == MAX_NUM_GROUPS - 1
