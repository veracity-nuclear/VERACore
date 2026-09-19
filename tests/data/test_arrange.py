"""VeraDataset.arrange across every dtype.

Expected results come from `expected()`, which builds fancy-index grids from
the semantic layout. It shares no code path with arrange (select, then
transpose, then split), so agreement between the two is meaningful.
dim_axes itself is pinned by test_dtype_semantics.py.

Sample data is arange(), so every element is unique and any axis mix-up
changes values. Dims that share a dtype have distinct sizes, and fixed axes
have size 2 so choosing the wrong fixed index is detectable.
"""

import itertools
import math

import numpy as np
import pytest

from vera_core.data.dtypes import (
    DerivationMethod,
    VeraAxes,
    VeraDataset,
    VeraDim,
    VeraDtype,
    build_core_dtypes,
)
from vera_core.data.readers.mock import build_source

from .conftest import make_source

T = VeraDtype
PY, PX, AX, AS, ND, GR, SF = (
    VeraDim.PIN_Y,
    VeraDim.PIN_X,
    VeraDim.AXIAL,
    VeraDim.ASSEMBLY,
    VeraDim.NODE,
    VeraDim.GROUP,
    VeraDim.SURFACE,
)
# Distinct within every dtype (checked by test_sizes_distinct_within_each_dtype).
SIZES = {PY: 2, PX: 3, AX: 4, AS: 5, ND: 2, GR: 3, SF: 6}
FIXED_SIZE = 2
KWARG = {SF: "surface", GR: "group", ND: "node", AX: "axial", AS: "assembly"}
ALL = list(VeraDtype)
WITH_DIMS = [d for d in ALL if d.dim_axes]


# ---------------------------------------------------------------- helpers


def sample(dtype):
    axes = dtype.dim_axes
    by_axis = {axis: dim for dim, axis in axes.items()}
    shape = tuple(SIZES[by_axis[i]] if i in by_axis else FIXED_SIZE for i in range(dtype.ndim))
    return VeraDataset(np.arange(math.prod(shape)).reshape(shape), dtype, "sample", "W")


def dims_of(dtype):
    """Semantic dims in physical order."""
    return tuple(sorted(dtype.dim_axes, key=dtype.dim_axes.get))


def units_of(dtype):
    """Selectable units: the pin pair counts as one, since pins are picked together."""
    dims = dims_of(dtype)
    return ((PY, PX),) * (PY in dims) + tuple((d,) for d in dims if d not in (PY, PX))


def to_kwargs(picks):
    kw = {KWARG[d]: v for d, v in picks.items() if d in KWARG}
    if PY in picks:
        kw["pin"] = (picks[PY], picks[PX])
    return kw


def expected(ds, order, split=(), picks=None):
    """Reference arrange via fancy indexing. Ignores dims absent from the
    dtype or already picked, as arrange documents."""
    picks = picks or {}
    axes = ds.dataset_type.dim_axes
    chosen = {d: v for d, v in picks.items() if d in axes}
    live_order = [d for d in order if d in axes and d not in chosen]
    live_split = [d for d in split if d in axes and d not in chosen]
    raw = np.asarray(ds)
    grids = np.indices([raw.shape[axes[d]] for d in live_order])
    out = []
    for combo in itertools.product(*(range(raw.shape[axes[d]]) for d in live_split)):
        index = [0] * raw.ndim  # fixed axes stay at 0
        for d, v in {**chosen, **dict(zip(live_split, combo))}.items():
            index[axes[d]] = v
        for k, d in enumerate(live_order):
            index[axes[d]] = grids[k]
        out.append(raw[tuple(index)])
    return out


def check(ds, order, split=(), picks=None):
    picks = picks or {}
    got = ds.arrange(order=order, split=split, **to_kwargs(picks))
    want = expected(ds, order, split, picks)
    assert len(got) == len(want)
    assert all(isinstance(g, VeraDataset) for g in got)
    assert {(g.dataset_type, g.name, g.physical_units) for g in got} == {
        (ds.dataset_type, ds.name, ds.physical_units)
    }
    assert {np.shape(g) for g in got} == {np.shape(w) for w in want}
    np.testing.assert_array_equal(np.stack(got), np.stack(want))
    return got


def partition(dtype, rng):
    """Randomly assign every unit to select, split or order."""
    picks, split, order = {}, [], []
    for unit in units_of(dtype):
        role = rng.integers(3)
        if role == 0:
            for d in unit:
                size = SIZES[d]
                picks[d] = int(rng.integers(-size, size))
        else:
            (split if role == 1 else order).extend(unit)
    rng.shuffle(split)
    rng.shuffle(order)
    return picks, tuple(split), tuple(order)


# ---------------------------------------------------------------- the oracle


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_sizes_distinct_within_each_dtype(dtype):
    sizes = [SIZES[d] for d in dims_of(dtype)]
    assert len(set(sizes)) == len(sizes)


def test_oracle_matches_hand_indexing():
    ds = sample(T.PIN)
    [got] = expected(ds, (AS, PX), picks={AX: 2, PY: 1})
    np.testing.assert_array_equal(got, np.asarray(ds)[1, :, 2, :].T)


def test_oracle_uses_fixed_index_zero():
    ds = sample(T.ASSEMBLY)
    [got] = expected(ds, (AX, AS))
    np.testing.assert_array_equal(got, np.asarray(ds)[0])


# ---------------------------------------------------------------- sweeps over every dtype


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_every_order_permutation(dtype):
    ds = sample(dtype)
    for order in itertools.permutations(dims_of(dtype)):
        check(ds, order)


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_every_split_choice(dtype):
    ds = sample(dtype)
    dims = dims_of(dtype)
    for k in range(len(dims) + 1):
        for split in itertools.permutations(dims, k):
            order = tuple(d for d in reversed(dims) if d not in split)
            check(ds, order, split)


@pytest.mark.parametrize("dtype", ALL, ids=str)
@pytest.mark.parametrize("pick", ["first", "last", "negative"])
def test_every_selection_subset(dtype, pick):
    ds = sample(dtype)
    units = units_of(dtype)
    index = {"first": lambda d: 0, "last": lambda d: SIZES[d] - 1, "negative": lambda d: -1}[pick]
    for k in range(len(units) + 1):
        for chosen in itertools.combinations(units, k):
            picks = {d: index(d) for unit in chosen for d in unit}
            order = tuple(d for d in reversed(dims_of(dtype)) if d not in picks)
            check(ds, order, picks=picks)


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_random_mixed_scenarios(dtype):
    ds = sample(dtype)
    rng = np.random.default_rng(dtype.value)
    for _ in range(60):
        picks, split, order = partition(dtype, rng)
        check(ds, order, split, picks)


@pytest.mark.parametrize("dtype", WITH_DIMS, ids=str)
def test_output_count_and_shape(dtype):
    ds = sample(dtype)
    rng = np.random.default_rng(1000 + dtype.value)
    for _ in range(20):
        picks, split, order = partition(dtype, rng)
        out = ds.arrange(order=order, split=split, **to_kwargs(picks))
        assert len(out) == math.prod(SIZES[d] for d in split)
        assert all(o.shape == tuple(SIZES[d] for d in order) for o in out)


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_outputs_are_datasets_with_source_metadata(dtype):
    ds = sample(dtype)
    split = dims_of(dtype)[:1]
    for out in ds.arrange(order=dims_of(dtype)[1:], split=split):
        assert isinstance(out, VeraDataset)
        assert (out.dataset_type, out.name, out.physical_units) == (dtype, "sample", "W")


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_physical_order_is_identity(dtype):
    ds = sample(dtype)
    [out] = ds.arrange(order=dims_of(dtype))
    np.testing.assert_array_equal(out, ds.select())


@pytest.mark.parametrize("dtype", WITH_DIMS, ids=str)
def test_full_selection_gives_one_scalar(dtype):
    ds = sample(dtype)
    picks = {d: 1 for d in dims_of(dtype)}
    [out] = check(ds, (), picks=picks)
    assert out.shape == ()


@pytest.mark.parametrize("dtype", WITH_DIMS, ids=str)
def test_split_everything_gives_one_scalar_per_element(dtype):
    ds = sample(dtype)
    out = check(ds, (), dims_of(dtype))
    assert len(out) == ds.select().size
    assert all(o.shape == () for o in out)


@pytest.mark.parametrize("dtype", WITH_DIMS, ids=str)
def test_split_outputs_concatenate_back_to_ordered_array(dtype):
    ds = sample(dtype)
    dims = dims_of(dtype)
    split, order = dims[:1], dims[1:]
    [whole] = ds.arrange(order=split + order)
    np.testing.assert_array_equal(np.stack(ds.arrange(order=order, split=split)), whole)


# ---------------------------------------------------------------- realistic use cases


def test_pin_map_for_one_assembly_level():
    ds = sample(T.PIN)
    [out] = ds.arrange(order=(PY, PX), axial=2, assembly=3)
    np.testing.assert_array_equal(out, np.asarray(ds)[:, :, 2, 3])


def test_pin_maps_stacked_by_level():
    ds = sample(T.PIN)
    maps = ds.arrange(order=(PY, PX), split=(AX,), assembly=0)
    assert len(maps) == SIZES[AX]
    for z, m in enumerate(maps):
        np.testing.assert_array_equal(m, np.asarray(ds)[:, :, z, 0])


def test_axial_profile_of_one_pin():
    ds = sample(T.PIN)
    [out] = ds.arrange(order=(AX,), pin=(1, 2), assembly=4)
    np.testing.assert_array_equal(out, np.asarray(ds)[1, 2, :, 4])


def test_assembly_radial_map_at_level():
    ds = sample(T.ASSEMBLY)
    [out] = ds.arrange(order=(AS,), axial=1)
    np.testing.assert_array_equal(out, np.asarray(ds)[0, 1, :])


def test_assembly_axial_profiles_as_columns():
    ds = sample(T.ASSEMBLY)
    [out] = ds.arrange(order=(AX, AS))
    assert out.shape == (SIZES[AX], SIZES[AS])


def test_per_group_assembly_maps():
    ds = sample(T.ASSY_ENERGY)
    maps = ds.arrange(order=(AS,), split=(GR,), axial=2)
    assert len(maps) == SIZES[GR]
    for g, m in enumerate(maps):
        np.testing.assert_array_equal(m, np.asarray(ds)[g, 0, 2, :])


def test_group_spectrum_at_one_node():
    ds = sample(T.COMP_NODAL_ENERGY)
    [out] = ds.arrange(order=(GR,), node=1, axial=1, assembly=4)
    np.testing.assert_array_equal(out, np.asarray(ds)[:, 1, 1, 4])


def test_surface_by_group_table_at_one_node():
    ds = sample(T.NODAL_SURFACE)
    [out] = ds.arrange(order=(GR, SF), node=0, axial=0, assembly=0)
    np.testing.assert_array_equal(out, np.asarray(ds)[:, :, 0, 0, 0].T)


def test_assembly_surface_split_by_surface():
    ds = sample(T.COMP_ASSY_SURFACE)
    faces = ds.arrange(order=(GR, AS), split=(SF,), axial=3)
    assert len(faces) == SIZES[SF]
    np.testing.assert_array_equal(faces[5], np.asarray(ds)[5, :, 0, 3, :])


def test_detector_traces_split_by_detector():
    ds = sample(T.POINT_DETECTOR)
    traces = ds.arrange(order=(AX,), split=(AS,))
    assert len(traces) == SIZES[AS]
    np.testing.assert_array_equal(traces[2], np.asarray(ds)[:, 2])


def test_nodal_values_transposed_node_last():
    ds = sample(T.NODAL)
    [out] = ds.arrange(order=(AS, AX, ND))
    np.testing.assert_array_equal(out, np.asarray(ds).transpose(2, 1, 0))


def test_radial_node_split_by_node():
    ds = sample(T.RADIAL_NODE)
    parts = ds.arrange(order=(AS,), split=(ND,))
    np.testing.assert_array_equal(np.stack(parts), np.asarray(ds))


def test_channel_uses_same_semantics_as_pin():
    pin, chan = sample(T.PIN), sample(T.CHANNEL)
    kw = dict(order=(PX, PY), axial=1, assembly=2)
    np.testing.assert_array_equal(pin.arrange(**kw)[0], chan.arrange(**kw)[0])


@pytest.mark.parametrize("dtype", [T.SCALAR, T.UNKNOWN], ids=str)
def test_zero_dimensional_dtypes_pass_through(dtype):
    ds = VeraDataset(np.float64(3.0), dtype)
    [out] = ds.arrange(order=(AX, AS), split=(GR,), axial=0, pin=(0, 0))
    assert out == 3.0


def test_loaded_scalar_shape_passes_through():
    """Regression: loaders store scalars as shape (1,), which SCALAR's
    zero-axis layout does not describe; arrange used to fail in transpose."""
    ds = VeraDataset(np.array([1.02]), T.SCALAR, "keff")
    [out] = ds.arrange(order=())
    assert np.ravel(out).tolist() == [1.02]


def test_unknown_with_axes_is_rejected():
    """UNKNOWN has no semantic dims, so arrange cannot place real axes.
    It fails in transpose with numpy's message, not a VeraDataset one."""
    with pytest.raises(ValueError, match="axes don't match array"):
        VeraDataset(np.zeros((2, 3)), T.UNKNOWN).arrange(order=())


@pytest.mark.parametrize(
    "dtype, kwargs",
    [
        (T.AXIAL, {"order": (), "split": (AX,)}),
        (T.PIN, {"order": (), "pin": (0, 0), "axial": 0, "assembly": 0}),
        (T.NODAL, {"order": (), "split": (ND,), "axial": 0, "assembly": 0}),
    ],
    ids=["split_1d", "select_all", "select_and_split"],
)
def test_fully_indexed_outputs_are_datasets(dtype, kwargs):
    """Regression: fully indexed outputs used to be numpy scalars."""
    out = sample(dtype).arrange(**kwargs)
    assert all(isinstance(o, VeraDataset) and o.physical_units == "W" for o in out)


# ---------------------------------------------------------------- split order


def test_split_is_row_major_in_given_order():
    ds = sample(T.PIN)
    out = ds.arrange(order=(PY, PX), split=(AX, AS))
    raw = np.asarray(ds)
    for k, (z, a) in enumerate(itertools.product(range(SIZES[AX]), range(SIZES[AS]))):
        np.testing.assert_array_equal(out[k], raw[:, :, z, a])


def test_reversing_split_order_changes_sequence():
    ds = sample(T.PIN)
    za = ds.arrange(order=(PY, PX), split=(AX, AS))
    az = ds.arrange(order=(PY, PX), split=(AS, AX))
    np.testing.assert_array_equal(za[1], az[SIZES[AX]])  # (z=0, a=1) in both
    assert not np.array_equal(za[1], az[1])


def test_split_on_selected_dim_is_ignored():
    ds = sample(T.PIN)
    out = ds.arrange(order=(PY, PX, AS), split=(AX,), axial=1)
    assert len(out) == 1


# ---------------------------------------------------------------- ignoring rules


def test_order_dims_absent_from_dtype_ignored():
    ds = sample(T.AXIAL)
    [out] = ds.arrange(order=(GR, AX, ND, SF))
    np.testing.assert_array_equal(out, ds)


def test_order_dims_already_selected_ignored():
    ds = sample(T.PIN)
    [out] = ds.arrange(order=(AX, PY, PX, AS), axial=2, assembly=1)
    np.testing.assert_array_equal(out, np.asarray(ds)[:, :, 2, 1])


def test_split_dims_absent_from_dtype_ignored():
    ds = sample(T.RADIAL_ASSEMBLY)
    [out] = ds.arrange(order=(AS,), split=(GR, AX))
    np.testing.assert_array_equal(out, ds)


def test_selectors_for_absent_dims_ignored():
    ds = sample(T.AXIAL)
    [out] = ds.arrange(order=(AX,), group=0, node=0, surface=0, assembly=0, pin=(0, 0))
    np.testing.assert_array_equal(out, ds)


def test_order_accepts_any_sequence():
    ds = sample(T.NODAL)
    as_list = ds.arrange(order=[AS, ND, AX])[0]
    as_tuple = ds.arrange(order=(AS, ND, AX))[0]
    np.testing.assert_array_equal(as_list, as_tuple)


def test_require_satisfied_by_selected_dim():
    ds = sample(T.PIN)
    ds.arrange(order=(PY, PX, AS), require=(AX,), axial=0)


def test_require_does_not_change_result():
    ds = sample(T.NODAL)
    a = ds.arrange(order=(AS, AX, ND))[0]
    b = ds.arrange(order=(AS, AX, ND), require=(ND, AX, AS))[0]
    np.testing.assert_array_equal(a, b)


def test_negative_indices_select_from_end():
    ds = sample(T.PIN)
    [out] = ds.arrange(order=(PY, PX), axial=-1, assembly=-2)
    np.testing.assert_array_equal(out, np.asarray(ds)[:, :, -1, -2])


# ---------------------------------------------------------------- validation


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"order": (AX, AX)}, "Duplicate dimensions in order"),
        ({"order": (), "split": (AS, AS)}, "Duplicate dimensions in split"),
        ({"order": (), "require": (ND, ND)}, "Duplicate dimensions in require"),
        ({"order": (AX,), "split": (AX,)}, "both order and split: axial"),
    ],
)
def test_structural_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sample(T.PIN).arrange(**kwargs)


def test_duplicates_rejected_even_for_absent_dims():
    with pytest.raises(ValueError, match="Duplicate"):
        sample(T.AXIAL).arrange(order=(AX, GR, GR))


def test_overlap_rejected_even_for_absent_dims():
    with pytest.raises(ValueError, match="both order and split: group"):
        sample(T.AXIAL).arrange(order=(AX, GR), split=(GR,))


def test_overlap_message_lists_all_sorted():
    with pytest.raises(ValueError, match="both order and split: assembly, axial"):
        sample(T.PIN).arrange(order=(AX, AS), split=(AS, AX))


def test_missing_required_lists_all_sorted():
    with pytest.raises(ValueError, match="ASSEMBLY is missing required dimensions: group, node"):
        sample(T.ASSEMBLY).arrange(order=(AX, AS), require=(ND, GR, AX))


def test_unhandled_lists_all_sorted():
    with pytest.raises(ValueError, match="leaves dimensions unhandled: assembly, pin_x, pin_y"):
        sample(T.PIN).arrange(order=(AX,))


def test_unhandled_after_partial_selection():
    with pytest.raises(ValueError, match="unhandled: axial$"):
        sample(T.PIN).arrange(order=(PY, PX), assembly=0)


def test_empty_order_on_dtype_with_dims_is_unhandled():
    with pytest.raises(ValueError, match="unhandled"):
        sample(T.AXIAL).arrange(order=())


def test_order_is_required_keyword():
    with pytest.raises(TypeError):
        sample(T.AXIAL).arrange()


def test_arguments_are_keyword_only():
    with pytest.raises(TypeError):
        sample(T.AXIAL).arrange((AX,))


def test_validation_precedes_indexing():
    with pytest.raises(ValueError, match="Duplicate"):
        sample(T.PIN).arrange(order=(AX, AX), axial=99)


def test_out_of_range_index_raises_index_error():
    with pytest.raises(IndexError):
        sample(T.PIN).arrange(order=(PY, PX, AS), axial=SIZES[AX])


def test_half_pin_selection_rejected():
    with pytest.raises(ValueError, match="together"):
        sample(T.PIN).arrange(order=(PX, AX, AS), pin=(0, None))


# ---------------------------------------------------------------- hazards


def test_output_keeps_source_dtype_even_when_layout_changes():
    """Documents a hazard: arrange returns datasets still tagged with the
    source dtype, whose axes no longer match it. Arranging the output again
    reads the axes by the stale tag without error."""
    ds = sample(T.PIN)
    [out] = ds.arrange(order=(AS, AX, PY, PX))
    assert out.dataset_type == T.PIN
    assert out.shape == (SIZES[AS], SIZES[AX], SIZES[PY], SIZES[PX])
    [again] = out.arrange(order=(PY, PX, AX, AS))  # "PY" is really assembly here
    np.testing.assert_array_equal(again, out)


def test_arrange_does_not_modify_source():
    ds = sample(T.NODAL_SURFACE)
    before = np.asarray(ds).copy()
    ds.arrange(order=(AS, AX, ND, GR, SF))
    ds.arrange(order=(AS,), split=(SF, GR, ND, AX))
    np.testing.assert_array_equal(ds, before)


# ---------------------------------------------------------------- shapes the readers produce


# Small core, sizes distinct so no two layouts collide.
CORE = dict(npiny=5, npinx=5, nax=7, nass=11, comp_nax=9, comp_nass=13, ndet=15, ndax=17)


def mapped_shapes():
    by_dtype = {}
    for shape, dtype in build_core_dtypes(**CORE).items():
        by_dtype.setdefault(dtype, []).append(shape)
    return by_dtype


MAPPED = mapped_shapes()


def arranged_shape(dtype, shape):
    """The shape arrange should return in physical order: fixed axes dropped.
    A zero-axis dtype keeps its stored shape, () or (1,)."""
    if not dtype.dim_axes:
        return shape
    live = set(dtype.dim_axes.values())
    return tuple(n for axis, n in enumerate(shape) if axis in live)


@pytest.mark.parametrize("dtype", sorted(MAPPED, key=str), ids=str)
def test_every_mapped_shape_is_arrangeable(dtype):
    """Every shape build_core_dtypes assigns to a dtype must be a valid
    input for that dtype's layout."""
    for shape in MAPPED[dtype]:
        ds = VeraDataset(np.broadcast_to(np.float64(1.0), shape), dtype)
        [out] = ds.arrange(order=dims_of(dtype))
        assert out.shape == arranged_shape(dtype, shape), shape


def test_mapped_shapes_cover_every_dtype_but_unknown_and_continuous():
    assert set(MAPPED) == set(ALL) - {T.UNKNOWN, T.CONTINOUS_DETECTOR}


@pytest.fixture(scope="module")
def loaded_state():
    return build_source(1).active_state


def test_every_loaded_dataset_is_arrangeable(loaded_state):
    names = loaded_state.full_core_keys
    assert names  # guards against a silently empty loop
    for name in names:
        ds = loaded_state.get(name)
        [out] = ds.arrange(order=dims_of(ds.dataset_type))
        np.testing.assert_array_equal(out, ds.select(), err_msg=name)
        assert out.name == name


def test_loaded_scalar_arranges_to_its_value(loaded_state):
    [out] = loaded_state.get("keff").arrange(order=())
    assert np.ravel(out).tolist() == [1.0]


@pytest.mark.parametrize(
    "axes",
    [a for a in VeraAxes if a is not VeraAxes.RADIAL_NODE],
    ids=lambda a: a.name,
)
def test_every_derived_dataset_is_arrangeable(axes):
    src = make_source(1)
    src.add_new_derived_dataset("pin_powers", "der", DerivationMethod.AVERAGE, axes)
    ds = src.active_state.get("der")
    [out] = ds.arrange(order=dims_of(ds.dataset_type))
    np.testing.assert_array_equal(out, ds.select())


# ---------------------------------------------------------------- data that does not match its tag


@pytest.mark.parametrize(
    "dtype, shape",
    [
        (T.PIN, (2, 3, 4)),  # one axis short
        (T.PIN, (2, 3, 4, 5, 6)),  # one axis extra
        (T.AXIAL, (4, 5)),
        (T.RADIAL_ASSEMBLY, (4, 5)),
        (T.RADIAL_NODE, (4,)),
        (T.RADIAL_NODE, (4, 7, 11)),  # NODAL data tagged RADIAL_NODE
        (T.ASSY_ENERGY, (3, 4, 5)),  # fixed axis missing
        (T.UNKNOWN, (2, 3)),
    ],
    ids=lambda v: str(v),
)
def test_mistagged_data_is_never_silently_arranged(dtype, shape):
    """Wrong-rank data must fail, not return a wrongly labelled array. The
    errors are numpy's (IndexError or ValueError), not a VeraDataset one."""
    ds = VeraDataset(np.zeros(shape), dtype)
    with pytest.raises((IndexError, ValueError)):
        ds.arrange(order=dims_of(dtype))


ZERO_AXIS = [T.SCALAR, T.UNKNOWN]


@pytest.mark.parametrize("dtype", ZERO_AXIS, ids=str)
@pytest.mark.parametrize("shape", [(), (1,)], ids=str)
def test_zero_axis_dtype_accepts_single_value(dtype, shape):
    ds = VeraDataset(np.full(shape, 2.5), dtype)
    [out] = ds.arrange(order=())
    assert out.shape == shape
    assert np.ravel(out).tolist() == [2.5]


@pytest.mark.parametrize("dtype", ZERO_AXIS, ids=str)
@pytest.mark.parametrize("shape", [(3,), (1, 1), (2, 5), (2, 3, 4)], ids=str)
def test_zero_axis_dtype_rejects_anything_else(dtype, shape):
    """Only () and (1,) describe a single value. A 1-D array of length > 1,
    or any higher-rank array, must not pass through unchecked."""
    with pytest.raises(ValueError):
        VeraDataset(np.zeros(shape), dtype).arrange(order=())


# ---------------------------------------------------------------- select returns datasets


@pytest.mark.parametrize("dtype", ALL, ids=str)
def test_full_select_returns_zero_d_dataset(dtype):
    ds = sample(dtype)
    out = ds.select({d: 1 for d in dims_of(dtype)})
    assert isinstance(out, VeraDataset)
    assert out.shape == ()
    assert (out.dataset_type, out.name, out.physical_units) == (dtype, "sample", "W")
