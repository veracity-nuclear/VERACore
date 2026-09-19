"""Slice builders on every dtype their DimSpec accepts."""

import numpy as np
import pytest

from vera_core.data.dtypes import VeraDim
from vera_core.data.analysis import (
    assembly_slice as new_as,
    assembly_surface_slice as new_ass,
    axial_lines as new_al,
    axial_slice as new_ax,
    core_slice as new_cs,
    surface_slice as new_ss,
)

from .slice_support import NAX, NPIN, SHAPES, SRC, FakeSource, T, equal


ALL_SPECS = {
    "core": (new_cs.SPEC, lambda t: new_cs.CoreSlice.create_core_slice(SRC, str(t), 1)),
    "assembly": (
        new_as.SPEC,
        lambda t: new_as.AssemblySlice.create_assembly_slice(SRC, str(t), 1, 4),
    ),
    "surface": (new_ss.SPEC, lambda t: new_ss.SurfaceSlice.create_surface_slice(SRC, str(t), 1)),
    "assembly_surface": (
        new_ass.SPEC,
        lambda t: new_ass.AssemblySurfaceSlice.create_assembly_surface_slice(SRC, str(t), 1, 4),
    ),
    "axial": (new_ax.SPEC, lambda t: new_ax.AxialSlice.create_axial_slice(SRC, str(t), 0, 4, "y")),
}
CASES = [(v, t) for v, (spec, _) in ALL_SPECS.items() for t in SHAPES if spec.supports(t)]


@pytest.mark.parametrize("view, dtype", CASES, ids=[f"{v}-{t}" for v, t in CASES])
def test_every_supported_dtype_builds(view, dtype):
    result = ALL_SPECS[view][1](dtype)
    assert result is not None and not isinstance(result, Exception)


def test_core_view_excludes_continuous_detector():
    assert not new_cs.SPEC.supports(T.CONTINOUS_DETECTOR)
    assert new_cs.SPEC.supports(T.POINT_DETECTOR)


@pytest.mark.parametrize("dtype", [T.NODAL, T.PIN, T.ASSEMBLY], ids=str)
@pytest.mark.parametrize("dim", ["x", "y"])
def test_out_of_range_selection_clips_to_last_row_or_column(dtype, dim):
    """Changed: the original sent a y-cut node index of 2 to column 0 and
    raised IndexError for an out-of-range pin. Both now clip to the edge."""
    build = new_ax.AxialSlice.create_axial_slice
    edge = {T.NODAL: 1, T.PIN: NPIN - 1, T.ASSEMBLY: 0}[dtype]
    assert equal(build(SRC, str(dtype), 99, 4, dim), build(SRC, str(dtype), edge, 4, dim))


# ---------------------------------------------------------------- bug fixes (pass 2)


def test_axial_column_returns_one_cell_bottom_first():
    s = new_ax.AxialSlice.create_axial_slice(SRC, "PIN", 1, 4, "x")
    np.testing.assert_array_equal(s.column(1), s.to_grid()[:, NPIN : 2 * NPIN])
    assert s.column(1).shape == (NAX, NPIN)


def test_axial_lines_records_requested_state_zero():
    SRC_STATES = FakeSource()
    SRC_STATES.states = [{"exposure": np.array([1.0])}] * 3
    SRC_STATES.active_state_index = 2
    lines = new_al.AxialLines.create_axial_lines(SRC_STATES, "AXIAL", {VeraDim.AXIAL: 0}, state=0)
    assert lines.states == [0]


def test_core_slice_negative_z_message():
    with pytest.raises(RuntimeError, match="z must be >= 0"):
        new_cs.CoreSlice.create_core_slice(SRC, "PIN", -1)
