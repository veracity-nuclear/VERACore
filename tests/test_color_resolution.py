"""The spec/style split: a style names the colormap, a spec names the range.
These fail unless ColorSpec.cmap defaults to None and resolve_color_specs
fills it in."""

import pytest
from conftest import StubSlice

from vera_core.data.analysis.color import (
    ColorScope,
    ColorSpec,
    array_range,
    resolve_color_specs,
    shared_group_specs,
    union_range,
)


def test_a_range_override_keeps_the_style_colormap():
    """Before the change, ColorSpec(vmin=..., vmax=...) silently reverted the
    colormap to jet, so a range and a colormap could not be set together."""
    specs = resolve_color_specs(
        StubSlice(2), ColorSpec(vmin=0.0, vmax=9.0), scope=ColorScope.SLICE_ALL, cmap="magma"
    )
    assert [spec.cmap for spec in specs] == ["magma", "magma"]
    assert [spec.range for spec in specs] == [(0.0, 9.0), (0.0, 9.0)]


def test_a_spec_that_names_a_colormap_keeps_it():
    specs = resolve_color_specs(
        StubSlice(1), ColorSpec(vmin=0.0, vmax=1.0, cmap="plasma"), cmap="magma"
    )
    assert specs[0].cmap == "plasma"


def test_the_fallback_uses_the_style_colormap():
    specs = resolve_color_specs(StubSlice(2), None, cmap="magma")
    assert [spec.cmap for spec in specs] == ["magma", "magma"]


def test_per_group_overrides_fall_back_where_they_are_missing():
    override = {1: ColorSpec(vmin=0.0, vmax=2.0, cmap="plasma")}
    specs = resolve_color_specs(StubSlice(3), override, cmap="magma")
    assert [spec.cmap for spec in specs] == ["magma", "plasma", "magma"]


def test_an_invalid_range_is_caught_by_group():
    with pytest.raises(ValueError, match="group 0"):
        resolve_color_specs(StubSlice(1), ColorSpec(vmin=1.0, vmax=0.0, cmap="jet"))


def test_shared_group_specs_spans_every_slice_per_group():
    specs = shared_group_specs([StubSlice(2, base=0.0), StubSlice(2, base=5.0)], cmap="magma")
    assert specs[0].range == (0.0, 6.0)
    assert specs[1].range == (1.0, 7.0)
    assert all(spec.cmap == "magma" for spec in specs)


def test_shared_group_specs_rejects_a_group_count_mismatch():
    with pytest.raises(ValueError):
        shared_group_specs([StubSlice(2), StubSlice(3)])


def test_shared_group_specs_of_nothing_is_empty():
    assert shared_group_specs([]) == []


def test_array_range_widens_a_flat_array():
    lo, hi = array_range([2.0, 2.0])
    assert hi > lo


def test_array_range_of_all_nan_falls_back():
    assert array_range([float("nan")]) == (0.0, 1.0)


def test_union_range_of_nothing():
    assert union_range([]) == (0.0, 1.0)
