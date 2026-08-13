"""Tests for the color contract.

resolve_color_specs is the single place a default color range is decided, so
these cover every shape it accepts and the two fallback paths.
"""

import numpy as np
import pytest
from conftest import make_core_slice

from vera_core.data.analysis.color import (
    ColorSpec,
    array_range,
    default_color_spec,
    resolve_color_specs,
    shared_color_spec,
)


class TestArrayRange:
    def test_ignores_nan(self):
        assert array_range(np.array([1.0, np.nan, 3.0])) == (1.0, 3.0)

    def test_all_nan_falls_back(self):
        assert array_range(np.array([np.nan, np.nan])) == (0.0, 1.0)

    def test_flat_array_widens(self):
        lo, hi = array_range(np.array([2.0, 2.0]))
        assert lo == 2.0 and hi > lo, "a zero-width range makes an unusable colorbar"

    def test_flat_zeros_widen(self):
        lo, hi = array_range(np.zeros(4))
        assert hi > lo


class TestColorSpec:
    def test_normalize_clips_at_endpoints(self, spec):
        got = spec.normalize(np.array([-5.0, 0.0, 5.0, 20.0]))
        assert list(got) == [0.0, 0.0, 0.5, 1.0]

    def test_normalize_preserves_nan(self, spec):
        assert np.isnan(spec.normalize(np.array([np.nan]))[0])

    def test_with_range_leaves_original_alone(self, spec):
        assert spec.with_range(1.0, 2.0).range == (1.0, 2.0)
        assert spec.range == (0.0, 10.0)

    @pytest.mark.parametrize(
        "bad, reason",
        [
            (ColorSpec(5.0, 1.0), "vmax below vmin"),
            (ColorSpec(1.0, 1.0), "zero-width range"),
            (ColorSpec(np.nan, 1.0), "non-finite endpoint"),
            (ColorSpec(0.0, 1.0, cmap=""), "empty colormap name"),
        ],
    )
    def test_validate_rejects(self, bad, reason):
        assert bad.validate(), reason

    def test_validate_accepts_a_good_spec(self, spec):
        assert spec.validate() == []


class TestResolveColorSpecs:
    def test_none_gives_each_slice_its_own_range(self, two_groups):
        specs = resolve_color_specs(two_groups)
        assert specs[0].range != specs[1].range

    def test_share_range_gives_one_scale(self, two_groups):
        specs = resolve_color_specs(two_groups, share_range=True)
        assert specs[0].range == specs[1].range
        lows = [s.value_range[0] for s in two_groups]
        highs = [s.value_range[1] for s in two_groups]
        assert specs[0].range == (min(lows), max(highs))

    def test_single_spec_applies_to_all(self, two_groups, spec):
        assert [s.range for s in resolve_color_specs(two_groups, spec)] == [spec.range] * 2

    def test_mapping_keys_on_group_index(self, two_groups, spec):
        specs = resolve_color_specs(two_groups, {1: spec})
        assert specs[1].range == spec.range
        assert specs[0].range == two_groups[0].value_range

    def test_mapping_falls_back_to_position_when_ungrouped(self, spec):
        """core_slices() leaves group None on a single-group dataset, so a
        dict keyed by index must still reach it."""
        ungrouped = [make_core_slice()]
        assert ungrouped[0].group is None
        assert resolve_color_specs(ungrouped, {0: spec})[0].range == spec.range

    def test_sequence_keys_on_position(self, two_groups, spec):
        specs = resolve_color_specs(two_groups, [None, spec])
        assert specs[0].range == two_groups[0].value_range
        assert specs[1].range == spec.range

    def test_short_sequence_falls_back(self, two_groups, spec):
        specs = resolve_color_specs(two_groups, [spec])
        assert specs[1].range == two_groups[1].value_range

    def test_callable_receives_the_slice(self, two_groups, spec):
        specs = resolve_color_specs(two_groups, lambda s: spec if s.group else None)
        assert specs[0].range == two_groups[0].value_range
        assert specs[1].range == spec.range

    def test_override_beats_share_range(self, two_groups, spec):
        specs = resolve_color_specs(two_groups, {0: spec}, share_range=True)
        assert specs[0].range == spec.range
        assert specs[1].range != spec.range, "group 1 keeps the shared fallback"

    def test_invalid_spec_raises_naming_the_group(self, two_groups):
        with pytest.raises(ValueError, match="group 1"):
            resolve_color_specs(two_groups, {1: ColorSpec(5.0, 1.0)})

    def test_cmap_carries_through(self, two_groups):
        specs = resolve_color_specs(two_groups, cmap="viridis")
        assert all(s.cmap == "viridis" for s in specs)


def test_shared_color_spec_of_nothing_is_the_default():
    assert shared_color_spec([]).range == (0.0, 1.0)


def test_default_color_spec_widens_a_flat_range():
    lo, hi = default_color_spec((3.0, 3.0)).range
    assert hi > lo
