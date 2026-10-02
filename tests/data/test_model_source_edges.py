"""VeraDataSource behavior not covered by test_model_source.py.

Several tests document hazards (behavior that is surprising but current).
They say so in their docstrings; a deliberate fix should update them.
"""

import numpy as np
import pytest

from vera_core.data.dtypes import DerivationMethod, VeraAxes, VeraDtype
from vera_core.data.model import VeraDataSource

from .source_support import make_core, make_source, make_state

T = VeraDtype


def reads(state, name):
    return state.source.loads.get(name, 0)


# ---------------------------------------------------------------- empty sources


@pytest.mark.xfail(strict=True, reason="setter returns before _active_state_index is ever set")
def test_empty_source_has_an_active_index():
    src = VeraDataSource(core=make_core(), states=[], provenance="x")
    assert src.active_state_index == 0


def test_first_added_state_becomes_active():
    core = make_core()
    src = VeraDataSource(core=core, states=[], provenance="x")
    state = make_state(core)
    src.add_state(state)
    assert src.active_state is state


# ---------------------------------------------------------------- state_idx handling


@pytest.mark.parametrize("requested, expected", [(99, 20.0), (-5, 0.0)])
def test_out_of_range_state_idx_is_clamped(requested, expected):
    """The _get_dataset docstring promises IndexError and rejection of
    negatives; the code clamps first, so neither happens."""
    assert make_source().get_dataset("exposure", state_idx=requested)[0] == expected


def test_metadata_lookups_honour_state_idx():
    src = make_source()
    src.states[2].add_derived_dataset("only_last", src.states[2].get("keff"))
    assert src.get_dataset_dtype("only_last") == T.UNKNOWN
    assert src.get_dataset_dtype("only_last", state_idx=2) == T.SCALAR


def test_get_dataset_shape_for_present_name():
    assert make_source().get_dataset_shape("pin_powers") == (3, 3, 5, 9)


# ---------------------------------------------------------------- state mutation


def test_removing_state_before_active_changes_active_state():
    """Documents a hazard: the active index is not shifted, so the active
    state silently becomes the next one."""
    src = make_source()
    src.active_state_index = 1
    before = src.active_state
    src.remove_state(0)
    assert src.active_state_index == 1
    assert src.active_state is not before


def test_replacing_active_state_leaves_replacement_uncached():
    """Documents a hazard: replace_state bypasses the active-index setter, so
    the new active state reads lazily."""
    src = make_source()
    new = make_state(src.core, 99.0, idx=5)
    src.replace_state(src.active_state_index, new)
    src.active_state.get("keff")
    src.active_state.get("keff")
    assert reads(new, "keff") == 2


def test_state_without_time_dataset_contributes_none():
    src = make_source()
    del src.states[1].source.arrays["exposure"]
    rebuilt = VeraDataSource(core=src.core, states=src.states, provenance="x")
    assert rebuilt.time_axes()["exposure"] == [0.0, None, 20.0]


def test_time_axes_read_via_get_when_source_has_no_sampler():
    src = make_source()
    assert not hasattr(src.states[0].source, "sample")
    assert src.time_axes()["exposure"] == [0.0, 10.0, 20.0]


def test_time_axes_are_plain_python_floats():
    assert all(type(v) is float for v in make_source().time_axes()["exposure"])


# ---------------------------------------------------------------- diffs


def test_diff_does_not_modify_reference_data():
    src = make_source()
    before = src.states[0].get("keff").copy()
    src.add_new_diff_dataset("keff", make_source(), "keff", "dk", ref_scale=3.0)
    np.testing.assert_array_equal(src.states[0].get("keff"), before)


def test_diff_outside_comparison_mesh_is_nan():
    """Interpolation does not extrapolate: levels beyond the comparison
    mesh's range become NaN."""
    ref = make_source()  # means 10..90
    comp = make_source(core=make_core(nax=2))  # means 25, 75
    ref.add_new_diff_dataset("axial_powers", comp, "axial_powers", "d")
    d = ref.states[0].get("d")
    assert np.isnan(d[[0, -1]]).all()
    assert np.isfinite(d[1:-1]).all()


def test_diff_default_units_differ_from_dataset_default():
    """Documents an inconsistency: diffs default to 'unitless', datasets
    everywhere else default to 'Unitless'."""
    src = make_source()
    src.add_new_diff_dataset("keff", make_source(), "keff", "dk")
    assert src.states[0].get("dk").physical_units == "unitless"


def test_diff_with_mismatched_dtypes_finds_no_states():
    with pytest.raises(ValueError, match="No overlapping"):
        make_source().add_new_diff_dataset("keff", make_source(), "axial_powers", "d")


def test_diff_scales_both_operands():
    ref, comp = make_source(), make_source()
    ref.add_new_diff_dataset("keff", comp, "keff", "d", ref_scale=2.0, comp_scale=0.5)
    expected = 2.0 * ref.states[1].get("keff") - 0.5 * comp.states[1].get("keff")
    np.testing.assert_allclose(ref.states[1].get("d"), expected)


# ---------------------------------------------------------------- derivations


def derived(method, axes):
    src = make_source(1)
    src.add_new_derived_dataset("pin_powers", "der", method, axes)
    return src.states[0].get("pin_powers"), src.states[0].get("der")


def test_rms_over_axial():
    data, out = derived(DerivationMethod.RMS, VeraAxes.AXIAL)
    np.testing.assert_allclose(out, np.sqrt((data**2).mean(axis=(0, 1, 3))))


def test_stddev_over_axial_uses_whole_core_mean():
    """Documents a hazard: the mean subtracted is the CORE average, not the
    per-level mean, so this is not the per-level standard deviation."""
    data, out = derived(DerivationMethod.STDDEV, VeraAxes.AXIAL)
    np.testing.assert_allclose(out, np.sqrt(((data - data.mean()) ** 2).mean(axis=(0, 1, 3))))
    assert not np.allclose(out, data.std(axis=(0, 1, 3)))


def test_derived_dataset_does_not_inherit_units():
    """Documents a hazard: an average of W data is tagged Unitless."""
    data, out = derived(DerivationMethod.AVERAGE, VeraAxes.AXIAL)
    assert data.physical_units == "W"
    assert out.physical_units == "Unitless"


def test_derivation_skips_states_missing_the_source_array():
    src = make_source()
    del src.states[1].source.arrays["pin_powers"]
    src.add_new_derived_dataset("pin_powers", "der", DerivationMethod.AVERAGE, VeraAxes.CORE)
    assert ["der" in s for s in src.states] == [True, False, True]


def test_close_calls_callback_before_calculator():
    order = []
    src = make_source(close_callback=lambda: order.append("callback"))
    src.vera_calculator.close = lambda: order.append("calculator")
    src.close()
    assert order == ["callback", "calculator"]
