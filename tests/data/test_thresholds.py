import numpy as np
import pytest

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.thresholds import THRESHOLD_OPS, apply_thresholds, threshold_mask

VALUES = np.array([1.0, 2.0, 3.0])


@pytest.mark.parametrize(
    "op, expected",
    [
        (">", [False, False, True]),
        (">=", [False, True, True]),
        ("<", [True, False, False]),
        ("<=", [True, True, False]),
        ("==", [False, True, False]),
        ("!=", [True, False, True]),
    ],
)
def test_each_op(op, expected):
    np.testing.assert_array_equal(threshold_mask(VALUES, [{"op": op, "value": 2.0}]), expected)


def test_ops_table_matches_literal():
    assert set(THRESHOLD_OPS) == {">", ">=", "<", "<=", "==", "!="}


def test_conditions_combine_with_and():
    mask = threshold_mask(np.arange(6.0), [{"op": ">", "value": 1}, {"op": "<", "value": 4}])
    np.testing.assert_array_equal(mask, [False, False, True, True, False, False])


def test_no_conditions_keeps_everything():
    assert threshold_mask(VALUES, []).all()


def test_mask_preserves_shape():
    assert threshold_mask(np.ones((2, 3, 4)), [{"op": ">", "value": 0}]).shape == (2, 3, 4)


def test_nan_fails_every_ordering_condition():
    for op in (">", ">=", "<", "<=", "=="):
        assert not threshold_mask(np.array([np.nan]), [{"op": op, "value": 0}])[0]


def test_unknown_op_raises_key_error():
    with pytest.raises(KeyError):
        threshold_mask(VALUES, [{"op": "~", "value": 0}])


def test_apply_replaces_rejected_values_with_nan():
    out = apply_thresholds(VALUES, [{"op": ">=", "value": 2.0}])
    np.testing.assert_array_equal(out, [np.nan, 2.0, 3.0])


def test_apply_promotes_integer_input_to_float():
    out = apply_thresholds(np.arange(3), [{"op": ">", "value": 0}])
    assert out.dtype.kind == "f"
    np.testing.assert_array_equal(out, [np.nan, 1.0, 2.0])


def test_apply_does_not_mutate_input():
    data = VALUES.copy()
    apply_thresholds(data, [{"op": ">", "value": 10}])
    np.testing.assert_array_equal(data, VALUES)


def test_apply_keeps_dataset_type():
    ds = VeraDataset(VALUES, VeraDtype.AXIAL)
    assert apply_thresholds(ds, []).dataset_type == VeraDtype.AXIAL


def test_apply_on_plain_array_is_unknown():
    assert apply_thresholds(VALUES, []).dataset_type == VeraDtype.UNKNOWN


def test_apply_keeps_name_and_units():
    ds = VeraDataset(VALUES, VeraDtype.AXIAL, "axial_powers", "W")
    out = apply_thresholds(ds, [{"op": ">", "value": 1}])
    assert (out.name, out.physical_units) == ("axial_powers", "W")
