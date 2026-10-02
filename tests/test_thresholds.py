import numpy as np

from vera_core.app.core import VeraDataset, VeraDtype
from vera_core.app.core.thresholds import apply_thresholds


def test_apply_thresholds_keeps_values_matching_all_conditions(pin_dataset):
    result = apply_thresholds(
        pin_dataset,
        [
            {"op": ">=", "value": 2.0},
            {"op": "<", "value": 4.0},
        ],
    )

    np.testing.assert_array_equal(
        np.isnan(result),
        np.array([[True, False], [False, True]]),
    )
    np.testing.assert_allclose(result[~np.isnan(result)], [2.0, 3.0])
    assert isinstance(result, VeraDataset)
    assert result.dataset_type is VeraDtype.PIN


def test_apply_thresholds_with_no_conditions_returns_equal_copy(pin_dataset):
    result = apply_thresholds(pin_dataset, [])

    np.testing.assert_array_equal(result, pin_dataset)
    assert result is not pin_dataset
    assert result.dataset_type is pin_dataset.dataset_type
