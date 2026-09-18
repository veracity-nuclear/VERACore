import operator
from typing import Literal, Sequence, TypedDict

import numpy as np

from .dtypes import VeraDataset, VeraDtype

THRESHOLD_OPS = {
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

ThresholdOp = Literal[">", ">=", "<", "<=", "==", "!="]


class ThresholdCondition(TypedDict):
    op: ThresholdOp
    value: float


def threshold_mask(
    array: VeraDataset | np.ndarray, conditions: Sequence[ThresholdCondition]
) -> np.ndarray:
    keep = np.ones(array.shape, dtype=bool)
    for c in conditions:
        keep &= THRESHOLD_OPS[c["op"]](array, c["value"])
    return keep


def apply_thresholds(
    array: VeraDataset | np.ndarray, conditions: Sequence[ThresholdCondition]
) -> VeraDataset:
    is_vera_dataset = isinstance(array, VeraDataset)
    dtype = array.dataset_type if is_vera_dataset else VeraDtype.UNKNOWN
    units = array.physical_units if is_vera_dataset else "Unitless"
    name = array.name if is_vera_dataset else None
    return VeraDataset(
        np.where(threshold_mask(array, conditions), array, np.nan),
        dataset_type=dtype,
        name=name,
        physical_units=units,
    )
