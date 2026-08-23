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
    dtype = array.dataset_type if isinstance(array, VeraDataset) else VeraDtype.UNKNOWN
    return VeraDataset(np.where(threshold_mask(array, conditions), array, np.nan), dtype)
