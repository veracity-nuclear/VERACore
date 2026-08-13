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


def apply_thresholds(
    array: VeraDataset | np.ndarray, conditions: Sequence[ThresholdCondition]
) -> VeraDataset:
    keep = np.ones(array.shape, dtype=bool)
    dtype = VeraDtype.UNKNOWN
    if isinstance(array, VeraDataset):
        dtype = array.dataset_type
    for c in conditions:
        keep &= THRESHOLD_OPS[c["op"]](array, c["value"])
    return VeraDataset(np.where(keep, array, np.nan), dtype)
