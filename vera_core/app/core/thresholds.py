import operator
import numpy as np
from . import VeraDataset

THRESHOLD_OPS = {
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

def apply_thresholds(array : VeraDataset, conditions) -> VeraDataset:
    keep = np.ones(array.shape, dtype=bool)
    dtype = array.dataset_type
    for c in conditions:
        keep &= THRESHOLD_OPS[c["op"]](array, c["value"])
    return VeraDataset(np.where(keep, array, np.nan), dtype)