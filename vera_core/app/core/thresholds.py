import operator
import numpy as np

THRESHOLD_OPS = {
    ">":  operator.gt,
    ">=": operator.ge,
    "<":  operator.lt,
    "<=": operator.le,
    "==": operator.eq,
    "!=": operator.ne,
}

def apply_thresholds(array, conditions):
    keep = np.ones(array.shape, dtype=bool)
    for c in conditions:
        keep &= THRESHOLD_OPS[c["op"]](array, c["value"])
    return np.where(keep, array, np.nan)