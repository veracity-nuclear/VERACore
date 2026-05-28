import numpy as np

from trame_server.core import State, Controller
from trame.widgets import html, vuetify
from vera_core.app.core import VeraDataRegistry

def format_label(file : str, key : str):
    return f"{file} | {key.replace('_', ' ').upper()}"

def array_range(array):
    lo = float(np.nanmin(array))
    hi = float(np.nanmax(array))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo == hi:
        eps = max(abs(hi) * 1e-9, 1e-12)
        return (lo, hi + eps)
    return (lo, hi)

def get_next_y_from_layout(layout):
    next_y = 0
    for item in layout:
        y, h = item.get("y", 0), item.get("h", 1)
        if y + h > next_y:
            next_y = y + h
    return next_y
