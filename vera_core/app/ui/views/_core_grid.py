import math
import numpy as np
from vera_core.app.core import VeraDataSource, VeraOutCore, VeraDataset

def nan_out_non_fuel_locs(array : np.ndarray, src : VeraDataSource, selected_layer : int, is_radial : bool):
    non_fuel_locs = src.core.non_fuel_locs
    if non_fuel_locs is None:
        return array
    rod_rows, rod_cols, layers, assy_id = non_fuel_locs
    keep = slice(None) if is_radial else (layers == selected_layer)
    idx = (assy_id[keep], rod_rows[keep], rod_cols[keep])
    new_array = array.copy()
    new_array[idx] = np.nan
    return new_array

def assembly_side(n: int) -> int:
    """Pin-side length for a cell of n values. n must be a perfect square."""
    if n <= 0:
        return 0
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"assembly cell length {n} is not a perfect square")
    return side

def format_for_vis(src : VeraDataSource, dataset : VeraDataset):
    cm = src.core.get_map(dataset)
    is_assembly_avg = dataset.is_assembly()
    core_width = cm.shape[0]
    result = []
    labels = []
    for i in range(core_width):
        line = [None] * core_width
        result.append(line)
        if is_assembly_avg:
            labels_line = [None] * core_width
            labels.append(labels_line)
        for j in range(core_width):
            index = cm[i, j] - 1
            if index == -1:
                continue
            if is_assembly_avg:
                line[j] = [float(dataset[index])]
                labels_line[j] = float(dataset[index])
            else:
                line[j] = np.ravel(dataset[index]).tolist()
    return result, labels    

def core_labels(core : VeraOutCore, is_comp: bool):
    """Return (x_labels, y_labels, max_core_cols) for the requested map."""
    x_labels = core.comp_core_map_column_labels if is_comp else core.reduced_core_map_column_labels
    y_labels = core.comp_core_map_row_labels if is_comp else core.reduced_core_map_row_labels
    max_core_cols = core.comp_core_map.shape[0] if is_comp else core.reduced_core_map.shape[0]
    return x_labels, y_labels, max_core_cols