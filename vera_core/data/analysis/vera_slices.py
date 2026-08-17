import math
from dataclasses import dataclass, replace
from typing import Self

import numpy as np

from ..dtypes import NUM_NODES, VeraDtype
from ..model import VeraOutCore
from .color import ColorScope, array_range, union_range


@dataclass(frozen=True, kw_only=True)
class GroupedSlice:
    data_groups: list[np.ndarray]
    state: int
    core_map: np.ndarray
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"

    aspect_ratio: float = 1.0
    """dx / dy of a pin cell, from the core."""

    dataset_range: tuple[float, float] | None = None
    """Finite (lo, hi) over the whole dataset, when the builder computed one.
    Only ColorScope.DATASET needs it."""

    @property
    def n_groups(self) -> int:
        return len(self.data_groups)

    def group_slice(self, group: int) -> Self:
        """This slice narrowed to one group, keeping geometry and labels."""
        return replace(self, data_groups=[self.data_groups[group]])

    def finite(self, group: int = 0) -> np.ndarray:
        """One group's finite values, flat. Empty when the group is all NaN."""
        values = np.asarray(self.data_groups[group], dtype=float).ravel()
        return values[np.isfinite(values)]

    def value_range(
        self, group: int = 0, scope: ColorScope = ColorScope.GROUP
    ) -> tuple[float, float]:
        """The (lo, hi) a colorbar should span for one group, at one scope."""
        scope = ColorScope(scope)
        if scope is ColorScope.GROUP:
            return array_range(self.data_groups[group])
        if scope is ColorScope.SLICE:
            return union_range([array_range(values) for values in self.data_groups])
        if self.dataset_range is None:
            raise ValueError(
                "this slice carries no dataset_range;"
                " build it with dataset_range=True to color at dataset scope"
            )
        return self.dataset_range

    def validate(self) -> list[str]:
        """Return contract violations common to all grouped slices."""
        problems: list[str] = []

        if not isinstance(self.data_groups, list):
            problems.append(
                f"data_groups must be list[np.ndarray], got {type(self.data_groups).__name__}"
            )
            return problems

        if not self.data_groups:
            problems.append("data_groups cannot be empty")
            return problems

        if not all(isinstance(group, np.ndarray) for group in self.data_groups):
            problems.append("data_groups must contain only np.ndarray objects")
            return problems

        if not isinstance(self.state, int):
            problems.append(f"state must be int, got {type(self.state).__name__}")

        if not isinstance(self.core_map, np.ndarray):
            problems.append(f"core_map must be np.ndarray, got {type(self.core_map).__name__}")
        elif self.core_map.ndim != 2:
            problems.append(f"core_map must be 2-D, got shape {self.core_map.shape}")

        if not isinstance(self.dtype, VeraDtype):
            problems.append(f"dtype must be VeraDtype, got {type(self.dtype).__name__}")

        if not isinstance(self.units, str):
            problems.append(f"units must be str, got {type(self.units).__name__}")

        if not isinstance(self.aspect_ratio, (int, float)):
            problems.append(f"aspect_ratio must be numeric, got {type(self.aspect_ratio).__name__}")
        elif not np.isfinite(self.aspect_ratio) or self.aspect_ratio <= 0:
            problems.append(f"aspect_ratio must be finite and > 0, got {self.aspect_ratio}")

        if self.dataset_range is not None:
            if not isinstance(self.dataset_range, tuple) or len(self.dataset_range) != 2:
                problems.append("dataset_range must be tuple[float, float] or None")
            else:
                lo, hi = self.dataset_range

                if not all(isinstance(value, (int, float, np.number)) for value in (lo, hi)):
                    problems.append("dataset_range values must be numeric")
                elif not all(np.isfinite(value) for value in (lo, hi)):
                    problems.append(f"dataset_range must be finite, got {self.dataset_range}")
                elif lo > hi:
                    problems.append(f"dataset_range lower bound {lo} exceeds upper bound {hi}")

        return problems


def assembly_side(n: int) -> int:
    """Pin-side length for a cell of n values. n must be a perfect square."""
    if n <= 0:
        return 0
    side = math.isqrt(n)
    if side * side != n:
        raise ValueError(f"assembly cell length {n} is not a perfect square")
    return side


def axis_labels(core: VeraOutCore, dtype: VeraDtype, is_comp: bool):
    """Column and row labels for the map this dtype renders against."""
    if is_comp and core.has_comp_core():
        return core.comp_core_map_column_labels, core.comp_core_map_row_labels
    return core.reduced_core_map_column_labels, core.reduced_core_map_row_labels


def core_labels(core: VeraOutCore, is_comp: bool):
    """Return (x_labels, y_labels, max_core_cols) for the requested map."""
    x_labels = core.comp_core_map_column_labels if is_comp else core.reduced_core_map_column_labels
    y_labels = core.comp_core_map_row_labels if is_comp else core.reduced_core_map_row_labels
    max_core_cols = core.comp_core_map.shape[0] if is_comp else core.reduced_core_map.shape[0]
    return x_labels, y_labels, max_core_cols


def convert_ji_to_node(selected_j, selected_i):
    return np.clip((selected_i + selected_j * int(NUM_NODES / 2)), 0, NUM_NODES - 1)
