import math
from dataclasses import dataclass, field, replace
from typing import Self

import numpy as np

from ..dtypes import NUM_NODES, VeraDataset, VeraDim, VeraDtype
from ..model import VeraDataSource, VeraOutCore
from .color import ColorScope, array_range, union_range
from .info import Info

CELL_DIMS: tuple[VeraDim, ...] = (VeraDim.NODE, VeraDim.PIN_Y, VeraDim.PIN_X)
"""Dims that lay values out inside one assembly, in arrange order."""


@dataclass(frozen=True, kw_only=True)
class DimSpec:
    """Which dtypes a view can draw"""

    requires: frozenset[VeraDim] = frozenset()
    forbids: frozenset[VeraDim] = frozenset()
    needs_cells: bool = False
    """Require a pin or node lattice inside each assembly."""
    detectors: bool = True
    exclude: frozenset[VeraDtype] = field(default_factory=frozenset)
    """Dtypes the dims admit but the view should not draw."""

    def supports(self, dtype: VeraDtype) -> bool:
        dims = dtype.dim_axes.keys()
        return (
            self.requires <= dims
            and not self.forbids & dims
            and (not self.needs_cells or bool(set(CELL_DIMS) & dims))
            and (self.detectors or not dtype.is_detector())
            and dtype not in self.exclude
        )

    def allowed(self) -> list[VeraDtype]:
        return [dtype for dtype in VeraDtype if self.supports(dtype)]


@dataclass(frozen=True, kw_only=True)
class GroupedSlice:
    data_groups: list[np.ndarray]
    state: int
    core_map: np.ndarray
    info: Info
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"

    aspect_ratio: float = 1.0
    """dx / dy of a pin cell, from the core."""

    dataset_ranges: list[tuple[float, float]] | None = None
    """Finite (lo, hi) over the whole dataset, when the builder computed one.
    Only ColorScope.DATASET_ needs it."""

    @property
    def n_groups(self) -> int:
        return len(self.data_groups)

    def group_slice(self, group: int) -> Self:
        """This slice narrowed to one group, keeping geometry and labels."""
        ranges = None if self.dataset_ranges is None else [self.dataset_ranges[group]]

        return replace(
            self,
            data_groups=[self.data_groups[group]],
            dataset_ranges=ranges,
        )

    def finite(self, group: int = 0) -> np.ndarray:
        """One group's finite values, flat. Empty when the group is all NaN."""
        values = np.asarray(self.data_groups[group], dtype=float).ravel()
        return values[np.isfinite(values)]

    def value_range(
        self, group: int = 0, scope: ColorScope = ColorScope.SLICE_ALL
    ) -> tuple[float, float]:
        """The (lo, hi) a colorbar should span for one group, at one scope."""
        scope = ColorScope(scope)
        if scope is ColorScope.SLICE_GROUP:
            return array_range(self.data_groups[group])
        if scope is ColorScope.SLICE_ALL:
            return union_range([array_range(values) for values in self.data_groups])
        if self.dataset_ranges is None:
            raise ValueError(
                "this slice carries no dataset_ranges; "
                "build it with dataset_ranges to color at dataset scope"
            )
        if scope is ColorScope.DATASET_GROUP:
            return self.dataset_ranges[group]
        return union_range(self.dataset_ranges)

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

        if self.dataset_ranges is not None:
            if not isinstance(self.dataset_ranges, list):
                problems.append("dataset_ranges must be list[tuple[float, float]] or None")
            else:
                if len(self.dataset_ranges) != self.n_groups:
                    problems.append(
                        f"dataset_ranges has {len(self.dataset_ranges)} ranges, "
                        f"expected {self.n_groups} "
                        f"for {self.n_groups} data groups"
                    )

                for group_index, value_range in enumerate(self.dataset_ranges):
                    if not isinstance(value_range, tuple) or len(value_range) != 2:
                        problems.append(
                            f"dataset_ranges[{group_index}] must be tuple[float, float]"
                        )
                        continue

                    lo, hi = value_range

                    if not all(isinstance(value, (int, float, np.number)) for value in (lo, hi)):
                        problems.append(f"dataset_ranges[{group_index}] values must be numeric")
                        continue

                    if not all(np.isfinite(value) for value in (lo, hi)):
                        problems.append(
                            f"dataset_ranges[{group_index}] must be finite, got {value_range}"
                        )
                        continue

                    if lo > hi:
                        problems.append(
                            f"dataset_ranges[{group_index}] "
                            f"lower bound {lo} exceeds upper bound {hi}"
                        )

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


def build_dataset_ranges(dataset: VeraDataset) -> list[tuple[float, float]]:
    ds_dtype = dataset.dataset_type
    if ds_dtype.has_energy_group_dim():
        e_group_dim = ds_dtype.energy_group_dim_idx
        dataset_ranges = [
            array_range(np.take(dataset, e_group, e_group_dim))
            for e_group in range(dataset.shape[e_group_dim])
        ]
        return dataset_ranges
    return [array_range(dataset)]


def get_dataset(src: VeraDataSource, array: str | VeraDataset, *, state_idx: int | None = None):
    if isinstance(array, VeraDataset):
        return array
    elif isinstance(array, str):
        return src.get_dataset(array, state_idx=state_idx)
    else:
        raise ValueError(
            "array must either be the name (str) of an vera dataset or a vera dataset itself"
        )
