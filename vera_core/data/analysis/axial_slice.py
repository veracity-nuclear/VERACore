from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..dtypes import NUM_NODES, VeraDtype
from ..model import VeraDataSource, VeraOutCore
from ..thresholds import ThresholdCondition, apply_thresholds
from .color import array_range

X_AXIS = "x"
Y_AXIS = "y"
"""An x cut runs along a core row and is indexed by the selected j pin; a y
cut runs along a column and is indexed by the selected i pin."""

FALLBACK_DISPLAY_WIDTH = 17
"""Assembly width in pins when the core does not report one."""

MAX_LABEL_WIDTH = 4
"""Cells wider than this hold too many values to label legibly."""

ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.PIN,
    VeraDtype.CHANNEL,
    VeraDtype.ASSEMBLY,
    VeraDtype.COMP_ASSY,
    VeraDtype.COMP_ASSY_ENERGY,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
    VeraDtype.NODAL,
]


@dataclass(frozen=True)
class AxialRequest:
    array: str
    state: int
    assembly: int = 0
    """Assembly index the cut passes through. Get it from
    core.reduced_core_map_assembly(i, j)."""

    pin: int = 0
    """Pin across the cut: the j index for an x cut, i for a y cut."""

    axis: str = X_AXIS
    src_id: str | None = None
    thresholds: Sequence[ThresholdCondition] = ()
    mask_reflected: bool = True
    group: int | None = None  # None means every group

    @property
    def is_x(self) -> bool:
        return self.axis == X_AXIS

    def label(self) -> str:
        """Short human-readable identifier."""
        stem = (
            f"{self.array} @ state {self.state}, {self.axis} cut through"
            f" assembly {self.assembly}, pin {self.pin}"
        )
        return stem if self.src_id is None else f"{self.src_id} · {stem}"


@dataclass(frozen=True)
class AxialSlice:
    data: np.ndarray
    """(n_layers, n_cols, cell_width). Row 0 is the lowest axial level."""

    x_edges: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    """(n_cols * cell_width + 1,) column boundaries in cm."""

    y_edges: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    """(n_layers + 1,) elevation boundaries in cm, ascending."""

    x_labels: list[str] = field(default_factory=list)
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"
    request: AxialRequest | None = None

    group: int | None = None
    n_groups: int = 1
    slice_value_range: tuple[float, float] = (0.0, 1.0)
    """(lo, hi) value range for the slice"""
    group_value_range: tuple[float, float] = (0.0, 1.0)
    """(lo, hi) value range for the group the slice belongs to"""
    dataset_value_range: tuple[float, float] = (0.0, 1.0)
    """(lo, hi) value range for the dataset the slice belong to
    (depending on dataset this may be identical to `group_value_range`)
    """

    @property
    def n_layers(self) -> int:
        return self.data.shape[0]

    @property
    def n_cols(self) -> int:
        return self.data.shape[1]

    @property
    def cell_width(self) -> int:
        """Distinct values across one assembly."""
        return self.data.shape[2]

    @property
    def labelable(self) -> bool:
        """False when a cell holds too many values to write in it."""
        return self.cell_width <= MAX_LABEL_WIDTH

    @property
    def elevations(self) -> np.ndarray:
        """Mid-height of each axial level, in cm."""
        return (self.y_edges[:-1] + self.y_edges[1:]) / 2

    @property
    def extent(self) -> tuple[float, float, float, float]:
        """(x_lo, x_hi, y_lo, y_hi) in cm."""
        return (
            float(self.x_edges[0]),
            float(self.x_edges[-1]),
            float(self.y_edges[0]),
            float(self.y_edges[-1]),
        )

    def as_image(self) -> np.ndarray:
        """(n_layers, n_cols * cell_width), the array pcolormesh draws."""
        return self.data.reshape(self.n_layers, self.n_cols * self.cell_width)

    def column(self, index: int) -> np.ndarray:
        """One assembly's values, as (n_layers, cell_width)."""
        return self.data[:, index]

    def finite(self) -> np.ndarray:
        """Every non-NaN value, flattened. For statistics and histograms."""
        flat = np.ravel(self.data)
        return flat[~np.isnan(flat)]

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems = []
        if self.data.ndim != 3:
            problems.append(f"data must be 3-D, got {self.data.shape}")
            return problems
        if len(self.y_edges) != self.n_layers + 1:
            problems.append(f"{len(self.y_edges)} y_edges for {self.n_layers} layers")
        if len(self.x_edges) != self.n_cols * self.cell_width + 1:
            problems.append(
                f"{len(self.x_edges)} x_edges for {self.n_cols} columns of {self.cell_width}"
            )
        if len(self.x_labels) not in (0, self.n_cols):
            problems.append(f"{len(self.x_labels)} x_labels for {self.n_cols} columns")
        if np.any(np.diff(self.y_edges) <= 0):
            problems.append("y_edges must ascend")
        for value_range in ("slice_value_range", "group_value_range", "dataset_value_range"):
            attr = getattr(self, value_range, None)
            if (
                not isinstance(attr, tuple)
                or len(attr) != 2
                or not all(isinstance(x, float) for x in attr)
            ):
                problems.append(f"{value_range} is not a valid range: {attr}")
            lo, hi = attr
            if not (np.isfinite(lo) and np.isfinite(hi)) or hi < lo:
                problems.append(f"{value_range} : {attr} is not a valid interval")
        if self.group is not None and not 0 <= self.group < self.n_groups:
            problems.append(f"group {self.group} outside n_groups {self.n_groups}")
        return problems

    def __repr__(self) -> str:
        name = self.request.array if self.request else "?"
        axis = self.request.axis if self.request else "?"
        grp = "" if self.group is None else f" group {self.group}/{self.n_groups}"
        return (
            f"<AxialSlice {name} {self.dtype.name} {axis}-cut"
            f" {self.n_layers}x{self.n_cols}x{self.cell_width}"
            f" [{self.slice_value_range[0]:.4g}, {self.slice_value_range[1]:.4g}]"
            f" {self.units}{grp}>"
        )


def node_pair(pin: int, is_x: bool) -> tuple[int, int]:
    """The two nodes a cut passes through, for nodal data."""
    half = NUM_NODES // 2
    if is_x:
        node = int(np.clip(pin * half, 0, NUM_NODES - 1))
        return (0, 1) if node in (0, 1) else (2, 3)
    node = int(np.clip(pin, 0, NUM_NODES - 1))
    return (0, 2) if node in (0, 2) else (1, 3)


def display_width(core: VeraOutCore) -> int:
    """Assembly width in pins, for laying columns out to scale."""
    width = core.core_shape[0]
    return int(width) if width else FALLBACK_DISPLAY_WIDTH


def _cut_columns(
    array: np.ndarray, dtype: VeraDtype, pin: int, columns: np.ndarray, is_x: bool
) -> np.ndarray:
    """Values along the cut as (n_layers, n_cols, cell_width), NaN where empty.

    columns holds assembly indices with -1 for positions the cut passes over
    that hold no assembly.
    """
    present = columns[columns > -1]
    if dtype in (VeraDtype.PIN, VeraDtype.CHANNEL):
        # (npy, npx, nax, nass) -> (nax, n_present, pins across)
        taken = array[pin, :, :, present] if is_x else array[:, pin, :, present]
        values = np.asarray(taken).transpose(2, 0, 1)
    elif dtype.is_assembly():
        # (nax, nass) -> one value per assembly
        values = np.asarray(array[:, present])[:, :, None]
    else:
        # (nodes, nax, nass)
        nodal = np.asarray(array)
        if nodal.shape[0] == 1:
            values = nodal[0][:, present][:, :, None]
        else:
            pair = list(node_pair(pin, is_x))
            values = nodal[pair][:, :, present].transpose(1, 2, 0)

    n_layers, _, cell_width = values.shape
    placed = np.full((n_layers, len(columns), cell_width), np.nan)
    placed[:, columns > -1] = values
    return placed


def _group_arrays(array, dtype: VeraDtype) -> list[np.ndarray]:
    """One array per energy group, shaped as _cut_columns() expects."""
    if dtype == VeraDtype.COMP_ASSY_ENERGY:
        return [array[g, 0] for g in range(np.shape(array)[0])]
    if dtype == VeraDtype.COMP_NODAL_ENERGY:
        return [array[g] for g in range(np.shape(array)[0])]
    if dtype.is_assembly():
        return [array[0]]
    return [np.asarray(array)]


def _column_labels(core: VeraOutCore, is_comp: bool, is_x: bool, n_cols: int):
    """Letters along a row cut, numbers down a column cut."""
    if is_x:
        return list(
            core.comp_core_map_column_labels if is_comp else core.reduced_core_map_column_labels
        )
    start = (core.comp_map_start_index if is_comp else core.reduced_core_map_start_index) + 1
    return [str(v) for v in range(start, start + n_cols)]


def axial_slices(source: VeraDataSource, request: AxialRequest) -> list[AxialSlice]:
    """Build the renderable axial cuts for one dataset and state.

    Returns one AxialSlice per energy group, or an empty list when the dataset
    has no axial view.
    """
    if request.axis not in (X_AXIS, Y_AXIS):
        raise ValueError(f"axis must be {X_AXIS!r} or {Y_AXIS!r}, got {request.axis!r}")

    dtype = source.get_dataset_dtype(request.array, request.state)
    if dtype not in ALLOWED_DTYPES:
        return []

    array = source.get_dataset(
        request.array,
        mask_reflected=request.mask_reflected,
        state_idx=request.state,
    )
    units = source.get_dataset_units(request.array, request.state)
    core = source.core

    array = np.array(array, dtype=float, copy=True)
    dataset_value_range = array_range(array)
    if dtype == VeraDtype.PIN and core.non_fuel_locs is not None:
        array[core.non_fuel_locs] = np.nan
    if request.thresholds:
        array = apply_thresholds(array, request.thresholds)

    is_comp = dtype.is_computational()
    picker = core.row_assembly_indices if request.is_x else core.col_assembly_indices
    columns = np.asarray(picker(request.assembly, is_comp, dtype.is_detector()))
    if not (columns > -1).any():
        raise ValueError(
            f"{request.label()}: the cut passes through no assemblies (columns {columns.tolist()})"
        )

    mesh = np.asarray(core.get_axial_mesh(dataset_type=dtype), dtype=float)
    y_edges = np.sort(np.ravel(mesh))
    width_cm = display_width(core) * float(core.pin_pitch)

    groups = _group_arrays(array, dtype)
    n_groups = len(groups)

    slices = []
    for group, values in enumerate(groups):
        group_dataset_range = array_range(values)
        data = _cut_columns(values, dtype, request.pin, columns, request.is_x)
        x_edges = np.linspace(0.0, len(columns) * width_cm, len(columns) * data.shape[2] + 1)
        slices.append(
            AxialSlice(
                data=data,
                x_edges=x_edges,
                y_edges=y_edges,
                x_labels=_column_labels(core, is_comp, request.is_x, len(columns)),
                dtype=dtype,
                units=units,
                request=request,
                group=group if n_groups > 1 else None,
                n_groups=n_groups,
                slice_value_range=array_range(data),
                group_value_range=group_dataset_range,
                dataset_value_range=dataset_value_range,
            )
        )
    return slices
