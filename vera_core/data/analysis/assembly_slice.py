from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import VeraDataSource
from ..thresholds import ThresholdCondition, apply_thresholds
from .color import array_range
from .core_slice import assembly_side

ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.PIN,
    VeraDtype.CHANNEL,
    VeraDtype.RADIAL,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
]

RADIAL_DTYPES = (VeraDtype.RADIAL,)
"""Radial datasets are already collapsed over z, so the request's z is unused."""


@dataclass(frozen=True)
class AssemblyRequest:
    array: str | VeraDataset
    state: int
    assembly: int = 0
    """Assembly index, from core.reduced_core_map_assembly(i, j)."""

    z: int = 0
    """Axial level. Ignored for radial datasets, which have no z."""

    src_id: str | None = None
    thresholds: Sequence[ThresholdCondition] = ()
    mask_reflected: bool = True
    group: int | None = None  # None means every group

    @property
    def is_dataset(self) -> bool:
        return isinstance(self.array, VeraDataset)

    @property
    def array_name(self) -> str:
        return self.array.name or "" if self.is_dataset else self.array

    def label(self) -> str:
        """Short identifier."""
        stem = f"{self.array_name} @ state {self.state}, assembly {self.assembly}, z {self.z}"
        return stem if self.src_id is None else f"{self.src_id} · {stem}"


@dataclass(frozen=True)
class AssemblySlice:
    data: np.ndarray
    """(side, side) pin values. NaN for guide tubes and threshold rejects."""

    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"
    request: AssemblyRequest | None = None

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

    aspect_ratio: float = 1.0
    """dx / dy of a pin cell, from the core."""

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the pin lattice."""
        return self.data.shape[0], self.data.shape[1]

    @property
    def side(self) -> int:
        """Pins across the assembly."""
        return self.data.shape[0]

    @property
    def is_empty(self) -> bool:
        return bool(np.isnan(self.data).all())

    def pin(self, row: int, col: int) -> float:
        """One pin's value. NaN when the position holds no fuel."""
        return float(self.data[row, col])

    def finite(self) -> np.ndarray:
        """Every non-NaN value, flattened. For statistics and histograms."""
        flat = np.ravel(self.data)
        return flat[~np.isnan(flat)]

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems = []
        if self.data.ndim != 2:
            problems.append(f"data must be 2-D, got {self.data.shape}")
            return problems
        if self.data.shape[0] != self.data.shape[1]:
            problems.append(f"lattice {self.data.shape} is not square")
        if len(self.x_labels) not in (0, self.data.shape[1]):
            problems.append(f"{len(self.x_labels)} x_labels for {self.data.shape[1]} columns")
        if len(self.y_labels) not in (0, self.data.shape[0]):
            problems.append(f"{len(self.y_labels)} y_labels for {self.data.shape[0]} rows")
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
        name = self.request.array_name if self.request else "?"
        grp = "" if self.group is None else f" group {self.group}/{self.n_groups}"
        return (
            f"<AssemblySlice {name} {self.dtype.name} {self.side}x{self.side}"
            f" [{self.slice_value_range[0]:.4g}, {self.slice_value_range[1]:.4g}]"
            f" {self.units}{grp}>"
        )


def _lattices(
    array, dtype: VeraDtype, assembly: int, z: int
) -> tuple[list[np.ndarray], list[tuple[float, float]]]:
    match dtype:
        case VeraDtype.PIN | VeraDtype.CHANNEL:
            return ([np.asarray(array[:, :, z, assembly])], [array_range(array)])
        case VeraDtype.RADIAL:
            return ([np.asarray(array[:, :, assembly])], [array_range(array)])
        case VeraDtype.COMP_NODAL:
            nodes = np.asarray(array[:, z, assembly])
            side = assembly_side(nodes.shape[0])
            return ([nodes.reshape(side, side)], [array_range(array)])
        case VeraDtype.COMP_NODAL_ENERGY:
            lattices = []
            group_value_ranges = []
            for group in range(np.shape(array)[0]):
                nodes = np.asarray(array[group, :, z, assembly])
                side = assembly_side(nodes.shape[0])
                lattices.append(nodes.reshape(side, side))
                group_value_ranges.append(array_range(array[group]))
            return (lattices, group_value_ranges)
        case _:
            raise RuntimeError(f"Assembly view cannot visualize a dataset of type {dtype}")


def nan_out_non_fuel_pins(
    lattice: np.ndarray, source: VeraDataSource, assembly: int, z: int, is_radial: bool
) -> np.ndarray:
    """Blank guide tubes and other non-fuel positions in one assembly."""
    locs = source.core.non_fuel_locs
    if locs is None:
        return lattice
    rows, cols, layers, assemblies = locs
    keep = assemblies == assembly
    if not is_radial:
        keep = keep & (layers == z)
    blanked = lattice.copy()
    blanked[rows[keep], cols[keep]] = np.nan
    return blanked


def assembly_slices(source: VeraDataSource, request: AssemblyRequest) -> list[AssemblySlice]:
    """Build the renderable pin lattices for one assembly, state and level.

    Returns one AssemblySlice per energy group, or an empty list when the
    dataset has no assembly view
    """
    if request.z < 0:
        raise ValueError(f"z must be >= 0, got {request.z}")

    dtype = (
        request.array.dataset_type
        if request.is_dataset
        else source.get_dataset_dtype(request.array, request.state)
    )
    if dtype not in ALLOWED_DTYPES:
        return []

    if request.is_dataset:
        array = request.array
        units = request.array.physical_units
    else:
        array = source.get_dataset(
            request.array,
            mask_reflected=request.mask_reflected,
            state_idx=request.state,
        )
        units = source.get_dataset_units(request.array, request.state)
    dataset_value_range = array_range(array)
    is_radial = dtype in RADIAL_DTYPES
    aspect_ratio = float(np.ravel(source.core.aspect_ratio)[0])

    lattices, group_value_ranges = _lattices(array, dtype, request.assembly, request.z)
    n_groups = len(lattices)
    assert n_groups == len(group_value_ranges)

    slices = []
    for group, lattice in enumerate(lattices):
        lattice = np.array(lattice, dtype=float, copy=True)
        if dtype in (VeraDtype.PIN, VeraDtype.RADIAL):
            lattice = nan_out_non_fuel_pins(lattice, source, request.assembly, request.z, is_radial)
        if request.thresholds:
            lattice = apply_thresholds(lattice, request.thresholds)
        if not np.isfinite(lattice).any():
            raise ValueError(
                f"{request.label()}: group {group} has no finite values."
                f" Lattice {lattice.shape}. The assembly may be masked by"
                f" symmetry -- try mask_reflected=False or another assembly."
            )
        side = lattice.shape[0]
        labels = [str(v) for v in range(1, side + 1)]
        slices.append(
            AssemblySlice(
                data=lattice,
                x_labels=labels,
                y_labels=labels,
                dtype=dtype,
                units=units,
                request=request,
                group=group if n_groups > 1 else None,
                n_groups=n_groups,
                slice_value_range=array_range(lattice),
                group_value_range=group_value_ranges[group],
                dataset_value_range=dataset_value_range,
                aspect_ratio=aspect_ratio,
            )
        )
    return slices
