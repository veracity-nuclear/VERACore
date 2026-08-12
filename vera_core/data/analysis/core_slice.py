import math
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import VeraDataSource, VeraOutCore
from ..thresholds import ThresholdCondition, apply_thresholds
from .color import array_range


@dataclass(frozen=True)
class SliceRequest:
    array: str | VeraDataset
    state: int
    z: int
    src_id: str | None = None
    thresholds: Sequence[ThresholdCondition] = ()
    mask_reflected: bool = True
    group: int | None = None  # None means every group

    def label(self) -> str:
        """Short human-readable identifier, e.g. 'pin_powers @ state 2, z 10'."""
        if isinstance(self.array, str):
            array_name = self.array
        else:
            array_name = self.array.name or ""
        stem = f"{array_name} @ state {self.state}, z {self.z}"
        return stem if self.src_id is None else f"{self.src_id} · {stem}"


@dataclass(frozen=True)
class CoreSlice:
    data: np.ndarray
    labels: np.ndarray | None = None
    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"
    request: SliceRequest | None = None

    group: int | None = None
    """Energy or surface group index, or None for ungrouped datasets."""

    n_groups: int = 1
    """How many groups the source dataset produced. 1 when ungrouped."""

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
    def cell_shape(self) -> tuple[int, ...]:
        """Per-assembly payload shape: (), (NUM_NODES,) or (npy, npx)."""
        return self.data.shape[2:]

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the assembly grid."""
        return self.data.shape[0], self.data.shape[1]

    @property
    def assembly_side(self) -> int:
        """Pins across one assembly, or 1 for assembly-level data."""
        return self.cell_shape[0] if len(self.cell_shape) == 2 else 1

    @property
    def core_cols(self) -> int:
        """Widest row of the assembly grid, for widget sizing."""
        return self.data.shape[1]

    @property
    def has_labels(self) -> bool:
        """True when any assembly carries label text."""
        return self.labels is not None and any(cell is not None for cell in np.ravel(self.labels))

    @property
    def is_empty(self) -> bool:
        return bool(np.isnan(self.data).all())

    # -- views ------------------------------------------------------------

    def as_image(self) -> np.ndarray:
        """Flatten per-assembly cells into one 2-D array for imshow"""
        if len(self.cell_shape) == 0:
            return self.data
        if len(self.cell_shape) != 2:
            raise ValueError(f"as_image() needs a 2-D cell, got cell_shape {self.cell_shape}")
        nr, nc, ch, cw = self.data.shape
        return self.data.transpose(0, 2, 1, 3).reshape(nr * ch, nc * cw)

    def assembly(self, row: int, col: int) -> np.ndarray:
        """The cell at a grid position. All-NaN when the position is empty."""
        return self.data[row, col]

    def finite(self) -> np.ndarray:
        """Every non-NaN value, flattened. For statistics and histograms."""
        flat = np.ravel(self.data)
        return flat[~np.isnan(flat)]

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed.

        Called by core_slices() in debug builds and by tests. Cheap: shape and
        length checks only, no data scanning beyond NaN counts.
        """
        problems = []
        if self.data.ndim < 2:
            problems.append(f"data must be at least 2-D, got {self.data.shape}")
            return problems
        if len(self.cell_shape) > 2:
            problems.append(f"cell_shape {self.cell_shape} has more than 2 dims")
        if len(self.x_labels) not in (0, self.data.shape[1]):
            problems.append(f"{len(self.x_labels)} x_labels for {self.data.shape[1]} columns")
        if len(self.y_labels) not in (0, self.data.shape[0]):
            problems.append(f"{len(self.y_labels)} y_labels for {self.data.shape[0]} rows")
        if self.labels is not None and self.labels.shape != self.grid_shape:
            problems.append(f"labels {self.labels.shape} does not match grid {self.grid_shape}")
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
        name = "?"
        if self.request and isinstance(self.request.array, str):
            name = self.request.array
        elif self.request and isinstance(self.request.array, VeraDataset):
            name = self.request.array.name or "?"
        name = self.request.array if self.request else "?"
        grp = "" if self.group is None else f" group {self.group}/{self.n_groups}"
        return (
            f"<CoreSlice {name} {self.dtype.name} grid{self.grid_shape}"
            f" cell{self.cell_shape} [{self.slice_value_range[0]:.4g},"
            f" {self.slice_value_range[1]:.4g}] {self.units}{grp}>"
        )


ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.PIN,
    VeraDtype.CHANNEL,
    VeraDtype.ASSEMBLY,
    VeraDtype.RADIAL,
    VeraDtype.RADIAL_ASSEMBLY,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
    VeraDtype.COMP_ASSY,
    VeraDtype.COMP_ASSY_ENERGY,
    VeraDtype.NODAL,
    VeraDtype.POINT_DETECTOR,
    VeraDtype.RADIAL_POINT_DETECTOR,
]


def nan_out_non_fuel_locs(
    array: np.ndarray, src: VeraDataSource, selected_layer: int, is_radial: bool
):
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


def _extract_layers(dataset: np.ndarray, dtype: VeraDtype, z: int) -> list[np.ndarray]:
    match dtype:
        case VeraDtype.PIN | VeraDtype.CHANNEL:
            return [dataset[:, :, z].swapaxes(0, 2).swapaxes(1, 2)]
        case VeraDtype.RADIAL:
            return [dataset.swapaxes(0, 2).swapaxes(1, 2)]
        case VeraDtype.POINT_DETECTOR:
            return [dataset[z, :]]
        case VeraDtype.ASSEMBLY | VeraDtype.COMP_ASSY:
            return [dataset[0, z, :]]
        case VeraDtype.RADIAL_ASSEMBLY | VeraDtype.RADIAL_POINT_DETECTOR:
            return [dataset]
        case VeraDtype.COMP_NODAL | VeraDtype.NODAL:
            return [dataset[:, z, :].swapaxes(0, 1)]
        case VeraDtype.COMP_ASSY_ENERGY:
            return [dataset[g, 0, z, :] for g in range(np.shape(dataset)[0])]
        case VeraDtype.COMP_NODAL_ENERGY:
            return [dataset[g, :, z, :].swapaxes(0, 1) for g in range(np.shape(dataset)[0])]
        case _:
            raise RuntimeError(f"Core view cannot visualize a dataset of type {dtype}")


def axis_labels(core: VeraOutCore, dtype: VeraDtype, is_comp: bool):
    """Column and row labels for the map this dtype renders against."""
    if is_comp and core.has_comp_core():
        return core.comp_core_map_column_labels, core.comp_core_map_row_labels
    return core.reduced_core_map_column_labels, core.reduced_core_map_row_labels


def place_on_core_map(layer: np.ndarray, core_map: np.ndarray) -> np.ndarray:
    """Lay per-entity values out on the assembly grid.

    core_map holds 1-based entity ids with 0 for empty positions. The result is
    (n_rows, n_cols, *cell_shape), NaN wherever the map is empty or the id falls
    outside the layer -- so downstream code has one sentinel, not two.
    """
    n_entities = layer.shape[0]
    cell_shape = layer.shape[1:]
    placed = np.full((*core_map.shape, *cell_shape), np.nan, dtype=float)
    ids = np.asarray(core_map, dtype=np.int64)
    rows, cols = np.nonzero(ids)
    for r, c in zip(rows, cols, strict=False):
        idx = ids[r, c] - 1
        if 0 <= idx < n_entities:
            placed[r, c] = layer[idx]
    return placed


def core_slices(source: VeraDataSource, request: SliceRequest) -> list[CoreSlice]:
    """Build the renderable core maps for one dataset, state and axial level.

    Returns one CoreSlice per energy or surface group, or a single-element list
    for ungrouped datasets. Returns an empty list when the dataset has no core
    view
    """
    if request.z < 0:
        raise ValueError(f"z must be >= 0, got {request.z}")

    if isinstance(request.array, str):
        dtype = source.get_dataset_dtype(request.array, request.state)
    else:
        dtype = request.array.dataset_type

    if dtype not in ALLOWED_DTYPES:
        return []

    if isinstance(request.array, str):
        dataset = source.get_dataset(
            request.array,
            mask_reflected=request.mask_reflected,
            state_idx=request.state,
        )
        units = source.get_dataset_units(request.array, request.state)
    else:
        dataset = request.array
        units = dataset.physical_units

    dataset_value_range = array_range(dataset)
    core = source.core
    core_map = core.get_map(dataset_type=dtype)
    is_comp = dtype.is_computational()
    x_labels, y_labels = axis_labels(core, dtype, is_comp)
    aspect_ratio = float(np.ravel(core.aspect_ratio)[0])

    layers = _extract_layers(dataset, dtype, request.z)
    n_groups = len(layers)

    if dtype in (VeraDtype.COMP_ASSY_ENERGY, VeraDtype.COMP_NODAL_ENERGY):
        group_value_ranges = [array_range(dataset[group]) for group in range(dataset.shape[0])]
    else:
        group_value_ranges = [dataset_value_range]
    assert n_groups == len(group_value_ranges)
    slices = []
    for group, layer in enumerate(layers):
        if dtype.has_fuel_pins():
            layer = nan_out_non_fuel_locs(layer, source, request.z, dtype == VeraDtype.RADIAL)
        if request.thresholds:
            layer = apply_thresholds(layer, request.thresholds)

        placed = place_on_core_map(np.asarray(layer), core_map)
        slices.append(
            CoreSlice(
                data=placed,
                labels=None,
                x_labels=list(x_labels),
                y_labels=[str(y) for y in y_labels],
                dtype=dtype,
                units=units,
                request=request,
                group=group if n_groups > 1 else None,
                n_groups=n_groups,
                slice_value_range=array_range(placed),
                group_value_range=group_value_ranges[group],
                dataset_value_range=dataset_value_range,
                aspect_ratio=aspect_ratio,
            )
        )
    return slices
