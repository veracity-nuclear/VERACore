from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import VeraDataSource
from ..thresholds import ThresholdCondition, apply_thresholds
from .color import array_range
from .core_slice import assembly_side, axis_labels, place_on_core_map

FACES: tuple[str, ...] = ("W", "N", "E", "S")
"""Order the reader delivers lateral faces in, and the order stored in the
last axis of SurfaceSlice.data."""

LATERAL_FACES = slice(0, 4)
"""The dataset holds [W, N, E, S, T, B]; only the four lateral faces have a
place on a radial map."""

ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.COMP_ASSY_SURFACE,
    VeraDtype.COMP_NODAL_SURFACE,
]


@dataclass(frozen=True)
class SurfaceRequest:
    array: str | VeraDataset
    state: int
    z: int
    src_id: str | None = None
    thresholds: Sequence[ThresholdCondition] = ()
    mask_reflected: bool = True
    group: int | None = None

    def label(self) -> str:
        """Short human-readable identifier."""
        stem = f"{self.array} @ state {self.state}, z {self.z}"
        return stem if self.src_id is None else f"{self.src_id} · {stem}"


@dataclass(frozen=True)
class SurfaceSlice:
    data: np.ndarray
    """(n_rows, n_cols, side, side, 4), last axis ordered as FACES."""

    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"
    request: SurfaceRequest | None = None
    faces: tuple[str, ...] = FACES

    group: int | None = None
    """Energy group index, or None for ungrouped datasets."""

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
        """(n_rows, n_cols) of the assembly grid."""
        return self.data.shape[0], self.data.shape[1]

    @property
    def node_side(self) -> int:
        """Nodes across one assembly. 1 for assembly-level surface data."""
        return self.data.shape[2]

    @property
    def n_nodes(self) -> int:
        return self.node_side**2

    @property
    def cell_shape(self) -> tuple[int, ...]:
        """Per-assembly payload shape, (side, side, 4)."""
        return self.data.shape[2:]

    @property
    def core_cols(self) -> int:
        return self.data.shape[1]

    @property
    def is_empty(self) -> bool:
        return bool(np.isnan(self.data).all())

    def assembly(self, row: int, col: int) -> np.ndarray:
        """The (side, side, 4) payload at a grid position, NaN when empty."""
        return self.data[row, col]

    def face(self, name: str) -> np.ndarray:
        """One face across the whole map, as (n_rows, n_cols, side, side)."""
        return self.data[..., self.faces.index(name)]

    def finite(self) -> np.ndarray:
        """Every non-NaN value, flattened. For statistics and histograms."""
        flat = np.ravel(self.data)
        return flat[~np.isnan(flat)]

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems = []
        if self.data.ndim != 5:
            problems.append(f"data must be 5-D, got {self.data.shape}")
            return problems
        if self.data.shape[2] != self.data.shape[3]:
            problems.append(f"node grid {self.cell_shape[:2]} is not square")
        if self.data.shape[4] != len(self.faces):
            problems.append(f"{self.data.shape[4]} face values for faces {self.faces}")
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
        name = self.request.array if self.request else "?"
        grp = "" if self.group is None else f" group {self.group}/{self.n_groups}"
        return (
            f"<SurfaceSlice {name} {self.dtype.name} grid{self.grid_shape}"
            f" nodes{self.node_side}x{self.node_side} [{self.slice_value_range[0]:.4g},"
            f" {self.slice_value_range[1]:.4g}] {self.units}{grp}>"
        )


def lateral_faces(dataset, group: int, z: int) -> np.ndarray:
    """One group and axial level as (n_assemblies, side, side, 4).

    The dataset is indexed (face, group, node, layer, assembly); this pulls the
    four lateral faces and folds the node axis into the square grid the view
    lays them out on.
    """
    radial = np.asarray(dataset[LATERAL_FACES, group, :, z, :])
    per_assembly = radial.transpose(2, 1, 0)  # (assembly, node, face)
    n_assemblies, n_nodes, n_faces = per_assembly.shape
    side = assembly_side(n_nodes)
    return per_assembly.reshape(n_assemblies, side, side, n_faces)


def surface_slices(source: VeraDataSource, request: SurfaceRequest) -> list[SurfaceSlice]:
    """Build the renderable surface maps for one dataset, state and level.

    Returns one SurfaceSlice per energy group, or an empty list when the
    dataset has no surface view.
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
        units = request.array.physical_units
    dataset_value_range = array_range(dataset[LATERAL_FACES])
    core = source.core
    core_map = core.get_map(dataset_type=dtype)
    x_labels, y_labels = axis_labels(core, dtype, dtype.is_computational())
    aspect_ratio = float(np.ravel(core.aspect_ratio)[0])
    n_groups = int(np.shape(dataset)[1])

    slices = []
    for group in range(n_groups):
        group_value_range = array_range(dataset[LATERAL_FACES, group])
        layer = lateral_faces(dataset, group, request.z)
        if request.thresholds:
            layer = apply_thresholds(layer, request.thresholds)
        placed = place_on_core_map(layer, core_map)
        slices.append(
            SurfaceSlice(
                data=placed,
                x_labels=list(x_labels),
                y_labels=[str(y) for y in y_labels],
                dtype=dtype,
                units=units,
                request=request,
                group=group if n_groups > 1 else None,
                n_groups=n_groups,
                slice_value_range=array_range(placed),
                group_value_range=group_value_range,
                dataset_value_range=dataset_value_range,
                aspect_ratio=aspect_ratio,
            )
        )
    return slices
