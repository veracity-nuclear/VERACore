from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..dtypes import VeraDataset, VeraDtype
from ..model import VeraDataSource
from ..thresholds import ThresholdCondition, apply_thresholds
from .color import array_range
from .core_slice import assembly_side
from .surface_slice import ALLOWED_DTYPES, FACES, LATERAL_FACES


@dataclass(frozen=True)
class AssemblySurfaceRequest:
    array: str | VeraDataset
    state: int
    assembly: int = 0
    """Assembly index, from core.reduced_core_map_assembly(i, j)."""

    z: int = 0
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
class AssemblySurfaceSlice:
    data: np.ndarray
    """(side, side, 4) node faces, last axis ordered as FACES."""

    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    dtype: VeraDtype = VeraDtype.UNKNOWN
    units: str = "unitless"
    request: AssemblySurfaceRequest | None = None
    faces: tuple[str, ...] = FACES

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
    """dx / dy of a node cell, from the core."""

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the node grid."""
        return self.data.shape[0], self.data.shape[1]

    @property
    def side(self) -> int:
        """Nodes across the assembly. 1 for assembly-level surface data."""
        return self.data.shape[0]

    @property
    def n_nodes(self) -> int:
        return self.side**2

    @property
    def is_empty(self) -> bool:
        return bool(np.isnan(self.data).all())

    def node(self, row: int, col: int) -> np.ndarray:
        """One node's four face values, in FACES order."""
        return self.data[row, col]

    def face(self, name: str) -> np.ndarray:
        """One face across the assembly, as (side, side)."""
        return self.data[..., self.faces.index(name)]

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
        if self.data.shape[0] != self.data.shape[1]:
            problems.append(f"node grid {self.grid_shape} is not square")
        if self.data.shape[2] != len(self.faces):
            problems.append(f"{self.data.shape[2]} face values for faces {self.faces}")
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
            f"<AssemblySurfaceSlice {name} {self.dtype.name}"
            f" {self.side}x{self.side} nodes [{self.slice_value_range[0]:.4g},"
            f" {self.slice_value_range[1]:.4g}] {self.units}{grp}>"
        )


def assembly_lateral_faces(dataset, group: int, z: int, assembly: int) -> np.ndarray:
    """One assembly as (side, side, 4).

    The dataset is indexed (face, group, node, layer, assembly);
    """
    radial = np.asarray(dataset[LATERAL_FACES, group, :, z, assembly])
    per_node = radial.transpose(1, 0)  # (node, face)
    side = assembly_side(per_node.shape[0])
    return per_node.reshape(side, side, per_node.shape[1])


def assembly_surface_slices(
    source: VeraDataSource, request: AssemblySurfaceRequest
) -> list[AssemblySurfaceSlice]:
    """Build the renderable node-face maps for one assembly, state and level.

    Returns one slice per energy group, or an empty list when the dataset has
    no assembly surface view
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
        dataset = request.array
        units = request.array.physical_units
    else:
        dataset = source.get_dataset(
            request.array,
            mask_reflected=request.mask_reflected,
            state_idx=request.state,
        )
        units = source.get_dataset_units(request.array, request.state)

    dataset_value_range = array_range(dataset[LATERAL_FACES])
    aspect_ratio = float(np.ravel(source.core.aspect_ratio)[0])
    n_groups = int(np.shape(dataset)[1])

    slices = []
    for group in range(n_groups):
        group_value_range = array_range(dataset[LATERAL_FACES, group])
        nodes = assembly_lateral_faces(dataset, group, request.z, request.assembly)
        nodes = np.array(nodes, dtype=float, copy=True)
        if request.thresholds:
            nodes = apply_thresholds(nodes, request.thresholds)
        if not np.isfinite(nodes).any():
            raise ValueError(
                f"{request.label()}: group {group} has no finite values."
                f" Nodes {nodes.shape}. The assembly may be masked by"
                f" symmetry -- try mask_reflected=False or another assembly."
            )
        labels = [str(v) for v in range(1, nodes.shape[0] + 1)]
        slices.append(
            AssemblySurfaceSlice(
                data=nodes,
                x_labels=labels,
                y_labels=labels,
                dtype=dtype,
                units=units,
                request=request,
                group=group if n_groups > 1 else None,
                n_groups=n_groups,
                slice_value_range=array_range(nodes),
                group_value_range=group_value_range,
                dataset_value_range=dataset_value_range,
                aspect_ratio=aspect_ratio,
            )
        )
    return slices
