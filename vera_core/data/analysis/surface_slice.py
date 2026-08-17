from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np

from ..dtypes import VeraDtype
from ..model import VeraDataSource
from .color import array_range
from .vera_slices import GroupedSlice, assembly_side, axis_labels

FACES: tuple[str, ...] = ("W", "N", "E", "S")
"""Order the reader delivers lateral faces in, and the order stored in the
last axis of SurfaceSlice.data."""

LATERAL_FACE_SLICE = slice(0, 4)
"""The dataset holds [W, N, E, S, T, B]; only the four lateral faces have a
place on a radial map."""

ALLOWED_DTYPES: list[VeraDtype] = [
    VeraDtype.COMP_ASSY_SURFACE,
    VeraDtype.COMP_NODAL_SURFACE,
]


@dataclass(frozen=True, kw_only=True)
class SurfaceSlice(GroupedSlice):
    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str | int] = field(default_factory=list)

    faces: ClassVar[tuple[str, ...]] = FACES
    ALLOWED_DTYPES: ClassVar[list[VeraDtype]] = ALLOWED_DTYPES

    def __post_init__(self):
        problems = self.validate()
        if problems:
            raise ValueError(f"Malformed SurfaceSlice, problems: {problems}")

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the assembly grid."""
        return int(self.core_map.shape[0]), int(self.core_map.shape[1])

    @property
    def node_side(self) -> int:
        """Nodes across one assembly. 1 for assembly-level surface data."""
        return assembly_side(self.data_groups[0].shape[1])

    @property
    def n_nodes(self) -> int:
        return self.data_groups[0].shape[1]

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems = super().validate()

        # Labels
        if not isinstance(self.x_labels, list):
            problems.append(f"x_labels must be list[str], got {type(self.x_labels).__name__}")
        elif not all(isinstance(label, str) for label in self.x_labels):
            problems.append("x_labels must contain only strings")

        if not isinstance(self.y_labels, list):
            problems.append(f"y_labels must be list[str], got {type(self.y_labels).__name__}")
        elif not all(isinstance(label, str) for label in self.y_labels) and not all(
            isinstance(label, int) for label in self.y_labels
        ):
            problems.append("y_labels must contain only strings")

        # Dtype
        if isinstance(self.dtype, VeraDtype) and self.dtype not in self.ALLOWED_DTYPES:
            problems.append(f"dtype {self.dtype} is not supported by SurfaceSlice")

        # Stop if common fields are unusable.
        if (
            not isinstance(self.data_groups, list)
            or not self.data_groups
            or not all(isinstance(group, np.ndarray) for group in self.data_groups)
        ):
            return problems

        if not isinstance(self.core_map, np.ndarray):
            return problems

        if self.core_map.ndim != 2:
            return problems

        # Group shape:
        # (n_faces, n_nodes, n_entities)
        first_shape = self.data_groups[0].shape

        if len(first_shape) != 3:
            problems.append(f"surface data groups must be 3-D, got shape {first_shape}")
            return problems

        n_faces, n_nodes, n_entities = first_shape

        if n_faces != len(self.faces):
            problems.append(
                f"surface group has {n_faces} faces, "
                f"expected {len(self.faces)} "
                f"({', '.join(self.faces)})"
            )

        if n_nodes <= 0:
            problems.append("surface group must contain at least one node")
        else:
            try:
                assembly_side(n_nodes)
            except ValueError as exc:
                problems.append(f"surface node count is invalid: {exc}")

        if n_entities <= 0:
            problems.append("surface group must contain at least one entity")

        # All energy groups must have identical geometry.
        for group_index, group in enumerate(
            self.data_groups[1:],
            start=1,
        ):
            if group.ndim != 3:
                problems.append(f"group {group_index} must be 3-D, got shape {group.shape}")
                continue

            if group.shape != first_shape:
                problems.append(
                    f"group {group_index} has shape {group.shape}, expected {first_shape}"
                )

        # Core-map values.
        if np.any(self.core_map < 0):
            problems.append(
                "core_map may contain 0 for empty positions, but no negative entity IDs"
            )

        if not np.issubdtype(self.core_map.dtype, np.integer):
            if not np.all(np.equal(self.core_map, np.floor(self.core_map))):
                problems.append("core_map must contain integer-valued entity IDs")

        positive_ids = self.core_map[self.core_map > 0]

        if positive_ids.size:
            max_entity_id = int(np.max(positive_ids))

            if max_entity_id > n_entities:
                problems.append(
                    f"core_map references entity {max_entity_id}, "
                    f"but surface groups contain only "
                    f"{n_entities} entities"
                )

        # Label geometry.
        n_rows, n_cols = self.core_map.shape

        if isinstance(self.x_labels, list):
            if len(self.x_labels) not in (0, n_cols):
                problems.append(
                    f"x_labels has length {len(self.x_labels)}, "
                    f"expected {n_cols} for {n_cols} columns"
                )

        if isinstance(self.y_labels, list):
            if len(self.y_labels) not in (0, n_rows):
                problems.append(
                    f"y_labels has length {len(self.y_labels)}, expected {n_rows} for {n_rows} rows"
                )

        return problems

    @classmethod
    def create_surface_slice(
        cls, vera_source: VeraDataSource, selected_array: str, z: int, state: int | None = None
    ) -> "SurfaceSlice | None":
        array = vera_source.get_dataset(selected_array, state_idx=state)
        array_dtype = array.dataset_type
        if array_dtype not in ALLOWED_DTYPES:
            return None
        core = vera_source.core
        is_comp = array_dtype.is_computational()
        n_energy = array.shape[1]
        grouped_datasets = [
            np.asarray(array[LATERAL_FACE_SLICE, g, :, z, :]) for g in range(n_energy)
        ]
        x_labels, y_labels = axis_labels(
            core,
            array_dtype,
            is_comp,
        )
        return SurfaceSlice(
            data_groups=grouped_datasets,
            state=state if state is not None else vera_source.active_state_index,
            units=array.physical_units,
            dtype=array_dtype,
            core_map=core.get_map(array),
            dataset_range=array_range(array),
            x_labels=x_labels,
            y_labels=y_labels,
        )

    def serialize_dataset_groups(self) -> list[list[list[list[float]]]]:
        cm = self.core_map
        n_rows, n_cols = self.grid_shape
        serialized = []
        for data_group in self.data_groups:
            grid = []
            for row in range(n_rows):
                line = []
                grid.append(line)
                for col in range(n_cols):
                    assembly_idx = int(cm[row, col]) - 1
                    if assembly_idx < 0:
                        line.append([])  # empty position
                        continue
                    # Each node -> [w, n, e, s] as python floats.
                    nodes = []
                    for node in range(self.n_nodes):
                        faces = data_group[:, node, assembly_idx]
                        nodes.append([float(v) for v in faces])
                    line.append(nodes)
            serialized.append(grid)
        return serialized
