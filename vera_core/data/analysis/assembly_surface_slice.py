from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from ..dtypes import VeraDim, VeraDtype
from ..model import VeraDataSource
from .info import create_info
from .surface_slice import ALLOWED_DTYPES, FACES, LATERAL_FACE_SLICE, SPEC
from .vera_slices import GroupedSlice, assembly_side, build_dataset_ranges, get_dataset


@dataclass(frozen=True, kw_only=True)
class AssemblySurfaceSlice(GroupedSlice):
    faces: ClassVar[tuple[str, ...]] = FACES
    ALLOWED_DTYPES: ClassVar[list[VeraDtype]] = ALLOWED_DTYPES

    def __post_init__(self):
        errors = self.validate()
        if errors:
            raise ValueError(f"Malformed AssemblySurfaceSlice, problems: {errors}")

    @property
    def x_labels(self) -> list[int]:
        return list(range(1, self.side + 1))

    @property
    def y_labels(self) -> list[int]:
        return list(range(1, self.side + 1))

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the node grid."""
        return self.side, self.side

    @property
    def side(self) -> int:
        """Nodes across the assembly. 1 for assembly-level surface data."""
        return assembly_side(self.data_groups[0].shape[1])

    @property
    def n_nodes(self) -> int:
        return self.side**2

    def to_grid(self, group: int = 0) -> np.ndarray:
        """One group laid out on the node grid: (side, side, n_faces), the
        last axis in `faces` order.

        Nodes run row-major, the order serialize_data_groups hands the web
        view, so a report and the web view put the same face in the same
        corner of the same cell.
        """
        values = np.asarray(self.data_groups[group], dtype=float)
        return values.T.reshape(self.side, self.side, len(self.faces))

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems = super().validate()

        # Stop if parent-level data_groups validation failed.
        if (
            not isinstance(self.data_groups, list)
            or not self.data_groups
            or not all(isinstance(group, np.ndarray) for group in self.data_groups)
        ):
            return problems

        if isinstance(self.dtype, VeraDtype):
            if self.dtype not in self.ALLOWED_DTYPES:
                problems.append(f"dtype {self.dtype} is not supported by AssemblySurfaceSlice")

        first_shape = self.data_groups[0].shape

        if len(first_shape) != 2:
            problems.append(f"surface data groups must be 2-D, got shape {first_shape}")
            return problems

        n_faces, n_nodes = first_shape

        # AssemblySurfaceSlice stores only the four lateral faces.
        if n_faces != 4:
            problems.append(f"surface group has {n_faces} faces, expected 4 lateral faces")

        if n_nodes <= 0:
            problems.append("surface group must contain at least one node")
        else:
            try:
                assembly_side(n_nodes)
            except ValueError as exc:
                problems.append(str(exc))

        # Every energy group must describe the same face/node geometry.
        for group_index, group in enumerate(self.data_groups[1:], start=1):
            if group.ndim != 2:
                problems.append(f"group {group_index} must be 2-D, got shape {group.shape}")
            elif group.shape != first_shape:
                problems.append(
                    f"group {group_index} has shape {group.shape}, expected {first_shape}"
                )

        return problems

    @classmethod
    def create_assembly_surface_slice(
        cls,
        vera_source: VeraDataSource,
        selected_array: str,
        z: int,
        assembly_id: int,
        state: int | None = None,
    ) -> "AssemblySurfaceSlice | None":
        array = get_dataset(vera_source, selected_array, state_idx=state)
        array_dtype = array.dataset_type
        if not SPEC.supports(array_dtype):
            return None
        groups = array.arrange(
            order=(VeraDim.SURFACE, VeraDim.NODE),
            split=(VeraDim.GROUP,),
            pad=(VeraDim.NODE,),
            axial=z,
            assembly=assembly_id,
        )
        grouped_datasets = [np.asarray(group[LATERAL_FACE_SLICE]) for group in groups]

        return AssemblySurfaceSlice(
            data_groups=grouped_datasets,
            state=state if state is not None else vera_source.active_state_index,
            core_map=vera_source.core.get_map(array),
            dtype=array.dataset_type,
            units=array.physical_units,
            dataset_ranges=build_dataset_ranges(array),
            info=create_info(vera_source, array, z=z, assembly=assembly_id, state_idx=state),
        )

    def serialize_data_groups(self) -> list[list[list]]:
        return [
            [data[:, node].tolist() for node in range(data.shape[1])] for data in self.data_groups
        ]
