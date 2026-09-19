from dataclasses import dataclass
from typing import ClassVar, Sequence

import numpy as np

from ..dtypes import VeraDim, VeraDtype
from ..model import VeraDataSource, nan_out_non_fuel
from ..thresholds import ThresholdCondition, apply_thresholds
from .info import create_info
from .vera_slices import (
    CELL_DIMS,
    DimSpec,
    GroupedSlice,
    assembly_side,
    build_dataset_ranges,
    get_dataset,
)

SPEC = DimSpec(
    requires=frozenset({VeraDim.ASSEMBLY}),
    forbids=frozenset({VeraDim.SURFACE}),
    needs_cells=True,
)
ALLOWED_DTYPES_: list[VeraDtype] = SPEC.allowed()


@dataclass(frozen=True)
class AssemblySlice(GroupedSlice):
    ALLOWED_DTYPES: ClassVar[list[VeraDtype]] = ALLOWED_DTYPES_

    def __post_init__(self):
        errors = self.validate()
        if errors:
            raise ValueError("Malformed Assembly Slice: ", errors)

    @property
    def side(self) -> int:
        """Pins across the assembly."""
        shape = self.data_groups[0].shape
        if len(shape) == 2:
            return shape[0]
        elif len(shape) == 1:
            return assembly_side(shape[0])
        else:
            raise RuntimeError("Invalid assembly shape:", shape)

    @property
    def x_labels(self) -> list[int]:
        return list(range(1, self.side + 1))

    @property
    def y_labels(self) -> list[str] | list[int]:
        return list(range(1, self.side + 1))

    def to_grid(self, group: int = 0) -> np.ndarray:
        """One group as the lattice: (side, side), origin at the top left.

        A nodal group arrives as a flat run of nodes and is laid out
        row-major, the order serialize_data_groups hands the web view, so a
        report and the web view put the same value in the same cell.
        """
        return np.asarray(self.data_groups[group], dtype=float).reshape(self.side, self.side)

    def validate(self) -> list[str]:
        """Return contract violations for an assembly slice."""
        problems = super().validate()
        if (
            not isinstance(self.data_groups, list)
            or not self.data_groups
            or not all(isinstance(group, np.ndarray) for group in self.data_groups)
        ):
            return problems

        if isinstance(self.dtype, VeraDtype):
            if self.dtype not in self.ALLOWED_DTYPES:
                problems.append(f"dtype {self.dtype} is not supported by AssemblySlice")

        first_shape = self.data_groups[0].shape

        if len(first_shape) not in (1, 2):
            problems.append(f"assembly group must be 1-D or 2-D, got shape {first_shape}")
            return problems
        for i, group in enumerate(self.data_groups[1:], start=1):
            if group.shape != first_shape:
                problems.append(f"group {i} has shape {group.shape}, expected {first_shape}")

        if len(first_shape) == 2:
            rows, cols = first_shape

            if rows != cols:
                problems.append(f"2-D assembly group must be square, got {first_shape}")

        else:
            try:
                assembly_side(first_shape[0])
            except ValueError as exc:
                problems.append(str(exc))

        if isinstance(self.dtype, VeraDtype) and self.dtype.dim_axes:
            expected_ndim = 2 if VeraDim.PIN_Y in self.dtype.dim_axes else 1
            if len(first_shape) != expected_ndim:
                problems.append(
                    f"{self.dtype} assembly data must be {expected_ndim}-D, got shape {first_shape}"
                )

        return problems

    @classmethod
    def create_assembly_slice(
        cls,
        vera_source: VeraDataSource,
        selected_array: str,
        z: int,
        assembly_id: int,
        state: int | None = None,
        thresholds_to_apply: Sequence[ThresholdCondition] | None = None,
    ) -> "AssemblySlice | None":
        array = get_dataset(vera_source, selected_array, state_idx=state)
        array_dtype: VeraDtype = array.dataset_type
        if not SPEC.supports(array_dtype):
            return None
        images_dataset = nan_out_non_fuel(array, vera_source.core.pin_volumes).arrange(
            order=CELL_DIMS,
            split=(VeraDim.GROUP,),
            require=(VeraDim.ASSEMBLY,),
            axial=z,
            assembly=assembly_id,
        )
        if thresholds_to_apply is not None:
            images_dataset = [
                apply_thresholds(image, thresholds_to_apply) for image in images_dataset
            ]

        return AssemblySlice(
            data_groups=images_dataset,
            state=state if state is not None else vera_source.active_state_index,
            core_map=vera_source.core.get_map(array),
            dtype=array_dtype,
            units=array.physical_units,
            dataset_ranges=build_dataset_ranges(array),
            info=create_info(vera_source, array, z=z, assembly=assembly_id, state_idx=state),
        )

    def serialize_data_groups(self) -> list[list[float]]:
        return [np.ravel(group).tolist() for group in self.data_groups]
