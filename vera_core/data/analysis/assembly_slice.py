from dataclasses import dataclass
from typing import ClassVar, Sequence

import numpy as np

from ..dtypes import VeraDtype
from ..model import VeraDataSource
from ..thresholds import ThresholdCondition, apply_thresholds
from .info import create_info
from .vera_slices import GroupedSlice, assembly_side, build_dataset_ranges, get_dataset

ALLOWED_DTYPES_: list[VeraDtype] = [
    VeraDtype.PIN,
    VeraDtype.COMP_PIN,
    VeraDtype.CHANNEL,
    VeraDtype.RADIAL,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
]

RADIAL_DTYPES = (VeraDtype.RADIAL,)
"""Radial datasets are already collapsed over z, so the request's z is unused."""


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

        if isinstance(self.dtype, VeraDtype):
            match self.dtype:
                case VeraDtype.PIN | VeraDtype.CHANNEL | VeraDtype.RADIAL:
                    if len(first_shape) != 2:
                        problems.append(
                            f"{self.dtype} assembly data must be 2-D, got shape {first_shape}"
                        )
                    elif first_shape[0] != first_shape[1]:
                        problems.append(
                            f"{self.dtype} assembly data must be square, got shape {first_shape}"
                        )

                case VeraDtype.COMP_NODAL | VeraDtype.COMP_NODAL_ENERGY:
                    if len(first_shape) != 1:
                        problems.append(
                            f"{self.dtype} assembly data must be 1-D, got shape {first_shape}"
                        )
                    else:
                        try:
                            assembly_side(first_shape[0])
                        except ValueError as exc:
                            problems.append(str(exc))

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
        if array_dtype not in ALLOWED_DTYPES_:
            return None
        match array_dtype:
            case VeraDtype.PIN | VeraDtype.CHANNEL | VeraDtype.COMP_PIN:
                images_dataset = [array[:, :, z, assembly_id].copy()]
            case VeraDtype.RADIAL:
                images_dataset = [array[:, :, assembly_id].copy()]
            case VeraDtype.COMP_NODAL:
                images_dataset = [array[:, z, assembly_id]]
            case VeraDtype.COMP_NODAL_ENERGY:
                num_energy_groups = np.shape(array)[0]
                images_dataset = [
                    array[energy_group, :, z, assembly_id]
                    for energy_group in range(num_energy_groups)
                ]
            case _:
                raise RuntimeError(
                    f"Assembly View cannot visualize datasets of type {str(array_dtype)}"
                )
        if (
            array_dtype in (VeraDtype.PIN, VeraDtype.RADIAL)
            and vera_source.core.non_fuel_locs is not None
            and vera_source.core.pin_volumes.shape == array.shape
        ):
            rows, cols, layers, assys = vera_source.core.non_fuel_locs
            in_image = (assys == assembly_id) & (layers == z)
            rod_ij = (rows[in_image], cols[in_image])
            for image in images_dataset:
                image[rod_ij] = np.nan
        if thresholds_to_apply is not None:
            for idx, image in enumerate(images_dataset):
                images_dataset[idx] = apply_thresholds(image, thresholds_to_apply)

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
