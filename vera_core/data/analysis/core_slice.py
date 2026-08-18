from dataclasses import dataclass, field
from typing import ClassVar, Sequence

import numpy as np

from ..dtypes import VeraDataset, VeraDim, VeraDtype
from ..model import VeraDataSource
from ..thresholds import ThresholdCondition, apply_thresholds
from .vera_slices import GroupedSlice, assembly_side, build_dataset_ranges, core_labels

ALLOWED_DTYPES_: list[VeraDtype] = [
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


@dataclass(frozen=True)
class CoreSlice(GroupedSlice):
    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)
    ALLOWED_DTYPES: ClassVar[list[VeraDtype]] = ALLOWED_DTYPES_

    def __post_init__(self):
        errors = self.validate()
        if errors:
            raise ValueError(f"Bad formed core slice, problems: {str(errors)}")

    @property
    def cell_shape(self) -> tuple[int, ...]:
        """Per-assembly payload shape: (), (NUM_NODES,) or (npy, npx)."""
        return self.data_groups[0].shape[1:]

    @property
    def grid_shape(self) -> tuple[int, int]:
        """(n_rows, n_cols) of the assembly grid."""
        core_shape = self.core_map.shape
        return (int(core_shape[0]), int(core_shape[1]))

    @property
    def max_columns(self) -> int:
        return int(self.core_map.shape[0])

    @property
    def assembly_side(self) -> int:
        """Pins or nodes across one assembly, or 1 for assembly-level data.

        A 1-D cell is a flat run of nodes, so its side is the square root, not
        the length.
        """
        if len(self.cell_shape) == 2:
            return self.cell_shape[0]
        if len(self.cell_shape) == 1:
            return assembly_side(self.cell_shape[0])
        return 1

    def to_grid(self, group: int = 0) -> np.ndarray:
        """One group laid out on the assembly grid"""
        side = self.assembly_side
        n_rows, n_cols = self.grid_shape
        values = self.data_groups[group]
        grid = np.full((n_rows * side, n_cols * side), np.nan)
        for row in range(n_rows):
            for col in range(n_cols):
                index = int(self.core_map[row, col]) - 1
                if index < 0:
                    continue
                cell = np.asarray(values[index], dtype=float)
                grid[row * side : (row + 1) * side, col * side : (col + 1) * side] = cell.reshape(
                    side, side
                )
        return grid

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems: list[str] = []

        problems = super().validate()

        if (
            not isinstance(self.data_groups, list)
            or not self.data_groups
            or not all(isinstance(group, np.ndarray) for group in self.data_groups)
            or not isinstance(self.core_map, np.ndarray)
        ):
            return problems

        if len(self.cell_shape) > 2:
            problems.append(f"cell_shape {self.cell_shape} has more than 2 dims")

        if isinstance(self.x_labels, list):
            if len(self.x_labels) not in (0, self.grid_shape[1]):
                problems.append(f"{len(self.x_labels)} x_labels for {self.grid_shape[1]} columns")

        if isinstance(self.y_labels, list):
            if len(self.y_labels) not in (0, self.grid_shape[0]):
                problems.append(f"{len(self.y_labels)} y_labels for {self.grid_shape[0]} rows")

        return problems

    @classmethod
    def create_core_slice(
        cls,
        vera_source: VeraDataSource,
        dataset_name: str,
        z: int,
        state_idx: int | None = None,
        thresholds: Sequence[ThresholdCondition] = [],
    ) -> "CoreSlice | None":
        if z < 0:
            raise RuntimeError(f"z must be < 0, z = {z}")
        dataset = vera_source.get_dataset(dataset_name, state_idx=state_idx)
        ds_dtype = dataset.dataset_type
        if ds_dtype not in ALLOWED_DTYPES_:
            return None
        is_comp = ds_dtype.is_computational()
        layer_list: list[VeraDataset] = dataset.arrange(
            order=(VeraDim.ASSEMBLY, VeraDim.NODE, VeraDim.PIN_Y, VeraDim.PIN_X),
            require=(VeraDim.ASSEMBLY,),
            split=(VeraDim.GROUP,),
            axial=z,
        )
        core = vera_source.core
        for idx, layer in enumerate(layer_list):
            if ds_dtype.has_fuel_pins():
                layer = nan_out_non_fuel_locs(layer, vera_source, z, ds_dtype == VeraDtype.RADIAL)
            if thresholds:
                layer = apply_thresholds(layer, thresholds)
            layer_list[idx] = layer

        x_labels, y_labels, _ = core_labels(core, is_comp)
        return cls(
            data_groups=layer_list,
            x_labels=x_labels,
            y_labels=y_labels,
            state=state_idx if state_idx is not None else vera_source.active_state_index,
            dtype=ds_dtype,
            units=dataset.physical_units,
            core_map=vera_source.core.get_map(dataset),
            dataset_ranges=build_dataset_ranges(dataset),
        )

    def serialize_data_groups(self) -> list[tuple[list, list]]:
        cm = self.core_map
        is_assembly_avg = self.dtype.is_assembly()
        core_width = cm.shape[0]
        results = []
        for dataset in self.data_groups:
            result = []
            labels = []
            for i in range(core_width):
                line = [None] * core_width
                result.append(line)
                if is_assembly_avg:
                    labels_line = [None] * core_width
                    labels.append(labels_line)
                for j in range(core_width):
                    index = cm[i, j] - 1
                    if index == -1:
                        continue
                    if is_assembly_avg:
                        line[j] = [float(dataset[index])]
                        labels_line[j] = float(dataset[index])
                    else:
                        line[j] = np.ravel(dataset[index]).tolist()
            results.append((result, labels))
        return results


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
