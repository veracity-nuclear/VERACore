from dataclasses import dataclass, field
from typing import ClassVar, Literal, Sequence

import numpy as np

from ..dtypes import VeraDtype
from ..model import VeraDataSource, VeraOutCore
from ..thresholds import ThresholdCondition, apply_thresholds
from .vera_slices import GroupedSlice, build_dataset_ranges, convert_ji_to_node

X_AXIS = "x"
Y_AXIS = "y"
"""An x cut runs along a core row and is indexed by the selected j pin; a y
cut runs along a column and is indexed by the selected i pin."""

X_SCALE = 3.0

FALLBACK_DISPLAY_SIZE = 17
"""Assembly width in pins when the core does not report one."""

MAX_LABEL_WIDTH = 4
"""Cells wider than this hold too many values to label legibly."""

ALLOWED_DTYPES_: list[VeraDtype] = [
    VeraDtype.PIN,
    VeraDtype.CHANNEL,
    VeraDtype.ASSEMBLY,
    VeraDtype.COMP_ASSY,
    VeraDtype.COMP_ASSY_ENERGY,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
    VeraDtype.NODAL,
]


@dataclass(frozen=True, kw_only=True)
class AxialSlice(GroupedSlice):
    cell_width: int
    """Distinct values across one assembly."""

    y_scale: float

    x_edges: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y_edges: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    """(n_layers + 1,) elevation boundaries in cm, ascending."""

    x_labels: list[str] = field(default_factory=list)
    y_labels: list[str] = field(default_factory=list)

    x_size: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y_size: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))

    assembly_indices: np.ndarray = field(default_factory=lambda: np.array([0, 1]))

    axis: Literal["x", "y"]

    ALLOWED_DTYPES: ClassVar[list[VeraDtype]] = ALLOWED_DTYPES_

    def __post_init__(self):
        errors = self.validate()
        if errors:
            raise ValueError("Malformed Axial Slice: ", errors)

    @property
    def n_layers(self) -> int:
        return self.data_groups[0].shape[0]

    @property
    def n_cols(self) -> int:
        return self.data_groups[0].shape[1]

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

    def column(self, index: int) -> np.ndarray:
        """One assembly's values, as (n_layers, cell_width)."""
        return self.data[:, index]

    @classmethod
    def create_axial_slice(
        cls,
        vera_source: VeraDataSource,
        selected_array: str,
        pin: int,
        assembly_id: int,
        dim: Literal["x", "y"],
        state: int | None = None,
        thresholds_to_apply: Sequence[ThresholdCondition] | None = None,
    ) -> "AxialSlice | None":
        array = vera_source.get_dataset(selected_array, state_idx=state)
        core = vera_source.core
        array_dtype: VeraDtype = array.dataset_type

        if array_dtype == VeraDtype.PIN and core.non_fuel_locs is not None:
            array[core.non_fuel_locs] = np.nan

        if thresholds_to_apply:
            array = apply_thresholds(array, thresholds_to_apply)

        is_comp = array_dtype.is_computational()
        is_detector = array_dtype.is_detector()

        if array_dtype not in ALLOWED_DTYPES_:
            return None

        is_x = dim == "x"
        if is_x:
            assembly_indices = core.row_assembly_indices(assembly_id, is_comp, is_detector)
        else:
            assembly_indices = core.col_assembly_indices(assembly_id, is_comp, is_detector)

        if array_dtype in (VeraDtype.COMP_NODAL_ENERGY, VeraDtype.COMP_ASSY_ENERGY):
            num_groups = array.shape[0]
            if array_dtype == VeraDtype.COMP_ASSY_ENERGY:
                group_arrays = [array[g, 0] for g in range(num_groups)]
            else:
                group_arrays = [array[g] for g in range(num_groups)]
        else:
            num_groups = 1
            if array_dtype.is_assembly():
                group_arrays = [array[0]]
            else:
                group_arrays = [array]

        mesh_pixels = core.get_axial_mesh_pixels(dataset_type=array_dtype)
        mesh_means = core.get_axial_mesh_means(dataset_type=array_dtype)
        y_size = mesh_pixels[::-1]
        y_labels = [np.round(m, 1) for m in mesh_means][::-1]

        axial_mesh = core.get_axial_mesh(dataset_type=array_dtype)
        y_edges = axial_mesh
        total_h = float(abs(axial_mesh[-1] - axial_mesh[0]))
        cm_per_pixel = total_h / mesh_pixels.sum()
        y_scale = float(X_SCALE * cm_per_pixel / core.pin_pitch)

        nb_cols = 0
        display_width = FALLBACK_DISPLAY_SIZE
        cell_width = 1
        data_groups = []
        for g in range(num_groups):
            grid, display_width, nb_cols, cell_width = _build_group_grid(
                group_arrays[g],
                array_dtype,
                is_x,
                core,
                pin,
                assembly_indices,
            )
            data_groups.append(grid)

        x_size = np.full(shape=(nb_cols,), fill_value=display_width)
        x_edges = np.arange(0.0, (nb_cols + 1) * vera_source.core.apitch, vera_source.core.apitch)

        if is_x:
            x_labels = (
                core.comp_core_map_column_labels if is_comp else core.reduced_core_map_column_labels
            )
        else:
            start_x = (
                core.comp_map_start_index if is_comp else core.reduced_core_map_start_index
            ) + 1
            x_labels = list(range(start_x, nb_cols + start_x))

        return AxialSlice(
            data_groups=data_groups,
            state=state if state is not None else vera_source.active_state_index,
            core_map=vera_source.core.get_map(array),
            units=array.physical_units,
            dtype=array.dataset_type,
            dataset_ranges=build_dataset_ranges(array),
            cell_width=cell_width,
            assembly_indices=assembly_indices,
            x_labels=x_labels,
            y_labels=y_labels,
            x_edges=x_edges,
            y_edges=y_edges,
            y_size=y_size,
            x_size=x_size,
            y_scale=y_scale,
            axis="x" if is_x else "y",
        )

    def column_edges(self) -> np.ndarray:
        """x boundaries of every drawn column: each assembly cell cut into
        cell_width equal parts, so a cell holding several values across its
        width gets one column per value."""
        offsets = np.linspace(0.0, 1.0, self.cell_width + 1)[:-1]
        widths = np.diff(self.x_edges)
        starts = self.x_edges[:-1, None] + widths[:, None] * offsets
        return np.append(starts.ravel(), self.x_edges[-1])

    def to_grid(self, group: int = 0) -> np.ndarray:
        """One group laid out on the cut: (n_layers, n_cells * cell_width).

        Row 0 is the lowest elevation, so rows line up with the ascending
        y_edges rather than with the top-down order the web view draws in.
        Empty core positions are NaN. The values themselves are the ones
        serialize_data_groups sends the web view, in the same order.
        """
        values = self.data_groups[group]
        width = self.cell_width
        grid = np.full((self.n_layers, len(self.assembly_indices) * width), np.nan)
        column = 0
        for cell, index in enumerate(self.assembly_indices):
            if index < 0:
                continue
            grid[:, cell * width : (cell + 1) * width] = values[
                :, column * width : (column + 1) * width
            ]
            column += 1
        return grid[::-1]

    def serialize_data_groups(self) -> list[list[list]]:
        serialized = []
        is_assembly = self.dtype.is_assembly()
        data_width = self.cell_width
        for image_data in self.data_groups:
            grid = []
            for j in range(self.n_layers):
                line = []
                grid.append(line)
                col = 0
                for i in self.assembly_indices:
                    if i < 0:
                        line.append(np.full(data_width, np.nan).tolist())
                        continue
                    if is_assembly:
                        line.append(np.full(data_width, image_data[j, col]).tolist())
                    else:
                        cell = image_data[j, col * data_width : (col + 1) * data_width]
                        line.append(np.ravel(cell).tolist())
                    col += 1
            serialized.append(grid)
        return serialized

    def validate(self) -> list[str]:
        """Return contract violations, empty when the slice is well formed."""
        problems = super().validate()

        # ------------------------------------------------------------------
        # Runtime types
        # ------------------------------------------------------------------

        if not isinstance(self.cell_width, int):
            problems.append(f"cell_width must be int, got {type(self.cell_width).__name__}")
        elif self.cell_width <= 0:
            problems.append(f"cell_width must be > 0, got {self.cell_width}")

        if self.axis not in ("x", "y"):
            problems.append(f"axis must be 'x' or 'y', got {self.axis!r}")

        if not isinstance(self.assembly_indices, np.ndarray):
            problems.append(
                f"assembly_indices must be np.ndarray, got {type(self.assembly_indices).__name__}"
            )

        if not isinstance(self.x_edges, np.ndarray):
            problems.append(f"x_edges must be np.ndarray, got {type(self.x_edges).__name__}")

        if not isinstance(self.y_edges, np.ndarray):
            problems.append(f"y_edges must be np.ndarray, got {type(self.y_edges).__name__}")

        if not isinstance(self.x_size, np.ndarray):
            problems.append(f"x_size must be np.ndarray, got {type(self.x_size).__name__}")

        if not isinstance(self.y_size, np.ndarray):
            problems.append(f"y_size must be np.ndarray, got {type(self.y_size).__name__}")

        if not isinstance(self.x_labels, list):
            problems.append(f"x_labels must be list, got {type(self.x_labels).__name__}")
        elif not all(isinstance(label, (str, int)) for label in self.x_labels):
            problems.append("x_labels must contain only strings or ints")

        if not isinstance(self.y_labels, list):
            problems.append(f"y_labels must be list, got {type(self.y_labels).__name__}")
        elif not all(isinstance(label, (str, int, float, np.number)) for label in self.y_labels):
            problems.append("y_labels must contain only strings or numeric values")

        # ------------------------------------------------------------------
        # data_groups
        # ------------------------------------------------------------------
        # Stop here if the parent-level data_groups contract is not usable.

        if (
            not isinstance(self.data_groups, list)
            or not self.data_groups
            or not all(isinstance(group, np.ndarray) for group in self.data_groups)
        ):
            return problems

        first_shape = self.data_groups[0].shape

        if len(first_shape) != 2:
            problems.append(f"axial data groups must be 2-D, got shape {first_shape}")
            return problems

        n_layers, n_data_cols = first_shape

        if n_layers <= 0:
            problems.append("axial slice must contain at least one axial layer")

        if n_data_cols <= 0:
            problems.append("axial slice must contain at least one horizontal value")

        # Every group must describe exactly the same geometry.
        for group_index, group in enumerate(self.data_groups[1:], start=1):
            if group.ndim != 2:
                problems.append(f"group {group_index} must be 2-D, got shape {group.shape}")
            elif group.shape != first_shape:
                problems.append(
                    f"group {group_index} has shape {group.shape}, expected {first_shape}"
                )

        # ------------------------------------------------------------------
        # assembly_indices
        # ------------------------------------------------------------------

        n_cells: int | None = None

        if isinstance(self.assembly_indices, np.ndarray):
            if self.assembly_indices.ndim != 1:
                problems.append(
                    f"assembly_indices must be 1-D, got shape {self.assembly_indices.shape}"
                )
            elif not np.issubdtype(
                self.assembly_indices.dtype,
                np.integer,
            ):
                problems.append(
                    f"assembly_indices must contain integers, "
                    f"got dtype {self.assembly_indices.dtype}"
                )
            else:
                n_cells = len(self.assembly_indices)

                if n_cells <= 0:
                    problems.append("assembly_indices cannot be empty")

                # -1 represents an empty position in the core.
                if np.any(self.assembly_indices < -1):
                    problems.append(
                        "assembly_indices may contain assembly indices "
                        "or -1, but no values below -1"
                    )

        # ------------------------------------------------------------------
        # x_edges
        # ------------------------------------------------------------------

        if isinstance(self.x_edges, np.ndarray):
            if self.x_edges.ndim != 1:
                problems.append(f"x_edges must be 1-D, got shape {self.x_edges.shape}")
            else:
                if len(self.x_edges) < 2:
                    problems.append("x_edges must contain at least two boundaries")

                if not np.all(np.isfinite(self.x_edges)):
                    problems.append("x_edges must contain only finite values")
                elif not np.all(np.diff(self.x_edges) > 0):
                    problems.append("x_edges must be strictly ascending")

                if n_cells is not None and len(self.x_edges) != n_cells + 1:
                    problems.append(
                        f"x_edges has length {len(self.x_edges)}, "
                        f"expected {n_cells + 1} for "
                        f"{n_cells} assembly cells"
                    )

        # ------------------------------------------------------------------
        # y_edges
        # ------------------------------------------------------------------

        if isinstance(self.y_edges, np.ndarray):
            if self.y_edges.ndim != 1:
                problems.append(f"y_edges must be 1-D, got shape {self.y_edges.shape}")
            else:
                expected_edges = n_layers + 1

                if len(self.y_edges) != expected_edges:
                    problems.append(
                        f"y_edges has length {len(self.y_edges)}, "
                        f"expected {expected_edges} for "
                        f"{n_layers} axial layers"
                    )

                if not np.all(np.isfinite(self.y_edges)):
                    problems.append("y_edges must contain only finite values")
                elif not np.all(np.diff(self.y_edges) > 0):
                    problems.append("y_edges must be strictly ascending")

        # ------------------------------------------------------------------
        # x_size
        # ------------------------------------------------------------------

        if isinstance(self.x_size, np.ndarray):
            if self.x_size.ndim != 1:
                problems.append(f"x_size must be 1-D, got shape {self.x_size.shape}")
            else:
                if n_cells is not None and len(self.x_size) != n_cells:
                    problems.append(
                        f"x_size has length {len(self.x_size)}, "
                        f"expected {n_cells} from assembly_indices"
                    )

                if not np.all(np.isfinite(self.x_size)):
                    problems.append("x_size must contain only finite values")
                elif np.any(self.x_size <= 0):
                    problems.append("x_size values must all be > 0")

        # ------------------------------------------------------------------
        # y_size
        # ------------------------------------------------------------------

        if isinstance(self.y_size, np.ndarray):
            if self.y_size.ndim != 1:
                problems.append(f"y_size must be 1-D, got shape {self.y_size.shape}")
            else:
                if len(self.y_size) != n_layers:
                    problems.append(f"y_size has length {len(self.y_size)}, expected {n_layers}")

                if not np.all(np.isfinite(self.y_size)):
                    problems.append("y_size must contain only finite values")
                elif np.any(self.y_size <= 0):
                    problems.append("y_size values must all be > 0")

        # ------------------------------------------------------------------
        # Labels
        # ------------------------------------------------------------------

        if isinstance(self.x_labels, list) and n_cells is not None:
            if len(self.x_labels) not in (0, n_cells):
                problems.append(f"x_labels has length {len(self.x_labels)}, expected {n_cells}")

        if isinstance(self.y_labels, list):
            if len(self.y_labels) not in (0, n_layers):
                problems.append(f"y_labels has length {len(self.y_labels)}, expected {n_layers}")

        # ------------------------------------------------------------------
        # Horizontal data geometry
        # ------------------------------------------------------------------

        if n_cells is not None and isinstance(self.cell_width, int) and self.cell_width > 0:
            n_real_cells = len(self.assembly_indices[self.assembly_indices >= 0])
            expected_data_cols = n_real_cells * self.cell_width

            if n_data_cols != expected_data_cols:
                problems.append(
                    f"data has {n_data_cols} horizontal values, "
                    f"but {n_cells} assembly cells * cell_width "
                    f"{self.cell_width} requires {expected_data_cols}"
                )

        return problems


def _nodal_node_pair(selected_pin, is_x: bool):
    """Pick the two nodes along the cut direction for this view's axis."""
    if is_x:
        node = int(convert_ji_to_node(selected_pin, 0))  # j picks the row
        return (0, 1) if node in (0, 1) else (2, 3)
    node = int(convert_ji_to_node(0, selected_pin))  # i picks the col
    return (0, 2) if node in (0, 2) else (1, 3)


def _build_group_grid(
    array_2d_or_nodal,
    array_dtype: VeraDtype,
    is_x: bool,
    core: VeraOutCore,
    selected_pin,
    assembly_indices,
):
    """label_count is the number of distinct values a
    cell holds, or 0 when a cell holds too many to label."""
    is_assembly = array_dtype.is_assembly()
    arr = array_2d_or_nodal
    assembly_data_indices = assembly_indices[assembly_indices > -1]
    if array_dtype in (VeraDtype.PIN, VeraDtype.CHANNEL):
        cell_width = arr.shape[0]
        if is_x:
            image_data = arr[selected_pin, :, :, assembly_data_indices]
        else:
            image_data = arr[:, selected_pin, :, assembly_data_indices]
        image_data = np.vstack(image_data).T
        data_width = display_width = cell_width
        label_count = cell_width

    elif is_assembly:
        cell_width = core.core_shape[0] or FALLBACK_DISPLAY_SIZE
        image_data = np.vstack(arr[:, assembly_data_indices])
        data_width = display_width = cell_width
        label_count = 1

    elif array_dtype in (
        VeraDtype.COMP_NODAL,
        VeraDtype.COMP_NODAL_ENERGY,
        VeraDtype.NODAL,
    ):
        nodal = arr[:, :, assembly_data_indices]  # (nodes, nax, ncols)
        n_nodes = nodal.shape[0]
        if n_nodes == 1:
            data_width = display_width = core.core_shape[0] or FALLBACK_DISPLAY_SIZE
            image_data = np.vstack(nodal[0])
            label_count = 1
        else:
            node_pair = _nodal_node_pair(selected_pin, is_x)
            data_width = len(node_pair)
            display_width = core.core_shape[0] or FALLBACK_DISPLAY_SIZE
            sel = nodal[list(node_pair)]
            sel = np.transpose(sel, (1, 2, 0))
            image_data = sel.reshape(sel.shape[0], -1)
            label_count = data_width
    else:
        raise RuntimeError(f"Axial view cannot visualize datasets of type {str(array_dtype)}")

    image_data = image_data[::-1, :]  # axial level 0 at the bottom
    nb_cols = image_data.shape[1] if is_assembly else image_data.shape[1] // data_width
    nb_cols = len(assembly_indices)

    return image_data, display_width, nb_cols, label_count
