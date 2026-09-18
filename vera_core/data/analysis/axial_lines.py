from collections.abc import Iterator
from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from beartype import beartype

from ..dtypes import Surface, VeraDim, VeraDtype
from ..model import VeraDataSource

type Line = tuple[
    tuple[np.ndarray, np.ndarray],
    str,
    str,
]
"""A single line represented as ((x, y), label identifier, mode)."""

type Indices = dict[VeraDim, int]
"""Each key is a dimension to index with its value"""

ALLOWED_DTYPES_ = [
    VeraDtype.PIN,
    VeraDtype.COMP_PIN,
    VeraDtype.CHANNEL,
    VeraDtype.AXIAL,
    VeraDtype.ASSEMBLY,
    VeraDtype.COMP_NODAL,
    VeraDtype.COMP_NODAL_ENERGY,
    VeraDtype.COMP_NODAL_SURFACE,
    VeraDtype.COMP_ASSY_SURFACE,
    VeraDtype.COMP_ASSY,
    VeraDtype.COMP_ASSY_ENERGY,
    VeraDtype.NODAL,
    VeraDtype.POINT_DETECTOR,
    VeraDtype.CONTINOUS_DETECTOR,
    VeraDtype.NODAL_ENERGY,
    VeraDtype.NODAL_SURFACE,
    VeraDtype.ASSY_SURFACE,
    VeraDtype.ASSY_ENERGY,
]


def region_segments(values, intervals):
    n = values.size
    x = np.full(3 * n, np.nan)
    y = np.full(3 * n, np.nan)
    x[0::3] = values
    x[1::3] = values
    y[0::3] = intervals[:, 0]
    y[1::3] = intervals[:, 1]
    return x, y


@beartype
@dataclass(frozen=True)
class AxialLines:
    axial_arrays: list[tuple[np.ndarray, np.ndarray]]
    identifiers: list[str]
    modes: list[str]
    states: list[int]

    ALLOWED_DTYPES: ClassVar[list[VeraDtype]] = ALLOWED_DTYPES_

    def __post_init__(self):
        if (
            len(self.axial_arrays) != len(self.identifiers)
            or len(self.axial_arrays) != len(self.modes)
            or len(self.axial_arrays) != len(self.states)
        ):
            raise ValueError("axial_arrays, identifiers, and modes must have the same length")

    def lines(self) -> Iterator[Line]:
        """
        Iterate over the axial line data.

        Yields:
            Line:
                A tuple containing:

                - A ``(x, y)`` tuple of NumPy arrays containing
                the coordinates/data for the line.
                - ``identifier``: A string identifying the line or source.
                - ``mode``: A string describing the mode associated with the line.

        The returned items correspond element-for-element across
        ``axial_arrays``, ``identifiers``, and ``modes``.
        """
        return zip(self.axial_arrays, self.identifiers, self.modes, strict=False)

    def axial_extents(self) -> tuple[float, float]:
        """The (lo, hi) the axial meshs span"""
        axial_min = min(np.nanmin(y) for _, y in self.axial_arrays)
        axial_max = max(np.nanmax(y) for _, y in self.axial_arrays)
        return float(axial_min), float(axial_max)

    def value_extents(self) -> tuple[float, float]:
        """The (lo, hi) the values span"""
        finite = [x[np.isfinite(x)] for x, _ in self.axial_arrays]
        finite = [values for values in finite if values.size]
        if not finite:
            return 0.0, 1.0
        return float(min(v.min() for v in finite)), float(max(v.max() for v in finite))

    @beartype
    @classmethod
    def create_axial_lines(
        cls,
        vera_sources: VeraDataSource | list[VeraDataSource],
        dataset_names: str | list[str] | list[list[str]],
        idxs: Indices | list[Indices] | list[list[Indices]],
        state: int | None = None,
    ) -> "AxialLines":
        axial_arrays: list[tuple[np.ndarray, np.ndarray]] = []
        identifiers: list[str] = []
        modes: list[str] = []
        states: list[int] = []

        for src, dataset_name, indices in _normalize_inputs(vera_sources, dataset_names, idxs):
            dataset = src.get_dataset(dataset_name, state_idx=state)
            units = dataset.physical_units
            units_label = f" ({units}) " if units != "unitless" else ""
            vdtype: VeraDtype = dataset.dataset_type
            if not indices or vdtype not in ALLOWED_DTYPES_:
                continue
            j = indices.get(VeraDim.PIN_Y, 0)
            i = indices.get(VeraDim.PIN_X, 0)
            node_idx = indices.get(VeraDim.NODE, 0)
            assembly_id = indices.get(VeraDim.ASSEMBLY, 0)
            surface_idx = indices.get(VeraDim.SURFACE, 0)

            mode = "lines" if vdtype != VeraDtype.POINT_DETECTOR else "lines+markers"
            identifier = ""

            if vdtype.has_assembly_id_dim():
                assembly_label = src.core.reduced_core_map_label(
                    assembly_id, vdtype.is_computational()
                )
                identifier += f" | {assembly_label}"

            if vdtype.is_detector():
                identifier += "Detector"

            if vdtype.has_pin_level_dim():
                identifier += f" @({i + 1},{j + 1})"
            elif vdtype.has_node_dim():
                identifier += f" @(NODE {node_idx + 1})"

            if vdtype.has_surface_dim():
                surface_str = Surface(surface_idx).str
                identifier += f" {surface_str}"

            grouped_axial_datasets = dataset.arrange(
                order=(VeraDim.AXIAL,),
                split=(VeraDim.GROUP,),
                require=(VeraDim.AXIAL,),
                pin=(j, i),
                surface=surface_idx,
                node=node_idx,
                assembly=assembly_id,
            )
            max_state = len(src.states) - 1
            recorded_state = src.active_state_index if not state else max(0, min(state, max_state))
            axial_mesh_means = src.core.get_axial_mesh_means(dataset_type=vdtype)
            for idx, axial_array in enumerate(grouped_axial_datasets):
                group_label = "" if len(grouped_axial_datasets) <= 1 else f" GROUP {idx + 1}"
                x, y = (
                    (axial_array, axial_mesh_means)
                    if axial_mesh_means.ndim != 2
                    else region_segments(axial_array, axial_mesh_means)
                )
                name = f"{src.name} | {dataset_name.replace('_', ' ').title()}{units_label}{identifier + group_label}"
                axial_arrays.append((x, y))
                identifiers.append(name)
                modes.append(mode)
                states.append(recorded_state)

        return cls(
            axial_arrays=axial_arrays,
            identifiers=identifiers,
            modes=modes,
            states=states,
        )


def _normalize_inputs(
    vera_sources: VeraDataSource | list[VeraDataSource],
    dataset_names: str | list[str] | list[list[str]],
    indices: Indices | list[Indices] | list[list[Indices]],
) -> list[tuple[VeraDataSource, str, Indices]]:

    sources = vera_sources if isinstance(vera_sources, list) else [vera_sources]

    n_sources = len(sources)

    if n_sources == 0:
        return []

    # normalize dataset_names
    if isinstance(dataset_names, str):  #
        names_per_source = [[dataset_names] for _ in sources]
    elif all(isinstance(name, str) for name in dataset_names):
        if n_sources == 1:
            # multiple names for one source
            names_per_source = [dataset_names]
        elif len(dataset_names) == n_sources:
            # One dataset per source
            names_per_source = [[name] for name in dataset_names]
        else:
            raise ValueError(
                "For multiple sources, a flat dataset_names list "
                "must contain exactly one dataset name per source. "
                "Use list[list[str]] to specify multiple datasets per source."
            )
    else:
        if len(dataset_names) != n_sources:
            raise ValueError("Nested dataset_names must contain exactly one list per data source.")
        # list of datasets for each source
        names_per_source = dataset_names

    job_count = sum(len(names) for names in names_per_source)

    if job_count == 0:
        raise ValueError("At least one dataset name is required.")

    if isinstance(indices, dict):
        indices_per_source = [[indices for _ in names] for names in names_per_source]
    elif all(isinstance(idx, dict) for idx in indices):
        if len(indices) != job_count:
            raise ValueError(
                f"Flat indices list must contain one entry per dataset. "
                f"Expected {job_count}, got {len(indices)}."
            )

        indices_per_source = []

        offset = 0
        for names in names_per_source:
            count = len(names)
            indices_per_source.append(indices[offset : offset + count])
            offset += count

    else:
        if len(indices) != n_sources:
            raise ValueError("Nested indices must contain exactly one list per data source.")

        indices_per_source = indices

        for i, (names, source_indices) in enumerate(
            zip(names_per_source, indices_per_source, strict=False)
        ):
            if len(source_indices) != len(names):
                raise ValueError(
                    f"Source {i} has {len(names)} dataset(s), "
                    f"but {len(source_indices)} indices specification(s)."
                )

    work_items: list[tuple[VeraDataSource, str, Indices]] = []

    for source, names, source_indices in zip(
        sources, names_per_source, indices_per_source, strict=False
    ):
        for name, idx in zip(names, source_indices, strict=False):
            work_items.append((source, name, idx))
    return work_items
