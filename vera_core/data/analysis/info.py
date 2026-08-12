from dataclasses import dataclass

import numpy as np

from ..dtypes import VeraDataset
from ..model import VeraDataSource

MISSING_EXPOSURE = "not recorded"


@dataclass(frozen=True)
class Info:
    exposure: float | None = None
    assembly_label: str | None = None
    elevation: float | None = None

    def caption(self) -> str:
        """The footer line.

        Example: ``Exposure 52.413 · (A-6) · Axial - 53.51``
        """
        exposure = MISSING_EXPOSURE if self.exposure is None else f"{self.exposure:g}"
        parts = [f"Exposure {exposure}"]
        if self.assembly_label is not None:
            parts.append(f"({self.assembly_label})")
        if self.elevation is not None:
            parts.append(f"Axial - {self.elevation:g}")
        return " · ".join(parts)


def _elevation(axial_mesh_means, z: int) -> float | None:
    """Height of axial level z, or None when z falls outside the mesh."""
    mesh = np.asarray(axial_mesh_means, dtype=float)
    if mesh.ndim == 2:
        mesh = mesh.mean(axis=1)
    if mesh.ndim != 1 or not 0 <= z < mesh.shape[0]:
        return None
    return float(np.round(mesh[z], 2))


def _exposure(source: VeraDataSource, state_idx: int) -> float | None:
    """Burnup of one state point, or None when the source does not record it.

    Reads the requested state rather than the active one, so a headless render
    of a non-active state reports its own exposure.
    """
    exposure = source.states[state_idx].get("exposure", None)
    if exposure is None or len(exposure) == 0:
        return None
    return float(np.round(exposure[0], 3))


@dataclass(frozen=True)
class GridPosition:
    """A position on the reduced core map"""

    row: int
    col: int

    @classmethod
    def from_ij(cls, i: int, j: int) -> "GridPosition":
        """Build from the vera convention, where i (x) is the column."""
        return cls(row=int(j), col=int(i))

    @property
    def ij(self) -> dict[str, int]:
        """The trame convention, for writing back to state."""
        return {"i": self.col, "j": self.row}


def create_info(
    source: VeraDataSource,
    request,
) -> Info:
    """Resolve the caption values for one request.

    Unknown values become None rather than raising, so a caption still renders
    when a source omits exposure or the axial mesh is short.
    """
    if not hasattr(request, "array"):
        raise RuntimeError
    if isinstance(request.array, str):
        dtype = source.get_dataset_dtype(request.array, request.state)
    elif isinstance(request.array, VeraDataset):
        dtype = request.array.dataset_type
    else:
        raise RuntimeError
    exposure = None
    elevation = None
    assembly_label = None
    if hasattr(request, "state"):
        exposure = _exposure(source, request.state)
    if hasattr(request, "z"):
        mesh = source.core.get_axial_mesh_means(dataset_type=dtype)
        elevation = _elevation(mesh, request.z)
    if hasattr(request, "assembly"):
        assembly_label = source.core.reduced_core_map_label(
            request.assembly, dtype.is_computational()
        )
    return Info(exposure=exposure, elevation=elevation, assembly_label=assembly_label)
