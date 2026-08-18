from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TypedDict

import numpy as np


class CoreOverride(TypedDict, total=False):
    npin: int
    nax: int


type FileOverrides = dict[str, CoreOverride]

NUM_ENERGY_GROUPS = 2
MAX_NUM_GROUPS = 8
NUM_DF = 6
NUM_NODES = 4
LATERAL_SURFACES = slice(0, 4)


@dataclass(frozen=True)
class _Info:
    assembly_id_idx: int | None = None  # None = no assembly axis
    axial_idx: int | None = None  # None = no axial axis
    group_idx: int | None = None  # None = no energy groups
    pin_idxs: tuple[int, int] | None = None  # None = no pin dim
    node_dim_idx: int | None = None  # None = no nodal dim
    surface_idx: int | None = None  # None = no surface dim
    fuel_pin: bool = False
    computational: bool = False
    nodal: bool = False
    assembly: bool = False
    surface: bool = False
    channel: bool = False
    detector: bool = False

    ndim: int = field(init=False)

    def __post_init__(self):
        # Collect all dimension indices.
        indices = {
            "assembly_id_idx": self.assembly_id_idx,
            "axial_idx": self.axial_idx,
            "group_idx": self.group_idx,
            "node_dim_idx": self.node_dim_idx,
            "surface_idx": self.surface_idx,
        }

        if self.pin_idxs is not None:
            if len(self.pin_idxs) != 2:
                raise ValueError(f"pin_idxs must contain exactly 2 indices, got {self.pin_idxs!r}")

            indices["pin_idxs[0]"] = self.pin_idxs[0]
            indices["pin_idxs[1]"] = self.pin_idxs[1]

        # Remove dimensions that don't exist.
        indices = {name: idx for name, idx in indices.items() if idx is not None}

        # Validate index types and values.
        for name, idx in indices.items():
            if not isinstance(idx, int) or isinstance(idx, bool):
                raise TypeError(f"{name} must be an int or None, got {type(idx).__name__}")

            if idx < 0:
                raise ValueError(f"{name} must be non-negative, got {idx}")

        ndim = len(indices)

        # Every axis must have a unique index.
        if len(set(indices.values())) != ndim:
            duplicates = {
                idx: [name for name, value in indices.items() if value == idx]
                for idx in set(indices.values())
                if list(indices.values()).count(idx) > 1
            }

            raise ValueError(f"Dimension indices must be unique; duplicates: {duplicates}")

        # Indices should describe exactly ndim axes:
        # e.g. ndim=3 -> {0, 1, 2}
        expected = set(range(ndim))
        actual = set(indices.values())

        if actual != expected:
            raise ValueError(
                f"Dimension indices must be contiguous from 0 to {ndim - 1}; "
                f"expected {sorted(expected)}, got {sorted(actual)}"
            )

        object.__setattr__(self, "ndim", ndim)


class VeraDtype(Enum):
    """Dataset Identifiers"""

    PIN = 1
    ASSEMBLY = 2
    AXIAL = 3
    NODAL = 4
    RADIAL = 5
    SCALAR = 6
    CORE = 6  # CORE is an alias for SCALAR
    RADIAL_ASSEMBLY = 7
    CHANNEL = 8
    CHANNEL_RADIAL = 9
    RADIAL_NODE = 10
    UNKNOWN = 11
    # COMP_ prefix means it uses computational core map for shape
    COMP_NODAL = 12
    COMP_NODAL_ENERGY = 13
    COMP_NODAL_SURFACE = 14
    COMP_ASSY = 15
    COMP_ASSY_ENERGY = 16
    COMP_ASSY_SURFACE = 17
    POINT_DETECTOR = 18
    RADIAL_POINT_DETECTOR = 19
    CONTINOUS_DETECTOR = 20

    def __str__(self):
        return self.name

    @property
    def str(self):
        return self.name

    @property
    def title(self):
        return str(self).upper()

    @property
    def _info(self):
        return _INFO[self]

    @property
    def axial_dim_idx(self) -> int:
        idx = self._info.axial_idx
        if idx is None:
            raise ValueError(f"{self} has no axial dimension")
        return idx

    @property
    def energy_group_dim_idx(self) -> int:
        idx = self._info.group_idx
        if idx is None:
            raise ValueError(f"{self} has no energy group dimension")
        return idx

    @property
    def pin_dim_idxs(self) -> tuple[int, int]:
        idxs = self._info.pin_idxs
        if idxs is None:
            raise ValueError(f"{self} has no pin dimensions")
        return idxs

    @property
    def node_dim_idx(self) -> int:
        idx = self._info.node_dim_idx
        if idx is None:
            raise ValueError(f"{self} has no dim dimensions")
        return idx

    @property
    def assembly_id_dim_idx(self) -> int:
        idx = self._info.assembly_id_idx
        if idx is None:
            raise ValueError(f"{self} has no assembly id dimensions")
        return idx

    @property
    def surface_dim_idx(self) -> int:
        idx = self._info.surface_idx
        if idx is None:
            raise ValueError(f"{self} has no surface dimensions")
        return idx

    @property
    def ndim(self) -> int:
        return self._info.ndim

    def has_axial_dim(self):
        return self._info.axial_idx is not None

    def has_energy_group_dim(self):
        return self._info.group_idx is not None

    def has_node_dim(self):
        return self._info.node_dim_idx is not None

    def has_pin_level_dim(self):
        return self._info.pin_idxs is not None

    def has_surface_dim(self):
        return self._info.surface_idx is not None

    def has_assembly_id_dim(self):
        return self._info.assembly_id_idx is not None

    def has_fuel_pins(self):
        return self._info.fuel_pin

    def is_computational(self):
        return self._info.computational

    def is_nodal(self):
        return self._info.nodal

    def is_assembly(self):
        return self._info.assembly

    def is_channel(self):
        return self._info.channel

    def is_detector(self):
        return self._info.detector

    def make_slice(
        self,
        *,
        surface_idx: int | None = None,
        group_idx: int | None = None,
        node_idx: int | None = None,
        pin_idxs: tuple[int, int] | None = None,
        axial_idx: int | None = None,
        assembly_id: int | None = None,
    ) -> tuple[int | slice, ...]:
        return make_slice(
            self,
            surface_idx=surface_idx,
            group_idx=group_idx,
            node_idx=node_idx,
            pin_idxs=pin_idxs,
            axial_idx=axial_idx,
            assembly_id=assembly_id,
        )


# the single place per-dtype facts are declared
_INFO = {
    VeraDtype.PIN: _Info(pin_idxs=(0, 1), axial_idx=2, assembly_id_idx=3, fuel_pin=True),
    VeraDtype.ASSEMBLY: _Info(node_dim_idx=0, axial_idx=1, assembly_id_idx=2, assembly=True),
    VeraDtype.AXIAL: _Info(axial_idx=0),
    VeraDtype.NODAL: _Info(node_dim_idx=0, axial_idx=1, assembly_id_idx=2, nodal=True),
    VeraDtype.RADIAL: _Info(pin_idxs=(0, 1), assembly_id_idx=2, fuel_pin=True),
    VeraDtype.SCALAR: _Info(),  # == CORE
    VeraDtype.RADIAL_ASSEMBLY: _Info(assembly_id_idx=0, assembly=True),
    VeraDtype.CHANNEL: _Info(pin_idxs=(0, 1), axial_idx=2, assembly_id_idx=3, channel=True),
    VeraDtype.CHANNEL_RADIAL: _Info(pin_idxs=(0, 1), assembly_id_idx=2, channel=True),
    VeraDtype.RADIAL_NODE: _Info(node_dim_idx=0, assembly_id_idx=1, nodal=True),
    VeraDtype.UNKNOWN: _Info(),
    VeraDtype.COMP_NODAL: _Info(
        node_dim_idx=0, axial_idx=1, assembly_id_idx=2, computational=True, nodal=True
    ),
    VeraDtype.COMP_NODAL_ENERGY: _Info(
        group_idx=0, node_dim_idx=1, axial_idx=2, assembly_id_idx=3, computational=True, nodal=True
    ),
    VeraDtype.COMP_NODAL_SURFACE: _Info(
        surface_idx=0,
        group_idx=1,
        node_dim_idx=2,
        axial_idx=3,
        assembly_id_idx=4,
        computational=True,
        nodal=True,
        surface=True,
    ),
    VeraDtype.COMP_ASSY: _Info(
        node_dim_idx=0, axial_idx=1, assembly_id_idx=2, computational=True, assembly=True
    ),
    VeraDtype.COMP_ASSY_ENERGY: _Info(
        group_idx=0,
        node_dim_idx=1,
        axial_idx=2,
        assembly_id_idx=3,
        computational=True,
        assembly=True,
    ),
    VeraDtype.COMP_ASSY_SURFACE: _Info(
        surface_idx=0,
        group_idx=1,
        node_dim_idx=2,
        axial_idx=3,
        assembly_id_idx=4,
        computational=True,
        assembly=True,
        surface=True,
    ),
    VeraDtype.POINT_DETECTOR: _Info(axial_idx=0, assembly_id_idx=1, assembly=True, detector=True),
    VeraDtype.RADIAL_POINT_DETECTOR: _Info(assembly_id_idx=0, assembly=True, detector=True),
    VeraDtype.CONTINOUS_DETECTOR: _Info(
        axial_idx=0, assembly_id_idx=1, assembly=True, detector=True
    ),
}


def make_slice(
    vdtype: "VeraDtype",
    *,
    surface_idx: int | None = None,
    group_idx: int | None = None,
    node_idx: int | None = None,
    pin_idxs: tuple[int, int] | None = None,
    axial_idx: int | None = None,
    assembly_id: int | None = None,
) -> tuple[int | slice, ...]:
    result = [slice(None)] * vdtype.ndim

    dims = [
        (vdtype._info.surface_idx, surface_idx),
        (vdtype._info.group_idx, group_idx),
        (vdtype._info.node_dim_idx, node_idx),
        (vdtype._info.axial_idx, axial_idx),
        (vdtype._info.assembly_id_idx, assembly_id),
    ]

    for dim, value in dims:
        if dim is not None and value is not None:
            result[dim] = value

    if vdtype._info.pin_idxs is not None and pin_idxs is not None:
        for dim, value in zip(vdtype._info.pin_idxs, pin_idxs, strict=True):
            result[dim] = value

    return tuple(result)


class VeraAxes(Enum):
    """Derivation Axes"""

    ASSEMBLY = 1
    AXIAL = 2
    RADIAL = 3
    CORE = 4
    NODE = 5
    RADIAL_ASSEMBLY = 6
    RADIAL_NODE = 7


class Surface(Enum):
    WEST = 0
    NORTH = 1
    EAST = 2
    SOUTH = 3
    TOP = 4
    BOTTOM = 5

    @property
    def str(self):
        return self.name


class DerivationMethod(StrEnum):
    AVERAGE = "Average"
    STDDEV = "Standard Deviation"
    RMS = "Root Mean Square"


class VeraDataset(np.ndarray):
    """Numpy array tagged with a vera dtype"""

    def __new__(
        cls,
        data,
        dataset_type: VeraDtype = VeraDtype.UNKNOWN,
        name: str | None = None,
        physical_units: str = "unitless",
    ):
        obj = np.asarray(data).view(cls)
        obj.dataset_type = dataset_type
        obj.physical_units = physical_units
        obj.name = name
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.dataset_type: VeraDtype = getattr(obj, "dataset_type", VeraDtype.UNKNOWN)
        self.physical_units: str = getattr(obj, "physical_units", "unitless")
        self.name: str | None = getattr(obj, "name", None)

    def is_computational(self) -> bool:
        return self.dataset_type.is_computational()

    def is_assembly(self) -> bool:
        return self.dataset_type.is_assembly()


def derive_recipe(
    src_id, source_array, name, method, axes, use_factors, exclude_non_fuel_rods
) -> dict:
    return {
        "kind": "derive",
        "src_id": src_id,
        "source_array": source_array,
        "name": name,
        "method": method,
        "axes": axes,
        "use_factor": use_factors,
        "exclude_fuel_rods": exclude_non_fuel_rods,
    }


def diff_recipe(
    ref_src_id,
    ref_array,
    comp_src_id,
    comp_array,
    name,
    interp_degree,
    ref_scale,
    comp_scale,
    units,
) -> dict:
    return {
        "kind": "diff",
        "ref_src_id": ref_src_id,
        "ref_array": ref_array,
        "comp_src_id": comp_src_id,
        "comp_array": comp_array,
        "name": name,
        "interp_degree": interp_degree,
        "ref_scale": ref_scale,
        "comp_scale": comp_scale,
        "units": units,
    }


def build_core_dtypes(
    npiny: int | None = None,
    npinx: int | None = None,
    nax: int | None = None,
    nass: int | None = None,
    comp_nax: int | None = None,
    comp_nass: int | None = None,
    ndet: int | None = None,
    ndax: int | None = None,
    continous_det: int | None = None,
) -> dict[tuple[int, ...], VeraDtype]:
    """Creates and returns a dict mapping dataset shapes to dataset identifier (enums)"""
    shape_to_dtype = {}
    if nax and nass:
        shape_to_dtype |= {
            (1, nax, nass): VeraDtype.ASSEMBLY,
            (nax,): VeraDtype.AXIAL,
            (nass,): VeraDtype.RADIAL_ASSEMBLY,
            (NUM_NODES, nax, nass): VeraDtype.NODAL,
            (NUM_NODES, nass): VeraDtype.RADIAL_NODE,
            (1,): VeraDtype.SCALAR,
            (): VeraDtype.SCALAR,
        }
    if ndet and ndax and (ndet != nass or ndax != ndet):
        shape_to_dtype |= {
            (ndax, ndet): VeraDtype.CONTINOUS_DETECTOR
            if continous_det
            else VeraDtype.POINT_DETECTOR,
        }
    if ndet and ndet != nass:
        shape_to_dtype |= {
            (ndet,): VeraDtype.RADIAL_POINT_DETECTOR,
        }
    if npiny and npinx and nax and nass:
        shape_to_dtype |= {
            (npiny, npinx, nax, nass): VeraDtype.PIN,
            (npiny + 1, npinx + 1, nax, nass): VeraDtype.CHANNEL,
            (npiny, npinx, nass): VeraDtype.RADIAL,
            (npiny + 1, npinx + 1, nass): VeraDtype.CHANNEL_RADIAL,
        }
    if comp_nax and comp_nass:
        shape_to_dtype |= {
            (NUM_NODES, comp_nax, comp_nass): VeraDtype.COMP_NODAL,
            (
                NUM_DF,
                NUM_ENERGY_GROUPS,
                NUM_NODES,
                comp_nax,
                comp_nass,
            ): VeraDtype.COMP_NODAL_SURFACE,
            (NUM_DF, 8, NUM_NODES, comp_nax, comp_nass): VeraDtype.COMP_NODAL_SURFACE,
            (
                NUM_ENERGY_GROUPS,
                NUM_NODES,
                comp_nax,
                comp_nass,
            ): VeraDtype.COMP_NODAL_ENERGY,
            (8, NUM_NODES, comp_nax, comp_nass): VeraDtype.COMP_NODAL_ENERGY,
            (1, comp_nax, comp_nass): VeraDtype.COMP_ASSY,
            (
                NUM_DF,
                NUM_ENERGY_GROUPS,
                1,
                comp_nax,
                comp_nass,
            ): VeraDtype.COMP_ASSY_SURFACE,
            (NUM_DF, 8, 1, comp_nax, comp_nass): VeraDtype.COMP_ASSY_SURFACE,
            (NUM_ENERGY_GROUPS, 1, comp_nax, comp_nass): VeraDtype.COMP_ASSY_ENERGY,
            (8, 1, comp_nax, comp_nass): VeraDtype.COMP_ASSY_ENERGY,
        }
    return shape_to_dtype
