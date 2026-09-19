import math
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import Mapping, Sequence, TypedDict

import numpy as np


class CoreOverride(TypedDict, total=False):
    npin: int
    nax: int


type FileOverrides = dict[str, CoreOverride]

MIN_NUM_GROUPS = 2
MAX_NUM_GROUPS = 51
NUM_DF = 6
NUM_NODES = 4
LATERAL_SURFACES = slice(0, 4)


class VeraDim(StrEnum):
    PIN_Y = "pin_y"
    PIN_X = "pin_x"
    AXIAL = "axial"
    ASSEMBLY = "assembly"
    NODE = "node"
    GROUP = "group"
    SURFACE = "surface"


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

    # Physical axes that have no semantic meaning and are always fixed.
    # Maps axis -> required index.
    fixed_idxs: dict[int, int] = field(default_factory=dict)

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

        for axis in self.fixed_idxs:
            indices[f"fixed_idxs[{axis}]"] = axis

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

        for axis, value in self.fixed_idxs.items():
            if not isinstance(axis, int) or isinstance(axis, bool):
                raise TypeError("fixed axis must be an int")

            if axis < 0:
                raise ValueError("fixed axis must be non-negative")

            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError("fixed index value must be an int")

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
    COMP_PIN = 21
    ASSY_ENERGY = 22
    NODAL_ENERGY = 23
    NODAL_SURFACE = 24
    ASSY_SURFACE = 26

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

    @property
    def dim_axes(self) -> dict[VeraDim, int]:
        """Map semantic dimensions to physical NumPy axes."""
        info = self._info

        axes: dict[VeraDim, int] = {}

        if info.pin_idxs is not None:
            axes[VeraDim.PIN_Y] = info.pin_idxs[0]
            axes[VeraDim.PIN_X] = info.pin_idxs[1]

        if info.axial_idx is not None:
            axes[VeraDim.AXIAL] = info.axial_idx

        if info.assembly_id_idx is not None:
            axes[VeraDim.ASSEMBLY] = info.assembly_id_idx

        if info.node_dim_idx is not None:
            axes[VeraDim.NODE] = info.node_dim_idx

        if info.group_idx is not None:
            axes[VeraDim.GROUP] = info.group_idx

        if info.surface_idx is not None:
            axes[VeraDim.SURFACE] = info.surface_idx

        return axes


# per-dtype facts
_INFO = {
    VeraDtype.PIN: _Info(pin_idxs=(0, 1), axial_idx=2, assembly_id_idx=3, fuel_pin=True),
    VeraDtype.ASSEMBLY: _Info(fixed_idxs={0: 0}, axial_idx=1, assembly_id_idx=2, assembly=True),
    VeraDtype.AXIAL: _Info(axial_idx=0),
    VeraDtype.NODAL: _Info(node_dim_idx=0, axial_idx=1, assembly_id_idx=2, nodal=True),
    VeraDtype.RADIAL: _Info(pin_idxs=(0, 1), assembly_id_idx=2, fuel_pin=True),
    VeraDtype.SCALAR: _Info(),  # == CORE
    VeraDtype.RADIAL_ASSEMBLY: _Info(assembly_id_idx=0, assembly=True),
    VeraDtype.CHANNEL: _Info(pin_idxs=(0, 1), axial_idx=2, assembly_id_idx=3, channel=True),
    VeraDtype.CHANNEL_RADIAL: _Info(pin_idxs=(0, 1), assembly_id_idx=2, channel=True),
    VeraDtype.RADIAL_NODE: _Info(node_dim_idx=0, assembly_id_idx=1, nodal=True),
    VeraDtype.UNKNOWN: _Info(),
    VeraDtype.COMP_PIN: _Info(
        pin_idxs=(0, 1),
        axial_idx=2,
        assembly_id_idx=3,
        fuel_pin=True,
        computational=True,
    ),
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
        fixed_idxs={0: 0}, axial_idx=1, assembly_id_idx=2, computational=True, assembly=True
    ),
    VeraDtype.COMP_ASSY_ENERGY: _Info(
        group_idx=0,
        fixed_idxs={1: 0},
        axial_idx=2,
        assembly_id_idx=3,
        computational=True,
        assembly=True,
    ),
    VeraDtype.COMP_ASSY_SURFACE: _Info(
        surface_idx=0,
        group_idx=1,
        fixed_idxs={2: 0},
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
    VeraDtype.NODAL_ENERGY: _Info(
        group_idx=0,
        node_dim_idx=1,
        axial_idx=2,
        assembly_id_idx=3,
        nodal=True,
    ),
    VeraDtype.NODAL_SURFACE: _Info(
        surface_idx=0,
        group_idx=1,
        node_dim_idx=2,
        axial_idx=3,
        assembly_id_idx=4,
        nodal=True,
        surface=True,
    ),
    VeraDtype.ASSY_ENERGY: _Info(
        group_idx=0,
        fixed_idxs={1: 0},
        axial_idx=2,
        assembly_id_idx=3,
        assembly=True,
    ),
    VeraDtype.ASSY_SURFACE: _Info(
        surface_idx=0,
        group_idx=1,
        fixed_idxs={2: 0},
        axial_idx=3,
        assembly_id_idx=4,
        assembly=True,
        surface=True,
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

    for axis, value in vdtype._info.fixed_idxs.items():
        result[axis] = value

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


def point_indices(
    vdtype: "VeraDtype",
    shape: Sequence[int],
    selection: Mapping[VeraDim, int],
    split: Sequence[VeraDim] = (VeraDim.GROUP,),
) -> list[tuple[int, ...]]:
    """Physical index tuples for one point, one per combination of split dims."""
    axes = vdtype.dim_axes
    if not axes:
        if math.prod(shape) != 1:
            raise ValueError(f"{vdtype} has no semantic dims to index shape {tuple(shape)}")
        return [(0,) * len(shape)]

    split = [dim for dim in split if dim in axes]
    open_dims = set(axes) - set(selection) - set(split)
    if open_dims:
        raise ValueError(
            f"selection leaves {vdtype} dims unindexed: "
            + ", ".join(sorted(dim.value for dim in open_dims))
        )

    points = []
    for combo in np.ndindex(*(shape[axes[dim]] for dim in split)):
        chosen = {**selection, **dict(zip(split, combo, strict=True))}
        pin = (chosen[VeraDim.PIN_Y], chosen[VeraDim.PIN_X]) if VeraDim.PIN_Y in axes else None
        index = vdtype.make_slice(
            surface_idx=chosen.get(VeraDim.SURFACE),
            group_idx=chosen.get(VeraDim.GROUP),
            node_idx=chosen.get(VeraDim.NODE),
            pin_idxs=pin,
            axial_idx=chosen.get(VeraDim.AXIAL),
            assembly_id=chosen.get(VeraDim.ASSEMBLY),
        )
        points.append(tuple(int(i) for i in index))
    return points


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
        physical_units: str = "Unitless",
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
        self.physical_units: str = getattr(obj, "physical_units", "Unitless")
        self.name: str | None = getattr(obj, "name", None)

    def is_computational(self) -> bool:
        return self.dataset_type.is_computational()

    def is_assembly(self) -> bool:
        return self.dataset_type.is_assembly()

    def select(
        self,
        indices: Mapping[VeraDim, int] | None = None,
    ) -> "VeraDataset":
        """
        Select semantic dimensions from the dataset.

        Dimensions that do not exist on this dataset type are ignored.

        Unselected semantic dimensions retain their original physical order.

        Parameters
        ----------
        indices
            Mapping from semantic dimension to the index to select.

            Example::

                {
                    VeraDim.NODE: node_idx,
                    VeraDim.ASSEMBLY: assembly_id,
                }

        Returns
        -------
        VeraDataset
            The selected dataset.
        """
        indices = dict(indices or {})

        dtype = self.dataset_type
        axis_map = dtype.dim_axes

        selected = {dim: value for dim, value in indices.items() if dim in axis_map}

        pin_idxs = None

        if VeraDim.PIN_Y in selected and VeraDim.PIN_X in selected:
            pin_idxs = (
                selected[VeraDim.PIN_Y],
                selected[VeraDim.PIN_X],
            )
        elif VeraDim.PIN_Y in selected or VeraDim.PIN_X in selected:
            raise ValueError("PIN_Y and PIN_X must be selected together")

        slice_ = dtype.make_slice(
            surface_idx=selected.get(VeraDim.SURFACE),
            group_idx=selected.get(VeraDim.GROUP),
            node_idx=selected.get(VeraDim.NODE),
            pin_idxs=pin_idxs,
            axial_idx=selected.get(VeraDim.AXIAL),
            assembly_id=selected.get(VeraDim.ASSEMBLY),
        )
        selection = self[slice_]
        if not isinstance(selection, VeraDataset):
            return VeraDataset(
                np.asarray(selection), self.dataset_type, self.name, self.physical_units
            )
        return selection

    def arrange(
        self,
        *,
        order: Sequence[VeraDim],
        split: Sequence[VeraDim] = (),
        require: Sequence[VeraDim] = (),
        pad: Sequence[VeraDim] = (),
        surface: int | None = None,
        group: int | None = None,
        node: int | None = None,
        pin: tuple[int, int] | None = None,
        axial: int | None = None,
        assembly: int | None = None,
    ) -> list["VeraDataset"]:
        """
        Select, reorder, and split a dataset using semantic dimensions.

        Selectors for dimensions that do not exist on the dataset are ignored.

        Every surviving semantic dimension must be explicitly accounted for
        by either ``order`` or ``split``.

        Parameters
        ----------
        order
            Desired semantic dimension order for each returned array.

            Dimensions in ``order`` that do not exist on this dtype are
            ignored.

        split
            Semantic dimensions to split into separate arrays.

            Dimensions in ``split`` that do not exist on this dtype are
            ignored.

        require
            Semantic dimensions that must exist on this dtype.

        pad
            Dimensions from ``order`` to keep in the output as length-1 axes
            when the dtype lacks them or a selector removed them, so callers
            get one shape across dtypes (e.g. a node axis for assembly data).

        surface
            Surface index to select, if the dtype has a surface dimension.

        group
            Energy-group index to select, if the dtype has a group dimension.

        node
            Node index to select, if the dtype has a node dimension.

        pin
            ``(pin_y, pin_x)`` indices to select, if the dtype has pin
            dimensions.

        axial
            Axial index to select, if the dtype has an axial dimension.

        assembly
            Assembly index to select, if the dtype has an assembly dimension.

        Returns
        -------
        list[VeraDataset]
            One arranged array if no active split dimensions exist, otherwise
            one array for every combination of split-dimension indices.
        """
        order = tuple(order)
        split = tuple(split)
        require = tuple(require)
        pad = tuple(pad)

        dtype = self.dataset_type
        axis_map = dtype.dim_axes

        # validate

        if len(set(order)) != len(order):
            raise ValueError(f"Duplicate dimensions in order: {order}")

        if len(set(split)) != len(split):
            raise ValueError(f"Duplicate dimensions in split: {split}")

        if len(set(require)) != len(require):
            raise ValueError(f"Duplicate dimensions in require: {require}")

        if len(set(pad)) != len(pad):
            raise ValueError(f"Duplicate dimensions in pad: {pad}")

        if not set(pad) <= set(order):
            raise ValueError(
                "pad dimensions must also appear in order: "
                + ", ".join(sorted(dim.value for dim in set(pad) - set(order)))
            )

        overlap = set(order) & set(split)

        if overlap:
            raise ValueError(
                "Dimensions cannot appear in both order and split: "
                + ", ".join(sorted(dim.value for dim in overlap))
            )

        missing_required = set(require) - set(axis_map)

        if missing_required:
            raise ValueError(
                f"{dtype} is missing required dimensions: "
                + ", ".join(sorted(dim.value for dim in missing_required))
            )

        # build semantic axis selections, if an axis does not exist in this dataset ignore it
        requested: dict[VeraDim, int | None] = {
            VeraDim.SURFACE: surface,
            VeraDim.GROUP: group,
            VeraDim.NODE: node,
            VeraDim.AXIAL: axial,
            VeraDim.ASSEMBLY: assembly,
        }

        if pin is not None:
            requested[VeraDim.PIN_Y] = pin[0]
            requested[VeraDim.PIN_X] = pin[1]

        selected = {
            dim: value for dim, value in requested.items() if value is not None and dim in axis_map
        }

        data = self.select(selected)

        selected_dims = set(selected)

        surviving = tuple(
            dim
            for dim, _axis in sorted(
                axis_map.items(),
                key=lambda item: item[1],
            )
            if dim not in selected_dims
        )  # these are the dims that weren't selected, they must be in required or ordering

        active_order = tuple(dim for dim in order if dim in surviving)  # order dims

        active_split = tuple(dim for dim in split if dim in surviving)  # split dims

        accounted_for = set(active_order) | set(active_split)

        unhandled = set(surviving) - accounted_for

        if unhandled:
            raise ValueError(
                f"Arrangement for {dtype} leaves dimensions unhandled: "
                + ", ".join(sorted(dim.value for dim in unhandled))
            )

        desired = active_split + active_order

        permutation = tuple(surviving.index(dim) for dim in desired)

        zero_axis_scalar = not dtype.dim_axes and data.shape == (1,)
        if permutation != tuple(range(data.ndim)) and not zero_axis_scalar:
            data = data.transpose(permutation)

        padded = tuple(dim for dim in order if dim in active_order or dim in pad)
        missing = [len(active_split) + i for i, dim in enumerate(padded) if dim not in active_order]
        if missing:
            data = np.expand_dims(data, missing)

        if not active_split:
            return [data]

        split_shape = data.shape[: len(active_split)]

        return [
            data[idx]
            if len(idx) != data.ndim
            else VeraDataset(data[idx], data.dataset_type, data.name, data.physical_units)
            for idx in np.ndindex(split_shape)
        ]


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
        for n_groups in range(MIN_NUM_GROUPS, MAX_NUM_GROUPS + 1):
            shape_to_dtype |= {
                (n_groups, 1, nax, nass): VeraDtype.ASSY_ENERGY,
                (NUM_DF, n_groups, 1, nax, nass): VeraDtype.ASSY_SURFACE,
                (n_groups, NUM_NODES, nax, nass): VeraDtype.NODAL_ENERGY,
                (NUM_DF, n_groups, NUM_NODES, nax, nass): VeraDtype.NODAL_SURFACE,
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
    if comp_nax and comp_nass and (comp_nax != nax or comp_nass != nass):
        shape_to_dtype |= {
            (npiny, npinx, comp_nax, comp_nass): VeraDtype.COMP_PIN,
            (NUM_NODES, comp_nax, comp_nass): VeraDtype.COMP_NODAL,
            (1, comp_nax, comp_nass): VeraDtype.COMP_ASSY,
        }
        for n_groups in range(MIN_NUM_GROUPS, MAX_NUM_GROUPS + 1):
            shape_to_dtype |= {
                (n_groups, 1, comp_nax, comp_nass): VeraDtype.COMP_ASSY_ENERGY,
                (NUM_DF, n_groups, 1, comp_nax, comp_nass): VeraDtype.COMP_ASSY_SURFACE,
                (n_groups, NUM_NODES, comp_nax, comp_nass): VeraDtype.COMP_NODAL_ENERGY,
                (NUM_DF, n_groups, NUM_NODES, comp_nax, comp_nass): VeraDtype.COMP_NODAL_SURFACE,
            }
    return shape_to_dtype
