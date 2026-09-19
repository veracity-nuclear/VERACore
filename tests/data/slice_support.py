"""Shared fixtures for slice tests: a real VeraOutCore (3x3 pins, 5 levels,
quarter-symmetric 5x5 map, computational map, zero-volume pins) and a
VeraDataSource holding one random dataset per dtype, named by the dtype."""

import dataclasses

import numpy as np

from vera_core.data.dtypes import NUM_DF, NUM_NODES, VeraDataset, VeraDtype
from vera_core.data.model import DatasetSource, VeraDataSource, VeraOutCore

T = VeraDtype
NPIN, NAX, NG = 3, 5, 2  # nax != NUM_NODES avoids the documented shape collision


class DictSource(DatasetSource):
    def __init__(self, arrays):
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}

    @property
    def provenance(self):
        return "<test>"

    def names(self):
        return list(self.arrays)

    def shape(self, name):
        return self.arrays[name].shape if name in self.arrays else None

    def load(self, name):
        return VeraDataset(self.arrays[name], name=name) if name in self.arrays else None


def quarter_map(n=5):
    half = n // 2
    q = np.arange(1, (half + 1) ** 2 + 1).reshape(half + 1, half + 1)
    full = np.zeros((n, n), dtype=int)
    full[half:, half:] = q
    full[: half + 1, half:] = q[::-1][: half + 1]
    full[half:, : half + 1] = q[:, ::-1][:, : half + 1]
    full[: half + 1, : half + 1] = q[::-1, ::-1]
    return full


def build_core():
    vols = np.ones((NPIN, NPIN, NAX, 9))
    vols[0, 0] = 0.0
    vols[2, 2, 1] = 0.0
    return VeraOutCore(
        DictSource(
            {
                "core_map": quarter_map(),
                "core_sym": np.array([4]),
                "axial_mesh": np.linspace(0.0, 100.0, NAX + 1),
                "npin": np.array([NPIN]),
                "pin_volumes": vols,
                "computational_core_map": quarter_map(),
                "computational_axial_mesh": np.linspace(0.0, 100.0, NAX + 1),
            }
        )
    )


CORE = build_core()
assert CORE.has_comp_core()
NASS = CORE.nass
SHAPES = {
    T.PIN: (NPIN, NPIN, NAX, NASS),
    T.COMP_PIN: (NPIN, NPIN, NAX, NASS),
    T.CHANNEL: (NPIN + 1, NPIN + 1, NAX, NASS),
    T.RADIAL: (NPIN, NPIN, NASS),
    T.CHANNEL_RADIAL: (NPIN + 1, NPIN + 1, NASS),
    T.ASSEMBLY: (1, NAX, NASS),
    T.COMP_ASSY: (1, NAX, NASS),
    T.ASSY_ENERGY: (NG, 1, NAX, NASS),
    T.COMP_ASSY_ENERGY: (NG, 1, NAX, NASS),
    T.ASSY_SURFACE: (NUM_DF, NG, 1, NAX, NASS),
    T.COMP_ASSY_SURFACE: (NUM_DF, NG, 1, NAX, NASS),
    T.RADIAL_ASSEMBLY: (NASS,),
    T.NODAL: (NUM_NODES, NAX, NASS),
    T.COMP_NODAL: (NUM_NODES, NAX, NASS),
    T.NODAL_ENERGY: (NG, NUM_NODES, NAX, NASS),
    T.COMP_NODAL_ENERGY: (NG, NUM_NODES, NAX, NASS),
    T.NODAL_SURFACE: (NUM_DF, NG, NUM_NODES, NAX, NASS),
    T.COMP_NODAL_SURFACE: (NUM_DF, NG, NUM_NODES, NAX, NASS),
    T.RADIAL_NODE: (NUM_NODES, NASS),
    T.AXIAL: (NAX,),
    T.POINT_DETECTOR: (NAX, NASS),
    T.CONTINOUS_DETECTOR: (NAX, NASS),
    T.RADIAL_POINT_DETECTOR: (NASS,),
    T.SCALAR: (1,),
}


class FakeSource(VeraDataSource):
    """Real core, in-memory datasets named by dtype."""

    name = "src"
    active_state_index = 0
    states = [{"exposure": np.array([1.5])}]
    core = CORE

    def __init__(self):
        rng = np.random.default_rng(0)
        self.data = {
            str(t): VeraDataset(rng.normal(size=s), t, str(t), "W") for t, s in SHAPES.items()
        }

    def get_dataset(self, name, *, state_idx=None, mask_reflected=True):
        return self.data[name].copy()

    def get_dataset_dtype(self, name, state_idx=None):
        return self.data[name].dataset_type


SRC = FakeSource()


def run(build):
    try:
        return build()
    except Exception as exc:  # the original raises for some allowed dtypes
        return exc


def equal(a, b):
    if isinstance(a, Exception) or isinstance(b, Exception):
        return type(a) is type(b)
    if a is None or b is None:
        return a is b
    if dataclasses.is_dataclass(a):
        fields = [f.name for f in dataclasses.fields(a)]
        extra = [m for m in ("serialize_data_groups", "serialize_dataset_groups") if hasattr(a, m)]
        return (
            type(a).__name__ == type(b).__name__
            and all(equal(getattr(a, f), getattr(b, f)) for f in fields)
            and all(equal(getattr(a, m)(), getattr(b, m)()) for m in extra)
        )
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, (np.ndarray, float, int, np.number)):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and np.array_equal(a, b, equal_nan=a.dtype.kind == "f")
    return a == b
