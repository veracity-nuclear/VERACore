"""In-memory cores, states and sources built only from dtypes and model.

Unlike readers.mock, nothing here is shared with production code, so these
builders cannot mask a reader bug. Default core: quarter-symmetric 5x5 map
(reduced map 1..9), 3x3 pins, 5 levels over 0..100 (means 10, 30, ..., 90).
nax=5 avoids the NODAL / CHANNEL_RADIAL shape collision that nax=4 causes.
"""

import numpy as np

from vera_core.data.dtypes import VeraDataset, VeraDtype
from vera_core.data.model import DatasetSource, VeraDataSource, VeraOutCore, VeraOutState

from .conftest import FakeCalculator
from .slice_support import quarter_map

NPIN, NAX, NASS = 3, 5, 9


class TypedSource(DatasetSource):
    """Dict-backed source. Loads are tagged by shape and counted."""

    def __init__(self, arrays, shape_to_dtype=None, units=None):
        self.arrays = {k: np.asarray(v) for k, v in arrays.items()}
        self.shape_to_dtype = shape_to_dtype or {}
        self.units = units or {}
        self.loads = {}

    @property
    def provenance(self):
        return "<test>"

    def names(self):
        return list(self.arrays)

    def shape(self, name):
        return self.arrays[name].shape if name in self.arrays else None

    def load(self, name):
        if name not in self.arrays:
            return None
        self.loads[name] = self.loads.get(name, 0) + 1
        data = self.arrays[name]
        dtype = self.shape_to_dtype.get(data.shape, VeraDtype.UNKNOWN)
        return VeraDataset(data.copy(), dtype, name, self.units.get(name, "Unitless"))


def core_arrays(npin=NPIN, nax=NAX, **extra):
    arrays = {
        "core_map": quarter_map(),
        "core_sym": np.array([4]),
        "axial_mesh": None if nax is None else np.linspace(0.0, 100.0, nax + 1),
        "npin": None if npin is None else np.array([npin]),
    }
    return {k: v for k, v in (arrays | extra).items() if v is not None}


def make_core(**kwargs):
    """kwargs go to core_arrays; pass a key as None to omit it."""
    return VeraOutCore(TypedSource(core_arrays(**kwargs)))


def state_arrays(core, exposure, seed=0):
    rng = np.random.default_rng(seed)
    npy, npx, nax, nass = core.core_shape
    return {
        "pin_powers": rng.uniform(0.5, 1.5, (npy, npx, nax, nass)),
        "axial_powers": rng.uniform(0.5, 1.5, nax),
        "keff": np.array([1.0 + exposure / 1000]),
        "exposure": np.array([float(exposure)]),
    }


def make_state(core, exposure=0.0, idx=0, **extra):
    arrays = state_arrays(core, exposure, seed=idx) | extra
    source = TypedSource(arrays, core.shape_to_dtype, units={"pin_powers": "W"})
    return VeraOutState(source, idx, core)


def make_source(n_states=3, *, core=None, calculator=True, **kwargs):
    core = core or make_core()
    states = [make_state(core, 10.0 * i, i) for i in range(n_states)]
    src = VeraDataSource(core=core, states=states, provenance="<test>", **kwargs)
    if calculator:
        src.vera_calculator = FakeCalculator()
    return src
