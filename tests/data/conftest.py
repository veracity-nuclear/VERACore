"""Fixtures shared by the data-layer tests.

Sources come from vera_core.data.readers.mock, so nothing touches disk except
the H5 reader tests. Derivation needs a VERAout calculator, which mock sources
lack; FakeCalculator stands in with plain NaN-aware means over PIN-shaped data.
"""

import numpy as np
import pytest

from vera_core.data.readers.mock import build_source


class FakeCalculator:
    """Mimics the VERAout reductions used by VeraDataSource._run_avg_over_axes.

    Input is PIN shaped: (npy, npx, nax, nass).
    """

    def __init__(self):
        self.closed = False
        self.h5f = self

    def close(self):
        self.closed = True

    def Average(self, data):
        return np.nanmean(data)

    def Axial(self, data):
        return np.nanmean(data, axis=(0, 1, 3))

    def Assembly(self, data):
        return np.nanmean(data, axis=(0, 1))

    def Radial(self, data):
        return np.nanmean(data, axis=2)

    def Radial_Assembly(self, data):
        return np.nanmean(data, axis=(0, 1, 2))

    def Node(self, data):
        _, _, nax, nass = data.shape
        return np.zeros((4, nax, nass))


def make_source(n_states: int = 3, *, calculator: bool = True, **kwargs):
    """build_source plus an optional FakeCalculator for derivation."""
    src = build_source(n_states, **kwargs)
    if calculator:
        src.vera_calculator = FakeCalculator()
    return src


@pytest.fixture
def source():
    """Three states, quarter-symmetric 5x5 map (9 assemblies), 3x3 pins, 4 levels."""
    return make_source()
