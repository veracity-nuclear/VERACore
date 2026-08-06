from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from vera_core.app.core import VeraDataset, VeraDtype


class FakeState(dict):
    """Small stand-in for trame State for pure unit tests."""

    def has(self, key: str) -> bool:
        return key in self

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


class FakeCore:
    def __init__(self, mesh=(10.0, 20.0, 30.0)):
        self.gross_axial_mesh = np.asarray(mesh, dtype=float)

    def get_axial_mesh_means(self, dataset_type=None):
        return self.gross_axial_mesh


class FakeSource:
    def __init__(
        self,
        *,
        path: str,
        mesh=(10.0, 20.0, 30.0),
        state_count: int = 1,
        time_axes: dict[str, list[float]] | None = None,
    ):
        self._file_path = path
        self._core = FakeCore(mesh)
        self._states = [SimpleNamespace() for _ in range(state_count)]
        self._active_state_index = 0
        self._time_axes = time_axes or {"state_count": list(range(state_count))}
        self.closed = False
        self.active_state_full_core_keys = []
        self.active_state_grouped_keys = []
        self.derived_calls = []
        self.diff_calls = []

    @property
    def file_path(self):
        return self._file_path

    @property
    def core(self):
        return self._core

    @property
    def states(self):
        return self._states

    @property
    def active_state_index(self):
        return self._active_state_index

    @active_state_index.setter
    def active_state_index(self, value):
        self._active_state_index = value

    def close(self):
        self.closed = True

    def time_axes(self):
        return self._time_axes

    def array_dtype(self, name):
        return VeraDtype.PIN if name == "pin_powers" else VeraDtype.UNKNOWN

    def add_new_derived_dataset(self, **kwargs):
        self.derived_calls.append(kwargs)

    def add_new_diff_dataset(self, *args):
        self.diff_calls.append(args)


@pytest.fixture
def pin_dataset():
    return VeraDataset(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        VeraDtype.PIN,
        "W/cm3",
    )
