import threading
from collections import OrderedDict

import numpy as np


class RomBasis:
    """Projection basis shared by every state of a dataset source that contains reduced states to be mapped to full state."""

    def __init__(
        self,
        basis_matrix: np.ndarray,
        full_state_shape: np.ndarray,
        derived_datasets: dict[dict[str : int | None]],
        cache_size: int = 4,
    ):
        self._flat_basis = basis_matrix
        self._full_state_shape = full_state_shape
        self._derived_dataset_shapes = {
            name: tuple(
                int(ndim)
                for dim_idx, ndim in enumerate(full_state_shape)
                if dim_idx != info["full_state_axis"]
            )
            for name, info in derived_datasets.items()
        }
        self._derived_datasets = derived_datasets
        self._names = list(derived_datasets)
        self._cache: OrderedDict = OrderedDict()
        self._cache_size = cache_size
        self._lock = threading.Lock()
        self._next_token = 0
        expected_rows = int(np.prod(full_state_shape))

        if basis_matrix.shape[0] != expected_rows:
            raise ValueError(
                f"Basis has {basis_matrix.shape[0]} rows, "
                f"but full_state_shape {full_state_shape} requires "
                f"{expected_rows} rows"
            )

    def names(self) -> list[str]:
        return self._names

    def shape(self, name: str) -> tuple[int, ...] | None:
        """Shape of the expanded dataset, without expanding it."""
        return self._derived_dataset_shapes.get(name)

    def new_token(self) -> int:
        """Cache identity for one state. Not reused, so entries cannot collide."""
        with self._lock:
            token = self._next_token
            self._next_token += 1
        return token

    def flat_index(self, name: str, indices) -> int | None:
        """Row of the full-state basis addressed by derived-dataset indices."""
        shape = self.shape(name)
        if shape is None:
            return None

        idx = indices if isinstance(indices, tuple) else (indices,)

        if len(idx) != len(shape):
            return None

        info = self._derived_datasets[name]
        axis = info.get("full_state_axis")
        dataset_idx = info.get("full_state_idx")

        if axis is not None:
            full_idx = list(idx)
            full_idx.insert(axis, dataset_idx)
            full_idx = tuple(full_idx)
        else:
            full_idx = idx

        try:
            return int(
                np.ravel_multi_index(
                    full_idx,
                    self._full_state_shape,
                )
            )
        except (ValueError, TypeError):
            return None

    def sample(self, name: str, flat_index: int, reduced_state, scale: float) -> float:
        """One element of the expansion, a dot product against one row."""
        return float(self._flat_basis[flat_index] @ reduced_state * scale)

    def expand(self, token: int, name: str, reduced_state, scale: float) -> np.ndarray:
        key = (token, name)
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        idx = self._derived_datasets[name].get("full_state_idx")
        axis = self._derived_datasets[name].get("full_state_axis")
        array = np.reshape(self._flat_basis @ reduced_state * scale, self._full_state_shape)
        if axis is not None and idx is not None:
            array = np.take(array, indices=idx, axis=axis)
        array.flags.writeable = False
        with self._lock:
            self._cache[key] = array
            self._cache.move_to_end(key)
            while len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return array
