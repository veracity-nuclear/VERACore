import tempfile, os
import numpy as np
from .vera_data import VeraDataSource
from .vera_out_file import VeraOutFile

class VeraDataRegistry:
    """Holds the open data sources keyed by id, with one marked as default.

    Provides lookup, the shared maximum state index, and fan-out of the active
    state across all sources.
    """
    def __init__(self):
        """Create an empty registry with no default source."""
        self._srcs : dict[str, VeraDataSource] = {}
        self.default_src_id : str = None
    
    def add_src(self, src : VeraDataSource, src_id : str):
        """Register a source under src_id, making it default if it's the first.
        Raises ValueError if src_id is already registered.
        """
        if src_id in self._srcs:
            raise ValueError(f"{src_id} already exsists in the registry, skipped adding")
        if self.default_src_id is None:
            self.default_src_id = src_id
        self._srcs[src_id] = src

    @property
    def default_src(self) -> VeraDataSource | None:
        """The default source, or None if the registry is empty."""
        if self.default_src_id is None:
            return None
        return self._srcs[self.default_src_id]
    
    @property
    def max_state(self) -> int:
        """Largest valid state index across all sources, or 0 if none."""
        if not self._srcs:
            return 0
        return max(max(len(src.states) for src in self._srcs.values()) - 1, 0)

    def get(self, src_id: str) -> VeraDataSource | None:
        """Return the source for src_id, or None if it isn't registered."""
        return self._srcs.get(src_id)
    
    def src_ids(self):
        """Return a view of all registered source ids."""
        return self._srcs.keys()
    
    def change_active_state(self, src_id : str, nstate : int) -> None:
        """Set the active state index for one source. Raises ValueError if src_id isn't registered."""
        if src_id not in self._srcs:
            raise ValueError(f"Could not find {src_id} in registry")
        self._srcs[src_id].active_state_index = nstate

    def change_all_active_state(self, nstate: int) -> None:
        """Set the active state index on every registered source."""
        for src in self._srcs.values():
            src.active_state_index = nstate
    
    def full_core_keys(self) -> dict[str, list[str]]:
        """Map each source id to its active state's full-core dataset names."""
        full_core_keys = {src_id : self._srcs[src_id].active_state_full_core_keys for src_id in self._srcs.keys()}
        return full_core_keys
    
    def shared_time_axes(self):
        if not self._srcs:
            return
        srcs = iter(self._srcs.values())
        shared_axes = set(next(srcs).time_axes())
        for src in srcs:
            shared_axes &= set(src.time_axes())
        return sorted(shared_axes)

    def remove_src(self, src_id: str) -> None:
        """Remove a source, close its file handles, and reassigns the default. Raises ValueError if src_id isn't registered."""
        if src_id not in self._srcs:
            raise ValueError(f"Could not find {src_id} in registry")
        src = self._srcs.pop(src_id)
        src.close()
        if self.default_src_id == src_id:
            self.default_src_id = next(iter(self._srcs), None)