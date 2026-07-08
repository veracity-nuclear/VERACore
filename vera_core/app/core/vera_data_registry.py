import tempfile, os
import numpy as np
from .vera_data import VeraDataSource, VeraOutCore, VeraDataset, VeraDtype
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
        self.gross_axial_mesh = np.asarray([], dtype=np.float64)
    
    def _compose_global_axial_mesh(self):
        global_axial_mesh = np.asarray([], dtype=np.float64)
        for src in self._srcs.values():
            core = src.core
            global_axial_mesh = np.union1d(global_axial_mesh, core.gross_axial_mesh)
        self.global_axial_mesh = global_axial_mesh

    def add_src(self, src : VeraDataSource, src_id : str):
        """Register a source under src_id, making it default if it's the first.
        Raises ValueError if src_id is already registered.
        """
        if src_id in self._srcs:
            raise ValueError(f"{src_id} already exsists in the registry, skipped adding")
        self._srcs[src_id] = src
        core = src.core
        if self.default_src_id is None:
            self.default_src_id = src_id
            self.global_axial_mesh = core.gross_axial_mesh
        else:
            self.global_axial_mesh = np.union1d(self.global_axial_mesh, core.gross_axial_mesh)

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
    
    def get_axial_index(self, z : np.float64):
        return int(np.searchsorted(self.global_axial_mesh, z))
    
    def src_axial_idx_to_global_idx(self, src_id : str, ds_dtype : VeraDtype, idx : int) -> int:
        if src_id not in self._srcs:
            raise ValueError("src_id not in stored src_ids")
        core = self._srcs[src_id].core
        src_axial_mesh = core.axial_mesh_means if not ds_dtype.is_computational() else core.comp_axial_mesh_means
        physical_layer = src_axial_mesh[idx]
        global_idx = self.get_axial_index(physical_layer)
        assert self.global_axial_mesh[global_idx] == physical_layer
        global_idx = np.clip(global_idx, 0, len(self.global_axial_mesh) - 1)
        return int(global_idx)
    
    def global_axial_idx_to_src_idx(self, src_id : str, ds_dtype : VeraDtype, idx : int):
        if src_id not in self._srcs:
            # FIXME, should probably not return 0
            return 0
        core = self._srcs[src_id].core
        physical_layer = self.global_axial_mesh[idx]
        src_axial_mesh = core.axial_mesh_means if not ds_dtype.is_computational() else core.comp_axial_mesh_means
        src_idx = np.searchsorted(src_axial_mesh, physical_layer)
        src_idx = np.clip(src_idx, 0, len(src_axial_mesh) - 1)
        return int(src_idx)

    def get(self, src_id: str) -> VeraDataSource | None:
        """Return the source for src_id, or None if it isn't registered."""
        return self._srcs.get(src_id)
    
    def get_ds_dtype(self, src_id : str, ds_name : str) -> VeraDtype:
        if src_id not in self._srcs:
            return VeraDtype.UNKNOWN
        src = self._srcs[src_id]
        return src.array_dtype(ds_name)
    
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
        self._compose_global_axial_mesh()
    
    def all_sources(self) -> dict[str, str]:
        src_paths = {}
        for src_id, src in self._srcs.items():
            src_paths[src_id] = src.file_path
        return src_paths

    def clear(self):
        for src in self._srcs.values():
            src.close()
        self._srcs = {}
        self.default_src_id = None