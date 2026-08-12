"""Top-level entry point.

vera = VeraCore().load_h5("myfile.h5")
vera.core_view.savefig("pin_powers", state=2, z=10)

"""

from pathlib import Path
from typing import TYPE_CHECKING

from .dtypes import VeraDataset
from .model import VeraDataSource
from .readers.h5 import open_vera_file_data_source
from .renders.assembly_surface_view import AssemblySurfaceView
from .renders.assembly_view import AssemblyView
from .renders.axial_view import AxialView
from .renders.core_surface_view import SurfaceView
from .renders.core_view import CoreView

VIEWS: dict[str, type] = {
    "core_view": CoreView,
    "surface_view": SurfaceView,
    "axial_view": AxialView,
    "assembly_view": AssemblyView,
    "assembly_surface_view": AssemblySurfaceView,
}


class VeraCore:
    """One loaded calculation and the views over it."""

    if TYPE_CHECKING:
        core_view: CoreView
        surface_view: SurfaceView
        axial_view: AxialView
        assembly_view: AssemblyView
        assembly_surface_view: AssemblySurfaceView

    def __init__(self, source: VeraDataSource | None = None):
        self._source = source
        self._views: dict[str, object] = {}

    def load_h5(self, path: str | Path) -> "VeraCore":
        """Load a VERAout file. Returns self so calls chain.

        Replaces any source already held, dropping cached views with it.
        """
        return self.set_source(open_vera_file_data_source(str(path)))

    def set_source(self, source: VeraDataSource) -> "VeraCore":
        """Attach an already-built source, e.g. from a stream or a test."""
        self._source = source
        self._views.clear()
        return self

    def get(self, array_name: str, src_id: str | None = None) -> VeraDataset:
        return self._source.get_dataset(array_name=array_name)

    @property
    def source(self) -> VeraDataSource:
        if self._source is None:
            raise RuntimeError("no source loaded; call load_h5() or set_source()")
        return self._source

    @property
    def datasets(self) -> list[str]:
        """Dataset names available on the active state."""
        return self.source.active_state.full_core_keys

    def close(self) -> None:
        self.source.close()
        self._source = None
        self._views.clear()

    def __getattr__(self, name: str) -> object:
        """Build a registered view on first access, then cache it.

        Only reached for attributes not found normally, so it cannot shadow a
        real method.
        """
        if name not in VIEWS:
            raise AttributeError(name)
        if name not in self._views:
            self._views[name] = VIEWS[name](self.source)
        return self._views[name]

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(VIEWS))

    def __repr__(self) -> str:
        if self._source is None:
            return "<VeraCore unloaded>"
        return f"<VeraCore {self._source.provenance} {len(self._source.states)} states>"
