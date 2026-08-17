"""Top-level entry point.

    vera = VeraCore.open("myfile.h5")
    vera.core_view.select("pin_powers", state=2, z=10).savefig("pin_powers.png")

One VeraCore is one loaded calculation. Comparing two files means two of
them; nothing here holds a registry of sources.
"""

from functools import cached_property
from pathlib import Path

from .dtypes import VeraDataset
from .model import VeraDataSource
from .readers.h5 import open_vera_file_data_source
from .renders.core_view import CoreView
from .renders.view import RenderOptions


class Vera:
    """One loaded calculation and the views over it.

    Views are built on first access and cached. Loading a new source drops
    them, so a view read afterwards is bound to the new one; a view a caller
    already holds is not, and points at a closed source.
    """

    def __init__(
        self,
        source: VeraDataSource | None = None,
        label: str | None = None,
        options: RenderOptions | None = None,
    ):
        self._source = source
        self.label = label
        """Names this calculation in headings, when a view asks for one."""
        self._options = options or RenderOptions()

    @classmethod
    def open(cls, path: str | Path, options: RenderOptions | None = None) -> "Vera":
        """Load a VERAout file. The usual way in."""
        return cls(options=options).load_h5(path)

    @property
    def options(self) -> RenderOptions:
        """Render defaults every view is built with. Setting them rebuilds
        the views, since a view is given its options once."""
        return self._options

    @options.setter
    def options(self, options: RenderOptions) -> None:
        self._options = options
        self._clear_views()

    def load_h5(self, path: str | Path) -> "Vera":
        """Load a VERAout file in place. Returns self so calls chain.

        Replaces any source already held, dropping cached views with it.
        """
        path = Path(path)
        return self.set_source(open_vera_file_data_source(str(path)), label=path.stem)

    def set_source(self, source: VeraDataSource, label: str | None = None) -> "Vera":
        """Attach an already-built source, e.g. from a stream or a test."""
        self._source = source
        self.label = label
        self._clear_views()
        return self

    # -- the calculation ---------------------------------------------------

    @property
    def source(self) -> VeraDataSource:
        if self._source is None:
            raise RuntimeError("no source loaded; call VeraCore.open() or set_source()")
        return self._source

    @property
    def dataset_names(self) -> list[str]:
        """Dataset names available on the active state."""
        return self.source.active_state.full_core_keys

    def dataset(self, name: str, state: int | None = None) -> VeraDataset:
        """One dataset by name, from the active state unless one is named."""
        if name not in self.dataset_names:
            raise KeyError(f"no dataset {name!r}; the active state has {self.dataset_names}")
        return self.source.get_dataset(array_name=name, state_idx=state)

    def close(self) -> None:
        self.source.close()
        self._source = None
        self._clear_views()

    def __enter__(self) -> "Vera":
        return self

    def __exit__(self, *exc_info) -> None:
        if self._source is not None:
            self.close()

    # -- the views ---------------------------------------------------------

    @cached_property
    def core_view(self) -> CoreView:
        return CoreView(self.source, self._options)

    # @cached_property
    # def core_surface_view(self) -> CoreSurfaceView:
    #     return CoreSurfaceView(self.source, self._options)

    # @cached_property
    # def axial_view(self) -> AxialView:
    #     return AxialView(self.source, self._options)

    # @cached_property
    # def assembly_view(self) -> AssemblyView:
    #     return AssemblyView(self.source, self._options)

    # @cached_property
    # def assembly_surface_view(self) -> AssemblySurfaceView:
    #     return AssemblySurfaceView(self.source, self._options)

    def _clear_views(self) -> None:
        """Drop every cached view, found from the class rather than a list
        that a new view could be left out of."""
        for name, attribute in type(self).__dict__.items():
            if isinstance(attribute, cached_property):
                self.__dict__.pop(name, None)

    def __repr__(self) -> str:
        if self._source is None:
            return "<VeraCore unloaded>"
        return f"<VeraCore {self._source.provenance} {len(self._source.states)} states>"
