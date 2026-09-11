"""Top-level entry point.

    vera = VeraVis.open("myfile.h5")
    vera.core_view.select("pin_powers", state=2, z=10).savefig("pin_powers.png")

One VeraVis is one loaded calculation. Comparing two of them is what a
VeraSet is for; nothing here holds a registry of sources.
"""

from functools import cached_property
from pathlib import Path

from .dtypes import VeraDataset
from .model import VeraDataSource
from .readers.h5 import open_vera_file_data_source
from .renders.assembly_surface_view import AssemblySurfaceView
from .renders.assembly_view import AssemblyView
from .renders.axial_plot_view import AxialLinesView
from .renders.axial_view import AxialView
from .renders.core_surface_view import SurfaceView
from .renders.core_view import CoreView
from .renders.view import RenderOptions


class VeraVis:
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
    def open(cls, path: str | Path, options: RenderOptions | None = None) -> "VeraVis":
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

    def load_h5(self, path: str | Path) -> "VeraVis":
        """Load a VERAout file in place. Returns self so calls chain.

        Replaces any source already held, dropping cached views with it.
        """
        path = Path(path)
        return self.set_source(open_vera_file_data_source(str(path)), label=path.stem)

    def set_source(self, source: VeraDataSource, label: str | None = None) -> "VeraVis":
        """Attach an already-built source, e.g. from a stream or a test."""
        self._source = source
        self.label = label
        self._clear_views()
        return self

    # -- the calculation ---------------------------------------------------

    @property
    def source(self) -> VeraDataSource:
        if self._source is None:
            raise RuntimeError("no source loaded; call VeraVis.open() or set_source()")
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

    def __enter__(self) -> "VeraVis":
        return self

    def __exit__(self, *exc_info) -> None:
        if self._source is not None:
            self.close()

    # -- the views ---------------------------------------------------------

    @cached_property
    def core_view(self) -> CoreView:
        return CoreView(self.source, self._options)

    @cached_property
    def core_surface_view(self) -> SurfaceView:
        return SurfaceView(self.source, self._options)

    @cached_property
    def axial_view(self) -> AxialView:
        return AxialView(self.source, self._options)

    @cached_property
    def assembly_view(self) -> AssemblyView:
        return AssemblyView(self.source, self._options)

    @cached_property
    def assembly_surface_view(self) -> AssemblySurfaceView:
        return AssemblySurfaceView(self.source, self._options)

    @cached_property
    def axial_plot(self) -> AxialLinesView:
        """The one view that is a plot rather than a map. A VeraSet's reads
        several calculations at once; this one reads only this calculation."""
        return AxialLinesView([self.source], self._options)

    def _clear_views(self) -> None:
        """Drop every cached view, found from the class rather than a list
        that a new view could be left out of."""
        for name, attribute in type(self).__dict__.items():
            if isinstance(attribute, cached_property):
                self.__dict__.pop(name, None)

    def __repr__(self) -> str:
        if self._source is None:
            return "<VeraVis unloaded>"
        return f"<VeraVis {self._source.provenance} {len(self._source.states)} states>"


class VeraSet:
    """Several loaded calculations, and the views that can read more than one.

        pair = VeraSet.open("a.h5", "b.h5")
        pair.axial_plot.select("pin_powers", assembly=0, pin=(8, 8)).savefig("cmp.png")

    A VeraVis is one calculation and a view on it is bound to that one
    source. Most views draw a map of one calculation and have nothing to say
    about a second; the ones that can hold two up against each other live
    here, so the single-source case does not grow a source list it never uses.

    This is not a registry: it holds the calculations a caller put in it, in
    the order they were given, and does not name or look them up. Closing it
    closes them, so a caller that still wants one should keep its own VeraVis.
    """

    def __init__(self, *veras: VeraVis, options: RenderOptions | None = None):
        if not veras:
            raise ValueError("a VeraSet needs at least one calculation")
        self.veras = list(veras)
        self._options = options or RenderOptions()

    @classmethod
    def open(cls, *paths: str | Path, options: RenderOptions | None = None) -> "VeraSet":
        """Load several VERAout files. The usual way in."""
        return cls(*(VeraVis.open(path) for path in paths), options=options)

    @property
    def sources(self) -> list[VeraDataSource]:
        """The loaded sources, in the order they were given."""
        return [vera.source for vera in self.veras]

    @property
    def labels(self) -> list[str | None]:
        return [vera.label for vera in self.veras]

    @property
    def options(self) -> RenderOptions:
        """Render defaults every view is built with. Setting them rebuilds
        the views, since a view is given its options once."""
        return self._options

    @options.setter
    def options(self, options: RenderOptions) -> None:
        self._options = options
        self._clear_views()

    def add(self, vera: VeraVis) -> "VeraSet":
        """Another calculation to compare against. Returns self so calls chain."""
        self.veras.append(vera)
        self._clear_views()
        return self

    # -- the views ---------------------------------------------------------

    @cached_property
    def axial_plot(self) -> AxialLinesView:
        return AxialLinesView(self.sources, self._options)

    def _clear_views(self) -> None:
        """Drop every cached view, found from the class rather than a list
        that a new view could be left out of."""
        for name, attribute in type(self).__dict__.items():
            if isinstance(attribute, cached_property):
                self.__dict__.pop(name, None)

    # -- the calculations --------------------------------------------------

    def close(self) -> None:
        """Close every calculation in the set."""
        for vera in self.veras:
            vera.close()
        self._clear_views()

    def __enter__(self) -> "VeraSet":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.veras)

    def __iter__(self):
        return iter(self.veras)

    def __getitem__(self, index):
        return self.veras[index]

    def __repr__(self) -> str:
        return f"<VeraSet {len(self.veras)}: {', '.join(str(label) for label in self.labels)}>"
