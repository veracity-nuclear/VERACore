"""Fixtures and stubs shared by the render tests.

The source is built here from plain dicts: one DictDatasetSource for the core
group, one per state. Nothing is read from disk, so a test that changes the
core changes one dict rather than a file.

A StubView keeps the view-level tests off the slice modules: it is the least
a view can be, so a failure there is a failure in view.py or canvas.py.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from vera_core.data.analysis.color import ColorScope, resolve_color_specs
from vera_core.data.readers.mock import DictDatasetSource
from vera_core.data.model import VeraDataSource, VeraOutCore, VeraOutState
from vera_core.data.renders import draw
from vera_core.data.renders.canvas import Canvas, block_layout
from vera_core.data.renders.styles import ViewStyle
from vera_core.data.renders.view import RenderOptions, Selection, View

NASS_SIDE = 5
"""Assemblies across the core map. The crop tests slice this grid."""

NPIN = 3
NAX = 4
N_STATES = 3

NASS = NASS_SIDE * NASS_SIDE


def core_arrays() -> dict[str, np.ndarray]:
    """The CORE group: a full-symmetry map numbered 1..25, its mesh, and pin
    volumes with one non-fuel position so the non-fuel masking path runs."""
    pin_volumes = np.ones((NPIN, NPIN, NAX, NASS))
    pin_volumes[0, 0] = 0.0
    return {
        "core_map": np.arange(1, NASS + 1, dtype=np.int64).reshape(NASS_SIDE, NASS_SIDE),
        "core_sym": np.array([1]),
        "axial_mesh": np.linspace(0.0, 100.0, NAX + 1),
        "npin": np.array([NPIN]),
        "pin_volumes": pin_volumes,
    }


def state_arrays(exposure: float) -> dict[str, np.ndarray]:
    """One state point. Seeded from the exposure, so a state's values are the
    same on every run and two states differ."""
    rng = np.random.default_rng(int(exposure * 100))
    return {
        "pin_powers": rng.random((NPIN, NPIN, NAX, NASS)),
        "pin_radial_powers": rng.random((NPIN, NPIN, NASS)),
        "assembly_powers": rng.random((1, NAX, NASS)),
        "axial_powers": rng.random((NAX,)),
        "keff": np.array([1.0 + exposure / 1000.0]),
        "exposure": np.array([exposure]),
    }


def build_test_source(n_states: int = N_STATES) -> VeraDataSource:
    """A complete in-memory source, assembled the way a reader assembles one."""
    core = VeraOutCore(DictDatasetSource(core_arrays()))
    states = [
        VeraOutState(
            DictDatasetSource(
                state_arrays(index * 10.0),
                provenance="<memory>",
                dataset_dtypes=core.shape_to_dtype,
                units={"pin_powers": "W"},
            ),
            index,
            core,
        )
        for index in range(n_states)
    ]
    return VeraDataSource(
        core=core,
        states=states,
        provenance="<memory>",
        filename=None,
        close_callback=None,
        state_caching=True,
    )


@pytest.fixture
def mock_source():
    """Three states over a 5x5 core map, 3x3 pins, 4 axial levels."""
    return build_test_source()


@pytest.fixture
def quiet_style():
    """Nothing that reaches a slice's caption or its value formatter."""
    return ViewStyle(show_caption=False, show_values=False, show_axis_labels=False)


class StubSlice:
    """The least resolve_color_specs and a canvas need: a group count, a
    range per group, and units for the bar."""

    def __init__(self, n_groups: int = 2, base: float = 0.0, state: int = 0):
        self.n_groups = n_groups
        self.base = base
        self.state = state
        self.units = "unitless"
        self.scopes: list[ColorScope] = []

    def value_range(self, group: int, scope: ColorScope) -> tuple[float, float]:
        self.scopes.append(scope)
        return (self.base + group, self.base + group + 1.0)


class StubView(View):
    """A view that draws a 2x2 block per group and records what it was given."""

    def __init__(self, source=None, options: RenderOptions | None = None, n_groups: int = 2):
        super().__init__(source or SimpleNamespace(states=list(range(N_STATES))), options)
        self.n_groups = n_groups
        self.shared_color_calls: list[tuple[list, ViewStyle]] = []
        self.last_slice: StubSlice | None = None
        self.last_specs: list = []

    def select(self, *, state: int = 0, z: int = 0) -> Selection:
        return Selection(self, state=state, z=z)

    def build_slice(self, selection: Selection) -> StubSlice:
        self.last_slice = StubSlice(
            self.n_groups, base=float(selection.state), state=selection.state
        )
        return self.last_slice

    def draw_panel(self, panel, slice_, spec, style):
        """Values shift with the state, so two frames of a sweep are not
        pixel-identical. A GIF writer drops a frame whose delta is empty, so
        a static stub would collapse a three-frame sweep into one."""
        return draw.cells(panel.ax, np.arange(4.0).reshape(2, 2) + slice_.base, spec, style)

    def render(self, slice_, selection, options) -> Canvas:
        style = options.style
        canvas = Canvas(
            slice_.n_groups,
            style=style,
            title=options.resolved_title(self, selection),
            caption="cap" if style.show_caption else None,
        )
        self.last_specs = resolve_color_specs(
            slice_, options.color, scope=style.color_scope, cmap=style.cmap
        )
        for panel, spec in zip(canvas.panels, self.last_specs, strict=True):
            mappable = self.draw_panel(panel, slice_, spec, style)
            canvas.colorbar(mappable, [panel], units=slice_.units)
        return canvas

    def render_collage(self, slices, selections, options, *, over, columns=None) -> Canvas:
        style = options.style
        n_groups, n_frames = slices[0].n_groups, len(slices)
        grid, cells = block_layout(n_groups, n_frames, columns)
        canvas = Canvas(len(cells), grid=grid, cells=cells, style=style)
        self.last_specs = resolve_color_specs(
            slices[0], options.color, scope=style.color_scope, cmap=style.cmap
        )
        for group in range(n_groups):
            block = canvas.panels[group * n_frames : (group + 1) * n_frames]
            for panel, slice_ in zip(block, slices, strict=True):
                self.draw_panel(panel, slice_, self.last_specs[group], style)
        return canvas

    def shared_color(self, slices, style):
        self.shared_color_calls.append((list(slices), style))
        return super().shared_color(slices, style)


@pytest.fixture
def stub_view():
    return StubView()
