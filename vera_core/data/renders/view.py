"""The selection interface: pick what to look at, then render it.

shot = vera.core_view.select("pin_powers", z=10)
shot.savefig("pin_powers.png")
"""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from matplotlib.figure import Figure

from ..analysis.color import (
    DEFAULT_CMAP,
    ColorScope,
    ColorSource,
    ColorSpec,
    shared_group_specs,
)
from .animation import DEFAULT_FPS, write_animation
from .canvas import DEFAULT_DPI, PANEL_WIDTH_IN, Canvas, write_figure
from .styles import ViewStyle

RESERVED = ("view", "params", "title", "options")
"""Selection's own attributes and methods; a view cannot select on these."""


@dataclass(frozen=True)
class RenderOptions:
    """Everything about how a slice is drawn, none of it about which slice.

    A view holds one as its defaults; a render call overrides fields of it.
    """

    style: ViewStyle = ViewStyle()
    color: ColorSource = None
    color_scope: ColorScope = ColorScope.SLICE_ALL
    title: str | bool = False
    """A heading, True for one the view derives, False for none."""
    caption: bool = True
    panel_width: float = PANEL_WIDTH_IN

    def resolved_title(self, view: "View", selection: "Selection") -> str | None:
        return view.default_title(selection) if self.title is True else (self.title or None)


class Selection:
    """What to look at, bound to the view that can render it.

    Built by View.select(), which names and defaults the parameters. They
    read back as attributes: shot.array, shot.z.
    """

    def __init__(self, view: "View", **params):
        clashes = sorted(set(params) & set(RESERVED))
        if clashes:
            raise TypeError(f"{clashes} are Selection attributes and cannot be selected on")
        self.view = view
        self.params = params
        self.title: str | bool = False
        """A heading for every render of this selection, unless the call
        names one. False leaves it to the view's options."""

    def __getattr__(self, name):
        params = self.__dict__.get("params", {})
        if name in params:
            return params[name]
        raise AttributeError(f"{type(self).__name__} has no {name!r}")

    def replace(self, **changes) -> "Selection":
        """The same view with some choices changed. Unknown names raise,
        since the new selection goes back through the view's select()."""
        selection = self.view.select(**{**self.params, **changes})
        selection.title = self.title
        return selection

    def slice(self):
        """The slice this renders, for callers that want the data itself."""
        return self.view.build_slice(self)

    def label(self) -> str:
        """The choices, for messages and repr."""
        shown = [
            (k, v)
            for k, v in self.params.items()
            if v is not None and not (isinstance(v, tuple) and not v)
        ]
        return " ".join(str(v) if k == "array" else f"{k}={v}" for k, v in shown)

    def savefig(
        self,
        path: str | Path,
        *,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        title: str | bool | None = None,
        caption: bool | None = None,
        panel_width: float | None = None,
        dpi: int = DEFAULT_DPI,
    ) -> Path:
        """Render and write to path. Format follows the suffix.

        Every option left None comes from the view's RenderOptions:

            style         a ViewStyle: theme, whether values are printed,
                          decimals, grid and label sizes
            color         one ColorSpec for every group, or {group: spec},
                          [spec, ...], or a function of the group index
            color_scope   how wide a span the colorbars cover: GROUP (this
                          group), SLICE (all groups on one scale), DATASET
            title         a heading string, True for one the view derives
                          from the selection, False for none
            caption       whether the view's caption line is drawn
            panel_width   inches across one panel, before cropping
        """
        return write_figure(
            self.figure(
                style=style,
                color=color,
                color_scope=color_scope,
                title=title,
                caption=caption,
                panel_width=panel_width,
            ),
            path,
            dpi,
        )

    def figure(
        self,
        *,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        title: str | bool | None = None,
        caption: bool | None = None,
        panel_width: float | None = None,
    ) -> Figure:
        """The finished figure, for callers that want to embed it or add to
        it. The options are savefig's, and mean the same thing."""
        return self.canvas(
            self.options(
                style=style,
                color=color,
                color_scope=color_scope,
                title=title,
                caption=caption,
                panel_width=panel_width,
            )
        ).figure()

    def savecollage(
        self,
        path: str | Path,
        *,
        over: str = "state",
        values: Sequence | None = None,
        columns: int = None,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        title: str | bool | None = None,
        caption: bool | None = None,
        panel_width: float | None = None,
        dpi: int = DEFAULT_DPI,
    ) -> Path:
        """Every frame of a sweep in one figure, written to path.

        A collage holds one choice up against itself: the same array at every
        state, on one scale, so a change between frames is a change in the
        data and not in the colorbar. Multi-group data keeps a scale and a
        bar per group, since a fast group would otherwise flatten a thermal
        one into a single color.

            over      which choice varies, "state" by default
            values    the values it takes, or None for every one the view
                      knows about
            columns   frames per row before a group's block wraps
            color     overrides the shared scale, one spec per group

        The rest are savefig's, and mean the same thing.
        """
        return write_figure(
            self.collage(
                over=over,
                values=values,
                columns=columns,
                style=style,
                color=color,
                title=title,
                caption=caption,
                panel_width=panel_width,
                color_scope=color_scope,
            ),
            path,
            dpi,
        )

    def collage(
        self,
        *,
        over: str = "state",
        values: Sequence | None = None,
        columns: int = None,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        title: str | bool | None = None,
        caption: bool | None = None,
        panel_width: float | None = None,
    ) -> Figure:
        """The finished collage figure. Options are savecollage's."""
        frames, slices, options = self._sweep(
            over, values, color, color_scope, style, title, caption, panel_width
        )
        return self.view.render_collage(
            slices, frames, options, over=over, columns=columns
        ).figure()

    def canvas(self, options: RenderOptions | None = None, slice_=None) -> Canvas:
        """The drawn canvas, before it is finished, for callers that want to
        reach the panels and annotate one. Takes whole options rather than
        fields of them; build a set with self.options() or replace().

        Pass slice_ to draw data already in hand, which is how a movie avoids
        reading every state twice.
        """
        if slice_ is None:
            slice_ = self.view.build_slice(self)
        return self.view.render(slice_, self, options or self.options())

    def savemovie(
        self,
        path: str | Path,
        *,
        over: str = "state",
        values: Sequence | None = None,
        fps: float = DEFAULT_FPS,
        loop: int = 0,
        style: ViewStyle | None = None,
        color: ColorSource = None,
        color_scope: ColorScope | None = None,
        title: str | bool | None = None,
        caption: bool | None = None,
        panel_width: float | None = None,
        dpi: int = DEFAULT_DPI,
    ) -> Path:
        """One still per frame of a sweep, combined into an animation.

        The same frames a collage lays out side by side, played in sequence
        instead. They share one scale per group for the same reason: a
        colorbar that moved between frames would show change that is not in
        the data.

            path      .gif, or .mp4 .avi .mov .webm .mkv with imageio and
                      ffmpeg installed, or no suffix for a directory of
                      numbered PNGs
            over      which choice varies, "state" by default
            values    the values it takes, or None for every one the view
                      knows about
            fps       frames per second
            loop      gif only: 0 forever, 1 once

        The rest are savefig's, and mean the same thing.
        """
        frames, slices, options = self._sweep(
            over, values, color, color_scope, style, title, caption, panel_width
        )
        figures = (
            frame.canvas(options, slice_).figure()
            for frame, slice_ in zip(frames, slices, strict=True)
        )
        return write_animation(figures, path, fps=fps, dpi=dpi, loop=loop)

    def _sweep(self, over, values, color, color_scope, style, title, caption, panel_width):
        """The frames of a sweep, their slices, and the options that hold
        every frame to one scale per group. Shared by collage and movie."""
        if over not in self.params:
            raise TypeError(f"{type(self.view).__name__} does not select on {over!r}")
        if values is None:
            values = self.view.sweep_values(over, self)
        frames = [self.replace(**{over: value}) for value in values]
        if not frames:
            raise ValueError(f"no {over} values to draw")
        slices = [frame.slice() for frame in frames]
        options = self.options(
            style=style,
            color=color if color is not None else self.view.shared_color(slices, color_scope),
            title=title,
            caption=caption,
            panel_width=panel_width,
            color_scope=color_scope,
        )
        return frames, slices, options

    def options(self, **named) -> RenderOptions:
        """The view's options, overridden by this selection's title, then by
        whatever is named here. None means 'leave it to the view'."""
        if named.get("title") is None and self.title is not False:
            named["title"] = self.title
        return replace(self.view.options, **{k: v for k, v in named.items() if v is not None})

    def __repr__(self) -> str:
        return f"<{type(self.view).__name__} selection {self.label()}>"


class View:
    """Base for render views, bound to one loaded source.

    A subclass implements:

        select(...)                        -> Selection, naming its inputs
        build_slice(selection)             -> the data to draw
        render(slice_, selection, options) -> a Canvas
        default_title(selection)           -> heading when title is True
        caption(slice_, selection)         -> the line under the figure

    render() builds its own Canvas and calls the artists it wants, so a view
    that needs a different panel shape, no colorbar or no map is not fighting
    a template.
    """

    def __init__(self, source, options: RenderOptions | None = None):  # noqa: D107
        self.source = source
        self.options = options or RenderOptions()

    def select(self, *args, **kwargs) -> Selection:
        raise NotImplementedError

    def build_slice(self, selection: Selection):
        raise NotImplementedError

    def render(self, slice_, selection: Selection, options: RenderOptions) -> Canvas:
        raise NotImplementedError

    def render_collage(
        self,
        slices,
        selections: "list[Selection]",
        options: RenderOptions,
        *,
        over: str,
        columns: int = None,
    ) -> Canvas:
        """Every frame in one canvas. Views that cannot do this say so."""
        raise NotImplementedError(f"{type(self).__name__} has no collage")

    def shared_color(self, slices, scope: ColorScope | None = None) -> list[ColorSpec] | None:
        """One scale per group, spanning every frame of a sweep.

        A sweep holds its frames to one scale so that a change between them
        is a change in the data and not in the colorbar. A view that does not
        color by value has no such scale, and says so with None.
        """
        cmap = self.options.color
        cmap = cmap.cmap if isinstance(cmap, ColorSpec) else DEFAULT_CMAP
        return shared_group_specs(slices, cmap=cmap, scope=scope)

    def sweep_values(self, over: str, selection: Selection) -> Sequence:
        """Everything `over` can be, when a sweep is not given values.

        States and axial levels are known from the source; a view that can
        enumerate its own other choices extends this. The selection is passed
        because what a choice ranges over can depend on what was picked: an
        array's axial extent is a property of that array, not of the view.
        """
        if over == "state":
            return range(len(self.source.states))
        if over == "z":
            return self.axial_levels(selection)
        raise ValueError(f"{type(self).__name__} cannot enumerate {over!r}; pass values")

    def axial_levels(self, selection: Selection) -> range:
        """Every axial level the selection's array has.

        Which mesh that is depends on the array: a computational one is on
        the computational mesh and a detector on the detector's, so the count
        comes from the source rather than from a single nax.
        """
        dtype = self.source.get_dataset_dtype(selection.array, selection.state)
        if not dtype.has_axial_dim():
            raise ValueError(
                f"{selection.array!r} reads as {dtype}, which has no axial dimension;"
                " check the name or pass values"
            )
        return range(len(self.source.core.get_axial_mesh_means(dataset_type=dtype)))

    def frame_title(self, selection: Selection, over: str) -> str:
        """The heading over one frame of a collage."""
        return f"{over.replace('_', ' ').capitalize()} {getattr(selection, over)}"

    def collage_caption(self, slices, selections: "list[Selection]", over: str) -> str:
        """The line under a collage. Says what is held fixed, not what varies:
        the frames are labelled with that."""
        return f"{len(selections)} frames over {over}"

    def default_title(self, selection: Selection) -> str:
        return selection.label()

    def caption(self, slice_, selection: Selection) -> str:
        """The line under the figure. Empty means none."""
        return ""
