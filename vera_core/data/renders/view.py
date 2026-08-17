"""The selection interface: pick what to look at, then render it.

    shot = vera.core_view.select("pin_powers", z=10)
    shot.savefig("pin_powers.png")

A view holds the source, the default render options, and the drawing. A
selection holds what the caller picked, and is the only thing a caller
renders from. Each view's select() names its own parameters, so that
signature is the one place a view's inputs are declared.

    draw.py       artists, one Axes at a time
    canvas.py     the figure and its panels
    view.py       the selection interface          <- you are here
    *_view.py     one concrete view, wiring the three together
"""

from dataclasses import dataclass, replace
from pathlib import Path

from matplotlib.figure import Figure

from ..analysis.color import ColorScope, ColorSource
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
    color_scope: ColorScope = ColorScope.GROUP
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
        shown = [(k, v) for k, v in self.params.items() if v is not None and v != ()]
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

    def canvas(self, options: RenderOptions | None = None) -> Canvas:
        """The drawn canvas, before it is finished, for callers that want to
        reach the panels and annotate one. Takes whole options rather than
        fields of them; build a set with self.options() or replace()."""
        return self.view.render(self.view.build_slice(self), self, options or self.options())

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

    def __init__(self, source, options: RenderOptions | None = None):
        self.source = source
        self.options = options or RenderOptions()

    def select(self, *args, **kwargs) -> Selection:
        raise NotImplementedError

    def build_slice(self, selection: Selection):
        raise NotImplementedError

    def render(self, slice_, selection: Selection, options: RenderOptions) -> Canvas:
        raise NotImplementedError

    def default_title(self, selection: Selection) -> str:
        return selection.label()

    def caption(self, slice_, selection: Selection) -> str:
        """The line under the figure. Empty means none."""
        return ""
