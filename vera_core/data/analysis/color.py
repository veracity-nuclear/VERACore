import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_CMAP = "jet"


def array_range(array) -> tuple[float, float]:
    """Finite (lo, hi) of an array, widened when flat and (0, 1) when empty."""
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lo = float(np.nanmin(array))
        hi = float(np.nanmax(array))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return (0.0, 1.0)
    if lo == hi:
        eps = max(abs(hi) * 1e-9, 1e-12)
        return (lo, hi + eps)
    return (lo, hi)


@dataclass(frozen=True)
class ColorSpec:
    """How values map to color. Backend-neutral.

    vmin/vmax are the colorbar endpoints, not the data range: a front end may
    narrow or widen them to saturate outliers.
    """

    vmin: float = 0.0
    vmax: float = 1.0
    cmap: str = DEFAULT_CMAP

    @property
    def range(self) -> tuple[float, float]:
        return (self.vmin, self.vmax)

    @property
    def span(self) -> float:
        return self.vmax - self.vmin

    def with_range(self, vmin: float, vmax: float) -> "ColorSpec":
        """A copy carrying a caller-supplied range, e.g. from a color editor."""
        return replace(self, vmin=float(vmin), vmax=float(vmax))

    def normalize(self, values) -> np.ndarray:
        """Values mapped to [0, 1], clipped at the endpoints. NaN stays NaN."""
        values = np.asarray(values, dtype=float)
        if self.span <= 0:
            return np.where(np.isnan(values), np.nan, 0.0)
        return np.clip((values - self.vmin) / self.span, 0.0, 1.0)

    def validate(self) -> list[str]:
        """Contract violations, empty when the spec is well formed."""
        problems = []
        if not (np.isfinite(self.vmin) and np.isfinite(self.vmax)):
            problems.append(f"range {self.range} is not finite")
        elif self.vmax <= self.vmin:
            problems.append(f"vmax {self.vmax} must exceed vmin {self.vmin}")
        if not self.cmap:
            problems.append("cmap must be a non-empty preset name")
        return problems

    def __repr__(self) -> str:
        return f"<ColorSpec {self.cmap} [{self.vmin:.4g}, {self.vmax:.4g}]>"


def default_color_spec(value_range: tuple[float, float], cmap: str = DEFAULT_CMAP) -> ColorSpec:
    """The starting spec for a slice. Pass CoreSlice.value_range.

    Takes the range rather than the slice so this module stays importable by
    core_slice.py without a cycle. A flat range is widened, so a hand-built
    slice cannot produce a spec that its own validate() rejects.
    """
    lo, hi = array_range(np.asarray(value_range, dtype=float))
    return ColorSpec(vmin=lo, vmax=hi, cmap=cmap)


def shared_color_spec(
    value_ranges: list[tuple[float, float]], cmap: str = DEFAULT_CMAP
) -> ColorSpec:
    """One spec spanning several ranges, for an explicit common scale.

    No longer backs any fallback: ColorScope reads ranges the source computed,
    which do not shift with whichever slices are on the page. Use this when
    the union over exactly what is displayed is what you mean -- comparing two
    states side by side, say -- and pass the result as color.

    Returns the default spec when the list is empty.
    """
    if not value_ranges:
        return ColorSpec(cmap=cmap)
    lo = min(float(r[0]) for r in value_ranges)
    hi = max(float(r[1]) for r in value_ranges)
    return default_color_spec((lo, hi), cmap=cmap)


class ColorScope(StrEnum):
    """How wide a span of data the colorbar covers.

        SLICE     this panel only -- most contrast, but the scale moves as
                  you page through levels or groups
        GROUP     every level of this slice's group -- panels stay comparable
                  across z, which is what makes an animation readable
        DATASET   the whole dataset -- every panel of it is comparable, at the
                  cost of contrast when groups differ by orders of magnitude

    The ranges come from the source, so they do not depend on which slices
    happen to be on the page. A union over only the displayed slices is a
    different quantity, and a moving one: pass shared_color_spec() explicitly
    if that is what you want.
    """

    SLICE = "slice"
    GROUP = "group"
    DATASET = "dataset"

    @property
    def attribute(self) -> str:
        """The slice attribute holding this scope's range."""
        return f"{self.value}_value_range"


@runtime_checkable
class RangedSlice(Protocol):
    """What resolve_color_specs() needs from a slice. Structural, so this
    module stays free of any dependency on a particular slice type."""

    group: int | None
    value_range: tuple[float, float]


ColorSource = (
    ColorSpec
    | Mapping[int, ColorSpec]
    | Sequence[ColorSpec | None]
    | Callable[[RangedSlice], ColorSpec | None]
    | None
)
"""Ways a caller can specify color for a set of slices.

    None                    each slice keeps its own data range
    ColorSpec               one spec for all of them
    {0: spec, 2: spec}      by group index, others fall back
    [spec, None, spec]      by position, None falls back
    lambda s: ...           computed, None falls back
"""


def scope_range(slice_: RangedSlice, scope: ColorScope) -> tuple[float, float]:
    """One slice's range at the requested scope.

    Raises AttributeError naming the scope rather than the attribute, since a
    slice type that predates the three-range contract fails here first.
    """
    try:
        return getattr(slice_, scope.attribute)
    except AttributeError:
        raise AttributeError(
            f"{type(slice_).__name__} has no {scope.attribute};"
            f" it cannot be colored at {scope.value} scope"
        ) from None


def _fallback_specs(slices: Sequence[RangedSlice], scope: ColorScope, cmap: str) -> list[ColorSpec]:
    """The spec each slice gets when the caller names nothing for it."""
    return [default_color_spec(scope_range(slice_, scope), cmap=cmap) for slice_ in slices]


def _override(color: ColorSource, slice_: RangedSlice, index: int) -> ColorSpec | None:
    """The caller's spec for one slice, or None to fall back."""
    if color is None:
        return None
    if isinstance(color, ColorSpec):
        return color
    if isinstance(color, Mapping):
        return color.get(slice_.group if slice_.group is not None else index)
    if callable(color):
        return color(slice_)
    return color[index] if index < len(color) else None


def resolve_color_specs(
    slices: Sequence[RangedSlice],
    color: ColorSource = None,
    *,
    scope: ColorScope = ColorScope.SLICE,
    cmap: str = DEFAULT_CMAP,
) -> list[ColorSpec]:
    """One ColorSpec per slice, from whatever shape the caller supplied."""
    scope = ColorScope(scope)
    specs = []
    for index, (slice_, fallback) in enumerate(
        zip(slices, _fallback_specs(slices, scope, cmap), strict=False)
    ):
        spec = _override(color, slice_, index) or fallback
        problems = spec.validate()
        if problems:
            label = index if slice_.group is None else f"group {slice_.group}"
            raise ValueError(f"color for {label}: {'; '.join(problems)}")
        specs.append(spec)
    return specs
