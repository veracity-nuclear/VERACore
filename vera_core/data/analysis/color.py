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


def union_range(ranges: Sequence[tuple[float, float]]) -> tuple[float, float]:
    """The range covering all of them, or (0, 1) when there are none."""
    if not ranges:
        return (0.0, 1.0)
    lo = min(float(low) for low, _ in ranges)
    hi = max(float(high) for _, high in ranges)
    return array_range(np.array([lo, hi]))


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
    """The starting spec for one range, e.g. CoreSlice.value_range().

    Takes the range rather than the slice so this module stays importable by
    core_slice.py without a cycle. A flat range is widened, so a hand-built
    spec cannot fail its own validate().
    """
    lo, hi = array_range(np.asarray(value_range, dtype=float))
    return ColorSpec(vmin=lo, vmax=hi, cmap=cmap)


def shared_color_spec(
    value_ranges: Sequence[tuple[float, float]], cmap: str = DEFAULT_CMAP
) -> ColorSpec:
    """One spec spanning several ranges, for an explicit common scale.

    Use it when the union over exactly what is displayed is what you mean --
    two states side by side, say -- and pass the result as color. Within one
    slice, ColorScope.SLICE already does this.
    """
    return default_color_spec(union_range(value_ranges), cmap=cmap)


class ColorScope(StrEnum):
    """How wide a span of data one panel's colorbar covers.

        GROUP     this group, this slice -- most contrast, but the scale
                  moves as you page through levels
        SLICE     every group of this slice -- panels are comparable with
                  each other, which is what makes a group comparison honest
        DATASET   the whole dataset -- every panel of it is comparable across
                  levels and states, at the cost of contrast

    DATASET needs a range the builder computed from the source; a slice that
    does not carry one raises rather than quietly narrowing the scope.
    """

    GROUP = "group"
    SLICE = "slice"
    DATASET = "dataset"


@runtime_checkable
class GroupedSlice(Protocol):
    """What resolve_color_specs() needs from a slice. Structural, so this
    module depends on no particular slice type."""

    @property
    def n_groups(self) -> int: ...

    def value_range(self, group: int, scope: "ColorScope") -> tuple[float, float]: ...


ColorSource = (
    ColorSpec
    | Mapping[int, ColorSpec]
    | Sequence[ColorSpec | None]
    | Callable[[int], ColorSpec | None]
    | None
)
"""Ways a caller can specify color for a slice's groups.

    None                    each group keeps its own range at the given scope
    ColorSpec               one spec for every group
    {0: spec, 2: spec}      by group index, others fall back
    [spec, None, spec]      by group index, None falls back
    lambda group: ...       computed, None falls back
"""


def _override(color: ColorSource, group: int) -> ColorSpec | None:
    """The caller's spec for one group, or None to fall back."""
    if color is None:
        return None
    if isinstance(color, ColorSpec):
        return color
    if isinstance(color, Mapping):
        return color.get(group)
    if callable(color):
        return color(group)
    return color[group] if group < len(color) else None


def resolve_color_specs(
    slice_: GroupedSlice,
    color: ColorSource = None,
    *,
    scope: ColorScope = ColorScope.GROUP,
    cmap: str = DEFAULT_CMAP,
) -> list[ColorSpec]:
    """One ColorSpec per group, from whatever shape the caller supplied."""
    scope = ColorScope(scope)
    specs = []
    for group in range(slice_.n_groups):
        fallback = default_color_spec(slice_.value_range(group, scope), cmap=cmap)
        spec = _override(color, group) or fallback
        problems = spec.validate()
        if problems:
            raise ValueError(f"color for group {group}: {'; '.join(problems)}")
        specs.append(spec)
    return specs
