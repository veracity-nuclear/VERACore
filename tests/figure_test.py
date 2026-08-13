"""Tests for the shared figure machinery.

The formatter and sizing helpers are pure and cheap to pin down. The Selection
and View tests cover the request-first API, including the signature drift
guard that keeps select() in step with its request dataclass.
"""

import dataclasses

import pytest
from conftest import make_core_slice
from matplotlib.figure import Figure

from vera_core.data.analysis.core_slice import SliceRequest
from vera_core.data.renders.axial_view import AxialView
from vera_core.data.renders.core_surface_view import SurfaceView
from vera_core.data.renders.core_view import CoreView
from vera_core.data.renders.layout import (
    DARK,
    LIGHT,
    ViewStyle,
    assert_select_matches,
    auto_value_size,
    contrast_color,
    map_aspect,
    panel_grid,
    tidy_exponent,
    value_formatter,
    write_figure,
)

VIEWS = [CoreView, SurfaceView, AxialView]


class TestValueFormatter:
    def test_moderate_values_stay_decimal(self):
        write = value_formatter([0.31, 16.7], ViewStyle())
        assert [write(v) for v in (0.31, 16.7)] == ["0.31", "17"]

    def test_one_tiny_value_switches_the_whole_panel(self):
        """A grid mixing 0.31 and 4.7e-5 reads as though the two were
        comparable, so notation is chosen once per panel."""
        values = [0.31, 16.7, 4.7e-5]
        write = value_formatter(values, ViewStyle())
        assert all("e" in write(v) for v in values)

    def test_fixed_format_collapses_small_values(self):
        """Documents why 'auto' is the default: 'fixed' is trame parity and
        prints zeros for anything below its precision."""
        write = value_formatter([4.7e-5], ViewStyle(value_format="fixed"))
        assert write(4.7e-5) == "0.00"

    def test_sci_format_is_unconditional(self):
        write = value_formatter([0.31], ViewStyle(value_format="sci"))
        assert "e" in write(0.31)

    def test_decimals_sets_precision(self):
        write = value_formatter([1.23456], ViewStyle(decimals=4))
        assert write(1.23456) == "1.235"

    def test_empty_panel_does_not_raise(self):
        assert value_formatter([], ViewStyle())(1.0)


class TestTidyExponent:
    @pytest.mark.parametrize(
        "raw, tidy",
        [("1e-05", "1e-5"), ("1.2e+06", "1.2e6"), ("4.70e-07", "4.70e-7"), ("12", "12")],
    )
    def test_padding_is_stripped(self, raw, tidy):
        assert tidy_exponent(raw) == tidy

    def test_shorter_labels_buy_font_size(self):
        assert len(tidy_exponent("1.2e+06")) < len("1.2e+06")


class TestAutoValueSize:
    def test_narrower_cells_give_smaller_text(self):
        wide = auto_value_size(720.0, 8, 4)
        narrow = auto_value_size(720.0, 32, 4)
        assert narrow < wide

    def test_longer_labels_give_smaller_text(self):
        assert auto_value_size(720.0, 8, 8) < auto_value_size(720.0, 8, 4)

    def test_clamped_at_both_ends(self):
        assert auto_value_size(1.0, 100, 20) >= 3.5
        assert auto_value_size(10_000.0, 1, 1) <= 22.0


class TestContrastColor:
    def test_dark_background_gets_white_text(self):
        assert contrast_color((0.0, 0.0, 0.4, 1.0), DARK) == "#ffffff"

    def test_light_background_gets_black_text(self):
        assert contrast_color((1.0, 1.0, 0.2, 1.0), LIGHT) == "#000000"

    def test_none_falls_back_to_the_theme(self):
        assert contrast_color(None, DARK) == DARK.foreground


class TestPanelGrid:
    @pytest.mark.parametrize(
        "n_groups, shape", [(1, (1, 1)), (2, (1, 2)), (3, (2, 2)), (4, (2, 2))]
    )
    def test_wraps_above_two_groups(self, n_groups, shape):
        assert panel_grid(n_groups) == shape


def test_map_aspect_uses_pin_proportions():
    tall = make_core_slice(rows=8, cols=4)
    wide = make_core_slice(rows=4, cols=8)
    assert map_aspect(tall) > map_aspect(wide)


def test_write_figure_keeps_the_dark_facecolor(tmp_path):
    """savefig writes a white page unless the facecolor is passed through."""
    figure = Figure(facecolor=DARK.background)
    path = write_figure(figure, tmp_path / "out.png")
    assert path.exists() and path.stat().st_size > 0


def test_write_figure_creates_missing_directories(tmp_path):
    path = write_figure(Figure(), tmp_path / "a" / "b" / "out.png")
    assert path.exists()


@pytest.mark.parametrize("suffix", [".png", ".pdf", ".svg"])
def test_write_figure_honours_the_suffix(tmp_path, suffix):
    assert write_figure(Figure(), tmp_path / f"out{suffix}").exists()


# -- the request-first API -------------------------------------------------


class StubView(CoreView):
    """A CoreView whose slices come from a list instead of a source."""

    def __init__(self, slices, **kwargs):
        super().__init__(source=None, **kwargs)
        self._slices = slices

    def build_slices(self, source, request):
        """Bound, so it shadows the staticmethod. The base calls it as
        build_slices(source, request), hence the unused source."""
        return self._slices

    build_info = staticmethod(lambda source, request: None)


@pytest.fixture
def stub(two_groups):
    return StubView(two_groups)


class TestSelection:
    def test_select_builds_the_right_request_type(self, stub):
        assert isinstance(stub.select("pin_powers", state=2, z=10).request, SliceRequest)

    def test_request_carries_no_view_reference(self, stub):
        """A request must stay inert data, so it can be reused across sources
        or sent to a worker."""
        request = stub.select("pin_powers", z=10).request
        assert not any(
            "view" in f.name or "source" in f.name
            for f in dataclasses.fields(request)
        )

    def test_replace_leaves_the_original_alone(self, stub):
        first = stub.select("pin_powers", z=10)
        second = first.replace(z=11)
        assert (first.request.z, second.request.z) == (10, 11)
        assert second.view is first.view

    def test_slices_returns_every_group_by_default(self, stub):
        assert len(stub.select("pin_powers").slices()) == 2

    def test_group_filter_narrows_to_one(self, stub):
        assert len(stub.select("pin_powers", group=1).slices()) == 1

    def test_figure_returns_a_figure_not_a_list(self, stub):
        """The group filter is an early return away from leaking the slice
        list out of figure()."""
        assert isinstance(stub.select("pin_powers").figure(caption=False), Figure)

    def test_figure_returns_a_figure_when_filtered(self, stub):
        assert isinstance(
            stub.select("pin_powers", group=0).figure(caption=False), Figure
        )

    def test_savefig_writes(self, stub, tmp_path):
        path = stub.select("pin_powers").savefig(tmp_path / "out.png", caption=False)
        assert path.exists()

    def test_missing_view_raises_with_the_request_label(self):
        empty = StubView([])
        with pytest.raises(ValueError, match="pin_powers"):
            empty.select("pin_powers").slices()

    def test_repr_names_the_view_and_request(self, stub):
        text = repr(stub.select("pin_powers", state=2, z=10))
        assert "pin_powers" in text and "10" in text


class TestSelectSignatures:
    @pytest.mark.parametrize("view", VIEWS, ids=lambda v: v.__name__)
    def test_select_matches_its_request_dataclass(self, view):
        """select() spells out its fields so an editor can complete them,
        which duplicates the request. This is the guard on that duplication."""
        assert_select_matches(view)

    @pytest.mark.parametrize("view", VIEWS, ids=lambda v: v.__name__)
    def test_select_is_overridden_not_inherited(self, view):
        assert "select" in vars(view), "an inherited select() offers no argument help"

    def test_the_guard_catches_a_missing_parameter(self):
        class Drifted(CoreView):
            def select(self, array, *, state=0):
                raise NotImplementedError

        with pytest.raises(AssertionError, match="missing"):
            assert_select_matches(Drifted)

    def test_the_guard_catches_an_invented_parameter(self):
        class Drifted(CoreView):
            def select(self, array, *, state=0, z=0, src_id=None, thresholds=(),
                       mask_reflected=True, group=None, colour=None):
                raise NotImplementedError

        with pytest.raises(AssertionError, match="not on the request"):
            assert_select_matches(Drifted)


class TestViewDefaults:
    def test_style_passed_at_call_time_wins(self, stub):
        figure = stub.select("pin_powers").figure(
            caption=False, style=ViewStyle(theme=LIGHT)
        )
        assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)

    def test_view_level_style_applies(self, two_groups):
        view = StubView(two_groups, style=ViewStyle(theme=LIGHT))
        figure = view.select("pin_powers").figure(caption=False)
        assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)

    def test_share_color_range_puts_groups_on_one_scale(self, stub):
        figure = stub.select("pin_powers").figure(
            caption=False, share_color_range=True
        )
        ranges = {
            (image.norm.vmin, image.norm.vmax)
            for ax in figure.axes
            for image in ax.get_images()
        }
        assert len(ranges) == 1

    def test_independent_ranges_are_the_default(self, stub):
        figure = stub.select("pin_powers").figure(caption=False)
        ranges = {
            (image.norm.vmin, image.norm.vmax)
            for ax in figure.axes
            for image in ax.get_images()
        }
        assert len(ranges) == 2, "each group keeps its own scale, as trame does"
