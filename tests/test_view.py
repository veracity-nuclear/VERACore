"""RenderOptions, Selection and the sweep. The refactor moved three fields
into the style, so the tests that matter most are the ones that prove the
call's style reaches the color scale."""

import pytest

from vera_core.data.analysis.color import ColorScope, ColorSpec
from vera_core.data.renders.styles import ViewStyle
from vera_core.data.renders.view import RenderOptions, Selection, View


# -- RenderOptions ---------------------------------------------------------


@pytest.mark.parametrize("field", ["caption", "color_scope", "panel_width"])
def test_moved_fields_are_gone_from_render_options(field):
    """They live on ViewStyle now. Failing loudly beats being ignored."""
    with pytest.raises(TypeError):
        RenderOptions(**{field: True})


class _TitleView(View):
    def default_title(self, selection):
        return "derived"


def test_resolved_title_modes():
    view = _TitleView(None)
    selection = Selection(view, array="flux")
    assert RenderOptions(title=True).resolved_title(view, selection) == "derived"
    assert RenderOptions(title="mine").resolved_title(view, selection) == "mine"
    assert RenderOptions(title=False).resolved_title(view, selection) is None
    assert RenderOptions(title="").resolved_title(view, selection) is None


# -- Selection -------------------------------------------------------------


def test_params_read_back_as_attributes(stub_view):
    selection = stub_view.select(state=2, z=7)
    assert selection.state == 2 and selection.z == 7
    with pytest.raises(AttributeError):
        selection.nonexistent


def test_reserved_names_cannot_be_selected_on(stub_view):
    with pytest.raises(TypeError):
        Selection(stub_view, title="x")


def test_replace_goes_back_through_select(stub_view):
    """An unknown name has to raise, which is what routing through select
    buys: the view's signature is the only place params are defined."""
    selection = stub_view.select(state=0)
    assert selection.replace(state=2).state == 2
    with pytest.raises(TypeError):
        selection.replace(nonexistent=1)


def test_replace_carries_the_selection_title(stub_view):
    selection = stub_view.select(state=0)
    selection.title = "kept"
    assert selection.replace(state=1).title == "kept"


def test_label_hides_none_and_empty_tuples(stub_view):
    selection = Selection(stub_view, array="flux", z=10, state=None, thresholds=())
    label = selection.label()
    assert "flux" in label and "z=10" in label
    assert "state" not in label and "thresholds" not in label


def test_options_overrides_view_defaults_only_where_named(stub_view):
    stub_view.options = RenderOptions(style=ViewStyle(cmap="magma"), title="default")
    options = stub_view.select().options(style=ViewStyle(cmap="viridis"))
    assert options.style.cmap == "viridis"
    assert options.title == "default"


def test_selection_title_beats_the_view_but_not_the_call(stub_view):
    selection = stub_view.select()
    selection.title = "from selection"
    assert selection.options().title == "from selection"
    assert selection.options(title="from call").title == "from call"


# -- rendering -------------------------------------------------------------


def test_style_cmap_reaches_every_spec(stub_view):
    stub_view.select().figure(style=ViewStyle(cmap="magma", show_caption=False))
    assert [spec.cmap for spec in stub_view.last_specs] == ["magma", "magma"]


def test_style_color_scope_reaches_the_slice(stub_view):
    stub_view.select().figure(style=ViewStyle(color_scope="slice_group", show_caption=False))
    assert set(stub_view.last_slice.scopes) == {ColorScope.SLICE_GROUP}


def test_explicit_color_still_wins_over_the_style(stub_view):
    """A fixed range is dataset-specific, so it stays on RenderOptions."""
    fixed = ColorSpec(vmin=-5.0, vmax=5.0, cmap="plasma")
    stub_view.select().figure(color=fixed, style=ViewStyle(cmap="magma", show_caption=False))
    assert all(spec.range == (-5.0, 5.0) for spec in stub_view.last_specs)
    assert all(spec.cmap == "plasma" for spec in stub_view.last_specs)


def test_show_caption_false_suppresses_the_footer(stub_view):
    figure = stub_view.select().figure(style=ViewStyle(show_caption=False))
    assert figure._supxlabel is None
    figure = stub_view.select().figure(style=ViewStyle(show_caption=True))
    assert figure._supxlabel is not None


@pytest.mark.parametrize("field", ["caption", "color_scope", "panel_width"])
def test_savefig_rejects_the_moved_arguments(stub_view, tmp_path, field):
    with pytest.raises(TypeError):
        stub_view.select().savefig(tmp_path / "x.png", **{field: True})


# -- sweeps ----------------------------------------------------------------


def test_sweep_uses_the_call_style_for_the_shared_scale(stub_view):
    """Before the refactor the colormap came from the view's default color,
    so a sweep and a still could disagree."""
    stub_view.select().collage(
        over="state", style=ViewStyle(cmap="magma", color_scope="slice_group")
    )
    slices, style = stub_view.shared_color_calls[-1]
    assert style.cmap == "magma"
    assert style.color_scope is ColorScope.SLICE_GROUP
    assert len(slices) == 3  # one per state in the stub source


def test_sweep_holds_every_frame_to_one_scale(stub_view):
    """Frame ranges differ by state, so an unshared scale would show change
    that is not in the data."""
    stub_view.select().collage(over="state", style=ViewStyle(show_caption=False))
    group_zero = stub_view.last_specs[0]
    assert group_zero.vmin == pytest.approx(0.0)  # state 0, group 0
    assert group_zero.vmax == pytest.approx(3.0)  # state 2, group 0


def test_explicit_color_skips_the_shared_scale(stub_view):
    stub_view.select().collage(over="state", color=ColorSpec(vmin=0.0, vmax=1.0, cmap="jet"))
    assert stub_view.shared_color_calls == []


def test_sweep_rejects_a_parameter_the_view_does_not_select_on(stub_view):
    with pytest.raises(TypeError):
        stub_view.select().collage(over="assembly")


def test_sweep_rejects_an_empty_value_list(stub_view):
    with pytest.raises(ValueError):
        stub_view.select().collage(over="state", values=[])


def test_a_view_without_a_scale_says_so(stub_view):
    """shared_color returning None is how the plot view opts out; the sweep
    must leave color unset rather than treating None as a failure."""
    stub_view.shared_color = lambda slices, style: None
    frames, slices, options = stub_view.select()._sweep("state", None, None, None, None)
    assert options.color is None
    assert len(frames) == len(slices) == 3
