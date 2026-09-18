"""The sweep-to-file path. A directory of PNGs avoids needing an encoder;
tmp_path keeps everything out of the working tree."""

import pytest
from PIL import Image

from vera_core.data.renders.animation import write_animation
from vera_core.data.renders.styles import ViewStyle

LOW_DPI = 30
"""Small frames: these tests check the wiring, not the pixels."""


def test_savemovie_writes_one_frame_per_sweep_value(stub_view, tmp_path):
    out = stub_view.select().savemovie(
        tmp_path / "frames", over="state", dpi=LOW_DPI, style=ViewStyle(show_caption=False)
    )
    assert sorted(p.name for p in out.iterdir()) == [
        "frame_0000.png",
        "frame_0001.png",
        "frame_0002.png",
    ]


def test_savemovie_writes_a_gif(stub_view, tmp_path):
    out = stub_view.select().savemovie(
        tmp_path / "sweep.gif", over="state", dpi=LOW_DPI, style=ViewStyle(show_caption=False)
    )
    with Image.open(out) as image:
        assert image.n_frames == 3


def test_a_movie_reuses_the_slices_the_sweep_already_built(stub_view, tmp_path):
    """Each frame is rendered from the slice _sweep read, so three states
    means three builds, not six."""
    builds = []
    original = stub_view.build_slice

    def counting_build(selection):
        builds.append(selection.state)
        return original(selection)

    stub_view.build_slice = counting_build
    stub_view.select().savemovie(tmp_path / "frames", over="state", dpi=LOW_DPI)
    assert builds == [0, 1, 2]


def test_an_unknown_suffix_is_rejected(tmp_path):
    from matplotlib.figure import Figure

    with pytest.raises(ValueError):
        write_animation([Figure()], tmp_path / "sweep.xyz")


def test_no_frames_is_an_error(tmp_path):
    with pytest.raises(ValueError):
        write_animation([], tmp_path / "frames")
