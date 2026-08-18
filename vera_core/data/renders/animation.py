"""Many figures, one animated file.

Sits beside canvas.py: a canvas is one image, this is a sequence of them.
"""

import io
from collections.abc import Iterable
from pathlib import Path

from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from PIL import Image

from .canvas import DEFAULT_DPI, PAD_INCHES

DEFAULT_FPS = 2.0
GIF_SUFFIX = ".gif"
VIDEO_SUFFIXES = (".mp4", ".avi", ".mov", ".webm", ".mkv")
MACRO_BLOCK = 16
"""Video encoders want dimensions in multiples of this. Padding to it keeps
the encoder from quietly resampling every frame."""


def write_animation(
    figures: Iterable[Figure],
    path: str | Path,
    *,
    fps: float = DEFAULT_FPS,
    dpi: int = DEFAULT_DPI,
    loop: int = 0,
) -> Path:
    """Write figures as one animation. The suffix picks the format.

        .gif                        Pillow only, loops by default
        .mp4 .avi .mov .webm .mkv   needs imageio and ffmpeg
        no suffix                   a directory of numbered PNGs

    loop is gif only: 0 forever, 1 once.

    Frames are held in memory to find the common size, so a long sweep at a
    high dpi costs real memory; lower dpi or panel_width if it bites.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()

    background = "#000000"
    frames: list[Image.Image] = []
    for figure in figures:
        background = to_hex(figure.get_facecolor())
        frames.append(_rasterize(figure, dpi))
    if not frames:
        raise ValueError("no frames to write")

    if not suffix:
        return _write_frames(frames, path)
    if suffix == GIF_SUFFIX:
        return _write_gif(_padded(frames, background), path, fps, loop)
    if suffix in VIDEO_SUFFIXES:
        return _write_video(_padded(frames, background, MACRO_BLOCK), path, fps)
    raise ValueError(
        f"cannot write {suffix or 'a directory'} as an animation;"
        f" use {GIF_SUFFIX}, one of {' '.join(VIDEO_SUFFIXES)}, or no suffix for PNG frames"
    )


def _rasterize(figure: Figure, dpi: int) -> Image.Image:
    """One figure as an image, cropped the way write_figure crops."""
    buffer = io.BytesIO()
    figure.savefig(
        buffer,
        format="png",
        dpi=dpi,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=PAD_INCHES,
    )
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("RGB")


def _padded(frames: list[Image.Image], background: str, multiple: int = 1) -> list[Image.Image]:
    """Every frame on the same canvas size, centered."""
    width = max(frame.width for frame in frames)
    height = max(frame.height for frame in frames)
    if multiple > 1:
        width += -width % multiple
        height += -height % multiple
    if all(frame.size == (width, height) for frame in frames):
        return frames
    padded = []
    for frame in frames:
        sheet = Image.new("RGB", (width, height), background)
        sheet.paste(frame, ((width - frame.width) // 2, (height - frame.height) // 2))
        padded.append(sheet)
    return padded


def _write_gif(frames: list[Image.Image], path: Path, fps: float, loop: int) -> Path:
    frames[0].save(
        path,
        save_all=True,
        append_images=frames[1:],
        duration=round(1000.0 / fps),
        loop=loop,
        optimize=True,
    )
    return path


def _write_video(frames: list[Image.Image], path: Path, fps: float) -> Path:
    try:
        import imageio.v2 as imageio
    except ImportError:
        raise ImportError(
            f"writing {path.suffix} needs imageio and ffmpeg;"
            f" install them or write {GIF_SUFFIX} instead"
        ) from None
    import numpy as np

    imageio.mimwrite(path, [np.asarray(frame) for frame in frames], fps=fps)
    return path


def _write_frames(frames: list[Image.Image], directory: Path) -> Path:
    """The frames as numbered PNGs, for a caller assembling them elsewhere."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(frames):
        frame.save(directory / f"frame_{index:04d}.png")
    return directory
