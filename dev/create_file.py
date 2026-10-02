#!/usr/bin/env python3
"""Clone a real VERA output file and mutate the copy on an interval.

The source file is opened read-only and never modified. CORE, INPUT and any
other non-STATE top-level group are copied verbatim, so core geometry, labels,
meshes and detector maps are exactly what the real reader expects.

Each tick cycles through the three things a refresh has to pick up:
    values change   pin arrays reweighted in every state
    dataset set     one real dataset added back, or dropped
    state count     a STATE group appended, cloned from the source

Reweighting is spatial, not uniform. A uniform multiply is invisible in a
color view because the color range is recomputed from the data, so everything
moves together. These patterns vary the multiplier across the assembly axis
and the pin plane, which changes the shape of the distribution.

Modes:
    replace  rebuild the whole file at a temp path, then os.replace onto the
             target. The reader's open handle keeps the old inode until it
             reopens by path, so there is no torn read.
    append   mutate the live file in place with 'a'. This is what a running
             solver does, and it is not safe to read concurrently.
"""

import argparse
import os
import tempfile
import time
from typing import NamedTuple

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py  # noqa: E402
import numpy as np  # noqa: E402

CORE_GROUP = "CORE"
STATE_PREFIX = "STATE_"
TIME_KEYS = ("exposure", "core_exposure", "exposure_efpd")
DEFAULT_SCALE_NAMES = ("pin_powers",)
PATTERNS = ("mixed", "tilt", "hotspot", "checker", "pin", "uniform")


class Scaling(NamedTuple):
    """Absolute (not cumulative) weighting applied to source values."""

    pattern: str
    amount: float
    phase: int


def _axis_profile(n: int, pattern: str, amount: float, phase: int) -> np.ndarray:
    """Multiplier along one axis, mean roughly 1."""
    if n <= 1:
        return np.ones(1)
    i = np.arange(n)
    t = i / (n - 1)
    if pattern == "tilt":
        return 1.0 + amount * (2.0 * t - 1.0)
    if pattern == "hotspot":
        center = (phase % n) / (n - 1)
        return 1.0 + amount * np.exp(-((t - center) ** 2) / (2 * 0.12**2))
    if pattern == "checker":
        return 1.0 + amount * np.where((i + phase) % 2 == 0, 1.0, -1.0)
    return np.ones(n)


def _pin_bump(ny: int, nx: int, amount: float, phase: int) -> np.ndarray:
    """Gaussian bump over the pin plane, orbiting the assembly center."""
    y = np.linspace(-1.0, 1.0, ny).reshape(ny, 1) if ny > 1 else np.zeros((1, 1))
    x = np.linspace(-1.0, 1.0, nx).reshape(1, nx) if nx > 1 else np.zeros((1, 1))
    angle = 0.7 * phase
    cy, cx = 0.55 * np.sin(angle), 0.55 * np.cos(angle)
    r2 = (y - cy) ** 2 + (x - cx) ** 2
    return 1.0 + amount * np.exp(-r2 / (2 * 0.35**2))


def reweight(values: np.ndarray, sc: Scaling) -> np.ndarray:
    """Apply the spatial pattern to an array of source values.

    The last axis is treated as the assembly index, which is what a core view
    tiles. Axes 0 and 1 are treated as the pin plane when present, which is
    what an assembly view shows.
    """
    out = np.asarray(values, dtype=np.float64)
    if out.size <= 1:
        return out * (1.0 + sc.amount)
    if sc.pattern == "uniform":
        return out * (1.0 + sc.amount * (sc.phase % 4))

    assembly = "tilt" if sc.pattern in ("mixed", "pin") else sc.pattern
    if sc.pattern != "pin":
        out = out * _axis_profile(out.shape[-1], assembly, sc.amount, sc.phase)
    if sc.pattern in ("mixed", "pin") and out.ndim >= 3:
        bump = _pin_bump(out.shape[0], out.shape[1], sc.amount, sc.phase)
        out = out * bump.reshape(bump.shape + (1,) * (out.ndim - 2))
    return out


def _apply(dataset: h5py.Dataset, sc: Scaling) -> None:
    if dataset.dtype.kind == "f":
        dataset[...] = reweight(dataset[...], sc)


def state_names(f: h5py.File) -> list[str]:
    return sorted(k for k in f if k.startswith(STATE_PREFIX))


def describe(source: str) -> None:
    """Print the source layout so dataset names can be picked by hand."""
    with h5py.File(source, "r") as f:
        print(source)
        if CORE_GROUP in f:
            for k in sorted(f[CORE_GROUP]):
                print(f"  CORE/{k:<30} {getattr(f[CORE_GROUP][k], 'shape', '<group>')}")
        names = state_names(f)
        print(f"  {len(names)} states")
        if names:
            for k in sorted(f[names[0]]):
                print(f"  {names[0]}/{k:<24} {getattr(f[names[0]][k], 'shape', '<group>')}")


def optional_datasets(f: h5py.File, scale_names: tuple[str, ...]) -> list[str]:
    """Leaf datasets in the first state that are safe to add and drop."""
    names = state_names(f)
    if not names:
        return []
    return sorted(
        k
        for k, v in f[names[0]].items()
        if isinstance(v, h5py.Dataset) and k not in TIME_KEYS and k not in scale_names
    )


def time_step(f: h5py.File) -> float:
    """Exposure increment between the first two source states, for cloned states."""
    names = state_names(f)
    if len(names) < 2 or "exposure" not in f[names[0]]:
        return 1.0
    first = float(np.asarray(f[names[0]]["exposure"]).ravel()[0])
    second = float(np.asarray(f[names[1]]["exposure"]).ravel()[0])
    return second - first if second > first else 1.0


def _copy_state(src_grp, dst_parent, name, dropped, sc, scale_names) -> None:
    grp = dst_parent.create_group(name)
    grp.attrs.update(src_grp.attrs)
    for key in src_grp:
        if key in dropped:
            continue
        src_grp.copy(key, grp)
        if key in scale_names and isinstance(grp[key], h5py.Dataset):
            _apply(grp[key], sc)


def _bump_time(dst, name, prev_name, step) -> None:
    """Push a cloned state's time axes past the state before it."""
    for key in TIME_KEYS:
        if key in dst[name] and key in dst[prev_name]:
            dst[name][key][...] = np.asarray(dst[prev_name][key]) + step


def build(source, out, n_states, sc, dropped, scale_names, step) -> None:
    """Write a complete file at `out` from `source`, replacing what's there."""
    out_dir = os.path.dirname(os.path.abspath(out)) or "."
    tmp_fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".h5.tmp")
    os.close(tmp_fd)
    try:
        with h5py.File(source, "r") as src, h5py.File(tmp_path, "w") as dst:
            dst.attrs.update(src.attrs)
            for key in src:
                if not key.startswith(STATE_PREFIX):
                    src.copy(key, dst, key)
            names = state_names(src)
            if not names:
                raise ValueError(f"{source} has no {STATE_PREFIX} groups")
            for idx in range(1, n_states + 1):
                target = f"{STATE_PREFIX}{idx:04d}"
                _copy_state(
                    src[names[(idx - 1) % len(names)]], dst, target, dropped, sc, scale_names
                )
                if idx > len(names):
                    _bump_time(dst, target, f"{STATE_PREFIX}{idx - 1:04d}", step)
        os.replace(tmp_path, out)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def append_reweight(source, out, sc, scale_names) -> None:
    """Rewrite scaled datasets from the source values, so weighting stays absolute."""
    with h5py.File(source, "r") as src, h5py.File(out, "a") as dst:
        src_names = state_names(src)
        for idx, grp in enumerate(state_names(dst), start=1):
            src_grp = src[src_names[(idx - 1) % len(src_names)]]
            for key in scale_names:
                ds = dst[grp].get(key)
                if isinstance(ds, h5py.Dataset) and key in src_grp and ds.dtype.kind == "f":
                    ds[...] = reweight(src_grp[key][...], sc)
        dst.flush()


def append_dataset(source, out, name, sc, scale_names) -> None:
    with h5py.File(source, "r") as src, h5py.File(out, "a") as dst:
        src_names = state_names(src)
        for idx, grp in enumerate(state_names(dst), start=1):
            if name in dst[grp]:
                continue
            src_grp = src[src_names[(idx - 1) % len(src_names)]]
            if name not in src_grp:
                continue
            src_grp.copy(name, dst[grp])
            if name in scale_names:
                _apply(dst[grp][name], sc)
        dst.flush()


def drop_dataset(out, name) -> None:
    with h5py.File(out, "a") as f:
        for grp in state_names(f):
            if name in f[grp]:
                del f[grp][name]
        f.flush()


def append_state(source, out, dropped, sc, scale_names, step) -> None:
    with h5py.File(source, "r") as src, h5py.File(out, "a") as dst:
        src_names = state_names(src)
        existing = state_names(dst)
        idx = len(existing) + 1
        target = f"{STATE_PREFIX}{idx:04d}"
        _copy_state(
            src[src_names[(idx - 1) % len(src_names)]], dst, target, dropped, sc, scale_names
        )
        if existing:
            _bump_time(dst, target, existing[-1], step)
        dst.flush()


def run(args) -> None:
    scale_names = tuple(args.scale)
    with h5py.File(args.source, "r") as src:
        available = state_names(src)
        if not available:
            raise SystemExit(f"{args.source} has no {STATE_PREFIX} groups")
        optional = args.optional or optional_datasets(src, scale_names)
        step = time_step(src)

    n_states = min(args.states, len(available)) if args.states else len(available)
    dropped = set(optional)
    sc = Scaling(args.pattern, args.amount, 0)
    build(args.source, args.out, n_states, sc, dropped, scale_names, step)
    print(f"[init] {args.out}: {n_states} states, cloned from {args.source}")
    print(f"       reweighting {', '.join(scale_names)} with {args.pattern} +/-{args.amount}")
    print(f"       held back: {', '.join(sorted(dropped)) or '(nothing)'}")

    tick = 0
    while args.ticks == 0 or tick < args.ticks:
        time.sleep(args.interval)
        tick += 1
        phase = tick % 3
        name = None

        if phase == 1:
            sc = sc._replace(phase=sc.phase + 1)
            action = f"reweighted {', '.join(scale_names)} at phase {sc.phase}"
        elif phase == 2:
            if not optional:
                print(f"[{tick}] no optional datasets in source, skipping")
                continue
            if dropped:
                name = sorted(dropped)[0]
                dropped.discard(name)
                action = f"added {name} to every state"
            else:
                name = sorted(optional)[0]
                dropped.add(name)
                action = f"dropped {name} from every state"
        else:
            n_states += 1
            action = f"appended {STATE_PREFIX}{n_states:04d} (now {n_states} states)"

        if args.mode == "replace":
            build(args.source, args.out, n_states, sc, dropped, scale_names, step)
        elif phase == 1:
            append_reweight(args.source, args.out, sc, scale_names)
        elif phase == 2:
            if name in dropped:
                drop_dataset(args.out, name)
            else:
                append_dataset(args.source, args.out, name, sc, scale_names)
        else:
            append_state(args.source, args.out, dropped, sc, scale_names, step)

        print(f"[{tick}] {action}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source", help="real VERA .h5 file, opened read-only")
    p.add_argument("out", nargs="?", help="mutating copy to point the app at")
    p.add_argument("--list", action="store_true", help="print source layout and exit")
    p.add_argument("--mode", choices=("replace", "append"), default="replace")
    p.add_argument("--interval", type=float, default=10.0, help="seconds between mutations")
    p.add_argument("--states", type=int, default=0, help="initial state count (0 = all)")
    p.add_argument("--ticks", type=int, default=0, help="stop after N mutations (0 = forever)")
    p.add_argument("--pattern", choices=PATTERNS, default="mixed", help="spatial weighting")
    p.add_argument("--amount", type=float, default=0.6, help="weighting strength, 0 = none")
    p.add_argument(
        "--scale", nargs="+", default=list(DEFAULT_SCALE_NAMES), help="datasets to reweight"
    )
    p.add_argument(
        "--optional", nargs="+", default=None, help="datasets to add and drop (default: auto)"
    )
    args = p.parse_args()

    if args.list:
        describe(args.source)
        return
    if not args.out:
        p.error("out path is required unless --list is given")
    if os.path.abspath(args.source) == os.path.abspath(args.out):
        p.error("source and out must differ; the source is never modified")
    try:
        run(args)
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
