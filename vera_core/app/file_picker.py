# file_picker.py
import argparse
import subprocess
import sys

FILTERS = {
    "h5": {"label": "HDF5 files", "exts": ["h5", "hdf5"], "mac_types": ["h5", "hdf5"]},
    "json": {
        "label": "Session files",
        "exts": ["json"],
        "mac_types": ["public.json", "json"],
    },
    "image": {
        "label": "Image files",
        "exts": ["png", "pdf", "svg", "jpg"],
        "mac_types": ["png", "pdf", "svg", "jpg", "jpeg"],
    },
}


def _osascript(script):
    """Run AppleScript; return trimmed stdout, or '' on cancel/error."""
    try:
        out = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _q(s):
    """Escape a string for an AppleScript double-quoted literal."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def _open_macos(prompt, mac_types):
    types = ", ".join(f'"{t}"' for t in mac_types)
    script = (
        'tell application "System Events" to activate\n'
        f'set f to choose file with prompt "{_q(prompt)}" of type {{{types}}}\n'
        "POSIX path of f"
    )
    return _osascript(script)


def _save_macos(prompt, default_name):
    script = (
        'tell application "System Events" to activate\n'
        f'set f to choose file name with prompt "{_q(prompt)}" default name "{_q(default_name)}"\n'
        "POSIX path of f"
    )
    return _osascript(script)


def _open_tk(prompt, label, exts):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    patterns = " ".join(f"*.{e}" for e in exts)
    path = filedialog.askopenfilename(
        title=prompt, filetypes=[(label, patterns), ("All files", "*.*")]
    )
    root.destroy()
    return path or ""


def _save_tk(prompt, default_name, label, exts):
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title=prompt,
        defaultextension=f".{exts[0]}",
        initialfile=default_name,
        filetypes=[(label, f"*.{exts[0]}"), ("All files", "*.*")],
    )
    root.destroy()
    return path or ""


def run_picker(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["open", "save"], default="open")
    p.add_argument("--filter", choices=list(FILTERS), default="h5")
    p.add_argument("--prompt", default="")
    p.add_argument("--name", default="")
    args = p.parse_args(argv)

    spec = FILTERS[args.filter]
    label, exts, mac_types = spec["label"], spec["exts"], spec["mac_types"]
    prompt = args.prompt or ("Save file" if args.mode == "save" else "Open file")
    is_mac = sys.platform == "darwin"

    if args.mode == "save":
        default_name = args.name or f"session.{exts[0]}"
        path = (
            _save_macos(prompt, default_name)
            if is_mac
            else _save_tk(prompt, default_name, label, exts)
        )
    else:
        path = _open_macos(prompt, mac_types) if is_mac else _open_tk(prompt, label, exts)

    sys.stdout.write(path)
    sys.stdout.flush()


if __name__ == "__main__":
    run_picker()
