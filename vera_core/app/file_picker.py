# file_picker.py
import sys
import subprocess

def _pick_macos():
    script = (
        'tell application "System Events" to activate\n'
        'set f to choose file with prompt "Open VERA Output file" '
        'of type {"h5", "hdf5"}\n'
        'POSIX path of f'
    )
    try:
        out = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True,
        )
    except Exception:
        return ""
    if out.returncode != 0:
        return ""
    return out.stdout.strip()

def _pick_tk():
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Open VERA Output file",
        filetypes=[("HDF5 files", "*.h5"), ("All files", "*.*")],
    )
    root.destroy()
    return path or ""

def run_picker():
    path = _pick_macos() if sys.platform == "darwin" else _pick_tk()
    sys.stdout.write(path)
    sys.stdout.flush()

if __name__ == "__main__":
    run_picker()