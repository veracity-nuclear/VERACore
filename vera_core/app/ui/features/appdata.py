from pathlib import Path
import json
from platformdirs import user_data_dir

APP_NAME = "VERACore"
APP_AUTHOR = "UnofficialVeracityNuclear"
MAX_RECENT = 10

from pathlib import Path
import json
from platformdirs import user_data_dir

APP_NAME = "VERACore"
APP_AUTHOR = "VeracityNuclear"
MAX_RECENT = 10

def _prefs_path() -> Path:
    d = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    d.mkdir(parents=True, exist_ok=True)
    return d / "prefs.json"

def load_prefs() -> tuple[list[str], dict]:
    """Return (recent_file_paths, core_overrides). Never raises."""
    try:
        data = json.loads(_prefs_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        data = {}
    if not isinstance(data, dict):
        data = {}

    recent = data.get("recent", [])
    if not isinstance(recent, list):
        recent = []
    recent = [p for p in recent if isinstance(p, str) and Path(p).is_file()][:MAX_RECENT]

    overrides = data.get("core_overrides", {})
    if not isinstance(overrides, dict):
        overrides = {}
    overrides = {k: v for k, v in overrides.items() if isinstance(v, dict) and Path(k).is_file()}
    return recent, overrides

def save_prefs(recent: list[str], core_overrides: dict) -> None:
    """Persist both. Write failures are silent."""
    try:
        _prefs_path().write_text(json.dumps({
            "recent": list(recent)[:MAX_RECENT],
            "core_overrides": dict(core_overrides),
        }, indent=2))
    except OSError:
        pass