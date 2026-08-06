import json
from pathlib import Path
from typing import Any

from platformdirs import user_data_dir

from vera_core.data.dtypes import CoreOverride, FileOverrides

APP_NAME = "VERACore"
APP_AUTHOR = "VeracityNuclear"
MAX_RECENT = 10


def _prefs_path() -> Path:
    directory = Path(user_data_dir(APP_NAME, APP_AUTHOR))
    directory.mkdir(parents=True, exist_ok=True)
    return directory / "prefs.json"


def _validate_recent(value: object) -> list[str]:
    """Return existing file paths from an untrusted JSON value."""
    if not isinstance(value, list):
        return []

    recent: list[str] = []

    for item in value:
        if not isinstance(item, str):
            continue

        path = Path(item)
        if not path.is_file():
            continue

        recent.append(item)

        if len(recent) >= MAX_RECENT:
            break

    return recent


def _validate_core_override(value: object) -> CoreOverride:
    """Validate overrides for one file."""
    if not isinstance(value, dict):
        return {}

    override: CoreOverride = {}

    npin = value.get("npin")
    if isinstance(npin, int) and not isinstance(npin, bool) and npin >= 0:
        override["npin"] = npin

    nax = value.get("nax")
    if isinstance(nax, int) and not isinstance(nax, bool) and nax > 0:
        override["nax"] = nax

    return override


def validate_file_overrides(value: object) -> FileOverrides:
    """Validate the file-path-to-core-overrides mapping."""
    if not isinstance(value, dict):
        return {}

    file_overrides: FileOverrides = {}

    for raw_path, raw_override in value.items():
        if not isinstance(raw_path, str):
            continue

        path = Path(raw_path)
        if not path.is_file():
            continue

        override = _validate_core_override(raw_override)
        file_overrides[raw_path] = override

    return file_overrides


def _load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object, returning an empty dictionary on failure."""
    try:
        raw_data: object = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}

    if not isinstance(raw_data, dict):
        return {}

    return raw_data


def load_prefs() -> tuple[list[str], FileOverrides]:
    """Return validated recent paths and file overrides. Never raises."""
    data = _load_json_object(_prefs_path())

    recent = _validate_recent(data.get("recent"))
    file_overrides = validate_file_overrides(data.get("file_overrides"))

    return recent, file_overrides


def save_prefs(recent: list[str], file_overrides: FileOverrides) -> None:
    """Persist validated preferences. Write failures are silent."""
    validated_recent = _validate_recent(recent)
    validated_overrides = validate_file_overrides(file_overrides)

    data = {
        "recent": validated_recent,
        "file_overrides": validated_overrides,
    }

    try:
        _prefs_path().write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass
