"""Read-modify-write of the user config, preserving comments.

Setup writes values the user later has to read and reason about (which mic,
which wheel button), so round-tripping through a plain dict-dump would be
actively hostile — it would strip every explanatory comment from the file the
first time setup touched it. tomlkit preserves them.

Only ever writes the *user* config. `config.default.toml` ships in the repo and
is never mutated.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any

from .config import default_config_path, user_config_path

log = logging.getLogger(__name__)


class UserConfigError(RuntimeError):
    pass


def _require_tomlkit():
    try:
        import tomlkit
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise UserConfigError(
            "writing config needs tomlkit: pip install tomlkit"
        ) from exc
    return tomlkit


def ensure_user_config(path: Path | None = None) -> Path:
    """Create the user config from the shipped defaults if it is missing."""
    target = path or user_config_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(default_config_path(), target)
        log.info("created %s", target)
    return target


def set_values(updates: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    """Merge {section: {key: value}} into the user config.

    Creates the file from defaults first if needed, so setup works on a machine
    that has never run `init-config`.
    """
    tomlkit = _require_tomlkit()
    target = ensure_user_config(path)

    doc = tomlkit.parse(target.read_text(encoding="utf-8"))
    for section, values in updates.items():
        if section not in doc:
            doc[section] = tomlkit.table()
        for key, value in values.items():
            doc[section][key] = value

    target.write_text(tomlkit.dumps(doc), encoding="utf-8")
    log.info("updated %s", target)
    return target


def describe_changes(updates: dict[str, dict[str, Any]]) -> str:
    """Render what would be written, for confirmation before writing."""
    lines: list[str] = []
    for section, values in updates.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            if isinstance(value, bool):
                rendered = str(value).lower()  # TOML booleans are lowercase
            elif isinstance(value, str):
                rendered = f'"{value}"'
            else:
                rendered = str(value)
            lines.append(f"{key} = {rendered}")
    return "\n".join(lines)
