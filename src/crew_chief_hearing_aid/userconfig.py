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
from pathlib import Path
from typing import Any

from .config import user_config_path

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


USER_TEMPLATE = """# crew_chief_hearing_aid — per-machine settings.
#
# This file is MERGED OVER the shipped config.default.toml, so it only needs
# what differs on this machine. Anything absent here falls through to the
# defaults and picks up future changes automatically.
#
# Deliberately NOT a copy of the defaults: copying them freezes the action list
# and key map at install time, so a later change to the shipped config would be
# silently shadowed by this file forever.
#
# To override the action list, paste [[intents]] blocks here — but note that
# doing so opts you out of updates to them.

[audio]
# Substring match against `crew_chief_hearing_aid devices`. Never an index.
input_device = ""

[ptt]
# Filled in by `crew_chief_hearing_aid setup-ptt`.
enabled = true
device_guid = ""
button_index = -1

[llm]
# Tier 4 needs ANTHROPIC_API_KEY in the environment. Never put a key in here.
enabled = true
"""


def ensure_user_config(path: Path | None = None) -> Path:
    """Create a minimal per-machine override if it is missing.

    Writes only machine-specific keys, not a copy of the defaults — see
    USER_TEMPLATE for why.
    """
    target = path or user_config_path()
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(USER_TEMPLATE, encoding="utf-8")
        log.info("created %s", target)
    return target


def shadows_shipped_intents(path: Path | None = None) -> bool:
    """True if the user config pins its own [[intents]], freezing the key map."""
    target = path or user_config_path()
    if not target.exists():
        return False
    try:
        import tomllib

        with target.open("rb") as fh:
            return bool(tomllib.load(fh).get("intents"))
    except Exception:  # noqa: BLE001 - advisory only
        return False


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
