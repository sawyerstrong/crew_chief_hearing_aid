"""Layered configuration.

`config.default.toml` ships in the repo and is the single source of truth for
defaults. A per-machine `%APPDATA%\\crew_chief_hearing_aid\\config.toml` is deep-merged over
it and is gitignored, because everything that differs between machines —
microphone name, key bindings, model paths — lives there.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .intent.phrases import Intent

APP_NAME = "crew_chief_hearing_aid"


def default_config_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "config.default.toml"


def user_config_dir() -> Path:
    if os.name == "nt":
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    else:  # pragma: no cover - the rig is Windows; this keeps tests portable
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_NAME


def user_config_path() -> Path:
    return user_config_dir() / "config.toml"


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass(frozen=True)
class Config:
    raw: dict[str, Any]
    intents: list[Intent]
    source_paths: tuple[Path, ...]

    def section(self, name: str) -> dict[str, Any]:
        value = self.raw.get(name, {})
        return value if isinstance(value, dict) else {}

    def get(self, section: str, key: str, fallback: Any = None) -> Any:
        return self.section(section).get(key, fallback)

    def intent_by_id(self, intent_id: str) -> Intent | None:
        return next((i for i in self.intents if i.id == intent_id), None)


def _build_intents(raw: dict[str, Any]) -> list[Intent]:
    entries = raw.get("intents", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("config defines no [[intents]]")

    intents: list[Intent] = []
    seen_ids: set[str] = set()
    seen_keys: dict[str, str] = {}
    for entry in entries:
        intent = Intent(
            id=str(entry["id"]),
            action=str(entry.get("action", "")),
            key=str(entry["key"]),
            phrases=tuple(str(p) for p in entry.get("phrases", ())),
        )
        if intent.id in seen_ids:
            raise ValueError(f"duplicate intent id {intent.id!r}")
        seen_ids.add(intent.id)

        # Two intents on one key means one of them silently never fires. That is
        # a config error worth failing at startup rather than mid-race.
        key_upper = intent.key.upper()
        if key_upper in seen_keys:
            raise ValueError(
                f"intents {seen_keys[key_upper]!r} and {intent.id!r} "
                f"are both bound to {intent.key!r}"
            )
        seen_keys[key_upper] = intent.id
        intents.append(intent)
    return intents


def load_config(user_path: Path | None = None, default_path: Path | None = None) -> Config:
    default_path = default_path or default_config_path()
    if not default_path.exists():
        raise FileNotFoundError(f"missing packaged defaults at {default_path}")

    with default_path.open("rb") as fh:
        raw = tomllib.load(fh)
    sources = [default_path]

    user_path = user_path if user_path is not None else user_config_path()
    if user_path.exists():
        with user_path.open("rb") as fh:
            override = tomllib.load(fh)
        # A user file that defines [[intents]] replaces the list wholesale
        # rather than merging positionally, which would be incoherent.
        raw = _deep_merge(raw, override)
        sources.append(user_path)

    return Config(raw=raw, intents=_build_intents(raw), source_paths=tuple(sources))
