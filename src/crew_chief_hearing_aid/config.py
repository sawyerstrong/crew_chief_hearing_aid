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

from .intent.phrases import Intent, parse_crewchief_config

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


def upstream_phrase_config_path() -> Path:
    """Vendored snapshot of CrewChief's speech_recognition_config.txt.

    Vendored rather than read from the install because CrewChief's ClickOnce
    layout does not reliably expose it, CI has no CrewChief at all, and G4
    requires the config to load on a machine that has never run the sim. A
    local user override, if present, takes precedence.
    """
    return Path(__file__).resolve().parent.parent.parent / "data" / (
        "speech_recognition_config.upstream.txt"
    )


def load_phrase_source() -> dict[str, list[str]]:
    """Parse the phrase corpus, preferring a local CrewChief override."""
    from .crewchief import user_phrase_override_path

    for path in (user_phrase_override_path(), upstream_phrase_config_path()):
        if path.is_file():
            text = path.read_text(encoding="utf-8", errors="replace")
            return parse_crewchief_config(text)
    return {}


def _build_intents(raw: dict[str, Any], phrase_source: dict[str, list[str]]) -> list[Intent]:
    entries = raw.get("intents", [])
    if not isinstance(entries, list) or not entries:
        raise ValueError("config defines no [[intents]]")

    intents: list[Intent] = []
    seen_ids: set[str] = set()
    seen_keys: dict[str, str] = {}
    for entry in entries:
        intent_id = str(entry["id"])
        phrases = tuple(str(p) for p in entry.get("phrases", ()))
        sre_key = str(entry.get("sre_key", "")).strip()

        # Explicit phrases win; otherwise import from CrewChief's own config.
        if not phrases and sre_key:
            if sre_key not in phrase_source:
                raise ValueError(
                    f"intent {intent_id!r} references sre_key {sre_key!r}, which is not "
                    f"in the phrase corpus. A CrewChief update may have renamed it."
                )
            phrases = tuple(phrase_source[sre_key])
        if not phrases:
            raise ValueError(
                f"intent {intent_id!r} has neither phrases nor a resolvable sre_key"
            )

        intent = Intent(
            id=intent_id,
            action=str(entry.get("action", "")),
            key=str(entry["key"]),
            phrases=phrases,
            description=str(entry.get("description", "")),
            sre_key=sre_key,
        )
        if intent.id in seen_ids:
            raise ValueError(f"duplicate intent id {intent.id!r}")
        seen_ids.add(intent.id)

        # Two intents on one key means one of them silently never fires. That is
        # a config error worth failing at startup rather than mid-race.
        # Normalised so ctrl+shift+F13 and shift+ctrl+F13 collide as they should.
        key_upper = _normalize_key_for_collision(intent.key)
        if key_upper in seen_keys:
            raise ValueError(
                f"intents {seen_keys[key_upper]!r} and {intent.id!r} "
                f"are both bound to {intent.key!r}"
            )
        seen_keys[key_upper] = intent.id
        intents.append(intent)
    return intents


def _normalize_key_for_collision(spec: str) -> str:
    try:
        from .output.keypress import normalize_key

        return normalize_key(spec)
    except Exception:
        # Non-Windows or an unparseable spec: fall back to a plain fold rather
        # than failing config load. keypress.preflight() reports the real error.
        return spec.upper()


def load_config(
    user_path: Path | None = None,
    default_path: Path | None = None,
    phrase_source: dict[str, list[str]] | None = None,
) -> Config:
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

    if phrase_source is None:
        phrase_source = load_phrase_source()

    return Config(
        raw=raw,
        intents=_build_intents(raw, phrase_source),
        source_paths=tuple(sources),
    )
