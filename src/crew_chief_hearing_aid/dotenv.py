"""Minimal .env loader.

Deliberately not python-dotenv: this is ~40 lines, the semantics matter here,
and a secret-bearing code path is worth being able to read end to end.

The one semantic that is load-bearing: **a real environment variable always
wins.** A `.env` file is a convenience for a single machine; an exported var is
an explicit act. Overriding it would silently ignore what the operator set.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

SECRET_KEYS = frozenset({"ANTHROPIC_API_KEY"})


def find_dotenv(start: Path | None = None) -> Path | None:
    """Nearest .env, searching upward from the package toward the repo root."""
    here = start or Path(__file__).resolve().parent
    for directory in (here, *here.parents):
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
        if (directory / ".git").exists():
            break  # do not escape the repo
    return None


def parse(text: str) -> dict[str, str]:
    """Parse KEY=VALUE lines. Supports `export`, quotes, and `#` comments."""
    out: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        # Strip one layer of matching quotes; a bare # only starts a comment
        # when unquoted, so an unquoted value keeps everything before it.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        elif "#" in value:
            value = value.split("#", 1)[0].strip()
        out[key] = value
    return out


def load(path: Path | None = None, *, override: bool = False) -> list[str]:
    """Load a .env into os.environ. Returns the names it set.

    Never logs a value — only names. `override=False` means an already-set
    environment variable is left alone.
    """
    path = path or find_dotenv()
    if path is None or not path.is_file():
        return []

    applied: list[str] = []
    for key, value in parse(path.read_text(encoding="utf-8")).items():
        if not value:
            continue
        if key in os.environ and not override:
            continue
        os.environ[key] = value
        applied.append(key)

    if applied:
        # Names only. Never the value, never a prefix — a project-scoped key's
        # suffix is as sensitive as the whole thing.
        log.debug("loaded from %s: %s", path.name, ", ".join(applied))
    return applied


def describe_source(key: str, path: Path | None = None) -> str:
    """Where a variable came from, for diagnostics. Never returns the value."""
    if key not in os.environ or not os.environ[key]:
        return "not set"
    path = path or find_dotenv()
    if path is not None and path.is_file():
        if key in parse(path.read_text(encoding="utf-8")):
            return f"set (from {path.name})"
    return "set (from environment)"
