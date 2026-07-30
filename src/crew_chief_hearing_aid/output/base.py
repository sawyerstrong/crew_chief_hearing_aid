"""Output sinks.

The pipeline depends only on this interface, which is what makes the eventual
move from synthetic keypresses to a forked CrewChief's named pipe a swap of one
class rather than a rewrite. Keypress is the v1 because it needs no changes to
CrewChief and therefore keeps the auto-updater.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..intent.phrases import Intent

log = logging.getLogger(__name__)


@runtime_checkable
class Sink(Protocol):
    def fire(self, intent: Intent) -> bool:
        """Dispatch the intent. Returns True if it was delivered."""
        ...

    def preflight(self, intents: list[Intent]) -> list[str]:
        """Return a list of human-readable problems, empty if ready.

        Called at startup so misconfiguration fails loudly at launch instead of
        silently no-op'ing mid-race.
        """
        ...

    def close(self) -> None: ...


class LogSink:
    """Dry run. Logs what would have fired without touching the system."""

    def __init__(self) -> None:
        self.fired: list[str] = []

    def fire(self, intent: Intent) -> bool:
        self.fired.append(intent.id)
        log.info("DRY RUN would fire %s (%s -> %s)", intent.id, intent.action, intent.key)
        return True

    def preflight(self, intents: list[Intent]) -> list[str]:
        return []

    def close(self) -> None:
        return None
