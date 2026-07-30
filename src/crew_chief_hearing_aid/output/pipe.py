"""Named-pipe sink for a forked CrewChief.

Not usable against stock CrewChief. This is the phase-2 output: it writes the
canonical command phrase to a named pipe that a patched CrewChief reads and
feeds into its recognised-text handler, which unlocks every voice command
rather than only the subset exposed in Add/Remove Actions (gap ahead/behind
being the notable omissions).

The C# side is roughly: refactor `sre_SpeechRecognized` so the body after
`e.Result.Text` is extracted into

    internal void handleRecognisedText(string text, float confidence)

then have both the SAPI event and a NamedPipeServerStream reader call it. Do
not call `getEventForSpeech` directly — it is only part of the dispatch, and
macros, iRacing pit commands and the mute toggles live in the other branches.
"""

from __future__ import annotations

import logging

from ..intent.phrases import Intent

log = logging.getLogger(__name__)


class NamedPipeSink:
    def __init__(self, pipe_name: str = "crewchief-voice", timeout_ms: int = 500) -> None:
        self.pipe_path = rf"\\.\pipe\{pipe_name}"
        self.timeout_ms = timeout_ms

    def _write(self, payload: str) -> bool:
        # Opened per-message rather than held: a long-lived handle goes stale
        # whenever CrewChief restarts, and CrewChief restarts more often than
        # this process does.
        try:
            with open(self.pipe_path, "w", encoding="utf-8") as pipe:
                pipe.write(payload + "\n")
            return True
        except OSError as exc:
            log.error("named pipe %s unavailable: %s", self.pipe_path, exc)
            return False

    def preflight(self, intents: list[Intent]) -> list[str]:
        missing = [i.id for i in intents if not i.phrases]
        problems = [f"intent {i!r} has no canonical phrase to send" for i in missing]
        try:
            with open(self.pipe_path, "w", encoding="utf-8"):
                pass
        except OSError:
            problems.append(
                f"named pipe {self.pipe_path} not reachable — is the patched CrewChief running?"
            )
        return problems

    def fire(self, intent: Intent) -> bool:
        # The first phrase is canonical: it is what CrewChief's own grammar
        # expects, whereas later entries are natural-language paraphrases meant
        # only for our matcher.
        return self._write(intent.phrases[0])

    def close(self) -> None:
        return None
