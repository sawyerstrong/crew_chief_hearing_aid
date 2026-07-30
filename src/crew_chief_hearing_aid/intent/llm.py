"""Tier 4: Claude Haiku 4.5 as a tool-calling intent router.

Only consulted when tiers 1-3 reject (D8). That cascade is what keeps the
network off the critical path for the common case -- most utterances resolve
locally in microseconds and never reach this module.

Three properties matter more than accuracy, because each maps to a failure this
project exists to avoid:

* **Forced tool choice.** `tool_choice={"type": "any"}` guarantees a structured
  call rather than prose, so an invalid tool name is *unrepresentable* rather
  than merely unlikely. This is the API-layer equivalent of a constrained
  grammar.
* **An explicit `no_match` tool.** Forcing a call without one would reintroduce
  exactly the confident-wrong-answer failure (P3) that motivated the project:
  a model obliged to pick something will pick the nearest thing.
* **It never raises.** A timeout, a dropped connection, a rate limit, or a
  missing API key returns "no intents" and the caller keeps the tier-3 result.
  Failing a voice command is acceptable; taking down the pipeline mid-race is
  not.

Parallel tool use is deliberately left enabled: "fuel and lap time ahead?" is a
legitimate two-tool answer, and multi-intent is one of the two things the LLM
tier buys under keypress output.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .phrases import Intent

log = logging.getLogger(__name__)

NO_MATCH_TOOL = "no_match"

SYSTEM_PROMPT = (
    "You route a sim racing driver's spoken request to one or more crew-chief "
    "commands. The transcript comes from speech recognition and may be terse, "
    "clipped, or slightly misheard.\n\n"
    "Call every tool the driver asked for -- a request like \"fuel and the gap "
    "ahead\" is two calls. Call them in the order the driver said them.\n\n"
    f"If the transcript is not a request for any listed command -- conversation, "
    f"radio chatter, someone else talking, a misfire -- call {NO_MATCH_TOOL}. "
    f"Guessing the nearest command is worse than {NO_MATCH_TOOL}: a wrong "
    "command costs the driver a lap."
)


@runtime_checkable
class MessagesClient(Protocol):
    """The slice of the Anthropic client this module uses.

    Narrow on purpose so tests can substitute a stub without the SDK, without
    an API key, and without a network call (AC6.6).
    """

    def create(self, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class RouteResult:
    intents: list[Intent]
    latency_ms: float
    failed: bool = False
    reason: str | None = None

    @property
    def matched(self) -> bool:
        return bool(self.intents)


def build_tools(intents: list[Intent]) -> list[dict[str, Any]]:
    """One zero-argument tool per intent, plus `no_match`.

    Zero-argument because a keypress carries no payload -- there is nowhere to
    put an extracted argument. Arguments become possible only with the
    named-pipe fork.
    """
    tools: list[dict[str, Any]] = [
        {
            "name": intent.id,
            "description": intent.description,
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
        for intent in intents
    ]
    tools.append(
        {
            "name": NO_MATCH_TOOL,
            "description": (
                "The transcript is not a request for any of the other commands. "
                "Use this for conversation, background speech, partial or garbled "
                "audio, or anything you are not confident about."
            ),
            "input_schema": {"type": "object", "properties": {}, "required": []},
        }
    )
    return tools


class HaikuRouter:
    def __init__(
        self,
        intents: list[Intent],
        *,
        client: MessagesClient | None = None,
        model: str = "claude-haiku-4-5",
        timeout_s: float = 2.5,
        max_tokens: int = 512,
    ) -> None:
        self.intents = intents
        self.model = model
        self.timeout_s = timeout_s
        self.max_tokens = max_tokens
        self._by_id = {i.id: i for i in intents}
        self._tools = build_tools(intents)
        self._client = client
        self._client_failed = False

    # -- client ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """True if a call could plausibly succeed. Never raises."""
        if self._client is not None:
            return not self._client_failed
        return bool(os.environ.get("ANTHROPIC_API_KEY"))

    def _get_client(self) -> MessagesClient | None:
        if self._client is not None:
            return self._client
        if self._client_failed:
            return None
        try:
            import anthropic

            # Retries are actively harmful here: a second attempt costs another
            # timeout while the driver waits, and the tier-3 fallback is already
            # sitting there. Fail fast, fall back.
            self._client = anthropic.Anthropic(
                max_retries=0, timeout=self.timeout_s
            ).messages
        except Exception as exc:  # noqa: BLE001 - never propagate into the pipeline
            log.error("could not construct Anthropic client: %s", exc)
            self._client_failed = True
            return None
        return self._client

    # -- routing --------------------------------------------------------

    def _parse(self, response: Any) -> list[Intent]:
        out: list[Intent] = []
        for block in getattr(response, "content", []) or []:
            if getattr(block, "type", None) != "tool_use":
                continue
            name = getattr(block, "name", "")
            if name == NO_MATCH_TOOL:
                # An explicit refusal. Honour it and discard anything else --
                # a response containing no_match is not a confident routing.
                return []
            intent = self._by_id.get(name)
            if intent is None:
                # Should be unrepresentable given the tool list, but a wrong
                # command is the exact failure being designed out, so drop it.
                log.warning("model returned unknown tool %r; ignoring", name)
                continue
            out.append(intent)
        return out

    def route(self, transcript: str) -> RouteResult:
        if not transcript.strip():
            return RouteResult([], 0.0, failed=False, reason="empty_transcript")

        client = self._get_client()
        if client is None:
            return RouteResult([], 0.0, failed=True, reason="no_client")

        t0 = time.perf_counter()
        try:
            response = client.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=SYSTEM_PROMPT,
                tools=self._tools,
                # Forces a structured call; an invalid tool name cannot be emitted.
                tool_choice={"type": "any"},
                messages=[{"role": "user", "content": transcript}],
            )
        except Exception as exc:  # noqa: BLE001 - AC6.4: never raise into the pipeline
            elapsed = (time.perf_counter() - t0) * 1000
            log.warning("Haiku route failed after %.0fms: %s", elapsed, exc)
            return RouteResult([], elapsed, failed=True, reason=type(exc).__name__)

        elapsed = (time.perf_counter() - t0) * 1000
        intents = self._parse(response)
        log.info(
            "Haiku routed %r -> %s in %.0fms",
            transcript,
            [i.id for i in intents] or "no_match",
            elapsed,
        )
        return RouteResult(intents, elapsed)
