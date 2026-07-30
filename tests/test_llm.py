"""Tier 4 tests. No network, no API key, no SDK — a stub client throughout.

AC6.6: CI must never make a network call. Everything here runs offline.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from crew_chief_hearing_aid.intent.llm import (
    NO_MATCH_TOOL,
    HaikuRouter,
    build_tools,
)
from crew_chief_hearing_aid.intent.phrases import Intent


@dataclass
class _Block:
    type: str
    name: str = ""


@dataclass
class _Response:
    content: list


class StubClient:
    """Returns queued responses; records the kwargs it was called with."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def tool_use(*names: str) -> _Response:
    return _Response([_Block("tool_use", n) for n in names])


@pytest.fixture
def intents():
    return [
        Intent(
            id="fuel_status",
            action="Report fuel",
            key="F15",
            description="Remaining fuel.",
            phrases=("hows my fuel",),
        ),
        Intent(
            id="car_ahead_last_lap",
            action="Car ahead lap time",
            key="F13",
            description="Lap time of the car ahead.",
            phrases=("car ahead lap time",),
        ),
    ]


class TestToolConstruction:
    def test_one_tool_per_intent_plus_no_match(self, intents):
        tools = build_tools(intents)
        assert [t["name"] for t in tools] == [
            "fuel_status",
            "car_ahead_last_lap",
            NO_MATCH_TOOL,
        ]

    def test_tools_are_zero_argument(self, intents):
        """A keypress carries no payload, so there is nowhere to put an
        argument. Arguments become possible only with the named-pipe fork."""
        for tool in build_tools(intents):
            assert tool["input_schema"]["properties"] == {}
            assert tool["input_schema"]["required"] == []

    def test_descriptions_are_carried_through(self, intents):
        tools = {t["name"]: t["description"] for t in build_tools(intents)}
        assert tools["fuel_status"] == "Remaining fuel."


class TestRouting:
    def test_single_tool_call(self, intents):
        router = HaikuRouter(intents, client=StubClient(tool_use("fuel_status")))
        result = router.route("hows the fuel looking")
        assert [i.id for i in result.intents] == ["fuel_status"]
        assert not result.failed

    def test_multi_intent_preserves_order(self, intents):
        """"fuel and the lap time ahead" is legitimately two calls."""
        router = HaikuRouter(
            intents, client=StubClient(tool_use("fuel_status", "car_ahead_last_lap"))
        )
        result = router.route("fuel and the lap time ahead")
        assert [i.id for i in result.intents] == ["fuel_status", "car_ahead_last_lap"]

    def test_no_match_returns_nothing(self, intents):
        router = HaikuRouter(intents, client=StubClient(tool_use(NO_MATCH_TOOL)))
        assert router.route("what's for dinner").intents == []

    def test_no_match_discards_other_calls(self, intents):
        """A response containing no_match is not a confident routing.

        Firing a command the model itself flagged as uncertain is precisely the
        P2 failure this project exists to avoid.
        """
        router = HaikuRouter(
            intents, client=StubClient(tool_use("fuel_status", NO_MATCH_TOOL))
        )
        assert router.route("mumble").intents == []

    def test_unknown_tool_name_is_dropped(self, intents):
        """Should be unrepresentable via tool_choice, but a wrong command is the
        exact failure being designed out — so drop rather than guess."""
        router = HaikuRouter(intents, client=StubClient(tool_use("hallucinated")))
        assert router.route("something").intents == []

    def test_text_blocks_are_ignored(self, intents):
        response = _Response([_Block("text"), _Block("tool_use", "fuel_status")])
        router = HaikuRouter(intents, client=StubClient(response))
        assert [i.id for i in router.route("fuel").intents] == ["fuel_status"]


class TestRequestShape:
    def test_forces_a_tool_call(self, intents):
        """tool_choice=any is what makes an invalid tool name unrepresentable."""
        client = StubClient(tool_use("fuel_status"))
        HaikuRouter(intents, client=client).route("fuel")
        assert client.calls[0]["tool_choice"] == {"type": "any"}

    def test_does_not_disable_parallel_tool_use(self, intents):
        """Multi-intent is one of only two things the LLM tier buys here."""
        client = StubClient(tool_use("fuel_status"))
        HaikuRouter(intents, client=client).route("fuel")
        assert "disable_parallel_tool_use" not in client.calls[0].get("tool_choice", {})

    def test_uses_the_configured_model(self, intents):
        client = StubClient(tool_use("fuel_status"))
        HaikuRouter(intents, client=client, model="claude-haiku-4-5").route("fuel")
        assert client.calls[0]["model"] == "claude-haiku-4-5"

    def test_sends_no_thinking_parameter(self, intents):
        """Haiku 4.5 predates adaptive thinking; omitting it means no thinking,
        which is what a latency-sensitive router wants."""
        client = StubClient(tool_use("fuel_status"))
        HaikuRouter(intents, client=client).route("fuel")
        assert "thinking" not in client.calls[0]


class TestFailureIsNeverFatal:
    """AC6.4 — every failure path returns empty, never raises."""

    def test_api_exception(self, intents):
        router = HaikuRouter(intents, client=StubClient(RuntimeError("connection reset")))
        result = router.route("fuel")
        assert result.intents == []
        assert result.failed
        assert result.reason == "RuntimeError"

    def test_timeout(self, intents):
        router = HaikuRouter(intents, client=StubClient(TimeoutError("timed out")))
        result = router.route("fuel")
        assert result.intents == []
        assert result.failed

    def test_empty_transcript_short_circuits(self, intents):
        client = StubClient()
        result = HaikuRouter(intents, client=client).route("   ")
        assert result.intents == []
        assert not result.failed
        assert client.calls == []  # no wasted API call

    def test_malformed_response(self, intents):
        router = HaikuRouter(intents, client=StubClient(_Response(None)))
        assert router.route("fuel").intents == []

    def test_missing_api_key_reports_unavailable(self, intents, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert not HaikuRouter(intents).available
