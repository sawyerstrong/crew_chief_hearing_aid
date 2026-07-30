"""Regression net for the shipped phrase map.

Every utterance in fixtures/utterances.jsonl must route to the expected intent
(or be rejected, where `intent` is null). Runs against the real
config.default.toml with the HashingEmbedder, so CI needs no model download and
the result is deterministic.

Add a line here whenever the session log shows a phrasing that missed. That is
the loop: race, read logs/utterances-*.jsonl, add the misses, fix the phrase map.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crew_chief_hearing_aid.config import load_config
from crew_chief_hearing_aid.intent.embedder import HashingEmbedder
from crew_chief_hearing_aid.intent.matcher import IntentMatcher

FIXTURE = Path(__file__).parent / "fixtures" / "utterances.jsonl"


def load_cases() -> list[dict]:
    cases = []
    for line in FIXTURE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("//"):
            cases.append(json.loads(line))
    return cases


@pytest.fixture(scope="module")
def matcher(tmp_path_factory):
    config = load_config(user_path=tmp_path_factory.mktemp("cfg") / "absent.toml")
    return IntentMatcher(
        config.intents,
        embedder=HashingEmbedder(),
        token_threshold=float(config.get("intent", "token_threshold", 0.72)),
        embed_threshold=float(config.get("intent", "embed_threshold", 0.60)),
        margin=float(config.get("intent", "margin", 0.05)),
    )


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["text"][:48])
def test_utterance_routes_correctly(matcher, case):
    result = matcher.match(case["text"])
    expected = case["intent"]

    if expected is None:
        assert not result.matched, (
            f"{case['text']!r} should have been rejected but matched "
            f"{result.intent.id} at {result.score:.3f} via {result.method}"
        )
    else:
        assert result.matched, (
            f"{case['text']!r} was rejected ({result.reject_reason}); "
            f"best was {result.runner_up_id or '-'} at {result.score:.3f} via {result.method}"
        )
        assert result.intent.id == expected, (
            f"{case['text']!r} -> {result.intent.id} (expected {expected}) "
            f"at {result.score:.3f} via {result.method}"
        )


class TestEvidenceGate:
    """The absolute-evidence gate, against the real corpus.

    IDF is corpus-relative, so `min_evidence` is only meaningful at realistic
    corpus size. These assert the gate against the shipped intent set rather
    than a toy fixture.
    """

    def test_filler_only_query_is_rejected(self, matcher):
        assert not matcher.match("what is my").matched

    def test_single_discriminative_word_is_enough(self, matcher):
        for word, expected in [("fuel", "fuel_status"), ("damage", "damage_report")]:
            result = matcher.match(word)
            assert result.matched, f"{word!r} rejected ({result.reject_reason})"
            assert result.intent.id == expected

    def test_terse_phrasing_resolves_without_the_embedder(self, matcher):
        """The case that motivated the tier-2 rewrite.

        Must resolve at tier 2 -- microseconds, deterministic, no model. If
        this starts matching via "embedding", the cheap path has regressed.
        """
        for text, expected in [
            ("car ahead laptime", "car_ahead_last_lap"),
            ("car behind laptime", "car_behind_last_lap"),
            ("car ahead lap time", "car_ahead_last_lap"),
        ]:
            result = matcher.match(text)
            assert result.matched, f"{text!r} rejected ({result.reject_reason})"
            assert result.intent.id == expected
            assert result.method == "token", f"{text!r} fell through to {result.method}"


def test_every_intent_has_coverage(matcher):
    """No intent should ship without at least one eval case."""
    covered = {c["intent"] for c in load_cases() if c["intent"]}
    defined = {i.id for i in matcher.intents}
    assert defined - covered == set(), f"intents with no eval coverage: {sorted(defined - covered)}"
