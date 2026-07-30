import pytest

from crew_chief_hearing_aid.intent.embedder import HashingEmbedder
from crew_chief_hearing_aid.intent.matcher import IntentMatcher, token_f1
from crew_chief_hearing_aid.intent.phrases import Intent, content_tokens


@pytest.fixture
def intents():
    return [
        Intent(
            id="car_ahead_last_lap",
            action="What's the car ahead's last lap time",
            key="F13",
            phrases=(
                "what is the car ahead's last lap time",
                "what's the guy in front running",
                "how fast is the car ahead going",
            ),
        ),
        Intent(
            id="car_behind_last_lap",
            action="What's the car behind's last lap time",
            key="F14",
            phrases=(
                "what is the car behind's last lap time",
                "what's the guy behind running",
                "how fast is the car behind going",
            ),
        ),
        Intent(
            id="fuel_status",
            action="Get fuel status",
            key="F15",
            phrases=(
                "how is my fuel looking",
                "what is my fuel status",
                "how many laps of fuel left",
            ),
        ),
    ]


@pytest.fixture
def matcher(intents):
    return IntentMatcher(intents, embedder=HashingEmbedder())


class TestTokenF1:
    def test_identical(self):
        assert token_f1(("a", "b"), ("a", "b")) == pytest.approx(1.0)

    def test_disjoint(self):
        assert token_f1(("a",), ("b",)) == 0.0

    def test_is_symmetric_not_containment(self):
        """The property the whole design rests on.

        A short registered phrase must NOT score 1.0 against a long utterance
        that contains it -- that containment behaviour is exactly how CrewChief's
        closed grammar fires the wrong command at high confidence.
        """
        short = content_tokens("lap time")
        long = content_tokens("what's the lap time of the car ahead")

        # Containment would score this 1.0 — every token of the short phrase is
        # present in the long one — and fire the wrong command.
        assert set(short) <= set(long)
        # F1 scores it 0.5, comfortably under the 0.72 default token_threshold.
        assert token_f1(short, long) == pytest.approx(0.5)
        assert token_f1(short, long) == pytest.approx(token_f1(long, short))


class TestExactTier:
    def test_exact_phrase(self, matcher):
        result = matcher.match("how is my fuel looking")
        assert result.intent.id == "fuel_status"
        assert result.method == "exact"
        assert result.score == 1.0

    def test_exact_ignores_case_punctuation_and_filler(self, matcher):
        result = matcher.match("Can you tell me, how is the fuel looking?")
        assert result.matched
        assert result.intent.id == "fuel_status"


class TestRejection:
    def test_unrelated_speech_is_rejected(self, matcher):
        result = matcher.match("what do you want for dinner tonight")
        assert not result.matched
        assert result.reject_reason in {"below_threshold", "ambiguous"}

    def test_empty_transcript_is_rejected(self, matcher):
        result = matcher.match("   ")
        assert not result.matched
        assert result.reject_reason == "empty"

    def test_zero_overlap_speech_rejects_without_crashing(self, intents):
        """Regression: no shared token with any intent left the score dict
        empty, which used to raise IndexError. Sim audio bleed and Discord
        chatter hit this constantly."""
        m = IntentMatcher(intents, embedder=None)
        result = m.match("the tyres are absolutely cooked mate")
        assert not result.matched
        assert result.reject_reason == "no_candidates"
        assert result.score == 0.0

    def test_zero_overlap_also_safe_with_embedder(self, matcher):
        result = matcher.match("qwertyuiop zxcvbnm")
        assert not result.matched

    def test_rejection_still_reports_best_candidate(self, matcher):
        result = matcher.match("what do you want for dinner tonight")
        # Needed for the tuning log: you want to see what it nearly matched.
        assert result.method in {"token", "embedding"}
        assert 0.0 <= result.score <= 1.0


class TestAmbiguityMargin:
    def test_near_tie_is_rejected(self, intents):
        """ahead/behind differ by one word and are the classic confusion pair."""
        strict = IntentMatcher(intents, embedder=None, token_threshold=0.3, margin=0.4)
        result = strict.match("how fast is the car going")
        assert not result.matched
        assert result.reject_reason == "ambiguous"

    def test_disambiguating_word_resolves_it(self, intents):
        m = IntentMatcher(intents, embedder=None, token_threshold=0.3, margin=0.05)
        ahead = m.match("how fast is the car ahead going")
        behind = m.match("how fast is the car behind going")
        assert ahead.intent.id == "car_ahead_last_lap"
        assert behind.intent.id == "car_behind_last_lap"


class TestConstruction:
    def test_rejects_empty_intent_list(self):
        with pytest.raises(ValueError, match="at least one intent"):
            IntentMatcher([])

    def test_rejects_duplicate_ids(self, intents):
        with pytest.raises(ValueError, match="duplicate intent ids"):
            IntentMatcher([intents[0], intents[0]])

    def test_works_without_embedder(self, intents):
        m = IntentMatcher(intents, embedder=None)
        assert m.match("how is my fuel looking").matched


class TestLogRecord:
    def test_shape(self, matcher):
        record = matcher.match("how is my fuel looking").as_log_record()
        assert set(record) == {
            "transcript",
            "intent",
            "score",
            "method",
            "runner_up",
            "runner_up_score",
            "reject_reason",
        }
        assert record["intent"] == "fuel_status"
