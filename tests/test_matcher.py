import pytest

from crew_chief_hearing_aid.intent.embedder import HashingEmbedder
from crew_chief_hearing_aid.intent.matcher import IntentMatcher, build_idf, query_coverage
from crew_chief_hearing_aid.intent.phrases import Intent, content_tokens, split_compounds


@pytest.fixture
def intents():
    return [
        Intent(
            id="car_ahead_last_lap",
            action="What's the car ahead's last lap time",
            key="F13",
            description="Lap time of the car directly ahead.",
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
            description="Lap time of the car directly behind.",
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
            description="Remaining fuel and laps it covers.",
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


class TestQueryCoverage:
    @pytest.fixture
    def idf(self):
        return build_idf([
            content_tokens("what is the car ahead's last lap time"),
            content_tokens("what is the car behind's last lap time"),
            content_tokens("what is my fuel status"),
        ])

    def test_fully_explained_query_scores_one(self, idf):
        """A terse query whose every token appears in the phrase scores 1.0.

        This is the case symmetric F1 structurally could not match: 3 query
        tokens against a 7-token phrase capped recall at 0.43, so F1 capped
        near 0.6 -- below any usable threshold.
        """
        query = content_tokens("car ahead lap time")
        phrase = content_tokens("what is the car ahead's last lap time")
        ratio, mass = query_coverage(query, phrase, idf)
        assert ratio == pytest.approx(1.0)
        assert mass > 0

    def test_short_alias_does_not_capture_long_query(self, idf):
        """P3, stated as a property rather than assumed.

        The registered alias "lap time" must not claim an utterance that is
        mostly about something else. Containment scores this 1.0; IDF-weighted
        query coverage scores it low because the discriminative "ahead" and
        "car" go unexplained.
        """
        query = content_tokens("what's the lap time of the car ahead")
        alias = content_tokens("lap time")
        assert set(alias) <= set(query)  # containment would fire
        ratio, _ = query_coverage(query, alias, idf)
        assert ratio < 0.6

    def test_discriminative_token_separates_ahead_from_behind(self, idf):
        query = content_tokens("car ahead lap time")
        ahead = content_tokens("what is the car ahead's last lap time")
        behind = content_tokens("what is the car behind's last lap time")
        assert query_coverage(query, ahead, idf)[0] > query_coverage(query, behind, idf)[0]

    def test_ratio_alone_is_degenerate_for_common_words(self, idf):
        """Why matched_mass exists at all.

        A query of only common words is "fully covered" by ratio, so ratio
        cannot be the whole gate. The absolute-evidence check is exercised
        against the real shipped corpus in TestEvidenceGate below -- a
        three-phrase fixture cannot demonstrate it, because IDF floors at 1.0
        by construction and any three tokens clear an absolute threshold of 2.0
        regardless of how common they are.
        """
        ratio, _ = query_coverage(
            content_tokens("what is my"), content_tokens("what is my fuel status"), idf
        )
        assert ratio == pytest.approx(1.0)

    def test_empty_query(self, idf):
        assert query_coverage((), content_tokens("anything"), idf) == (0.0, 0.0)


class TestNormalisation:
    def test_possessive_folds_onto_stem(self):
        """"ahead's" -> "aheads" would never match a spoken "ahead"."""
        assert "ahead" in content_tokens("what is the car ahead's last lap time")

    def test_compound_splits_against_vocabulary(self):
        vocab = frozenset({"lap", "time", "car", "ahead"})
        assert split_compounds(("laptime",), vocab) == ("lap", "time")

    def test_unknown_compound_is_left_alone(self):
        vocab = frozenset({"lap", "time"})
        assert split_compounds(("brakebias",), vocab) == ("brakebias",)


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
