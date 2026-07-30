from pathlib import Path

import pytest

from crew_chief_hearing_aid.intent.phrases import (
    Intent,
    canonical_key,
    content_tokens,
    normalize,
    parse_crewchief_config,
)

FIXTURE = Path(__file__).parent / "fixtures" / "speech_recognition_config_sample.txt"


class TestNormalize:
    def test_lowercases_and_strips_punctuation(self):
        assert normalize("What's the GAP ahead?") == "whats the gap ahead"

    def test_collapses_whitespace(self):
        assert normalize("  gap    ahead \n") == "gap ahead"

    def test_apostrophe_variants_collapse(self):
        # Whisper emits both spellings depending on decoder temperature.
        assert normalize("what's my lap time") == normalize("whats my lap time")

    def test_strips_accents(self):
        assert normalize("café") == "cafe"

    def test_empty(self):
        assert normalize("   ") == ""


class TestContentTokens:
    def test_drops_filler(self):
        assert content_tokens("can you tell me the gap ahead") == ("gap", "ahead")

    def test_keeps_meaningful_words(self):
        assert content_tokens("what is my fuel status") == ("what", "is", "my", "fuel", "status")

    def test_canonical_key_is_order_preserving(self):
        assert canonical_key("the gap ahead please") == "gap ahead"


class TestParseCrewChiefConfig:
    @pytest.fixture
    def text(self):
        return FIXTURE.read_text(encoding="utf-8")

    def test_parses_keys(self, text):
        parsed = parse_crewchief_config(text)
        assert "WHAT_WAS_MY_LAST_LAP_TIME" in parsed
        assert "WHATS_MY_POSITION" in parsed

    def test_skips_comments_and_malformed_lines(self, text):
        parsed = parse_crewchief_config(text)
        assert "NOT_A_SETTING_LINE" not in parsed
        assert not any(k.startswith("#") for k in parsed)

    def test_drops_empty_values(self, text):
        assert "EMPTY_VALUE" not in parse_crewchief_config(text)

    def test_drops_aliases_contained_in_the_canonical_phrase(self, text):
        alts = parse_crewchief_config(text, drop_short_aliases=True)["WHAT_WAS_MY_LAST_LAP_TIME"]
        # All three later alternatives are token-subsets of the canonical one,
        # so they add no matching power and only contribute greedy attractors.
        assert alts == ["what's my last lap time"]

    def test_keeps_aliases_that_add_new_tokens(self, text):
        alts = parse_crewchief_config(text, drop_short_aliases=True)["WHATS_MY_GAP_IN_FRONT"]
        assert "what's the gap ahead" in alts
        # "in front" is not present in the canonical phrase, so it survives.
        assert "what's the gap in front" in alts
        # ...but the bare short forms do not.
        assert "gap ahead" not in alts
        assert "gap in front" not in alts

    def test_keeps_short_aliases_when_disabled(self, text):
        alts = parse_crewchief_config(text, drop_short_aliases=False)["WHAT_WAS_MY_LAST_LAP_TIME"]
        assert "lap time" in alts

    def test_always_keeps_first_alternative(self, text):
        # WHATS_MY_POSITION's only alternative is 3 content words; a stricter
        # min would otherwise delete the entry entirely.
        parsed = parse_crewchief_config(text, drop_short_aliases=True, min_alias_words=99)
        assert parsed["WHATS_MY_POSITION"] == ["what's my position"]

    def test_deduplicates_by_canonical_key(self, text):
        alts = parse_crewchief_config(text, drop_short_aliases=False)["WHATS_MY_GAP_IN_FRONT"]
        keys = [canonical_key(a) for a in alts]
        assert len(keys) == len(set(keys))


class TestIntent:
    def test_rejects_missing_phrases(self):
        with pytest.raises(ValueError, match="no phrases"):
            Intent(id="x", action="a", key="F13", phrases=())

    def test_rejects_missing_key(self):
        with pytest.raises(ValueError, match="no output key"):
            Intent(id="x", action="a", key="", phrases=("hello",))
