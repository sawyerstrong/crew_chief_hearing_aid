"""Phrase normalisation and CrewChief speech_recognition_config.txt parsing.

CrewChief's SRE config maps a command key to colon-separated alternatives:

    WHAT_WAS_MY_LAST_LAP_TIME = what's my last lap time:last lap time:lap time

The short trailing aliases are why CrewChief's closed-grammar recogniser
collapses long utterances onto the wrong command at high confidence: "lap time"
is a substring of many longer sentences, so it wins acoustically while being the
wrong intent. `drop_short_aliases` exists to strip those attractors when
importing a phrase set.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_APOSTROPHE = re.compile(r"[’'`]")
_PUNCT = re.compile(r"[^\w\s]")
_WS = re.compile(r"\s+")

# Spoken forms that carry no disambiguating signal. Removing them makes
# "what's the gap ahead" and "gap ahead" land in the same place without
# needing a short alias registered as its own phrase.
_FILLER = frozenset(
    {
        "a",
        "an",
        "the",
        "please",
        "can",
        "you",
        "tell",
        "me",
        "give",
        "just",
        "hey",
        "ok",
        "okay",
        "so",
        "um",
        "uh",
    }
)


def normalize(text: str) -> str:
    """Lowercase, strip accents and punctuation, collapse whitespace.

    Apostrophes are removed rather than expanded so that "what's" and "whats"
    -- which Whisper emits interchangeably depending on decoder temperature --
    normalise identically.
    """
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    # Apostrophes are deleted, not replaced with a space, so "what's" collapses
    # to "whats" rather than splitting into "what s". Everything else becomes a
    # separator.
    text = _APOSTROPHE.sub("", text)
    text = _PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def _depluralize(token: str) -> str:
    """Fold possessives and simple plurals onto a common stem.

    Needed because apostrophe stripping turns "ahead's" into "aheads", which
    would otherwise never match a spoken "ahead" -- and "ahead" is the single
    most discriminative token separating the car-ahead and car-behind intents.

    Deliberately crude: no stemmer, no wordlist. Over-folding is *mostly*
    harmless because query and corpus go through the same function and fold
    together -- but the `-us`/`-is`/`-as`/`-os` guard is load-bearing, not
    cosmetic: without it "status" becomes "statu" and "bias" becomes "bia",
    which is both wrong and unreadable in the diagnostic log.
    """
    if len(token) > 3 and token.endswith("es") and token[-3] in "sxzh":
        return token[:-2]
    if len(token) > 2 and token.endswith("s") and token[-2] not in "suioa":
        return token[:-1]
    return token


def content_tokens(text: str) -> tuple[str, ...]:
    """Normalised, stemmed tokens with filler words removed."""
    return tuple(
        _depluralize(t) for t in normalize(text).split() if t not in _FILLER
    )


def split_compounds(tokens: tuple[str, ...], vocabulary: frozenset[str]) -> tuple[str, ...]:
    """Split run-together words against a known vocabulary.

    "laptime" -> ("lap", "time") when both halves are in the corpus. Speech is
    transcribed as it is spoken, and people say compounds as one word; Whisper
    faithfully reproduces that. Rather than hardcode a compound list, any token
    absent from the vocabulary is tried against every binary split, and the
    split is accepted only when *both* halves are known words.
    """
    out: list[str] = []
    for token in tokens:
        if token in vocabulary or len(token) < 4:
            out.append(token)
            continue
        for i in range(2, len(token) - 1):
            left, right = token[:i], token[i:]
            if left in vocabulary and right in vocabulary:
                out.extend((left, right))
                break
        else:
            out.append(token)
    return tuple(out)


def canonical_key(text: str) -> str:
    """Order-preserving content-word key, used for cheap exact matching."""
    return " ".join(content_tokens(text))


@dataclass(frozen=True)
class Intent:
    """One thing the user can ask for.

    `action` is the CrewChief action label as it appears in Add/Remove Actions;
    it is documentation for the human doing the key binding, not something the
    code sends. The wire format is whatever `output.key` is bound to.
    """

    id: str
    action: str
    key: str
    phrases: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.phrases:
            raise ValueError(f"intent {self.id!r} has no phrases")
        if not self.key:
            raise ValueError(f"intent {self.id!r} has no output key")


def parse_crewchief_config(
    text: str,
    *,
    drop_short_aliases: bool = True,
    min_alias_words: int = 3,
) -> dict[str, list[str]]:
    """Parse a CrewChief speech_recognition_config.txt into {KEY: [phrases]}.

    Lines are `KEY = alt1:alt2:alt3`; `#` starts a comment. Blank alternatives
    and duplicates are dropped.

    The first alternative is always kept -- CrewChief treats it as canonical.

    With `drop_short_aliases`, later alternatives are discarded when either:

    * their content tokens are a **subset** of any already-kept alternative, or
    * they have fewer than `min_alias_words` content words.

    The subset rule is the principled one. An alternative contained within a
    phrase we already hold adds no matching power here -- symmetric token
    overlap already scores "whats my lap time" at 0.89 against "whats my last
    lap time" -- while contributing a short attractor that can capture
    unrelated longer sentences. Checking against every kept alternative rather
    than only the first matters: "gap in front" is not contained in the
    canonical "whats the gap ahead", but it is contained in the second
    alternative "whats the gap in front".

    The word-count floor is a backstop for short alternatives that are not
    strict subsets of anything.
    """
    out: dict[str, list[str]] = {}
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue

        alts: list[str] = []
        seen: set[str] = set()
        kept_tokens: list[frozenset[str]] = []
        for i, alt in enumerate(value.split(":")):
            alt = alt.strip()
            if not alt:
                continue
            tokens = frozenset(content_tokens(alt))
            if drop_short_aliases and i > 0 and alts:
                if any(tokens <= kept for kept in kept_tokens):
                    continue
                if len(tokens) < min_alias_words:
                    continue
            ck = canonical_key(alt)
            if not ck or ck in seen:
                continue
            seen.add(ck)
            alts.append(alt)
            kept_tokens.append(tokens)

        if alts:
            out[key] = alts
    return out
