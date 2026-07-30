"""Transcript -> intent matching.

Three tiers, cheapest first:

1. exact canonical-key match        (no model, ~microseconds)
2. symmetric token overlap (F1)     (no model, ~microseconds)
3. embedding cosine similarity      (model, ~1ms with model2vec)

The reason this is a ladder rather than one embedding lookup is that the first
two tiers are deterministic and testable in CI without downloading anything,
and they resolve the large majority of utterances -- you say the same handful of
things every race.

Two properties matter more than raw accuracy, because they are exactly what
CrewChief's closed SAPI grammar could not provide:

* **A real reject.** If nothing clears its threshold, `match` returns
  `intent=None`. A closed grammar must always emit its nearest phrase, which is
  how you get a confident wrong command.
* **Ambiguity rejection.** If the top two intents are within `margin` of each
  other, the result is rejected even if the best score is high. Firing the wrong
  command is worse than asking again.

Note that tier 2 is *symmetric* (F1 of precision and recall over content
tokens), not containment. Containment would let a short registered phrase like
"lap time" score 1.0 against "what's the lap time of the car ahead" -- which is
precisely the collapse this project exists to avoid. F1 scores that pair at
2*(2/2 * 2/7)/(2/2 + 2/7) = 0.44 and correctly declines to match.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .embedder import Embedder, l2_normalize
from .phrases import Intent, canonical_key, content_tokens, split_compounds


@dataclass(frozen=True)
class MatchResult:
    transcript: str
    intent: Intent | None
    score: float
    method: str
    runner_up_id: str | None = None
    runner_up_score: float = 0.0
    reject_reason: str | None = None

    @property
    def matched(self) -> bool:
        return self.intent is not None

    def as_log_record(self) -> dict:
        return {
            "transcript": self.transcript,
            "intent": self.intent.id if self.intent else None,
            "score": round(self.score, 4),
            "method": self.method,
            "runner_up": self.runner_up_id,
            "runner_up_score": round(self.runner_up_score, 4),
            "reject_reason": self.reject_reason,
        }


def build_idf(phrase_tokens: list[tuple[str, ...]]) -> dict[str, float]:
    """Smoothed inverse document frequency over the phrase corpus.

    A token appearing in one phrase is highly discriminative ("ahead", "fuel");
    one appearing in most phrases carries almost no signal ("what", "my", "is").
    Weighting by IDF is what separates the car-ahead and car-behind intents,
    whose phrases are otherwise near-identical.
    """
    n = len(phrase_tokens)
    df: dict[str, int] = {}
    for tokens in phrase_tokens:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def query_coverage(
    query: tuple[str, ...],
    phrase: tuple[str, ...],
    idf: dict[str, float],
    default_idf: float = 1.0,
) -> tuple[float, float]:
    """How much of what the user *said* is explained by this phrase.

    Returns (ratio, matched_mass) where ratio is IDF-weighted coverage in [0,1]
    and matched_mass is the absolute IDF weight matched.

    Asymmetric on purpose, and in the opposite direction from containment.
    Two cases have to be separated, and symmetric F1 blocks both:

    * P3 (must reject): user says "whats the lap time of the car ahead" and a
      short registered alias "lap time" tries to claim it. Coverage is low
      because the high-IDF "ahead" is unmatched.
    * Terse input (must accept): user says "car ahead laptime" against the
      phrase "what is the car ahead's last lap time". Every query token is
      explained, so coverage is 1.0 -- even though most of the phrase is
      unmatched, which is exactly what F1 penalised.

    `matched_mass` exists because ratio alone has a degenerate case: a query of
    one common word ("what") is fully covered by any phrase containing it.
    Requiring absolute evidence as well as proportion rules that out.
    """
    if not query:
        return 0.0, 0.0
    phrase_set = set(phrase)
    total = 0.0
    matched = 0.0
    for token in set(query):
        weight = idf.get(token, default_idf)
        total += weight
        if token in phrase_set:
            matched += weight
    if total <= 0.0:
        return 0.0, 0.0
    return matched / total, matched


class IntentMatcher:
    def __init__(
        self,
        intents: list[Intent],
        embedder: Embedder | None = None,
        *,
        token_threshold: float = 0.72,
        embed_threshold: float = 0.60,
        margin: float = 0.05,
        min_evidence: float = 2.0,
    ) -> None:
        if not intents:
            raise ValueError("IntentMatcher requires at least one intent")
        ids = [i.id for i in intents]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate intent ids")

        self.intents = intents
        self.embedder = embedder
        self.token_threshold = token_threshold
        self.embed_threshold = embed_threshold
        self.margin = margin
        self.min_evidence = min_evidence

        self._by_id = {i.id: i for i in intents}
        # canonical phrase key -> intent id. First writer wins, so an earlier
        # intent keeps the phrase if two intents register the same wording.
        self._exact: dict[str, str] = {}
        self._tokens: list[tuple[str, tuple[str, ...]]] = []
        self._corpus: list[tuple[str, str]] = []  # (intent_id, phrase)
        for intent in intents:
            for phrase in intent.phrases:
                key = canonical_key(phrase)
                if not key:
                    continue
                self._exact.setdefault(key, intent.id)
                self._tokens.append((intent.id, content_tokens(phrase)))
                self._corpus.append((intent.id, phrase))

        # Vocabulary drives compound splitting ("laptime" -> "lap time"); IDF
        # drives tier-2 weighting. Both are derived from the corpus, so adding
        # an intent automatically re-tunes them.
        self._vocabulary = frozenset(t for _, toks in self._tokens for t in toks)
        self._idf = build_idf([toks for _, toks in self._tokens])

        self._corpus_vecs: np.ndarray | None = None

    def prepare_query(self, transcript: str) -> tuple[str, ...]:
        """Normalise, stem, and compound-split a transcript into query tokens."""
        return split_compounds(content_tokens(transcript), self._vocabulary)

    # -- tier 3 support -------------------------------------------------

    def _ensure_corpus_vectors(self) -> np.ndarray | None:
        if self.embedder is None:
            return None
        if self._corpus_vecs is None:
            phrases = [p for _, p in self._corpus]
            self._corpus_vecs = l2_normalize(self.embedder.encode(phrases))
        return self._corpus_vecs

    def warmup(self) -> None:
        """Force model load and corpus encoding ahead of first use.

        Call this at startup. Otherwise the first command of the session eats
        the model load, which is the one time you least want the latency.
        """
        self._ensure_corpus_vectors()

    # -- matching -------------------------------------------------------

    def _best_two(
        self, scores: dict[str, float]
    ) -> tuple[tuple[str, float], tuple[str, float] | None]:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return ranked[0], (ranked[1] if len(ranked) > 1 else None)

    def _finalize(
        self,
        transcript: str,
        scores: dict[str, float],
        threshold: float,
        method: str,
    ) -> MatchResult:
        if not scores:
            # Speech sharing no token with any intent -- sim audio bleed, a
            # Discord aside, someone else in the room. Common enough that this
            # must reject cleanly rather than raise.
            return MatchResult(
                transcript, None, 0.0, method, reject_reason="no_candidates"
            )
        (best_id, best_score), runner = self._best_two(scores)
        runner_id, runner_score = (runner if runner else (None, 0.0))

        if best_score < threshold:
            return MatchResult(
                transcript, None, best_score, method,
                runner_id, runner_score, reject_reason="below_threshold",
            )
        if runner is not None and (best_score - runner_score) < self.margin:
            return MatchResult(
                transcript, None, best_score, method,
                runner_id, runner_score, reject_reason="ambiguous",
            )
        return MatchResult(
            transcript, self._by_id[best_id], best_score, method, runner_id, runner_score
        )

    def match(self, transcript: str) -> MatchResult:
        key = canonical_key(transcript)
        if not key:
            return MatchResult(transcript, None, 0.0, "none", reject_reason="empty")

        # Tier 1: exact
        if key in self._exact:
            return MatchResult(transcript, self._by_id[self._exact[key]], 1.0, "exact")

        # Tier 2: IDF-weighted query coverage
        query = self.prepare_query(transcript)
        token_scores: dict[str, float] = {}
        for intent_id, phrase_tokens in self._tokens:
            ratio, mass = query_coverage(query, phrase_tokens, self._idf)
            # Absolute evidence gate: a query of only low-information words is
            # fully "covered" by anything containing them. Score it as zero
            # rather than letting the ratio carry it.
            score = ratio if mass >= self.min_evidence else 0.0
            if score > token_scores.get(intent_id, 0.0):
                token_scores[intent_id] = score
        result = self._finalize(transcript, token_scores, self.token_threshold, "token")
        if result.matched:
            return result

        # Tier 3: embeddings
        vecs = self._ensure_corpus_vectors()
        if vecs is None:
            return result  # carries the tier-2 reject reason

        qv = l2_normalize(self.embedder.encode([transcript]))[0]
        sims = vecs @ qv
        embed_scores: dict[str, float] = {}
        for (intent_id, _), sim in zip(self._corpus, sims, strict=True):
            sim = float(sim)
            if sim > embed_scores.get(intent_id, -1.0):
                embed_scores[intent_id] = sim
        return self._finalize(transcript, embed_scores, self.embed_threshold, "embedding")
