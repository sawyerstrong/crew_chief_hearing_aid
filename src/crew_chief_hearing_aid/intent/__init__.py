from .embedder import Embedder, HashingEmbedder, Model2VecEmbedder, SentenceTransformerEmbedder
from .matcher import IntentMatcher, MatchResult, token_f1
from .phrases import Intent, canonical_key, content_tokens, normalize, parse_crewchief_config

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "Intent",
    "IntentMatcher",
    "MatchResult",
    "Model2VecEmbedder",
    "SentenceTransformerEmbedder",
    "canonical_key",
    "content_tokens",
    "normalize",
    "parse_crewchief_config",
    "token_f1",
]


def build_embedder(kind: str, model_name: str | None = None) -> Embedder:
    """Factory used by the pipeline; keeps config strings out of call sites."""
    kind = (kind or "").lower()
    if kind in {"model2vec", "static"}:
        return Model2VecEmbedder(model_name or "minishlab/potion-base-8M")
    if kind in {"sentence-transformers", "st", "minilm"}:
        return SentenceTransformerEmbedder(
            model_name or "sentence-transformers/all-MiniLM-L6-v2"
        )
    if kind in {"hashing", "none", "offline"}:
        return HashingEmbedder()
    raise ValueError(f"unknown embedder {kind!r}")
