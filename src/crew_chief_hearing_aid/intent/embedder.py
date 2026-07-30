"""Sentence embedding backends.

Default is model2vec static embeddings: ~30MB, numpy-only inference, sub-
millisecond per phrase, no torch. For a fixed set of short command phrases that
is more than enough, and it keeps the install off a 2GB torch dependency on a
machine whose GPU is busy rendering VR.

`SentenceTransformerEmbedder` is available behind the `st` extra if static
embeddings turn out to be too blunt for your phrasing.
"""

from __future__ import annotations

import zlib
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Embedder(Protocol):
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 array. Rows need not be normalised."""
        ...


def l2_normalize(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float32)
    if mat.ndim == 1:
        mat = mat[None, :]
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    np.maximum(norms, 1e-12, out=norms)
    return mat / norms


class Model2VecEmbedder:
    """Static embeddings via model2vec. Lazy-loads so import stays cheap."""

    def __init__(self, model_name: str = "minishlab/potion-base-8M") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from model2vec import StaticModel

            self._model = StaticModel.from_pretrained(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load().encode(texts), dtype=np.float32)


class SentenceTransformerEmbedder:
    """Heavier contextual embeddings. Requires the `st` extra."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._load().encode(texts), dtype=np.float32)


class HashingEmbedder:
    """Deterministic character-ngram hashing embedder.

    Not competitive with a learned model, but it needs no download and no
    network, which makes it the right default for CI and a survivable fallback
    if model fetching fails on the rig mid-session. Captures enough lexical
    overlap to be better than nothing.
    """

    def __init__(self, dim: int = 512, ngram: tuple[int, int] = (3, 5)) -> None:
        self.dim = dim
        self.ngram = ngram

    def encode(self, texts: list[str]) -> np.ndarray:
        # zlib.crc32, not builtin hash(): str hashing is salted per-process by
        # PYTHONHASHSEED, which would make cached corpus vectors incomparable
        # with query vectors across restarts.
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        lo, hi = self.ngram
        for i, text in enumerate(texts):
            padded = f" {text} "
            for n in range(lo, hi + 1):
                for j in range(len(padded) - n + 1):
                    gram = padded[j : j + n].encode("utf-8")
                    out[i, zlib.crc32(gram) % self.dim] += 1.0
        return out
