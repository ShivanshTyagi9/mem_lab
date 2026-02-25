"""
embedder.py — Lightweight embedding engine.

Strategy:
1. If sentence-transformers is available, use it (best quality).
2. Otherwise fall back to a TF-IDF style sparse embedding over numpy.
   Not as good, but zero external dependencies and still useful for
   conflict detection and leaf ranking.

The system is designed so embeddings are only used for:
  a) Conflict detection (comparing new fact vs branch candidates)
  b) Leaf-level ranking (when a branch has too many nodes)
Both are small-scale operations, so even simple embeddings work well.
"""

import numpy as np
import re
from typing import Protocol


class EmbedderProtocol(Protocol):
    def encode(self, text: str) -> np.ndarray: ...
    def encode_batch(self, texts: list[str]) -> list[np.ndarray]: ...


class FallbackEmbedder:
    """
    TF-IDF style sparse embedding using numpy only.
    Builds vocabulary from seen texts, produces L2-normalized vectors.
    Good enough for semantic similarity within a memory domain.
    """

    def __init__(self, dim: int = 512):
        self.dim = dim
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.ones(dim)

    def _tokenize(self, text: str) -> list[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z]{2,}", text)
        # simple bigrams for better semantic capture
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return tokens + bigrams

    def _token_to_index(self, token: str) -> int:
        # deterministic hash into vocab space
        return hash(token) % self.dim

    def encode(self, text: str) -> np.ndarray:
        tokens = self._tokenize(text)
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokens:
            idx = self._token_to_index(token)
            vec[idx] += 1.0
        # L2 normalize
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [self.encode(t) for t in texts]


class SentenceTransformerEmbedder:
    """Wrapper around sentence-transformers if available."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        return self._model.encode(text, normalize_embeddings=True)

    def encode_batch(self, texts: list[str]) -> list[np.ndarray]:
        return list(self._model.encode(texts, normalize_embeddings=True))


def create_embedder() -> EmbedderProtocol:
    """
    Try to load sentence-transformers, fall back to numpy embedder.
    Called once at MemorySystem init.
    """
    try:
        embedder = SentenceTransformerEmbedder()
        print("[agentmem] Using sentence-transformers for embeddings (high quality)")
        return embedder
    except ImportError:
        print("[agentmem] sentence-transformers not found, using fallback embedder")
        print("[agentmem] Install with: pip install sentence-transformers")
        return FallbackEmbedder()
