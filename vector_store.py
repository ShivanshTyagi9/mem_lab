"""
vector_store.py — Embedding model + vector similarity search

Two layers:
  EmbeddingModel  — turns text into a float vector.
  VectorStore     — stores those vectors and queries them by cosine similarity.

Backend priority:
  Embeddings: SentenceTransformers (local, free) → OpenAI (API) → raises
  VectorDB:   ChromaDB (persistent, fast)        → NumpyStore (in-memory fallback)

Install the good path:
  pip install sentence-transformers chromadb

Minimum path (no extra deps, slower, in-memory only):
  pip install openai           # if using OpenAI backend already

The NumpyStore fallback is automatic — no config needed.
"""

import json
import math
import os
from typing import List, Dict, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Embedding Model
# ---------------------------------------------------------------------------

class EmbeddingModel:
    """
    Converts text to dense vectors for semantic similarity.

    Tries SentenceTransformers first (local, no API cost, ~22MB model).
    Falls back to OpenAI text-embedding-3-small if ST is not installed.
    """

    # Singleton cache — load model once
    _instance: Optional["EmbeddingModel"] = None
    _model_name: str = ""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self._dim: int = 0
        self._backend: str = ""
        self._model: Any = None

        # Try sentence-transformers first
        try:
            from sentence_transformers import SentenceTransformer
            self._model     = SentenceTransformer(model_name)
            self._dim       = self._model.get_sentence_embedding_dimension()
            self._backend   = "sentence_transformers"
            self._st_name   = model_name
            return
        except ImportError:
            pass

        # Try OpenAI embeddings
        try:
            import openai as _openai_test  # noqa: F401
            self._backend = "openai"
            self._dim     = 1536   # text-embedding-3-small
            return
        except ImportError:
            pass

        raise RuntimeError(
            "No embedding backend available.\n"
            "  Option 1 (recommended): pip install sentence-transformers\n"
            "  Option 2: pip install openai  (requires OPENAI_API_KEY)"
        )

    def embed(self, text: str) -> List[float]:
        """Embed a single string. Returns a list of floats."""
        text = text.strip()
        if not text:
            return [0.0] * self._dim

        if self._backend == "sentence_transformers":
            return self._model.encode(text, normalize_embeddings=True).tolist()

        if self._backend == "openai":
            from llm import get_backend, OpenAIBackend
            b = get_backend()
            if isinstance(b, OpenAIBackend):
                resp = b._client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text,
                )
                return resp.data[0].embedding
            raise RuntimeError("OpenAI backend not active — set OPENAI_API_KEY.")

        return [0.0] * self._dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of strings efficiently."""
        texts = [t.strip() for t in texts]
        if not texts:
            return []

        if self._backend == "sentence_transformers":
            return self._model.encode(
                texts, normalize_embeddings=True, show_progress_bar=False
            ).tolist()

        # Fallback: embed one by one
        return [self.embed(t) for t in texts]

    @property
    def dimension(self) -> int:
        return self._dim

    @classmethod
    def get(cls, model_name: str = "all-MiniLM-L6-v2") -> "EmbeddingModel":
        """Shared singleton — model is loaded once."""
        if cls._instance is None or cls._model_name != model_name:
            cls._instance  = cls(model_name)
            cls._model_name = model_name
        return cls._instance


# ---------------------------------------------------------------------------
# NumpyVectorStore — pure-Python fallback (no extra deps beyond numpy)
# ---------------------------------------------------------------------------

class NumpyVectorStore:
    """
    In-memory cosine similarity store using numpy.

    Fine for up to ~10k nodes. Runs without ChromaDB.
    Not persistent — must be serialized explicitly.
    """

    def __init__(self, dim: int):
        self._dim = dim
        # node_id → {"embedding": [...], "metadata": {...}}
        self._store: Dict[str, Dict] = {}

    def add(self, node_id: str, embedding: List[float], metadata: Dict = None):
        self._store[node_id] = {
            "embedding": list(embedding),
            "metadata":  metadata or {},
        }

    def delete(self, node_id: str):
        self._store.pop(node_id, None)

    def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Returns list of (node_id, cosine_similarity) sorted descending.
        where: optional dict of exact metadata filters, e.g. {"is_latest": True}
        """
        if not self._store:
            return []

        import numpy as np
        q = np.array(embedding, dtype=float)
        q_norm = q / (np.linalg.norm(q) + 1e-9)

        scores: List[Tuple[str, float]] = []
        for nid, data in self._store.items():
            # Apply metadata filter
            if where:
                meta = data.get("metadata", {})
                if not all(meta.get(k) == v for k, v in where.items()):
                    continue

            v = np.array(data["embedding"], dtype=float)
            v_norm = v / (np.linalg.norm(v) + 1e-9)
            sim = float(np.dot(q_norm, v_norm))
            scores.append((nid, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def find_similar(
        self, embedding: List[float], threshold: float = 0.92
    ) -> Optional[str]:
        """
        Find an existing node with cosine similarity >= threshold.
        Returns node_id or None.
        Used for deduplication on ingestion.
        """
        results = self.query(embedding, top_k=1)
        if results and results[0][1] >= threshold:
            return results[0][0]
        return None

    def __len__(self) -> int:
        return len(self._store)

    # Serialization — embeddings stored as JSON in the .umc archive
    def to_dict(self) -> dict:
        return {
            "dim":   self._dim,
            "store": self._store,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "NumpyVectorStore":
        s = cls(dim=d["dim"])
        s._store = d.get("store", {})
        return s


# ---------------------------------------------------------------------------
# ChromaVectorStore — persistent, production-grade backend
# ---------------------------------------------------------------------------

class ChromaVectorStore:
    """
    ChromaDB-backed vector store. Persistent to disk.
    Requires: pip install chromadb

    Stores embeddings separately from graph nodes.
    The node_id links back to the MemoryGraph.
    """

    def __init__(self, persist_dir: str, collection_name: str = "memories"):
        import chromadb
        self._dir    = persist_dir
        self._name   = collection_name
        os.makedirs(persist_dir, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._col    = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, node_id: str, embedding: List[float], metadata: Dict = None):
        # ChromaDB requires string values in metadata
        safe_meta = {}
        if metadata:
            for k, v in metadata.items():
                if isinstance(v, bool):
                    safe_meta[k] = str(v)  # "True" / "False"
                elif isinstance(v, (int, float, str)):
                    safe_meta[k] = v
                else:
                    safe_meta[k] = str(v)

        # Upsert handles re-embedding updated nodes
        self._col.upsert(
            ids=[node_id],
            embeddings=[embedding],
            metadatas=[safe_meta] if safe_meta else [{}],
        )

    def delete(self, node_id: str):
        try:
            self._col.delete(ids=[node_id])
        except Exception:
            pass

    def query(
        self,
        embedding: List[float],
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """Returns list of (node_id, similarity_score) sorted descending."""
        n_results = min(top_k, max(1, self._col.count()))

        kwargs: Dict = {
            "query_embeddings": [embedding],
            "n_results": n_results,
        }
        if where:
            # ChromaDB where requires string values
            chroma_where = {k: str(v) if isinstance(v, bool) else v
                            for k, v in where.items()}
            kwargs["where"] = chroma_where

        try:
            result = self._col.query(**kwargs)
        except Exception:
            return []

        ids         = result["ids"][0] if result["ids"] else []
        distances   = result["distances"][0] if result["distances"] else []

        # ChromaDB cosine distance = 1 - cosine_similarity
        return [(nid, 1.0 - dist) for nid, dist in zip(ids, distances)]

    def find_similar(
        self, embedding: List[float], threshold: float = 0.92
    ) -> Optional[str]:
        results = self.query(embedding, top_k=1)
        if results and results[0][1] >= threshold:
            return results[0][0]
        return None

    def __len__(self) -> int:
        return self._col.count()

    def to_dict(self) -> dict:
        """
        Export all embeddings as a dict for archiving in .umc.
        ChromaDB's own persistence handles its directory,
        but we also keep a portable copy in the archive.
        """
        try:
            result = self._col.get(include=["embeddings", "metadatas"])
            store = {}
            for nid, emb, meta in zip(
                result["ids"], result["embeddings"], result["metadatas"]
            ):
                store[nid] = {"embedding": list(emb), "metadata": meta}
            return {"dim": len(result["embeddings"][0]) if result["embeddings"] else 0, "store": store}
        except Exception:
            return {"dim": 0, "store": {}}

    @classmethod
    def from_dict(cls, d: dict, persist_dir: str) -> "ChromaVectorStore":
        """Restore a ChromaVectorStore from an exported dict."""
        vs = cls(persist_dir=persist_dir)
        store = d.get("store", {})
        for nid, data in store.items():
            vs.add(nid, data["embedding"], data.get("metadata", {}))
        return vs


# ---------------------------------------------------------------------------
# VectorStore — unified interface, picks the right backend automatically
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Public interface. Picks ChromaDB if available, else NumpyStore.

    Usage:
      vs = VectorStore(persist_dir="./memory_db")
      vs.add("node-id-123", embedding, {"is_latest": True, "mem_type": "fact"})
      results = vs.query(query_embedding, top_k=10)
      # → [("node-id-123", 0.94), ("node-id-456", 0.87), ...]
    """

    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "memories",
        embed_model: Optional[EmbeddingModel] = None,
    ):
        self._embed = embed_model or EmbeddingModel.get()
        self._backend_name: str = ""

        if persist_dir:
            try:
                self._backend = ChromaVectorStore(persist_dir, collection_name)
                self._backend_name = "chromadb"
                return
            except ImportError:
                pass

        # Fallback
        self._backend = NumpyVectorStore(dim=self._embed.dimension)
        self._backend_name = "numpy"

    # ── Core operations ───────────────────────────────────────────────────

    def add(self, node_id: str, text: str, metadata: Dict = None):
        """Embed text and store it."""
        embedding = self._embed.embed(text)
        self._backend.add(node_id, embedding, metadata)

    def add_with_embedding(self, node_id: str, embedding: List[float], metadata: Dict = None):
        """Store a pre-computed embedding directly."""
        self._backend.add(node_id, embedding, metadata)

    def delete(self, node_id: str):
        self._backend.delete(node_id)

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: Optional[Dict] = None,
    ) -> List[Tuple[str, float]]:
        """
        Semantic search. Returns (node_id, score) pairs, score in [0, 1].
        Higher = more similar.
        """
        embedding = self._embed.embed(query_text)
        return self._backend.query(embedding, top_k=top_k, where=where)

    def find_duplicate(
        self,
        text: str,
        threshold: float = 0.92,
        existing_id: Optional[str] = None,
    ) -> Optional[str]:
        """
        Check if semantically identical content already exists.
        Returns node_id of duplicate, or None.
        existing_id: exclude this id from the check (used when updating a node).
        """
        embedding = self._embed.embed(text)
        results   = self._backend.query(embedding, top_k=3)
        for nid, score in results:
            if score >= threshold and nid != existing_id:
                return nid
        return None

    # ── Persistence ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize for embedding in .umc archive."""
        return {
            "backend": self._backend_name,
            "data":    self._backend.to_dict(),
        }

    @classmethod
    def from_dict(
        cls,
        d: dict,
        persist_dir: Optional[str] = None,
        embed_model: Optional[EmbeddingModel] = None,
    ) -> "VectorStore":
        """Restore from a serialized dict."""
        vs = cls.__new__(cls)
        vs._embed        = embed_model or EmbeddingModel.get()
        vs._backend_name = d.get("backend", "numpy")
        data             = d.get("data", {})

        if persist_dir and vs._backend_name == "chromadb":
            try:
                vs._backend = ChromaVectorStore.from_dict(data, persist_dir)
                return vs
            except ImportError:
                vs._backend_name = "numpy"

        vs._backend = NumpyVectorStore.from_dict(data)
        vs._backend_name = "numpy"
        return vs

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def __len__(self) -> int:
        return len(self._backend)

    def __repr__(self) -> str:
        return f"<VectorStore backend={self._backend_name} n={len(self)}>"
