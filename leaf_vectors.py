"""
leaf_vectors.py — Layer 3: Local Leaf Vector Ranker.

Only activated when a tree branch returns more nodes than fit in context.
Each leaf maintains a tiny numpy vector store — typically 30-50 vectors.
This is the LAST resort, not the primary retrieval mechanism.
"""

import numpy as np
import json
from pathlib import Path
from typing import Optional


class LeafVectorStore:
    """
    Minimal per-leaf vector store using pure numpy.
    No FAISS, no external dependencies.
    Scales fine at leaf level (30-100 vectors per leaf).
    """

    def __init__(self):
        # leaf_key → { node_id: embedding }
        self._stores: dict[str, dict[str, np.ndarray]] = {}

    def _leaf_key(self, tree_path: str) -> str:
        """Convert tree_path to a safe filename key."""
        return tree_path.strip("/").replace("/", "_")

    def add(self, tree_path: str, node_id: str, embedding: np.ndarray):
        key = self._leaf_key(tree_path)
        if key not in self._stores:
            self._stores[key] = {}
        self._stores[key][node_id] = embedding

    def remove(self, tree_path: str, node_id: str):
        key = self._leaf_key(tree_path)
        if key in self._stores:
            self._stores[key].pop(node_id, None)

    def rank(
        self,
        tree_path: str,
        query_embedding: np.ndarray,
        node_ids: list[str],
        top_k: int = 8,
    ) -> list[str]:
        """
        Rank node_ids within a leaf by cosine similarity to query.
        Returns top_k node_ids sorted by relevance.
        """
        key = self._leaf_key(tree_path)
        store = self._stores.get(key, {})

        if not store:
            return node_ids[:top_k]

        q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
        scored = []

        for nid in node_ids:
            if nid in store:
                emb = store[nid]
                emb_norm = emb / (np.linalg.norm(emb) + 1e-9)
                score = float(np.dot(q_norm, emb_norm))
                scored.append((nid, score))
            else:
                scored.append((nid, 0.0))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [nid for nid, _ in scored[:top_k]]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, layer3_dir: Path):
        layer3_dir.mkdir(parents=True, exist_ok=True)
        # save index of which leaves exist
        index = list(self._stores.keys())
        with open(layer3_dir / "index.json", "w") as f:
            json.dump(index, f)
        # save each leaf as compressed numpy
        for key, store in self._stores.items():
            if store:
                np.savez_compressed(
                    str(layer3_dir / f"{key}.npz"),
                    **{nid: emb for nid, emb in store.items()}
                )

    def load(self, layer3_dir: Path):
        if not layer3_dir.exists():
            return
        index_path = layer3_dir / "index.json"
        if not index_path.exists():
            return
        with open(index_path) as f:
            keys = json.load(f)
        for key in keys:
            npz_path = layer3_dir / f"{key}.npz"
            if npz_path.exists():
                data = np.load(str(npz_path))
                self._stores[key] = {nid: data[nid] for nid in data.files}
