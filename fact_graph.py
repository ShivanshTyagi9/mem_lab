"""
fact_graph.py — Layer 2: Temporal Fact Graph.

Nodes = raw natural language facts (never decomposed into triplets).
Edges = temporal and relational metadata only.

Conflict detection uses cosine similarity over numpy vectors (no external vector DB).
Each node stores a cached embedding so conflict search stays fast and local.
"""

import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional
import networkx as nx

from models import MemoryNode, MemoryEdge, EdgeType


class FactGraph:

    def __init__(self):
        self._graph = nx.DiGraph()
        # node_id → numpy embedding (for conflict detection only)
        self._embeddings: dict[str, np.ndarray] = {}

    # ── Node Operations ───────────────────────────────────────────────────────

    def add_node(self, node: MemoryNode, embedding: Optional[np.ndarray] = None):
        """Add a memory node to the graph."""
        self._graph.add_node(node.id, **node.to_dict())
        if embedding is not None:
            self._embeddings[node.id] = embedding

    def get_node(self, node_id: str) -> Optional[MemoryNode]:
        if node_id not in self._graph:
            return None
        data = self._graph.nodes[node_id]
        return MemoryNode.from_dict(data)

    def get_nodes(self, node_ids: list[str]) -> list[MemoryNode]:
        nodes = []
        for nid in node_ids:
            n = self.get_node(nid)
            if n:
                nodes.append(n)
        return nodes

    def update_confidence(self, node_id: str, delta: float):
        if node_id in self._graph:
            current = self._graph.nodes[node_id].get("confidence", 0.9)
            self._graph.nodes[node_id]["confidence"] = max(0.0, min(1.0, current + delta))

    # ── Edge Operations ───────────────────────────────────────────────────────

    def add_edge(self, from_id: str, to_id: str, edge_type: EdgeType):
        """Draw a typed, timestamped edge between two nodes."""
        ts = datetime.utcnow().isoformat()
        self._graph.add_edge(from_id, to_id, type=edge_type, timestamp=ts)

        # keep node's own edge list in sync
        if from_id in self._graph:
            node_data = self._graph.nodes[from_id]
            edges = node_data.get("edges", [])
            edges.append({"to": to_id, "type": edge_type, "timestamp": ts})
            self._graph.nodes[from_id]["edges"] = edges

    def get_edge_chain(self, node_id: str, depth: int = 2) -> list[MemoryNode]:
        """
        Walk the edge graph from a node up to `depth` hops.
        Returns connected nodes — gives agent full temporal context of a fact.
        """
        visited = set()
        result = []

        def walk(nid, remaining):
            if remaining == 0 or nid in visited:
                return
            visited.add(nid)
            node = self.get_node(nid)
            if node:
                result.append(node)
            # walk both successors and predecessors to get full chain
            for neighbor in list(self._graph.successors(nid)) + list(self._graph.predecessors(nid)):
                walk(neighbor, remaining - 1)

        walk(node_id, depth)
        return result

    # ── Conflict Detection ────────────────────────────────────────────────────

    def find_conflicts(
        self,
        new_embedding: np.ndarray,
        candidate_node_ids: list[str],
        threshold: float = 0.82,
    ) -> list[tuple[str, float]]:
        """
        Find existing nodes that semantically conflict with a new fact.
        Only searches within candidate_node_ids (from tree branch) — never the full graph.
        Returns [(node_id, similarity_score)] sorted by similarity desc.
        """
        if not candidate_node_ids:
            return []

        new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-9)
        conflicts = []

        for nid in candidate_node_ids:
            if nid not in self._embeddings:
                continue
            existing = self._embeddings[nid]
            existing_norm = existing / (np.linalg.norm(existing) + 1e-9)
            similarity = float(np.dot(new_norm, existing_norm))
            if similarity >= threshold:
                conflicts.append((nid, similarity))

        return sorted(conflicts, key=lambda x: x[1], reverse=True)

    def all_node_ids(self) -> list[str]:
        return list(self._graph.nodes)

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, graph_path: Path, embeddings_path: Path):
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        with open(graph_path, "wb") as f:
            pickle.dump(self._graph, f)
        np.savez_compressed(str(embeddings_path), **{
            k: v for k, v in self._embeddings.items()
        })

    def load(self, graph_path: Path, embeddings_path: Path):
        if graph_path.exists():
            with open(graph_path, "rb") as f:
                self._graph = pickle.load(f)
        emb_path = Path(str(embeddings_path) + ".npz")
        if emb_path.exists():
            data = np.load(str(emb_path))
            self._embeddings = {k: data[k] for k in data.files}
