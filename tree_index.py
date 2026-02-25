"""
tree_index.py — Layer 1: Hierarchical index tree.

Purpose: eliminate 99% of data before any real search happens.
This is just a nested dict of node_ids — no vectors, no embeddings.
Tree traversal is the first and cheapest filtering step.
"""

import json
from pathlib import Path
from typing import Optional
from functools import reduce


class TreeIndex:
    """
    Lightweight hierarchical index. Structure:
    {
      "semantic": {
        "people": {
          "aryan": {"_nodes": ["node_001", "node_002"]},
        },
        "concepts": {
          "healthtech": {"_nodes": ["node_089"]}
        }
      },
      "episodic": {
        "2026-feb": {
          "week-1": {"_nodes": ["node_150"]}
        }
      },
      "procedural": {
        "code_requests": {"_nodes": ["node_301"]}
      }
    }
    """

    def __init__(self):
        self._tree: dict = {
            "semantic": {},
            "episodic": {},
            "procedural": {}
        }

    # ── Read ──────────────────────────────────────────────────────────────────

    def get_nodes_at_path(self, path: str) -> list[str]:
        """Return all node_ids under a given tree path (including sub-paths)."""
        parts = [p for p in path.strip("/").split("/") if p]
        subtree = self._navigate(parts)
        if subtree is None:
            return []
        return self._collect_nodes(subtree)

    def get_nodes_for_type(self, memory_type: str) -> list[str]:
        """Return all node_ids under a memory type branch."""
        subtree = self._tree.get(memory_type, {})
        return self._collect_nodes(subtree)

    def suggest_path(self, memory_type: str, content: str) -> str:
        """
        Suggest a tree_path for a new node based on content heuristics.
        This is intentionally simple — the LLM extraction prompt does the real work.
        """
        content_lower = content.lower()
        base = memory_type

        if memory_type == "semantic":
            # crude keyword routing
            people_words = ["i ", "my ", "user", "aryan", "he ", "she ", "they "]
            concept_words = ["technology", "system", "method", "concept", "framework", "tool"]
            if any(w in content_lower for w in people_words):
                return f"semantic/people/user"
            elif any(w in content_lower for w in concept_words):
                return f"semantic/concepts/general"
            else:
                return f"semantic/facts/general"

        elif memory_type == "episodic":
            from datetime import datetime
            now = datetime.utcnow()
            month = now.strftime("%Y-%b").lower()
            week = f"week-{(now.day - 1) // 7 + 1}"
            return f"episodic/{month}/{week}"

        elif memory_type == "procedural":
            if "code" in content_lower or "implement" in content_lower:
                return f"procedural/code_requests"
            elif "ask" in content_lower or "clarif" in content_lower:
                return f"procedural/clarification_triggers"
            else:
                return f"procedural/general"

        return base

    # ── Write ─────────────────────────────────────────────────────────────────

    def add_node(self, node_id: str, tree_path: str):
        """Register a node_id at a given tree path, creating branches as needed."""
        parts = [p for p in tree_path.strip("/").split("/") if p]
        node = self._tree
        for part in parts:
            node = node.setdefault(part, {})
        node.setdefault("_nodes", [])
        if node_id not in node["_nodes"]:
            node["_nodes"].append(node_id)

    def remove_node(self, node_id: str, tree_path: str):
        """Remove a node_id from a tree path."""
        parts = [p for p in tree_path.strip("/").split("/") if p]
        subtree = self._navigate(parts)
        if subtree and "_nodes" in subtree:
            subtree["_nodes"] = [n for n in subtree["_nodes"] if n != node_id]

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self._tree, f, indent=2)

    def load(self, path: Path):
        if path.exists():
            with open(path) as f:
                self._tree = json.load(f)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _navigate(self, parts: list[str]) -> Optional[dict]:
        node = self._tree
        for part in parts:
            if part not in node:
                return None
            node = node[part]
        return node

    def _collect_nodes(self, subtree: dict) -> list[str]:
        """Recursively collect all _nodes lists under a subtree."""
        result = []
        if isinstance(subtree, dict):
            result.extend(subtree.get("_nodes", []))
            for key, val in subtree.items():
                if key != "_nodes":
                    result.extend(self._collect_nodes(val))
        return list(set(result))

    def summary(self) -> dict:
        """Return a summary of node counts per branch."""
        def count(d):
            if not isinstance(d, dict):
                return 0
            total = len(d.get("_nodes", []))
            for k, v in d.items():
                if k != "_nodes":
                    total += count(v)
            return total

        return {
            "semantic": count(self._tree.get("semantic", {})),
            "episodic": count(self._tree.get("episodic", {})),
            "procedural": count(self._tree.get("procedural", {})),
        }
