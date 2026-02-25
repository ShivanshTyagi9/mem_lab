"""
models.py — Core data structures for the AgentMem system.
Nodes are raw natural language facts. Edges are temporal/relational metadata only.
"""

from dataclasses import dataclass, field
from typing import Optional, Literal
from datetime import datetime
import uuid


MemoryType = Literal["semantic", "episodic", "procedural"]

EdgeType = Literal[
    "supersedes",       # new fact replaces old
    "superseded_by",    # inverse
    "contradicts",      # conflict — both kept, agent reasons over both
    "reinforces",       # same fact confirmed again, confidence increases
    "refines",          # adds detail without replacing
    "follows",          # episodic chain: this happened after that
    "triggers",         # procedural: this condition → that action
]


@dataclass
class MemoryEdge:
    to: str                  # target node_id
    type: EdgeType
    timestamp: str           # ISO format


@dataclass
class MemoryNode:
    content: str             # raw natural language fact — never decomposed
    memory_type: MemoryType
    tree_path: str           # e.g. "semantic/people/aryan"
    id: str = field(default_factory=lambda: "node_" + uuid.uuid4().hex[:8])
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    confidence: float = 0.9
    edges: list[MemoryEdge] = field(default_factory=list)
    source_session: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "tree_path": self.tree_path,
            "timestamp": self.timestamp,
            "confidence": self.confidence,
            "edges": [{"to": e.to, "type": e.type, "timestamp": e.timestamp} for e in self.edges],
            "source_session": self.source_session,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        edges = [MemoryEdge(to=e["to"], type=e["type"], timestamp=e["timestamp"]) for e in d.get("edges", [])]
        return cls(
            id=d["id"],
            content=d["content"],
            memory_type=d["memory_type"],
            tree_path=d["tree_path"],
            timestamp=d["timestamp"],
            confidence=d.get("confidence", 0.9),
            edges=edges,
            source_session=d.get("source_session"),
        )
