"""
models.py — Core data structures for uMemory

Design principles:
  - Nodes hold WHOLE FACTS (not triplets). A node is a self-contained unit of knowledge.
  - The Graph holds relationships between facts. Not the storage — the wiring.
  - Tree branch index kept for backwards compatibility and secondary lookup.
  - Temporal chains preserve history without destroying it.

v1.1 changes:
  - MemoryNode.source_kb: which markdown KB section this node belongs to
  - MemoryNode.reinforce(): boost salience on duplicate detection
  - MemoryGraph.all_latest(): flat list of current nodes (used by vector retrieval)
  - MemoryGraph.add_edge(): deduplicates edges
"""

import uuid
import time
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Set, Any


class MemType:
    FACT       = "fact"
    PREFERENCE = "preference"
    EPISODE    = "episode"
    PROCEDURAL = "procedural"
    DERIVED    = "derived"
    TASK       = "task"       # Actionable item with optional due date and status


# Task status values
class TaskStatus:
    PENDING   = "pending"
    DONE      = "done"
    CANCELLED = "cancelled"


DECAY_RATES = {
    MemType.FACT:       0.000_001,
    MemType.PREFERENCE: 0.000_002,
    MemType.EPISODE:    0.000_020,
    MemType.PROCEDURAL: 0.000_000_5,
    MemType.DERIVED:    0.000_005,
    MemType.TASK:       0.000_005,   # Tasks decay slowly — they persist until done/cancelled
}

BASE_SALIENCE = {
    MemType.FACT:       0.90,
    MemType.PREFERENCE: 0.85,
    MemType.EPISODE:    0.70,
    MemType.PROCEDURAL: 0.95,
    MemType.DERIVED:    0.80,
    MemType.TASK:       0.95,   # High salience — tasks are urgent by nature
}

TREE_BRANCHES = ["People", "Tasks", "Technical", "Preferences", "Events", "General"]


class Rel:
    UPDATES        = "UPDATES"
    EXTENDS        = "EXTENDS"
    DERIVES        = "DERIVES"
    CONSOLIDATES   = "CONSOLIDATES"
    TEMPORAL_CHAIN = "TEMPORAL_CHAIN"


@dataclass
class MemoryNode:
    """
    A single unit of memory. One whole fact, preference, episode, or procedure.

    v1.1: source_kb tracks which markdown KB section this node was routed to.
    """

    content:    str
    mem_type:   str
    tree_path:  List[str]
    confidence: float = 1.0
    source_kb:  str   = ""    # "facts" | "skills" | "processes" | "people" | "preferences" | "events" | "tasks"

    # ── Task-specific fields (only populated when mem_type == "task") ─────
    task_title:  str            = ""               # short title, e.g. "Meeting Red"
    task_status: str            = TaskStatus.PENDING  # pending | done | cancelled
    due_date:    Optional[str]  = None             # ISO string "YYYY-MM-DD HH:MM" or None
    completed_at: Optional[float] = None           # unix timestamp when marked done

    id:            str   = field(default_factory=lambda: str(uuid.uuid4()))
    created_at:    float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count:  int   = 0
    is_latest:     bool  = True
    salience:      float = field(init=False)
    decay_rate:    float = field(init=False)

    # ── Temporal chain fields ─────────────────────────────────────────────
    # When this node was replaced by a newer version.
    # None = still the current truth (is_latest=True).
    superseded_at:   Optional[float] = None

    # ID of the very first node in this fact's history.
    # All versions of the same fact share the same chain_root_id.
    # For root nodes (first version ever), chain_root_id == self.id.
    chain_root_id:   Optional[str]   = None

    # 1-based version counter. First ever fact = 1, first update = 2, etc.
    chain_position:  int             = 1

    # Human-readable reason why this fact replaced the previous version.
    # Set by detect_relationship when it returns UPDATES.
    update_reason:   str             = ""

    def __post_init__(self):
        self.salience   = BASE_SALIENCE.get(self.mem_type, 0.80)
        self.decay_rate = DECAY_RATES.get(self.mem_type, 0.000_010)

    def strength(self) -> float:
        elapsed = time.time() - self.last_accessed
        base    = self.salience * math.exp(-self.decay_rate * elapsed)
        boost   = min(0.25, self.access_count * 0.025)
        return min(1.0, (base + boost) * self.confidence)

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1

    def reinforce(self, amount: float = 0.05):
        """
        Boost salience directly — called when a duplicate is detected on ingest.
        Repeated mention of a fact strengthens it without creating a new node.
        """
        self.salience      = min(1.0, self.salience + amount)
        self.last_accessed = time.time()
        self.access_count  += 1

    def mark_superseded(self, reason: str = ""):
        """
        Called when a newer version of this fact is ingested.
        Records the exact timestamp this fact stopped being the current truth.
        """
        self.is_latest      = False
        self.superseded_at  = time.time()

    def age_seconds(self) -> float:
        """How long ago this fact was first believed."""
        return time.time() - self.created_at

    def belief_duration_seconds(self) -> Optional[float]:
        """
        How long this fact was the current truth.
        Returns None if still the latest (still believed).
        Returns a float of seconds if superseded.
        """
        if self.superseded_at is None:
            return None
        return self.superseded_at - self.created_at

    def timeline_label(self) -> str:
        """
        Short human-readable label for display in a timeline.
        e.g.  "v2  2024-03-01 → 2024-06-15  (believed 106 days)"
        """
        created  = time.strftime("%Y-%m-%d", time.localtime(self.created_at))
        if self.superseded_at:
            ended    = time.strftime("%Y-%m-%d", time.localtime(self.superseded_at))
            duration = self.superseded_at - self.created_at
            days     = int(duration / 86400)
            dur_str  = f"{days}d" if days >= 1 else f"{int(duration/3600)}h"
            return f"v{self.chain_position}  {created} → {ended}  (believed {dur_str})"
        return f"v{self.chain_position}  {created} → present"

    def branch(self) -> str:
        return self.tree_path[0] if self.tree_path else "General"

    def to_dict(self) -> dict:
        return {
            "id":              self.id,
            "content":         self.content,
            "mem_type":        self.mem_type,
            "tree_path":       self.tree_path,
            "confidence":      self.confidence,
            "source_kb":       self.source_kb,
            "created_at":      self.created_at,
            "last_accessed":   self.last_accessed,
            "access_count":    self.access_count,
            "is_latest":       self.is_latest,
            "salience":        self.salience,
            "decay_rate":      self.decay_rate,
            # temporal chain fields
            "superseded_at":   self.superseded_at,
            "chain_root_id":   self.chain_root_id,
            "chain_position":  self.chain_position,
            "update_reason":   self.update_reason,
            # task fields
            "task_title":      self.task_title,
            "task_status":     self.task_status,
            "due_date":        self.due_date,
            "completed_at":    self.completed_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryNode":
        node = cls.__new__(cls)
        d.setdefault("source_kb", "")
        d.setdefault("superseded_at",  None)
        d.setdefault("chain_root_id",  None)
        d.setdefault("chain_position", 1)
        d.setdefault("update_reason",  "")
        d.setdefault("task_title",     "")
        d.setdefault("task_status",    TaskStatus.PENDING)
        d.setdefault("due_date",       None)
        d.setdefault("completed_at",   None)
        node.__dict__.update(d)
        return node


class MemoryGraph:
    """
    The wiring layer — not storage.

    nodes:  id → MemoryNode
    edges:  source_id → [(Rel, target_id)]
    index:  branch_name → set of node_ids  (secondary index, vector is primary)
    """

    def __init__(self):
        self.nodes: Dict[str, MemoryNode]  = {}
        self.edges: Dict[str, List[Tuple]] = defaultdict(list)
        self.index: Dict[str, Set[str]]    = defaultdict(set)

    def add_node(self, node: MemoryNode):
        self.nodes[node.id] = node
        self.index[node.branch()].add(node.id)

    def add_edge(self, src_id: str, rel: str, tgt_id: str):
        existing = self.edges.get(src_id, [])
        if (rel, tgt_id) not in existing:
            self.edges[src_id].append((rel, tgt_id))

    def supersede(self, node_id: str, reason: str = ""):
        """Mark a node as no longer the latest truth. Records superseded_at timestamp."""
        if node_id in self.nodes:
            self.nodes[node_id].mark_superseded(reason=reason)

    def remove_node(self, node_id: str):
        node = self.nodes.pop(node_id, None)
        if node:
            self.index[node.branch()].discard(node_id)
        self.edges.pop(node_id, None)
        for src in list(self.edges):
            self.edges[src] = [(r, t) for r, t in self.edges[src] if t != node_id]
            if not self.edges[src]:
                del self.edges[src]

    def nodes_in_branch(self, branch: str) -> List[MemoryNode]:
        ids = self.index.get(branch, set())
        return [self.nodes[i] for i in ids if i in self.nodes]

    def latest_in_branch(self, branch: str) -> List[MemoryNode]:
        return [n for n in self.nodes_in_branch(branch) if n.is_latest]

    def all_latest(self) -> List[MemoryNode]:
        """Flat list of all current (non-superseded) nodes."""
        return [n for n in self.nodes.values() if n.is_latest]

    def temporal_chain(self, node_id: str) -> List[MemoryNode]:
        """
        Walk the TEMPORAL_CHAIN from a node backwards to the root.
        Returns [newest → ... → oldest].
        """
        chain: List[MemoryNode] = []
        current = node_id
        visited: set = set()
        while current and current not in visited:
            visited.add(current)
            node = self.nodes.get(current)
            if node:
                chain.append(node)
            next_id = None
            for rel, tgt in self.edges.get(current, []):
                if rel == Rel.TEMPORAL_CHAIN:
                    next_id = tgt
                    break
            current = next_id
        return chain

    def get_full_timeline(self, node_id: str) -> List[Dict]:
        """
        Complete version history of a fact, sorted oldest → newest (chronological).

        Each entry:
          {
            "version":         int,          # 1-based
            "content":         str,
            "is_current":      bool,
            "believed_from":   str,          # "YYYY-MM-DD HH:MM:SS"
            "believed_until":  str | None,   # None if still current
            "belief_duration": str,          # e.g. "45 days", "3 hours"
            "update_reason":   str,          # why it was replaced
            "node_id":         str,
          }
        """
        chain = self.temporal_chain(node_id)        # newest → oldest
        chain_asc = list(reversed(chain))           # oldest → newest

        result = []
        for node in chain_asc:
            believed_from = time.strftime(
                "%Y-%m-%d %H:%M:%S", time.localtime(node.created_at)
            )
            believed_until  = None
            belief_duration = "still believed"

            if node.superseded_at is not None:
                believed_until = time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime(node.superseded_at)
                )
                secs = node.superseded_at - node.created_at
                if secs >= 86400:
                    belief_duration = f"{int(secs / 86400)} days"
                elif secs >= 3600:
                    belief_duration = f"{int(secs / 3600)} hours"
                else:
                    belief_duration = f"{int(secs / 60)} minutes"

            result.append({
                "version":         node.chain_position,
                "content":         node.content,
                "is_current":      node.is_latest,
                "believed_from":   believed_from,
                "believed_until":  believed_until,
                "belief_duration": belief_duration,
                "update_reason":   node.update_reason,
                "node_id":         node.id,
            })

        return result

    def get_chain_for_root(self, chain_root_id: str) -> List[MemoryNode]:
        """
        Find all versions of a fact using chain_root_id (O(n) but avoids edge walking).
        Returns nodes sorted oldest → newest by chain_position.
        """
        members = [
            n for n in self.nodes.values()
            if n.chain_root_id == chain_root_id
        ]
        return sorted(members, key=lambda n: n.chain_position)

    def get_pending_tasks(self) -> List[MemoryNode]:
        """Return all task nodes that are still pending, sorted by due_date."""
        tasks = [
            n for n in self.nodes.values()
            if n.mem_type == MemType.TASK
            and n.task_status == TaskStatus.PENDING
            and n.is_latest
        ]
        # Sort: tasks with due_date first (chronologically), then undated tasks
        dated   = sorted([t for t in tasks if t.due_date], key=lambda t: t.due_date)
        undated = [t for t in tasks if not t.due_date]
        return dated + undated

    def mark_task_done(self, node_id: str) -> bool:
        """Mark a task as done. Returns True if found and updated."""
        node = self.nodes.get(node_id)
        if node and node.mem_type == MemType.TASK:
            node.task_status  = TaskStatus.DONE
            node.completed_at = time.time()
            return True
        return False

    def mark_task_cancelled(self, node_id: str) -> bool:
        """Mark a task as cancelled."""
        node = self.nodes.get(node_id)
        if node and node.mem_type == MemType.TASK:
            node.task_status = TaskStatus.CANCELLED
            return True
        return False

    def get_related(
        self,
        node_id: str,
        rel_type: Optional[str] = None,
    ) -> List[MemoryNode]:
        result = []
        for rel, tgt_id in self.edges.get(node_id, []):
            if rel_type is None or rel == rel_type:
                node = self.nodes.get(tgt_id)
                if node:
                    result.append(node)
        return result

    def stats(self) -> dict:
        nodes   = list(self.nodes.values())
        by_type: Dict[str, int] = defaultdict(int)
        for n in nodes:
            by_type[n.mem_type] += 1
        return {
            "total_nodes":  len(nodes),
            "latest_nodes": sum(1 for n in nodes if n.is_latest),
            "superseded":   sum(1 for n in nodes if not n.is_latest),
            "by_type":      dict(by_type),
            "total_edges":  sum(len(v) for v in self.edges.values()),
            "branches":     {k: len(v) for k, v in self.index.items()},
        }

    def to_dict(self) -> dict:
        return {
            "nodes": {nid: n.to_dict() for nid, n in self.nodes.items()},
            "edges": {k: list(v) for k, v in self.edges.items()},
            "index": {k: list(v) for k, v in self.index.items()},
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryGraph":
        g = cls()
        g.nodes = {nid: MemoryNode.from_dict(n) for nid, n in d["nodes"].items()}
        g.edges = defaultdict(list, {
            k: [tuple(e) for e in v] for k, v in d["edges"].items()
        })
        g.index = defaultdict(set, {
            k: set(v) for k, v in d.get("index", {}).items()
        })
        return g
