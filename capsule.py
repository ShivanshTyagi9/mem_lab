"""
capsule.py — MemoryCapsule v1.1: graph + vector store + markdown KB

Three layers that work together:
  ┌─────────────────────────────────────────────────────────────┐
  │  Markdown KB (kb_store.py)                                  │
  │  Structured synthesis: skills.md, people.md, facts.md ...   │
  │  → Injected as rich context blocks into agent system prompts │
  ├─────────────────────────────────────────────────────────────┤
  │  Vector Store (vector_store.py)                             │
  │  Semantic similarity over all nodes — no LLM at query time  │
  │  → Primary retrieval index                                  │
  ├─────────────────────────────────────────────────────────────┤
  │  Memory Graph (models.py)                                   │
  │  Relationships, temporal chains, decay scoring              │
  │  → Secondary index, graph expansion, history traversal      │
  └─────────────────────────────────────────────────────────────┘

Ingestion pipeline:
  text → extract_memories (LLM) → for each memory:
    1. dedup check (vector similarity, no LLM)
    2. classify_into_tree (LLM) — done in batch to reduce calls
    3. get_relationship_candidates (vector similarity, not full branch scan)
    4. detect_relationship (LLM, now on 8 candidates max, not 50+)
    5. add to graph + vector_store + kb_store (quick_append)

Query pipeline — NO LLM CALLS:
  text → embed → vector_search → graph_expand → hybrid_score → kb_inject → return

.umc archive (ZIP):
  capsule.json  — graph nodes + edges + index + metadata
  vectors.json  — all embeddings (portable, backend-independent)
  kb.json       — all KB section markdown strings
"""

import json
import os
import time
import zipfile
from collections import defaultdict
from typing import List, Optional, Dict, Any

from models import MemoryNode, MemoryGraph, MemType, TaskStatus, Rel, TREE_BRANCHES
from vector_store import VectorStore, EmbeddingModel
from kb_store import KBStore
from retrieval import (
    retrieve,
    retrieve_with_history,
    check_duplicate,
    get_relationship_candidates,
)
from llm import (
    extract_memories,
    classify_into_tree,
    detect_relationship,
    consolidate_episodes,
)


CAPSULE_VERSION = "1.1"
CAPSULE_EXT     = ".umc"


class MemoryCapsule:
    """
    Plug-and-play memory for AI agents.

    Quick start:
    ────────────
      cap = MemoryCapsule()
      cap.ingest("Alex moved from Google SWE to Stripe PM in 2024, Seattle.")
      cap.ingest("Alex leads a team of 5 on payments infrastructure.")
      cap.ingest("Alex just got promoted to VP of Product at Stripe.")

      # Query — pure vector + graph, no LLM
      cap.query("Where does Alex work?")
      # → ["Alex was promoted to VP of Product at Stripe.", ...]

      # Rich query with KB context block (for system prompt injection)
      result = cap.query_full("Where does Alex work?")
      result["context_block"]   # ready-to-inject markdown
      result["kb_sections"]     # [(section_name, markdown), ...]

      # History of superseded facts
      cap.query_with_history("Alex job")
      # → {"current": [...], "history": [...]}

      # Consolidate episodes into derived facts, regenerate KB
      cap.consolidate(verbose=True)

      # Export to a single portable file
      cap.export("session.umc")

      # Restore anywhere — vector embeddings + KB included
      cap = MemoryCapsule.load("session.umc")
    """

    def __init__(
        self,
        name: str = "capsule",
        persist_dir: Optional[str] = None,
        embed_model: Optional[EmbeddingModel] = None,
        raw_log_path: Optional[str] = None,
    ):
        self.graph    = MemoryGraph()
        self.name     = name
        self.vectors  = VectorStore(
            persist_dir=persist_dir,
            embed_model=embed_model,
        )
        self.kb       = KBStore()
        self._meta    = {
            "name":         name,
            "created_at":   time.time(),
            "version":      CAPSULE_VERSION,
            "ingest_count": 0,
        }
        # Raw memory log — every extracted memory written as markdown
        self._raw_log_path: str = raw_log_path or f"{name}_raw_memories.md"
        self._ensure_raw_log()

    # ──────────────────────────────────────────────────────────────────────
    # RAW MEMORY LOG
    # ──────────────────────────────────────────────────────────────────────

    def _ensure_raw_log(self):
        """Create the raw memory log file if it doesn't exist."""
        if not os.path.exists(self._raw_log_path):
            ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            with open(self._raw_log_path, "w", encoding="utf-8") as f:
                f.write(f"# {self.name} — Raw Memory Log\n")
                f.write(f"_Created: {ts}_\n\n")
                f.write("---\n\n")

    def _log_raw_memory(self, node: MemoryNode, source_text: str = ""):
        """
        Append a single memory node to the raw markdown log.

        Format:
          ## 2024-03-04 14:22:01
          **Type:** preference  |  **Section:** preferences  |  **Confidence:** 0.95
          The user loves their girlfriend's ass and wants to have anal sex with her.

          > _Source:_ "I find my gf hot and love her ass so much and i want to fuck her ass"
        """
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(node.created_at))
        lines = [
            f"## {ts}\n",
            f"**Type:** {node.mem_type}  |  "
            f"**Section:** {node.source_kb or 'general'}  |  "
            f"**Confidence:** {node.confidence:.2f}\n\n",
            f"{node.content}\n",
        ]
        if source_text:
            # Truncate long source text for readability
            preview = source_text[:200].replace("\n", " ").strip()
            if len(source_text) > 200:
                preview += "..."
            lines.append(f"\n> _Source:_ \"{preview}\"\n")
        lines.append("\n---\n\n")

        with open(self._raw_log_path, "a", encoding="utf-8") as f:
            f.writelines(lines)

    def _log_dedup(self, existing_content: str, incoming_text: str = ""):
        """Log when a duplicate was detected and an existing memory was reinforced."""
        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        preview = incoming_text[:120].replace("\n", " ").strip()
        with open(self._raw_log_path, "a", encoding="utf-8") as f:
            f.write(f"## {ts} _(reinforced)_\n")
            f.write(f"**Duplicate detected** — reinforced existing memory:\n\n")
            f.write(f"{existing_content}\n")
            if preview:
                f.write(f"\n> _Trigger:_ \"{preview}\"\n")
            f.write("\n---\n\n")

    # ──────────────────────────────────────────────────────────────────────
    # INGESTION
    # ──────────────────────────────────────────────────────────────────────

    def ingest(self, text: str, hint: Optional[str] = None, verbose: bool = False) -> List[MemoryNode]:
        """
        Main ingestion entry point. Feed it any text.

        Improvements over v1.0:
          - Deduplication: similar content strengthens existing node instead of
            creating a duplicate.
          - Relationship candidates come from vector search (top-8 similar),
            not full branch scan (was O(branch_size) LLM context).
          - classify_into_tree calls batched at end if >1 memory extracted.
          - KB sections updated cheaply via quick_append (LLM only on consolidate).
        """
        self._meta["ingest_count"] += 1
        created = []

        # Step 1: Extract memories (LLM — this is the right place for it)
        try:
            extracted = extract_memories(text, hint=hint)
        except Exception as e:
            print(f"[ingest] ERROR in extract_memories: {e}")
            return []

        if verbose:
            print(f"[ingest] extracted {len(extracted)} memory units")

        for item in extracted:
            content     = (item.get("content") or "").strip()
            mem_type    = item.get("mem_type", MemType.FACT)
            confidence  = float(item.get("confidence", 1.0))
            is_temporal = bool(item.get("is_temporal", False))
                # Router hint bias (does NOT modify content, only storage category)
            if hint:
                if mem_type == MemType.FACT:
                    map_cat = {
                        "task": MemType.TASK,
                        "procedural": MemType.PROCEDURAL,
                        "fact": MemType.FACT,
                        "episode": MemType.EPISODE,
                        "preference": MemType.PREFERENCE,
                        "ignore": MemType.FACT,
                    }
                    mem_type = map_cat.get(hint, mem_type)
            if not content:
                continue

            valid_types = (MemType.FACT, MemType.PREFERENCE,
                           MemType.EPISODE, MemType.PROCEDURAL, MemType.TASK)
            if mem_type not in valid_types:
                mem_type = MemType.FACT

            # ── Deduplication ─────────────────────────────────────────────
            dup_node = check_duplicate(content, self.vectors, self.graph)
            if dup_node is not None:
                dup_node.reinforce(amount=0.05)
                self._log_dedup(dup_node.content, text)
                if verbose:
                    print(f"[ingest] DEDUP: reinforced existing node "
                          f"'{dup_node.content[:60]}'")
                continue

            # ── Tree classification (LLM) ─────────────────────────────────
            try:
                tree_path = classify_into_tree(content)
            except Exception as e:
                if verbose:
                    print(f"[ingest] classify error: {e}")
                tree_path = ["Tasks"] if mem_type == MemType.TASK else ["General"]

            # ── Create the node ───────────────────────────────────────────
            node = MemoryNode(
                content=content,
                mem_type=mem_type,
                tree_path=tree_path,
                confidence=confidence,
            )

            if is_temporal:
                node.decay_rate = 0.001
                node.salience   = 0.6

            # ── Populate task-specific fields ─────────────────────────────
            if mem_type == MemType.TASK:
                node.task_title  = item.get("task_title", "").strip()
                node.due_date    = item.get("due_date", "") or None
                node.task_status = TaskStatus.PENDING
                node.tree_path   = ["Tasks"]

            # ── Route to KB section ───────────────────────────────────────
            section = self.kb.route_node_to_section(node)
            node.source_kb = section

            # ── Relationship detection ────────────────────────────────────
            # Use vector search to find the top-8 most similar existing nodes.
            # OLD: sent entire branch (up to 50+ nodes) to LLM.
            # NEW: only send top-8 semantically similar nodes → much cheaper.
            rel_candidates = get_relationship_candidates(
                content, self.vectors, self.graph, top_k=8
            )
            try:
                rel = detect_relationship(content, rel_candidates)
            except Exception as e:
                if verbose:
                    print(f"[ingest] relationship detection error: {e}")
                rel = None

            # ── Add to graph + vector store + KB ─────────────────────────
            self.graph.add_node(node)
            self.vectors.add(
                node.id,
                node.content,
                metadata={
                    "mem_type":  node.mem_type,
                    "is_latest": node.is_latest,
                    "source_kb": node.source_kb,
                },
            )
            self.kb.quick_append(section, node)
            self._log_raw_memory(node, source_text=text)
            created.append(node)

            if verbose:
                print(f"[ingest] + [{mem_type}] [{section}] {content[:80]}")

            # ── Wire relationships ────────────────────────────────────────
            if rel and isinstance(rel, dict):
                relationship = rel.get("relationship", "NONE")
                target_id    = rel.get("target_id")
                reason       = rel.get("reason", "")

                if relationship != "NONE" and target_id and target_id in self.graph.nodes:
                    self.graph.add_edge(node.id, relationship, target_id)

                    if relationship == Rel.UPDATES:
                        old_node = self.graph.nodes[target_id]

                        # ── Build temporal chain metadata ─────────────────
                        # The new node inherits the chain root from the old one.
                        # If the old node was the first in its chain, its own id is the root.
                        root_id = old_node.chain_root_id or old_node.id
                        node.chain_root_id  = root_id
                        node.chain_position = old_node.chain_position + 1
                        node.update_reason  = reason

                        # Supersede the old node — records superseded_at timestamp
                        self.graph.supersede(target_id, reason=reason)

                        # Temporal chain edge: new → old
                        self.graph.add_edge(node.id, Rel.TEMPORAL_CHAIN, target_id)

                        # Update vector store metadata for superseded node
                        self.vectors.add(
                            target_id,
                            old_node.content,
                            metadata={"is_latest": False},
                        )

                        if verbose:
                            print(
                                f"[ingest]   ↳ UPDATES v{old_node.chain_position}→v{node.chain_position}: "
                                f"'{old_node.content[:50]}'"
                            )
                            if reason:
                                print(f"[ingest]      reason: {reason}")

                    elif verbose:
                        print(f"[ingest]   ↳ {relationship}: {target_id[:8]}...")

        return created

    def ingest_many(self, texts: List[str], verbose: bool = False) -> int:
        total = 0
        for text in texts:
            nodes = self.ingest(text, verbose=verbose)
            total += len(nodes)
        return total

    # ──────────────────────────────────────────────────────────────────────
    # QUERY — NO LLM CALLS
    # ──────────────────────────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 5) -> List[str]:
        """
        Fast semantic retrieval. No LLM call.

        Pipeline: embed → vector search → graph expand → hybrid score → return

        Hybrid score = 0.6 × semantic_similarity + 0.4 × memory_strength
        Graph expansion: EXTENDS/DERIVES neighbours pulled for richer context.
        """
        result = retrieve(
            query, self.graph, self.vectors, self.kb, top_k=top_k
        )
        return result["memories"]

    def query_full(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Full retrieval result with KB sections and context block.

        Returns:
          {
            "memories":     [str],             # top_k content strings
            "nodes":        [MemoryNode],      # for direct node access
            "scores":       [(id, score)],     # debug: per-node hybrid scores
            "kb_sections":  [(name, markdown)],# relevant KB sections
            "context_block": str,              # ready-to-inject into system prompt
          }
        """
        return retrieve(
            query, self.graph, self.vectors, self.kb, top_k=top_k
        )

    def query_with_history(self, query: str, top_k: int = 5) -> Dict[str, List]:
        """
        Returns current memories + temporal history (superseded facts).

        Returns:
          {
            "memories":  [str],  # current (is_latest=True)
            "history":   [str],  # superseded, newest first
            "kb_sections": [...],
            "context_block": str,
          }
        """
        result = retrieve_with_history(
            query, self.graph, self.vectors, self.kb, top_k=top_k
        )
        return {
            "current":       result["memories"],
            "history":       result.get("history", []),
            "kb_sections":   result.get("kb_sections", []),
            "context_block": result.get("context_block", ""),
        }

    def query_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Rich results with per-node metadata. For agent introspection/debugging.
        """
        result = retrieve(
            query, self.graph, self.vectors, self.kb, top_k=top_k
        )
        scores_map = dict(result["scores"])
        output = []
        for node in result["nodes"]:
            related = self.graph.get_related(node.id)
            history = self.graph.temporal_chain(node.id)
            output.append({
                "content":      node.content,
                "mem_type":     node.mem_type,
                "source_kb":    node.source_kb,
                "strength":     round(node.strength(), 4),
                "hybrid_score": scores_map.get(node.id, 0.0),
                "confidence":   node.confidence,
                "tree_path":    node.tree_path,
                "chain_position": node.chain_position,
                "related":      [r.content for r in related],
                "history":      [h.content for h in history[1:]],
            })
        return output

    def query_timeline(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Retrieve facts and their full version timelines.

        For each matched fact, returns the complete history of how that
        fact changed over time — who the user worked for, where they lived,
        who they were dating, etc.

        Returns a list of timeline objects, one per matched fact:
          [
            {
              "current_fact": str,              # what is true now
              "versions": [                     # full history, oldest first
                {
                  "version":         1,
                  "content":         "User works at Google",
                  "is_current":      False,
                  "believed_from":   "2024-01-01 10:00:00",
                  "believed_until":  "2024-06-15 14:32:00",
                  "belief_duration": "165 days",
                  "update_reason":   "changed employer from Google to Stripe",
                  "node_id":         "abc123",
                },
                {
                  "version":         2,
                  "content":         "User works at Stripe",
                  "is_current":      True,
                  ...
                }
              ]
            },
            ...
          ]
        """
        result = retrieve(
            query, self.graph, self.vectors, self.kb, top_k=top_k
        )

        timelines = []
        for node in result["nodes"]:
            chain = self.graph.get_full_timeline(node.id)
            if len(chain) <= 1 and chain:
                # Single-version fact — still show it for completeness
                timelines.append({
                    "current_fact": node.content,
                    "versions":     chain,
                    "has_history":  False,
                })
            elif chain:
                timelines.append({
                    "current_fact": node.content,
                    "versions":     chain,
                    "has_history":  True,
                    "version_count": len(chain),
                })

        return timelines

    # ──────────────────────────────────────────────────────────────────────
    # TASK MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────

    def get_pending_tasks(self) -> List[MemoryNode]:
        """Return all pending task nodes, sorted by due_date."""
        return self.graph.get_pending_tasks()

    def mark_task_done(self, node_id: str) -> bool:
        """Mark a task as done by node_id. Returns True if found."""
        return self.graph.mark_task_done(node_id)

    def mark_task_cancelled(self, node_id: str) -> bool:
        """Mark a task as cancelled by node_id."""
        return self.graph.mark_task_cancelled(node_id)

    # ──────────────────────────────────────────────────────────────────────
    # GRAPH EXPORT (for visualization)
    # ──────────────────────────────────────────────────────────────────────

    def export_graph_json(self, path: Optional[str] = None) -> str:
        """
        Export the full graph as a standalone JSON file for visualization.

        The JSON contains nodes and edges in a format ready for D3/vis.js/etc.
        Each node includes: id, content, mem_type, source_kb, strength,
        is_latest, chain_position, due_date, task_status, tree_path, created_at.

        File is named {capsule_name}_graph.json by default.
        """
        if path is None:
            path = f"{self.name}_graph.json"

        nodes_out = []
        for node in self.graph.nodes.values():
            nodes_out.append({
                "id":             node.id,
                "content":        node.content,
                "mem_type":       node.mem_type,
                "source_kb":      node.source_kb,
                "tree_path":      node.tree_path,
                "branch":         node.branch(),
                "is_latest":      node.is_latest,
                "strength":       round(node.strength(), 4),
                "salience":       round(node.salience, 4),
                "confidence":     node.confidence,
                "chain_position": node.chain_position,
                "chain_root_id":  node.chain_root_id,
                "update_reason":  node.update_reason,
                "created_at":     node.created_at,
                "created_label":  time.strftime(
                    "%Y-%m-%d %H:%M", time.localtime(node.created_at)
                ),
                "superseded_at":  node.superseded_at,
                # task fields
                "task_title":     node.task_title,
                "task_status":    node.task_status,
                "due_date":       node.due_date,
            })

        edges_out = []
        for src_id, edge_list in self.graph.edges.items():
            for rel, tgt_id in edge_list:
                if src_id in self.graph.nodes and tgt_id in self.graph.nodes:
                    edges_out.append({
                        "source":   src_id,
                        "target":   tgt_id,
                        "relation": rel,
                    })

        payload = {
            "capsule_name": self.name,
            "exported_at":  time.strftime("%Y-%m-%d %H:%M:%S"),
            "stats":        self.graph.stats(),
            "nodes":        nodes_out,
            "edges":        edges_out,
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[export_graph] {path} ({len(nodes_out)} nodes, {len(edges_out)} edges)")
        return path

    # ──────────────────────────────────────────────────────────────────────
    # KB SECTIONS
    # ──────────────────────────────────────────────────────────────────────

    def get_kb_section(self, section_name: str) -> str:
        """Get the markdown content of a knowledge base section."""
        return self.kb.get_section(section_name)

    def get_all_kb_sections(self) -> Dict[str, str]:
        """Get all populated KB sections."""
        return self.kb.get_all_populated()

    # ──────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ──────────────────────────────────────────────────────────────────────

    def consolidate(
        self,
        min_episodes: int = 3,
        regenerate_kb: bool = True,
        verbose: bool = False,
    ) -> int:
        """
        Merge accumulated episodes into stable derived facts.
        If regenerate_kb=True (default), also rebuilds all KB sections via LLM.

        Episodes decay fast. Consolidation creates a durable derived fact
        while individual episodes can fade.
        """
        by_branch: Dict[str, List[MemoryNode]] = defaultdict(list)
        for node in self.graph.nodes.values():
            if node.mem_type == MemType.EPISODE and node.is_latest:
                by_branch[node.branch()].append(node)

        consolidated_count = 0

        for branch, episodes in by_branch.items():
            if len(episodes) < min_episodes:
                continue

            summary = consolidate_episodes(episodes, branch)
            if not summary:
                continue

            paths     = [e.tree_path for e in episodes]
            tree_path = list(max(set(map(tuple, paths)), key=paths.count))

            derived = MemoryNode(
                content=summary,
                mem_type=MemType.DERIVED,
                tree_path=tree_path,
                confidence=0.85,
            )
            derived.source_kb = self.kb.route_node_to_section(derived)

            self.graph.add_node(derived)
            self.vectors.add(derived.id, derived.content, {
                "mem_type":  derived.mem_type,
                "is_latest": True,
                "source_kb": derived.source_kb,
            })

            for e in episodes:
                self.graph.add_edge(derived.id, Rel.CONSOLIDATES, e.id)

            consolidated_count += 1
            if verbose:
                print(f"[consolidate] {branch}: {len(episodes)} episodes → '{summary[:60]}'")

        # Regenerate KB sections from the updated graph
        if regenerate_kb:
            if verbose:
                print("[consolidate] regenerating KB sections...")
            self.kb.regenerate_all(self.graph.all_latest(), verbose=verbose)

        if verbose and consolidated_count == 0:
            print("[consolidate] nothing to consolidate")

        return consolidated_count

    def decay(self, threshold: float = 0.05, verbose: bool = False) -> int:
        """
        Prune memories whose strength has dropped below the threshold.
        Removes from graph AND vector store.
        """
        to_remove = [
            nid for nid, n in self.graph.nodes.items()
            if n.strength() < threshold
        ]
        for nid in to_remove:
            self.graph.remove_node(nid)
            self.vectors.delete(nid)

        if verbose and to_remove:
            print(f"[decay] pruned {len(to_remove)} nodes (threshold={threshold})")

        return len(to_remove)

    # ──────────────────────────────────────────────────────────────────────
    # PLUG / UNPLUG
    # ──────────────────────────────────────────────────────────────────────

    def export(self, path: Optional[str] = None) -> str:
        """
        Plug-out: pack the entire capsule into a single .umc file.

        .umc archive contains:
          capsule.json  — graph nodes + edges + tree index + metadata
          vectors.json  — all node embeddings (portable, backend-independent)
          kb.json       — all KB section markdown strings
        """
        if path is None:
            path = f"{self.name}{CAPSULE_EXT}"

        self._meta["exported_at"] = time.time()
        self._meta["node_count"]  = len(self.graph.nodes)

        capsule_payload = {
            "meta":  self._meta,
            "graph": self.graph.to_dict(),
        }
        vectors_payload = self.vectors.to_dict()
        kb_payload      = self.kb.to_dict()

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("capsule.json", json.dumps(capsule_payload, indent=2))
            z.writestr("vectors.json", json.dumps(vectors_payload))
            z.writestr("kb.json",      json.dumps(kb_payload, indent=2))

        size_kb = os.path.getsize(path) / 1024
        print(
            f"[export] {path} "
            f"({len(self.graph.nodes)} nodes, "
            f"{len(self.vectors)} vectors, "
            f"{len(self.kb.get_all_populated())} KB sections, "
            f"{size_kb:.1f} KB)"
        )
        return path

    @classmethod
    def load(cls, path: str, persist_dir: Optional[str] = None) -> "MemoryCapsule":
        """
        Plug-in: restore a capsule from a .umc file.
        All three layers (graph, vectors, KB) are restored immediately.
        No warm-up. First query is fast.
        """
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            capsule_data = json.loads(z.read("capsule.json"))
            vectors_data = json.loads(z.read("vectors.json")) if "vectors.json" in names else {}
            kb_data      = json.loads(z.read("kb.json"))      if "kb.json"      in names else {}

        meta = capsule_data.get("meta", {})
        name = meta.get("name", "capsule")

        cap          = cls.__new__(cls)
        cap.name     = name
        cap._meta    = meta
        cap.graph    = MemoryGraph.from_dict(capsule_data["graph"])
        cap.vectors  = VectorStore.from_dict(vectors_data, persist_dir=persist_dir)
        cap.kb       = KBStore.from_dict(kb_data)
        cap._raw_log_path = f"{name}_raw_memories.md"
        cap._ensure_raw_log()

        n = len(cap.graph.nodes)
        v = len(cap.vectors)
        print(f"[load] {path} → {n} nodes, {v} vectors, KB: {cap.kb}")
        return cap

    # ──────────────────────────────────────────────────────────────────────
    # MERGE
    # ──────────────────────────────────────────────────────────────────────

    def merge(
        self,
        other: "MemoryCapsule",
        name: Optional[str] = None,
        prefer: str = "self",
    ) -> "MemoryCapsule":
        """
        Create a new capsule combining knowledge from self and other.

        prefer: "self" | "other" | "newest" — conflict resolution strategy.
        Deduplication by exact content match (lowercased).
        Embeddings, edges, and KB sections are all merged.
        """
        merged = MemoryCapsule(name=name or f"{self.name}+{other.name}")
        seen: Dict[str, str] = {}   # content_lower → node_id in merged

        def _add(node: MemoryNode):
            key = node.content.lower().strip()
            if key in seen:
                existing_id = seen[key]
                existing    = merged.graph.nodes[existing_id]
                if prefer == "other" or (
                    prefer == "newest" and node.created_at > existing.created_at
                ):
                    merged.graph.remove_node(existing_id)
                    merged.vectors.delete(existing_id)
                    merged.graph.add_node(node)
                    merged.vectors.add(node.id, node.content, {
                        "mem_type": node.mem_type, "is_latest": node.is_latest
                    })
                    seen[key] = node.id
            else:
                merged.graph.add_node(node)
                merged.vectors.add(node.id, node.content, {
                    "mem_type": node.mem_type, "is_latest": node.is_latest
                })
                seen[key] = node.id

        for node in self.graph.nodes.values():
            _add(node)
        for src_id, edges in self.graph.edges.items():
            if src_id in merged.graph.nodes:
                for rel, tgt_id in edges:
                    if tgt_id in merged.graph.nodes:
                        merged.graph.add_edge(src_id, rel, tgt_id)

        other_id_map: Dict[str, str] = {}
        for old_id, node in other.graph.nodes.items():
            key = node.content.lower().strip()
            _add(node)
            other_id_map[old_id] = seen.get(key, node.id)

        for src_id, edges in other.graph.edges.items():
            m_src = other_id_map.get(src_id)
            if m_src and m_src in merged.graph.nodes:
                for rel, tgt_id in edges:
                    m_tgt = other_id_map.get(tgt_id)
                    if m_tgt and m_tgt in merged.graph.nodes:
                        merged.graph.add_edge(m_src, rel, m_tgt)

        # Merge KB: prefer non-empty sections
        for section_name, content in self.kb.get_all_populated().items():
            merged.kb.set_section(section_name, content)
        for section_name, content in other.kb.get_all_populated().items():
            if not merged.kb.get_section(section_name):
                merged.kb.set_section(section_name, content)

        merged._meta["merged_from"]  = [self.name, other.name]
        merged._meta["ingest_count"] = (
            self._meta.get("ingest_count", 0) +
            other._meta.get("ingest_count", 0)
        )

        print(
            f"[merge] {self.name} ({len(self.graph.nodes)} nodes) + "
            f"{other.name} ({len(other.graph.nodes)} nodes) → "
            f"{merged.name} ({len(merged.graph.nodes)} nodes)"
        )
        return merged

    # ──────────────────────────────────────────────────────────────────────
    # INTROSPECTION
    # ──────────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        s = self.graph.stats()
        s["name"]          = self.name
        s["ingest_count"]  = self._meta.get("ingest_count", 0)
        s["created_at"]    = self._meta.get("created_at")
        s["vector_count"]  = len(self.vectors)
        s["vector_backend"] = self.vectors.backend_name
        s["kb_sections"]   = list(self.kb.get_all_populated().keys())
        return s

    def dump_branch(self, branch: str, include_superseded: bool = False) -> List[dict]:
        nodes = self.graph.nodes_in_branch(branch)
        if not include_superseded:
            nodes = [n for n in nodes if n.is_latest]
        return [
            {
                "content":   n.content,
                "type":      n.mem_type,
                "source_kb": n.source_kb,
                "strength":  round(n.strength(), 4),
                "is_latest": n.is_latest,
                "path":      n.tree_path,
            }
            for n in sorted(nodes, key=lambda n: n.strength(), reverse=True)
        ]

    def __repr__(self) -> str:
        s = self.graph.stats()
        return (
            f"<MemoryCapsule '{self.name}' "
            f"nodes={s['total_nodes']} "
            f"latest={s['latest_nodes']} "
            f"vectors={len(self.vectors)} "
            f"kb={self.kb}>"
        )
