"""
memory_system.py — Core orchestrator for the 3-layer AgentMem system.

This is the main interface. An agent uses this class to:
  - plug()    load a .agentmem file and activate memory
  - retrieve() get relevant memory for a query (called each turn)
  - update()  ingest new information mid-session
  - eject()   compress, save, and return the updated .agentmem file
"""

import json
import zipfile
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Callable
import uuid

from models import MemoryNode, MemoryEdge
from tree_index import TreeIndex
from fact_graph import FactGraph
from leaf_vectors import LeafVectorStore
from embedder import create_embedder
from extractor import MemoryExtractor


# Max nodes to inject into context before triggering leaf-level ranking
CONTEXT_NODE_LIMIT = 12

# Max nodes from tree before triggering leaf ranking
LEAF_RANKING_THRESHOLD = 15


class MemorySystem:
    """
    3-layer plug-and-play memory for AI agents.

    Layer 1: TreeIndex      — hierarchical path-based routing
    Layer 2: FactGraph      — temporal fact graph with conflict edges
    Layer 3: LeafVectorStore — local ranking within large branches

    Usage:
        mem = MemorySystem(llm_fn=my_llm_callable)
        mem.plug("path/to/agent.agentmem")          # load memory
        context = mem.retrieve("user's question")   # each turn
        mem.update(conversation_history)            # after turn or session
        mem.eject("path/to/agent.agentmem")         # save + return file
    """

    def __init__(self, llm_fn: Optional[Callable] = None):
        """
        llm_fn: callable(prompt: str) -> str
          Any function that takes a string prompt and returns a string response.
          Examples: Claude API call, GPT call, local model call.
          If None, extraction is disabled (memory can still be read).
        """
        self._tree = TreeIndex()
        self._graph = FactGraph()
        self._leaves = LeafVectorStore()
        self._embedder = create_embedder()
        self._extractor = MemoryExtractor(llm_fn) if llm_fn else None
        self._session_id = f"session_{uuid.uuid4().hex[:8]}"
        self._session_delta: list[MemoryNode] = []
        self._is_loaded = False
        self._work_dir: Optional[Path] = None
        self._meta: dict = {}

    # ══════════════════════════════════════════════════════════════════════════
    # PLUG — Load a .agentmem file and activate memory
    # ══════════════════════════════════════════════════════════════════════════

    def plug(self, agentmem_path: str) -> dict:
        """
        Load a .agentmem file (zip) into memory.
        Returns a summary of what was loaded.

        If the file doesn't exist, initializes a fresh empty memory.
        """
        agentmem_path = Path(agentmem_path)
        self._work_dir = Path(tempfile.mkdtemp(prefix="agentmem_"))

        if agentmem_path.exists():
            # unzip into work dir
            with zipfile.ZipFile(agentmem_path, "r") as zf:
                zf.extractall(self._work_dir)
            print(f"[agentmem] Plugged: {agentmem_path.name}")
        else:
            print(f"[agentmem] No existing memory found at {agentmem_path} — starting fresh")

        # load each layer
        self._tree.load(self._work_dir / "layer_1" / "tree_index.json")
        self._graph.load(
            self._work_dir / "layer_2" / "graph.gpickle",
            self._work_dir / "layer_2" / "embeddings"
        )
        self._leaves.load(self._work_dir / "layer_3")

        # load metadata
        meta_path = self._work_dir / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self._meta = json.load(f)
        else:
            self._meta = {
                "created": datetime.utcnow().isoformat(),
                "sessions": 0,
                "agent_type": "generic",
                "version": "1.0",
            }

        self._meta["sessions"] = self._meta.get("sessions", 0) + 1
        self._meta["last_session"] = self._session_id
        self._is_loaded = True

        summary = self._tree.summary()
        print(f"[agentmem] Loaded — semantic: {summary['semantic']}, episodic: {summary['episodic']}, procedural: {summary['procedural']} nodes")
        return summary

    # ══════════════════════════════════════════════════════════════════════════
    # RETRIEVE — Get relevant memory for a query (called each agent turn)
    # ══════════════════════════════════════════════════════════════════════════

    def retrieve(self, query: str, top_k: int = CONTEXT_NODE_LIMIT) -> str:
        """
        Main retrieval method. Called each turn before the agent responds.
        Returns a formatted string ready to inject into a system prompt.

        Pipeline:
          Layer 1: Tree routing → candidate node_ids
          Layer 2: Graph walk   → expand with edge chains
          Layer 3: Leaf ranking → trim to top_k if still too many
        """
        if not self._is_loaded:
            return ""

        # ── Layer 1: Route query through tree ─────────────────────────────────
        if self._extractor:
            routing = self._extractor.route_query(query, [])
            primary_path = routing.get("primary_path", "semantic/people/user")
            secondary_paths = routing.get("secondary_paths", [])
        else:
            primary_path = "semantic/people/user"
            secondary_paths = ["semantic/facts/general", "procedural/general"]

        candidate_ids = self._tree.get_nodes_at_path(primary_path)
        for path in secondary_paths:
            candidate_ids.extend(self._tree.get_nodes_at_path(path))
        candidate_ids = list(set(candidate_ids))

        if not candidate_ids:
            # fallback: grab from all semantic nodes
            candidate_ids = self._tree.get_nodes_for_type("semantic")[:20]

        # ── Layer 2: Graph walk — load nodes + edge chains ────────────────────
        all_nodes: list[MemoryNode] = []
        seen_ids = set()

        for nid in candidate_ids:
            if nid in seen_ids:
                continue
            chain = self._graph.get_edge_chain(nid, depth=1)
            for node in chain:
                if node.id not in seen_ids:
                    all_nodes.append(node)
                    seen_ids.add(node.id)

        # always include procedural nodes (rules apply to all queries)
        procedural_ids = self._tree.get_nodes_for_type("procedural")
        for nid in procedural_ids[:5]:
            if nid not in seen_ids:
                node = self._graph.get_node(nid)
                if node:
                    all_nodes.append(node)
                    seen_ids.add(nid)

        # ── Layer 3: Rank if too many ─────────────────────────────────────────
        if len(all_nodes) > top_k:
            query_emb = self._embedder.encode(query)
            # rank within primary path leaf
            ranked_ids = self._leaves.rank(
                primary_path,
                query_emb,
                [n.id for n in all_nodes],
                top_k=top_k
            )
            id_set = set(ranked_ids)
            all_nodes = [n for n in all_nodes if n.id in id_set]

        # ── Format for injection ──────────────────────────────────────────────
        return self._format_context(all_nodes)

    # ══════════════════════════════════════════════════════════════════════════
    # UPDATE — Ingest new memories (mid-session or end-of-session)
    # ══════════════════════════════════════════════════════════════════════════

    def update(self, conversation: list[dict]) -> list[str]:
        """
        Extract new memories from a conversation and add them to the system.
        Returns list of new node IDs created.

        Call this:
          - At end of session (before eject)
          - Or mid-session every N turns for long conversations
        """
        if not self._extractor:
            print("[agentmem] No LLM function provided — skipping extraction")
            return []

        extractions = self._extractor.extract_from_conversation(conversation)
        new_ids = []

        for ext in extractions:
            node_id = self._ingest_extraction(ext)
            if node_id:
                new_ids.append(node_id)

        print(f"[agentmem] Extracted {len(new_ids)} new memory nodes from conversation")
        return new_ids

    def add_memory_directly(
        self,
        content: str,
        memory_type: str = "semantic",
        tree_path: Optional[str] = None,
        confidence: float = 0.9,
    ) -> str:
        """
        Directly add a memory node without LLM extraction.
        Useful for programmatic memory injection.
        Returns the new node_id.
        """
        if not tree_path:
            tree_path = self._tree.suggest_path(memory_type, content)

        node = MemoryNode(
            content=content,
            memory_type=memory_type,
            tree_path=tree_path,
            confidence=confidence,
            source_session=self._session_id,
        )
        embedding = self._embedder.encode(content)
        self._graph.add_node(node, embedding)
        self._tree.add_node(node.id, tree_path)
        self._leaves.add(tree_path, node.id, embedding)
        self._session_delta.append(node)
        return node.id

    # ══════════════════════════════════════════════════════════════════════════
    # EJECT — Save and return the updated .agentmem file
    # ══════════════════════════════════════════════════════════════════════════

    def eject(self, output_path: str) -> str:
        """
        Compress and save all memory to a .agentmem zip file.
        Returns the path to the saved file.

        This is the 'remove' step — after this, the agent's memory is clean.
        The returned file can be stored, shared, or re-plugged later.
        """
        if not self._is_loaded or not self._work_dir:
            raise RuntimeError("No memory loaded. Call plug() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # save all layers to work dir
        self._tree.save(self._work_dir / "layer_1" / "tree_index.json")
        self._graph.save(
            self._work_dir / "layer_2" / "graph.gpickle",
            self._work_dir / "layer_2" / "embeddings"
        )
        self._leaves.save(self._work_dir / "layer_3")

        # update metadata
        self._meta["last_ejected"] = datetime.utcnow().isoformat()
        self._meta["total_nodes"] = self._graph.node_count()
        self._meta["session_nodes_added"] = len(self._session_delta)
        with open(self._work_dir / "meta.json", "w") as f:
            json.dump(self._meta, f, indent=2)

        # zip work dir → output path
        if output_path.exists():
            output_path.unlink()

        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in self._work_dir.rglob("*"):
                if file.is_file():
                    zf.write(file, file.relative_to(self._work_dir))

        # cleanup work dir
        shutil.rmtree(self._work_dir)
        self._work_dir = None
        self._is_loaded = False
        self._session_delta = []

        print(f"[agentmem] Ejected → {output_path} ({output_path.stat().st_size // 1024}kb)")
        return str(output_path)

    # ══════════════════════════════════════════════════════════════════════════
    # INTERNAL
    # ══════════════════════════════════════════════════════════════════════════

    def _ingest_extraction(self, ext: dict) -> Optional[str]:
        """Process one extracted fact — detect conflicts, create node, update graph."""
        content = ext.get("content", "").strip()
        if not content:
            return None

        memory_type = ext.get("memory_type", "semantic")
        tree_path = ext.get("tree_path") or self._tree.suggest_path(memory_type, content)
        confidence = float(ext.get("confidence", 0.9))
        contradicts_hint = ext.get("contradicts_hint")

        # embed the new fact
        embedding = self._embedder.encode(content)

        # conflict detection: search only within the relevant tree branch
        branch_node_ids = self._tree.get_nodes_at_path(tree_path)
        conflicts = self._graph.find_conflicts(embedding, branch_node_ids, threshold=0.80)

        # create the new node
        node = MemoryNode(
            content=content,
            memory_type=memory_type,
            tree_path=tree_path,
            confidence=confidence,
            source_session=self._session_id,
        )

        # handle conflicts
        for conflicting_id, similarity in conflicts:
            if similarity > 0.92:
                # very high similarity = reinforcement, not conflict
                self._graph.update_confidence(conflicting_id, +0.05)
                node.edges.append(MemoryEdge(
                    to=conflicting_id,
                    type="reinforces",
                    timestamp=datetime.utcnow().isoformat()
                ))
            elif similarity > 0.80:
                # moderate similarity = likely supersedes or contradicts
                if contradicts_hint:
                    node.edges.append(MemoryEdge(
                        to=conflicting_id,
                        type="supersedes",
                        timestamp=datetime.utcnow().isoformat()
                    ))
                    # reduce confidence of old node
                    self._graph.update_confidence(conflicting_id, -0.3)
                    # draw back-edge on old node
                    self._graph.add_edge(conflicting_id, node.id, "superseded_by")
                else:
                    node.edges.append(MemoryEdge(
                        to=conflicting_id,
                        type="contradicts",
                        timestamp=datetime.utcnow().isoformat()
                    ))

        # add to all three layers
        self._graph.add_node(node, embedding)
        self._tree.add_node(node.id, tree_path)
        self._leaves.add(tree_path, node.id, embedding)
        self._session_delta.append(node)

        return node.id

    def _format_context(self, nodes: list[MemoryNode]) -> str:
        """Format nodes for injection into agent system prompt."""
        if not nodes:
            return ""

        # group by type for clean presentation
        semantic = [n for n in nodes if n.memory_type == "semantic"]
        episodic = [n for n in nodes if n.memory_type == "episodic"]
        procedural = [n for n in nodes if n.memory_type == "procedural"]

        lines = ["=== MEMORY ==="]

        if semantic:
            lines.append("\n[KNOWLEDGE]")
            for n in semantic:
                conf_marker = "" if n.confidence > 0.7 else " [uncertain]"
                lines.append(f"• {n.content}{conf_marker}")
                # include superseded info so agent knows history
                for edge in n.edges:
                    if edge.type == "supersedes":
                        old_node = self._graph.get_node(edge.to)
                        if old_node:
                            lines.append(f"  ↳ previously: {old_node.content}")

        if episodic:
            lines.append("\n[PAST EVENTS]")
            for n in sorted(episodic, key=lambda x: x.timestamp, reverse=True)[:5]:
                lines.append(f"• {n.content}")

        if procedural:
            lines.append("\n[BEHAVIORAL RULES]")
            for n in procedural:
                lines.append(f"• {n.content}")

        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    def stats(self) -> dict:
        """Return current memory statistics."""
        tree_summary = self._tree.summary()
        return {
            "total_nodes": self._graph.node_count(),
            "session_id": self._session_id,
            "session_nodes_added": len(self._session_delta),
            **tree_summary,
            "is_loaded": self._is_loaded,
        }
