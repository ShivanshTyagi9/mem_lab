"""
retrieval.py — Hybrid retrieval pipeline

Replaces the LLM-based branch routing entirely.

Old pipeline:
  query(text) → LLM(reformulate_query) → filter by branch → sort by decay → return

Problems:
  - LLM call on every query (slow, expensive)
  - Ranking by recency, not relevance (wrong answer at rank 1)
  - Wrong branch → answer lost entirely
  - EXTENDS edges completely ignored

New pipeline:
  query(text) → embed(text) → vector_search → graph_expand → hybrid_score → return

  No LLM during retrieval. Vector search is pure inference (fast).
  Relevance = 0.6 × semantic_similarity + 0.4 × memory_strength
  Graph expansion pulls EXTENDS neighbours for richer context.
  KB sections injected as structured markdown, not raw bullets.

The whole pipeline is ~2-10ms on CPU with SentenceTransformers,
vs ~300-1500ms per LLM call in the old system.
"""

from typing import List, Dict, Optional, Tuple, Any

from models import MemoryNode, MemoryGraph, Rel
from vector_store import VectorStore
from kb_store import KBStore


# ---------------------------------------------------------------------------
# Score weights (tunable)
# ---------------------------------------------------------------------------

VECTOR_WEIGHT  = 0.60   # semantic similarity from embedding
DECAY_WEIGHT   = 0.40   # memory strength (recency + salience + access count)
MIN_STRENGTH   = 0.03   # nodes below this are treated as decayed — skip them
DEDUP_THRESHOLD = 0.92  # cosine similarity above this = duplicate on ingest


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve(
    query: str,
    graph: MemoryGraph,
    vector_store: VectorStore,
    kb_store: KBStore,
    top_k: int = 5,
    vector_candidates: int = 15,
    expand_graph: bool = True,
    include_kb: bool = True,
    min_score: float = 0.0,
) -> Dict[str, Any]:
    """
    Full hybrid retrieval. Returns a rich result dict.

    Args:
      query             — user's query text
      graph             — MemoryGraph (for strength + graph expansion)
      vector_store      — VectorStore (for semantic search)
      kb_store          — KBStore (for structured section injection)
      top_k             — number of individual memories to return
      vector_candidates — how many vector results to fetch before re-ranking
      expand_graph      — whether to follow EXTENDS edges for extra context
      include_kb        — whether to include KB sections in context
      min_score         — minimum hybrid score to include in results

    Returns:
      {
        "memories":    [str],              # top_k content strings
        "nodes":       [MemoryNode],       # the actual nodes (for touch())
        "scores":      [(node_id, score)], # debug: per-node scores
        "kb_sections": [(name, markdown)], # relevant KB sections
        "context_block": str,             # ready-to-inject prompt block
      }
    """
    if not graph.nodes:
        return _empty_result()

    # ── Step 1: Vector similarity search ─────────────────────────────────
    # No LLM call. Pure embedding inference + cosine similarity.
    vec_results = vector_store.query(query, top_k=vector_candidates)
    # vec_results = [(node_id, cosine_sim), ...]

    # ── Step 2: Collect valid nodes ───────────────────────────────────────
    # Filter to is_latest=True and above min strength threshold.
    seen_ids: set = set()
    candidates: List[Tuple[MemoryNode, float]] = []  # (node, vector_score)

    for node_id, vec_score in vec_results:
        if node_id in seen_ids:
            continue
        node = graph.nodes.get(node_id)
        if node is None:
            continue
        if not node.is_latest:
            continue
        if node.strength() < MIN_STRENGTH:
            continue
        candidates.append((node, vec_score))
        seen_ids.add(node_id)

    # ── Step 3: Graph expansion via EXTENDS edges ─────────────────────────
    # For each top-5 candidate, pull nodes connected by EXTENDS.
    # These are contextually related memories that might not be in the
    # vector top-k but are semantically linked by the graph.
    if expand_graph and candidates:
        top_for_expansion = candidates[:5]
        expansion_ids: List[str] = []
        for node, _ in top_for_expansion:
            for rel, tgt_id in graph.edges.get(node.id, []):
                if rel in (Rel.EXTENDS, Rel.DERIVES) and tgt_id not in seen_ids:
                    expansion_ids.append(tgt_id)

        for node_id in expansion_ids:
            node = graph.nodes.get(node_id)
            if node and node.is_latest and node.strength() >= MIN_STRENGTH:
                # Assign a modest vector score (we don't have the real one)
                # Use a slight discount to rank below direct hits
                candidates.append((node, 0.55))
                seen_ids.add(node_id)

    if not candidates:
        # Vector store has no matches — fall back to highest-strength nodes
        all_latest = [n for n in graph.nodes.values() if n.is_latest]
        top_by_strength = sorted(all_latest, key=lambda n: n.strength(), reverse=True)
        candidates = [(n, 0.5) for n in top_by_strength[:vector_candidates]]

    # ── Step 4: Hybrid scoring ────────────────────────────────────────────
    # score = vector_weight × similarity + decay_weight × strength
    # Both components are in [0, 1].
    scored: List[Tuple[MemoryNode, float]] = []
    for node, vec_score in candidates:
        hybrid = (VECTOR_WEIGHT * vec_score) + (DECAY_WEIGHT * node.strength())
        if hybrid >= min_score:
            scored.append((node, hybrid))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_nodes = [node for node, _ in scored[:top_k]]
    top_scores = [(node.id, round(score, 4)) for node, score in scored[:top_k]]

    # ── Step 5: Touch top nodes (spacing effect) ──────────────────────────
    for node in top_nodes:
        node.touch()

    # ── Step 6: KB section injection ─────────────────────────────────────
    kb_sections = []
    if include_kb:
        kb_sections = kb_store.get_relevant_sections(query, max_sections=2)

    # ── Step 7: Build context block ───────────────────────────────────────
    memories = [n.content for n in top_nodes]
    context_block = kb_store.build_context_block(kb_sections, memories)

    return {
        "memories":     memories,
        "nodes":        top_nodes,
        "scores":       top_scores,
        "kb_sections":  kb_sections,
        "context_block": context_block,
    }


def _empty_result() -> Dict[str, Any]:
    return {
        "memories":      [],
        "nodes":         [],
        "scores":        [],
        "kb_sections":   [],
        "context_block": "",
    }


# ---------------------------------------------------------------------------
# Ingestion-time deduplication check
# ---------------------------------------------------------------------------

def check_duplicate(
    content: str,
    vector_store: VectorStore,
    graph: MemoryGraph,
    threshold: float = DEDUP_THRESHOLD,
) -> Optional[MemoryNode]:
    """
    Before creating a new node, check if semantically identical content
    already exists. Returns the existing MemoryNode if duplicate, else None.

    Uses vector similarity — much better than exact string match.

    When a duplicate is found, the caller should:
      - Boost the existing node's access count (reinforcement)
      - Skip creating the new node
    """
    dup_id = vector_store.find_duplicate(content, threshold=threshold)
    if dup_id and dup_id in graph.nodes:
        node = graph.nodes[dup_id]
        if node.is_latest:
            return node
    return None


# ---------------------------------------------------------------------------
# Relationship detection scope limiter
# ---------------------------------------------------------------------------

def get_relationship_candidates(
    content: str,
    vector_store: VectorStore,
    graph: MemoryGraph,
    top_k: int = 8,
) -> List[MemoryNode]:
    """
    Get the most relevant existing nodes to compare against for relationship
    detection (UPDATES / EXTENDS / DERIVES).

    OLD approach: send ALL nodes in the branch to the LLM.
    NEW approach: vector search → top-k most similar nodes only.

    This reduces the LLM context for detect_relationship() from
    potentially 50+ nodes down to 8, without losing quality.
    Because: if the new memory doesn't have vector similarity to an old one,
    it almost certainly doesn't update or extend it either.
    """
    vec_results = vector_store.query(content, top_k=top_k)

    candidates: List[MemoryNode] = []
    for node_id, score in vec_results:
        if score < 0.40:
            # Very low similarity — not a plausible relationship
            continue
        node = graph.nodes.get(node_id)
        if node and node.is_latest:
            candidates.append(node)

    return candidates


# ---------------------------------------------------------------------------
# History retrieval (for query_with_history)
# ---------------------------------------------------------------------------

def retrieve_with_history(
    query: str,
    graph: MemoryGraph,
    vector_store: VectorStore,
    kb_store: KBStore,
    top_k: int = 5,
) -> Dict[str, Any]:
    """
    Retrieve current memories AND their temporal history (superseded nodes).

    Returns the standard retrieve() result plus:
      "history": [str]  — content of superseded nodes related to the query
    """
    result = retrieve(
        query, graph, vector_store, kb_store, top_k=top_k
    )

    # For each returned node, walk its temporal chain
    history_contents = []
    seen_history: set = set()

    for node in result["nodes"]:
        chain = graph.temporal_chain(node.id)
        for hist_node in chain[1:]:   # exclude the node itself (index 0)
            if hist_node.id not in seen_history and not hist_node.is_latest:
                history_contents.append(hist_node.content)
                seen_history.add(hist_node.id)

    result["history"] = history_contents[:top_k]
    return result
