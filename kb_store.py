"""
kb_store.py — Markdown Knowledge Base

Maintains structured, human-readable markdown files that summarise
the memory graph by domain. These are the "sections" agents inject
into their system prompts instead of raw bullet lists.

Sections (maps to TREE_BRANCHES + mem_types):
  skills.md      — what the user/subject can do (Technical + Procedural)
  processes.md   — workflows, procedures, step-by-step instructions
  people.md      — people, relationships, social context
  facts.md       — stable facts (job, location, identity)
  preferences.md — likes, dislikes, habits, patterns
  events.md      — episodic memories, specific occurrences

Design:
  - The KB is a derived artifact generated from the graph, NOT the primary store.
  - Individual memories live in graph + vector store.
  - The KB provides synthesis and human-readable context.
  - regenerate() calls LLM once per section (cheap because it's batch).
  - get_relevant_sections() is LLM-free (keyword + fuzzy match).
  - Sections are stored in the .umc archive as markdown text.
"""

import re
import time
from typing import Dict, List, Optional, Tuple

from models import MemoryNode, MemType


# ---------------------------------------------------------------------------
# Section definitions
# ---------------------------------------------------------------------------

# Each section: name → {description, branches, mem_types, query_keywords}
SECTIONS: Dict[str, Dict] = {
    "tasks": {
        "title":     "Tasks & Reminders",
        "desc":      "Pending tasks, reminders, scheduled meetings, to-dos.",
        "branches":  ["Tasks"],
        "mem_types": ["task"],
        "keywords":  ["task", "todo", "remind", "schedule", "meeting", "appointment",
                      "deadline", "due", "book", "call", "send", "finish", "complete",
                      "pending", "need to", "have to", "must", "don't forget"],
    },
    "facts": {
        "title":     "Core Facts",
        "desc":      "Stable, objective facts: identity, location, role, background.",
        "branches":  ["People", "General"],
        "mem_types": [MemType.FACT, MemType.DERIVED],
        "keywords":  ["who", "what", "where", "name", "job", "work", "live",
                      "location", "background", "identity", "role", "company"],
    },
    "skills": {
        "title":     "Skills & Expertise",
        "desc":      "Technical skills, domain knowledge, tools the subject uses.",
        "branches":  ["Technical"],
        "mem_types": [MemType.FACT, MemType.PROCEDURAL],
        "keywords":  ["skill", "know", "can", "build", "code", "use", "expert",
                      "tech", "stack", "language", "tool", "framework", "api"],
    },
    "processes": {
        "title":     "Processes & Workflows",
        "desc":      "How things are done: steps, procedures, workflows.",
        "branches":  ["Technical", "Tasks"],
        "mem_types": [MemType.PROCEDURAL],
        "keywords":  ["how", "process", "step", "workflow", "procedure", "deploy",
                      "setup", "configure", "run", "build", "pipeline", "flow"],
    },
    "people": {
        "title":     "People & Relationships",
        "desc":      "People mentioned: colleagues, friends, family, their context.",
        "branches":  ["People"],
        "mem_types": [MemType.FACT, MemType.EPISODE],
        "keywords":  ["person", "people", "team", "colleague", "friend", "family",
                      "meet", "relationship", "manager", "report", "lead", "boss"],
    },
    "preferences": {
        "title":     "Preferences & Habits",
        "desc":      "What the subject likes, dislikes, prefers, their patterns.",
        "branches":  ["Preferences"],
        "mem_types": [MemType.PREFERENCE],
        "keywords":  ["prefer", "like", "hate", "dislike", "habit", "always",
                      "never", "usually", "tend", "enjoy", "favourite", "want"],
    },
    "events": {
        "title":     "Recent Events & Episodes",
        "desc":      "Specific things that happened: meetings, decisions, occurrences.",
        "branches":  ["Events", "General"],
        "mem_types": [MemType.EPISODE],
        "keywords":  ["happened", "event", "met", "went", "did", "yesterday",
                      "last week", "recently", "today", "meeting", "offsite",
                      "started", "finished", "completed"],
    },
}

SECTION_NAMES = list(SECTIONS.keys())


# ---------------------------------------------------------------------------
# KBStore
# ---------------------------------------------------------------------------

class KBStore:
    """
    Manages 6 markdown knowledge base sections.

    The KB is generated from the memory graph via LLM synthesis
    (during consolidate/export), then injected into agent prompts.

    For queries, get_relevant_sections() returns the 1-2 most relevant
    sections using keyword matching — no LLM needed at query time.
    """

    def __init__(self):
        # section_name → markdown string
        self._sections: Dict[str, str] = {name: "" for name in SECTION_NAMES}
        self._generated_at: Dict[str, float] = {}
        self._node_count_at_gen: Dict[str, int] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Generation — builds markdown from a list of MemoryNodes
    # ──────────────────────────────────────────────────────────────────────

    def regenerate_section(
        self,
        section_name: str,
        nodes: List[MemoryNode],
        verbose: bool = False,
    ) -> str:
        """
        Use the LLM to synthesise a list of memory nodes into a clean
        markdown section. Updates the section in-place.

        Only regenerate if the node list has changed meaningfully.
        Returns the generated markdown.
        """
        if not nodes:
            self._sections[section_name] = ""
            return ""

        spec = SECTIONS.get(section_name, {})

        # Build a concise node list for the LLM
        node_lines = "\n".join(
            f"- [{n.mem_type}] {n.content}" for n in nodes if n.is_latest
        )
        if not node_lines.strip():
            self._sections[section_name] = ""
            return ""

        from llm import _chat  # lazy import to avoid circular
        prompt = f"""
You are synthesising memory notes into a clean, structured markdown knowledge base section.

Section: {spec.get('title', section_name)}
Purpose: {spec.get('desc', '')}

Raw memory notes (latest facts):
{node_lines}

Task: Write a structured markdown section that:
1. Uses ## subheadings to organise by sub-topic (e.g. ## Current Role, ## Background)
2. Uses concise bullet points — one fact per line
3. Merges overlapping or redundant points
4. Omits trivial filler
5. Is written in third person ("The user ..." or use the subject's name)
6. Is under 300 words

Start with: # {spec.get('title', section_name)}
"""
        try:
            content = _chat(prompt)
        except Exception as e:
            if verbose:
                print(f"[kb] LLM error for section '{section_name}': {e}")
            # Fallback: plain bullet list
            content = f"# {spec.get('title', section_name)}\n\n" + node_lines

        self._sections[section_name] = content.strip()
        self._generated_at[section_name] = time.time()
        self._node_count_at_gen[section_name] = len(nodes)

        if verbose:
            print(f"[kb] regenerated '{section_name}' ({len(nodes)} nodes → "
                  f"{len(content)} chars)")

        return content.strip()

    def regenerate_all(
        self,
        all_nodes: List[MemoryNode],
        verbose: bool = False,
    ) -> Dict[str, str]:
        """
        Regenerate all sections from a flat list of nodes.
        Routes each node to the right section(s) by branch + mem_type.
        """
        # Route nodes to sections
        buckets: Dict[str, List[MemoryNode]] = {name: [] for name in SECTION_NAMES}
        for node in all_nodes:
            if not node.is_latest:
                continue
            placed = False
            for section_name, spec in SECTIONS.items():
                branch_match  = node.branch() in spec["branches"]
                type_match    = node.mem_type in spec["mem_types"]
                if branch_match and type_match:
                    buckets[section_name].append(node)
                    placed = True
            if not placed:
                # Default: put facts in facts, everything else in events
                if node.mem_type in (MemType.FACT, MemType.DERIVED):
                    buckets["facts"].append(node)
                else:
                    buckets["events"].append(node)

        results = {}
        for section_name, nodes in buckets.items():
            if nodes:
                results[section_name] = self.regenerate_section(
                    section_name, nodes, verbose=verbose
                )
        return results

    # ──────────────────────────────────────────────────────────────────────
    # Update — append new facts to an existing section (fast, no LLM)
    # ──────────────────────────────────────────────────────────────────────

    def quick_append(self, section_name: str, node: MemoryNode):
        """
        Append a single memory to a section without regenerating it.
        Used during ingestion to keep sections in sync cheaply.
        Full regeneration happens on consolidate()/export().
        """
        if section_name not in self._sections:
            return

        line = f"- {node.content}\n"
        current = self._sections[section_name]

        if not current:
            spec    = SECTIONS.get(section_name, {})
            current = f"# {spec.get('title', section_name)}\n\n"

        self._sections[section_name] = current.rstrip() + "\n" + line

    def route_node_to_section(self, node: MemoryNode) -> str:
        """
        Determine which KB section a node belongs to.
        Returns section name.
        """
        for section_name, spec in SECTIONS.items():
            if node.branch() in spec["branches"] and node.mem_type in spec["mem_types"]:
                return section_name

        # Default routing
        type_defaults = {
            MemType.FACT:       "facts",
            MemType.PREFERENCE: "preferences",
            MemType.EPISODE:    "events",
            MemType.PROCEDURAL: "processes",
            MemType.DERIVED:    "facts",
            "task":             "tasks",
        }
        return type_defaults.get(node.mem_type, "facts")

    # ──────────────────────────────────────────────────────────────────────
    # Query — no LLM, keyword matching
    # ──────────────────────────────────────────────────────────────────────

    def get_relevant_sections(
        self,
        query: str,
        max_sections: int = 2,
    ) -> List[Tuple[str, str]]:
        """
        Return the most relevant populated sections for a query.
        Uses keyword overlap — no LLM call.

        Returns: [(section_name, markdown_content), ...]
        """
        query_lower  = query.lower()
        query_tokens = set(re.findall(r'\b\w+\b', query_lower))

        scored: List[Tuple[float, str]] = []
        for section_name, spec in SECTIONS.items():
            content = self._sections.get(section_name, "")
            if not content.strip():
                continue

            keywords = set(spec.get("keywords", []))
            overlap  = len(query_tokens & keywords)
            # Also check query words against section content
            content_hits = sum(
                1 for tok in query_tokens if len(tok) > 3 and tok in content.lower()
            )
            score = overlap * 2 + content_hits
            if score > 0:
                scored.append((score, section_name))

        scored.sort(reverse=True)
        result = []
        for _, name in scored[:max_sections]:
            content = self._sections.get(name, "")
            if content.strip():
                result.append((name, content))
        return result

    def get_section(self, section_name: str) -> str:
        """Get the markdown content of a named section."""
        return self._sections.get(section_name, "")

    def get_all_populated(self) -> Dict[str, str]:
        """Return all non-empty sections."""
        return {k: v for k, v in self._sections.items() if v.strip()}

    def set_section(self, section_name: str, content: str):
        """Manually set a section's content (e.g. when loading from archive)."""
        self._sections[section_name] = content

    def needs_regeneration(self, section_name: str, current_node_count: int) -> bool:
        """
        Heuristic: regenerate if the node count has changed by >20%
        or if it's never been generated.
        """
        prev = self._node_count_at_gen.get(section_name, 0)
        if prev == 0:
            return True
        return abs(current_node_count - prev) / max(prev, 1) > 0.20

    # ──────────────────────────────────────────────────────────────────────
    # Prompt injection
    # ──────────────────────────────────────────────────────────────────────

    def build_context_block(
        self,
        relevant_sections: List[Tuple[str, str]],
        individual_memories: List[str],
    ) -> str:
        """
        Assemble the full memory context block for injection into a system prompt.

        Structure:
          [KB CONTEXT — structured knowledge]
          (markdown sections)

          [RELEVANT MEMORIES — specific facts]
          (bullet list)
        """
        parts = []

        if relevant_sections:
            parts.append("### Knowledge Base")
            for section_name, content in relevant_sections:
                # Include first 600 chars of each section to stay token-efficient
                trimmed = content[:600].strip()
                if len(content) > 600:
                    trimmed += "\n*(... truncated)*"
                parts.append(trimmed)

        if individual_memories:
            parts.append("\n### Specific Memories")
            for m in individual_memories:
                parts.append(f"- {m}")

        if not parts:
            return ""

        return "\n\n".join(parts)

    # ──────────────────────────────────────────────────────────────────────
    # Serialization
    # ──────────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "sections":              self._sections,
            "generated_at":          self._generated_at,
            "node_count_at_gen":     self._node_count_at_gen,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "KBStore":
        kb = cls()
        kb._sections              = d.get("sections", {name: "" for name in SECTION_NAMES})
        kb._generated_at          = d.get("generated_at", {})
        kb._node_count_at_gen     = d.get("node_count_at_gen", {})
        return kb

    def __repr__(self) -> str:
        populated = sum(1 for v in self._sections.values() if v.strip())
        return f"<KBStore sections={populated}/{len(SECTION_NAMES)} populated>"
