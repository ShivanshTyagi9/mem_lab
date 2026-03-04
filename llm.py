"""
llm.py — LLM pipeline for uMemory

All model interactions live here. The prompts are provider-agnostic.
Swap backends by changing one line — no prompt edits needed.

Supported backends:
  OpenAIBackend   — any OpenAI-compatible endpoint (gpt-4o-mini default)
  AnthropicBackend — Claude via anthropic SDK (claude-haiku default)

Usage:
  # Auto-detect from environment
  backend = auto_backend()

  # Explicit
  from llm import AnthropicBackend, set_backend
  set_backend(AnthropicBackend())
"""

import re
import json
import os
from abc import ABC, abstractmethod
from typing import List, Optional, Dict
from datetime import datetime
try:
    from dateutil import parser as _dateparser
    _HAS_DATEUTIL = True
except Exception:
    _HAS_DATEUTIL = False

from models import MemoryNode, TREE_BRANCHES


# ---------------------------------------------------------------------------
# LLM Backend abstraction
# ---------------------------------------------------------------------------

_ROUTER_PROMPT = """Classify the user's message into ONE category from:
task, procedural, fact, episode, preference, ignore

Respond ONLY with the single category name (lowercase).
Do not add any other text.
Message:
"""

class LLMBackend(ABC):
    """Base class for all LLM providers. One method to implement: complete()."""

    @abstractmethod
    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        """Send a prompt, return the text response."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"


class OpenAIBackend(LLMBackend):
    """
    OpenAI (or any OpenAI-compatible) backend.
    Works with OpenAI, Together, Groq, Ollama, etc.

    Env vars:
      OPENAI_API_KEY   (required for OpenAI)
      OPENAI_BASE_URL  (optional — override endpoint)
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        from openai import OpenAI
        self.model = model
        self._client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        )

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def chat(self, messages: list, temperature: float = 0.7) -> str:
        """Multi-turn chat for agent conversations."""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def __repr__(self) -> str:
        return f"<OpenAIBackend model={self.model}>"


class AnthropicBackend(LLMBackend):
    """
    Anthropic Claude backend.

    Env vars:
      ANTHROPIC_API_KEY  (required)

    Default model: claude-haiku-4-5-20251001  (fast + cheap, good for extraction)
    For higher quality: claude-sonnet-4-6
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
    ):
        import anthropic
        self.model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY")
        )

    def complete(self, prompt: str, temperature: float = 0.0) -> str:
        msg = self._client.messages.create(
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return msg.content[0].text.strip()

    def chat(self, messages: list, system: str = "", temperature: float = 0.7) -> str:
        """
        Multi-turn chat for agent conversations.
        messages: [{"role": "user"|"assistant", "content": str}, ...]
        The system prompt is passed separately (Claude API design).
        """
        kwargs = dict(
            model=self.model,
            max_tokens=2048,
            messages=messages,
            temperature=temperature,
        )
        if system:
            kwargs["system"] = system
        msg = self._client.messages.create(**kwargs)
        return msg.content[0].text.strip()

    def __repr__(self) -> str:
        return f"<AnthropicBackend model={self.model}>"


# ---------------------------------------------------------------------------
# Global backend — set once, used by all pipeline functions
# ---------------------------------------------------------------------------

_backend: Optional[LLMBackend] = None


def auto_backend() -> LLMBackend:
    """
    Detect a backend from environment variables.
    Prefers Anthropic if ANTHROPIC_API_KEY is set, falls back to OpenAI.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return AnthropicBackend()
    if os.getenv("OPENAI_API_KEY"):
        return OpenAIBackend()
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."
    )


def get_backend() -> LLMBackend:
    global _backend
    if _backend is None:
        _backend = auto_backend()
    return _backend


def set_backend(backend: LLMBackend):
    """Override the global backend. Call this before any pipeline functions."""
    global _backend
    _backend = backend


def _chat(prompt: str, temperature: float = 0.0) -> str:
    """Internal: send a prompt through the active backend."""
    return get_backend().complete(prompt, temperature=temperature)


# ---- pipeline-specific backends (new) ----
_pipeline_backends = {
    # default names; you can change these later with set_pipeline_backend()
    "chat": None,        # e.g. gpt-4o-mini
    "extract": None,     # e.g. gpt-4.1-nano
    "graph": None,       # e.g. gpt-4.1-nano
    "summarize": None,   # e.g. gpt-4o-mini
}

def set_pipeline_backend(stage: str, backend: LLMBackend):
    """Call this at startup to assign a backend for a particular pipeline stage:
       stage ∈ { 'chat', 'extract', 'graph', 'summarize' }"""
    _pipeline_backends[stage] = backend

def _get_backend_for(stage: str) -> LLMBackend:
    """Return specific backend if configured, else fall back to global backend."""
    b = _pipeline_backends.get(stage)
    if b is not None:
        return b
    return get_backend()


def _chat_with(stage: str, prompt: str, temperature: float = 0.0) -> str:
    """Use a backend chosen for a pipeline stage."""
    backend = _get_backend_for(stage)
    return backend.complete(prompt, temperature=temperature)

# ---------------------------------------------------------------------------
# JSON parsing helpers
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> Optional[list]:
    """Pull the first JSON array from a string, even if wrapped in prose."""
    m = re.search(r'\[.*\]', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> Optional[dict]:
    """Pull the first JSON object from a string."""
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if not m:
        # Try a broader match
        m = re.search(r'\{.*\}', text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None

def route_message(text: str, backend: Optional[LLMBackend] = None) -> str:
    """
    Lightweight LLM router: returns one of
    'task' | 'procedural' | 'fact' | 'episode' | 'preference' | 'ignore'
    Uses the passed backend if provided, else the global backend.
    """
    b = backend or get_backend()
    resp = b.complete(_ROUTER_PROMPT + "" + text, temperature=0.0)
    return resp.strip().lower().splitlines()[0].strip()

# ---------------------------------------------------------------------------
# 1. Memory extraction
#    Also acts as the noise filter — empty list = nothing worth storing.
# ---------------------------------------------------------------------------

def extract_memories(text: str, hint: Optional[str] = None) -> List[dict]:
    """
    Extract atomic memory units from text, including tasks and action items.

    Returns list of:
      {
        "content":     str,
        "mem_type":    "fact|preference|episode|procedural|task",
        "confidence":  float 0-1,
        "is_temporal": bool,
        # task-only fields (only present when mem_type == "task"):
        "task_title":  str,    # short title e.g. "Meeting Red"
        "due_date":    str,    # "YYYY-MM-DD HH:MM" or "" if not specified
      }
    """

    preserve_note = (
        "PRESERVE EXACT MEANING AND LANGUAGE — THIS IS THE MOST IMPORTANT RULE:\n"
        "- Never sanitize, soften, euphemize, or generalize any statement.\n"
        "- If the user uses explicit, sexual, crude, or blunt language, keep it exactly.\n"
        "- Resolve pronouns but preserve content words exactly as expressed.\n\n"
    )

    # If a hint was provided by the router, include it as a short instruction for categorization.
    hint_line = f"Routing hint: {hint}\n\n" if hint else ""

    raw = _chat(f"""
You are a memory extraction system for a personal AI assistant.
Your only job is to faithfully extract and preserve what the user said.
You are a recorder, not an editor. Never judge, filter, or reword content.
{preserve_note}
{hint_line}
Rules:
1. Each memory must be a COMPLETE, SELF-CONTAINED SENTENCE.
   Resolve pronouns — "my name is X" → "The user's name is X".
   "my girlfriend Shreya" → "The user's girlfriend is named Shreya".

2. Do NOT decompose into triplets. The full sentence is the unit.

3. Memory types:
   fact        — names, relationships, jobs, locations, anything stable
   preference  — likes, dislikes, habits, desires, attractions, turn-ons
   episode     — a specific past event or thing that happened
   procedural  — how to do something, step-by-step workflows
   task        — an action item, reminder, meeting, to-do, or scheduled event
                 that needs to happen in the future

4. TASK detection — classify as "task" if the message mentions:
   - scheduling or booking something ("schedule a meeting", "book a call")
   - a future appointment or event ("meeting on Monday", "call at 3pm")
   - something the user needs to do ("remind me to...", "I need to send...")
   - a reminder ("don't forget to...", "remember to...")
   For tasks, also extract:
     task_title: a short 2-5 word label (e.g. "Meeting Red", "Send report", "Call dentist")
     due_date:   the date/time if mentioned, as "YYYY-MM-DD HH:MM" (use current year if unspecified)
                 Leave "" if no date was given.

5. is_temporal=true only for time-bound non-task facts: "exam tomorrow"

6. Skip ONLY pure filler: "ok", "thanks", "sounds good", "lol", "how are you"

7. PRESERVE EXACT MEANING AND LANGUAGE — THIS IS THE MOST IMPORTANT RULE:
   - Never sanitize, soften, euphemize, or generalize any statement.
   - If the user uses explicit, sexual, crude, or blunt language, keep it.
   - The stored memory must reflect exactly what the user expressed.

Return ONLY a JSON array. Return [] for pure filler:
[
  {{
    "content": "...",
    "mem_type": "fact|preference|episode|procedural|task",
    "confidence": 0.0-1.0,
    "is_temporal": false,
    "task_title": "",
    "due_date": ""
  }}
]

Text:
{text}
""")
    return _extract_json_array(raw) or []


# ---------------------------------------------------------------------------
# 2. Tree classification
# ---------------------------------------------------------------------------

def classify_into_tree(content: str) -> List[str]:
    """
    Map a memory's content to a tree path for branch-based routing.

    Returns a path like ["People"] or ["People", "Alex"] or ["Technical", "Python"].
    """
    raw = _chat(f"""
Classify this memory into the most specific tree path possible.

Top-level branches: {TREE_BRANCHES}

If the content is about a specific named person, go one level deeper:
  ["People", "Alex"] not just ["People"]

If it's about a specific technology or project:
  ["Technical", "Python"] or ["Technical", "API Design"]

Return ONLY a JSON array of strings. Maximum 2 levels deep.
Examples:
  ["People", "Alex"]
  ["Tasks"]
  ["Technical", "Payments"]
  ["Preferences"]
  ["Events"]
  ["General"]

Content: "{content}"
""")
    result = _extract_json_array(raw)
    if not result or not isinstance(result, list) or not result[0]:
        return ["General"]
    if result[0] not in TREE_BRANCHES:
        return ["General"]
    return result


# ---------------------------------------------------------------------------
# 3. Relationship detection
# ---------------------------------------------------------------------------

def detect_relationship(
    new_content: str,
    candidates: List[MemoryNode],
    max_candidates: int = 20,
) -> Optional[dict]:
    """
    Determine how a new memory relates to existing memories.

    Returns: {
        "relationship": "UPDATES|EXTENDS|DERIVES|NONE",
        "target_id":    "existing node id or null",
        "reason":       "brief explanation of what changed / why"
    }

    The reason is stored on the new node as update_reason for timeline display.
    When candidates have prior versions, those are shown to the LLM so it
    understands "this is already v2 of a fact about X's job" and can detect
    a v3 update correctly.
    """
    if not candidates:
        return {"relationship": "NONE", "target_id": None, "reason": ""}

    latest = [n for n in candidates if n.is_latest][:max_candidates]
    if not latest:
        return {"relationship": "NONE", "target_id": None, "reason": ""}

    # Build candidate text, including prior version count for context
    cand_lines = []
    for n in latest:
        version_hint = ""
        if n.chain_position > 1:
            version_hint = f" [already v{n.chain_position} of this fact]"
        cand_lines.append(
            f'  ID="{n.id}"{version_hint}\n  Content: "{n.content}"'
        )
    cand_text = "\n".join(cand_lines)

    raw = _chat(f"""
A new memory has arrived. Determine how it relates to the existing memories.

New memory:
"{new_content}"

Existing memories (latest version of each fact):
{cand_text}

Relationship definitions:
  UPDATES  — the new memory CONTRADICTS or SUPERSEDES an existing one.
             The old fact is no longer true. The new fact replaces it.
             Examples:
               "Alex works at OpenAI" when existing says "Alex works at Stripe"
               "User broke up with Shreya" when existing says "User is dating Shreya"
               "User moved to Mumbai" when existing says "User lives in Delhi"

  EXTENDS  — the new memory ADDS DETAIL to an existing one. Both remain true.
             Example: "Alex leads a team of 5" alongside "Alex is PM at Stripe"

  DERIVES  — the new memory is a LOGICAL INFERENCE from existing memories.

  NONE     — no meaningful relationship. Genuinely new, independent fact.

Instructions:
- If UPDATES: pick the SINGLE existing memory being replaced.
  Also write a brief reason explaining what specifically changed
  (e.g. "changed employer from Stripe to OpenAI").
- If EXTENDS or DERIVES: pick the best matching memory.
- If NONE: reason can be empty.
- Do not manufacture relationships that aren't clearly there.

Return ONLY valid JSON, no prose:
{{
  "relationship": "UPDATES|EXTENDS|DERIVES|NONE",
  "target_id": "<existing id or null>",
  "reason": "<brief explanation of what changed, or empty string>"
}}
""")

    result = _extract_json_object(raw)
    if not result:
        return {"relationship": "NONE", "target_id": None, "reason": ""}

    # Validate target_id exists
    target_id = result.get("target_id")
    valid_ids  = {n.id for n in latest}
    if target_id and target_id not in valid_ids:
        result["target_id"]    = None
        result["relationship"] = "NONE"

    result.setdefault("reason", "")
    return result


# ---------------------------------------------------------------------------
# 4. Consolidation
# ---------------------------------------------------------------------------

def consolidate_episodes(episodes: List[MemoryNode], branch: str) -> Optional[str]:
    """
    Synthesise a list of episodic memories into a single stable derived fact.
    Returns the summary string, or None if consolidation isn't meaningful.
    """
    if not episodes:
        return None

    ep_text = "\n".join(f"- {e.content}" for e in episodes)

    raw = _chat(f"""
You are synthesising episodic memories into a stable semantic summary.

These are specific events that happened (episodes):
{ep_text}

Task: Write a single, concise factual statement that captures the stable pattern
or insight behind these episodes. This will become a "derived fact" memory.

Rules:
- Write one sentence only.
- It should be a stable truth, not a list of events.
- If these episodes don't form a coherent pattern worth summarising, reply: SKIP

Output only the sentence (or SKIP):
""")

    summary = raw.strip()
    if not summary or summary.upper() == "SKIP":
        return None
    return summary


# ---------------------------------------------------------------------------
# 5. Query reformulation
# ---------------------------------------------------------------------------

def reformulate_query(query: str) -> dict:
    """
    Given a user query, determine which tree branch to search and the core need.

    Returns: {"tree_path": [...], "core_query": "..."}
    """
    raw = _chat(f"""
Analyse this query for a memory retrieval system.

Query: "{query}"

Return ONLY JSON:
{{
  "tree_path": ["branch"],
  "core_query": "simplified essence of what's being asked"
}}

Available branches: {TREE_BRANCHES}
If about a specific named person: ["People", "Name"]
If about a specific technology: ["Technical", "Name"]
""")

    result = _extract_json_object(raw)
    if not result:
        return {"tree_path": ["General"], "core_query": query}

    path = result.get("tree_path", ["General"])
    if not path or path[0] not in TREE_BRANCHES:
        path = ["General"]

    return {"tree_path": path, "core_query": result.get("core_query", query)}
