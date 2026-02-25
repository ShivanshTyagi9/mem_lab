"""
extractor.py — LLM-powered memory extraction and compression.

This is the intelligence layer of the system.
The extraction prompt is the single most important piece — it determines
what gets stored, how it's typed, and whether conflicts are flagged.
"""

import json
import re
from typing import Optional


EXTRACTION_PROMPT = """You are a memory extraction system for an AI agent. Analyze the conversation below and extract memorable facts.

For each fact, output:
- content: the raw fact as a single clear, self-contained sentence
- memory_type: one of "semantic" (knowledge about the user/world), "episodic" (events that occurred), "procedural" (behavioral rules/patterns)
- tree_path: where it belongs in the index. Use format:
    semantic/people/user — facts about the user themselves
    semantic/people/<name> — facts about specific named people
    semantic/concepts/<topic> — conceptual knowledge, tools, domains
    semantic/facts/general — other factual knowledge
    episodic/<year>-<month-abbr>/week-<n> — things that happened (e.g. episodic/2026-feb/week-3)
    procedural/code_requests — coding preferences and patterns
    procedural/clarification_triggers — when agent should ask questions
    procedural/general — other behavioral rules
- confidence: 0.0 to 1.0 (how certain are you this is a durable fact?)
- contradicts_hint: a short phrase describing what this might contradict in existing memory, or null if nothing

Rules:
- Do NOT decompose facts into triplets. Keep each as natural language.
- Only extract durable facts, not conversational filler.
- Max 8 facts per session. Prefer quality over quantity.
- Skip facts about the AI itself.
- Procedural memories must be actionable rules: "When X, do Y"

Conversation:
{conversation}

Respond ONLY with a valid JSON array. No explanation, no markdown.
Example:
[
  {{
    "content": "The user prefers Python over JavaScript for backend work",
    "memory_type": "semantic",
    "tree_path": "semantic/people/user",
    "confidence": 0.9,
    "contradicts_hint": null
  }}
]"""


COMPRESSION_PROMPT = """You are a memory compression system. The following episodic memories are old enough to be compressed into semantic facts.

Episodes to compress:
{episodes}

Extract 2-4 durable semantic facts that capture the lasting knowledge from these episodes.
Format: same JSON array as extraction output.
Discard narrative details. Keep only what would be useful to know months from now.

Respond ONLY with a valid JSON array."""


ROUTER_PROMPT = """Given this user message, which memory types are most relevant to retrieve?

Message: {message}

Respond with a JSON object:
{{
  "primary_path": "the most relevant tree path to search first",
  "secondary_paths": ["other relevant paths, max 2"],
  "reasoning": "one sentence"
}}

Tree paths available:
- semantic/people/user
- semantic/concepts/<topic>  
- semantic/facts/general
- episodic/<recent periods>
- procedural/code_requests
- procedural/clarification_triggers
- procedural/general"""


def parse_json_response(text: str) -> Optional[list | dict]:
    """Safely parse LLM JSON output, handling markdown fences."""
    # strip markdown fences if present
    text = re.sub(r"```json\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # try to find JSON array or object in the text
        array_match = re.search(r"\[.*\]", text, re.DOTALL)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
        obj_match = re.search(r"\{.*\}", text, re.DOTALL)
        if obj_match:
            try:
                return json.loads(obj_match.group())
            except json.JSONDecodeError:
                pass
        return None


class MemoryExtractor:
    """
    Uses an LLM to extract, route, and compress memories.
    Agnostic to LLM provider — takes a callable that accepts a prompt and returns text.
    """

    def __init__(self, llm_fn):
        """
        llm_fn: callable(prompt: str) -> str
        Can wrap any LLM — Claude, GPT, local model, etc.
        """
        self._llm = llm_fn

    def extract_from_conversation(self, conversation: list[dict]) -> list[dict]:
        """
        Extract memory candidates from a conversation history.
        Returns list of raw extraction dicts (not yet MemoryNodes).
        """
        conv_text = self._format_conversation(conversation)
        prompt = EXTRACTION_PROMPT.format(conversation=conv_text)
        response = self._llm(prompt)
        parsed = parse_json_response(response)
        if not isinstance(parsed, list):
            return []
        # validate structure
        valid = []
        for item in parsed:
            if isinstance(item, dict) and "content" in item and "memory_type" in item:
                valid.append(item)
        return valid

    def route_query(self, message: str, existing_paths: list[str]) -> dict:
        """
        Determine which tree paths to search for a given query.
        Returns routing decision dict.
        """
        prompt = ROUTER_PROMPT.format(message=message)
        response = self._llm([
            {"role": "system", "content": "You are a routing classifier."},
            {"role": "user", "content": prompt}
        ])
        parsed = parse_json_response(response)
        if isinstance(parsed, dict) and "primary_path" in parsed:
            return parsed
        # fallback: search semantic/people/user + recent episodic
        return {
            "primary_path": "semantic/people/user",
            "secondary_paths": ["semantic/facts/general"],
            "reasoning": "fallback routing"
        }

    def compress_episodes(self, episode_nodes: list[dict]) -> list[dict]:
        """
        Compress old episodic nodes into semantic facts.
        Called during periodic compression passes.
        """
        if not episode_nodes:
            return []
        episodes_text = "\n".join([f"- {n['content']}" for n in episode_nodes])
        prompt = COMPRESSION_PROMPT.format(episodes=episodes_text)
        response = self._llm(prompt)
        parsed = parse_json_response(response)
        if not isinstance(parsed, list):
            return []
        return parsed

    def _format_conversation(self, conversation: list[dict]) -> str:
        lines = []
        for msg in conversation:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
