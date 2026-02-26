import uuid
import json
import time
import math
import os
from dotenv import load_dotenv
import pickle
import zipfile
from collections import defaultdict
from typing import List, Dict, Optional
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key= os.getenv("OPENAI_API_KEY"))


# =========================
# Node
# =========================

class MemoryNode:
    def __init__(
        self,
        content: str,
        node_type: str,
        tree_path: List[str],
        confidence: float = 1.0,
        salience: float = 0.8,
        decay_rate: float = 0.01,
        embedding: Optional[List[float]] = None
    ):
        self.id = str(uuid.uuid4())
        self.content = content
        self.type = node_type  # fact | preference | episode | procedural | derived
        self.tree_path = tree_path
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.isLatest = True
        self.confidence = confidence
        self.salience = salience
        self.decay_rate = decay_rate
        self.embedding = embedding

    def strength(self):
        delta = time.time() - self.last_accessed
        return self.salience * math.exp(-self.decay_rate * delta)

    def to_dict(self):
        return self.__dict__

    @staticmethod
    def from_dict(data):
        node = MemoryNode(
            data["content"],
            data["type"],
            data["tree_path"],
            data["confidence"],
            data["salience"],
            data["decay_rate"],
            data["embedding"]
        )
        node.__dict__.update(data)
        return node


# =========================
# Graph
# =========================

class MemoryGraph:
    def __init__(self):
        self.nodes: Dict[str, MemoryNode] = {}
        self.edges = defaultdict(list)  # node_id -> list of (relationship, target_id)

    def add_node(self, node: MemoryNode):
        self.nodes[node.id] = node

    def add_edge(self, source_id: str, relationship: str, target_id: str):
        self.edges[source_id].append((relationship, target_id))

    def get_subtree_nodes(self, path_prefix: List[str]):
        result = []
        for node in self.nodes.values():
            if node.tree_path[:len(path_prefix)] == path_prefix:
                result.append(node)
        return result

    def to_dict(self):
        return {
            "nodes": {nid: node.to_dict() for nid, node in self.nodes.items()},
            "edges": dict(self.edges)
        }

    @staticmethod
    def from_dict(data):
        graph = MemoryGraph()
        graph.nodes = {nid: MemoryNode.from_dict(n) for nid, n in data["nodes"].items()}
        graph.edges = defaultdict(list, data["edges"])
        return graph


# =========================
# Tree Router
# =========================

import re

def classify_tree_path(text: str):

    try:
        response = client.responses.create(
            model="gpt-4.1",
            input=f"""
Classify this text into a tree path.

Choose from:
People
Tasks
Technical
Preferences
General

Return ONLY a JSON array.
Example:
["People"]

Text:
{text}
"""
        )

        raw = response.output_text.strip()

        # Extract JSON array from response safely
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return ["General"]

    except Exception:
        return ["General"]


# =========================
# Memory Extraction
# =========================

def extract_memories(text: str):

    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=f"""
Extract atomic memory units from the text.

Return a JSON array of objects:
[
  {{
    "content": "...",
    "type": "fact | preference | episode | procedural"
  }}
]

Text:
{text}
"""
        )

        raw = response.output_text.strip()

        match = re.search(r'\[.*\]', raw, re.DOTALL)
        if match:
            return json.loads(match.group())

        return []

    except Exception:
        return []


# =========================
# Relationship Detection
# =========================

def detect_relationship(new_node: MemoryNode, candidates: List[MemoryNode]):

    if not candidates:
        return None

    try:
        candidate_text = "\n".join(
            [f"{n.id}: {n.content}" for n in candidates]
        )

        response = client.responses.create(
            model="gpt-4o-mini",
            input=f"""
New memory:
{new_node.content}

Existing memories:
{candidate_text}

Does it UPDATES, EXTENDS, DERIVES, or NONE?

Return JSON:
{{
  "relationship": "UPDATES | EXTENDS | DERIVES | NONE",
  "target_id": "existing_node_id_or_null"
}}
"""
        )

        raw = response.output_text.strip()

        import re
        match = re.search(r'\{.*\}', raw, re.DOTALL)

        if match:
            data = json.loads(match.group())
            return data

        return None

    except Exception:
        return None


# =========================
# Capsule
# =========================

class MemoryCapsule:

    def __init__(self):
        self.graph = MemoryGraph()

    def add_text(self, text: str):
        extracted = extract_memories(text)

        for item in extracted:
            tree_path = classify_tree_path(item["content"])
            node = MemoryNode(
                content=item["content"],
                node_type=item["type"],
                tree_path=tree_path
            )

            candidates = self.graph.get_subtree_nodes(tree_path[:1])
            rel = detect_relationship(node, candidates)

            self.graph.add_node(node)

            if rel and isinstance(rel, dict):
                relationship = rel.get("relationship")
                target_id = rel.get("target_id")

                if relationship and relationship != "NONE" and target_id in self.graph.nodes:

                    self.graph.add_edge(node.id, relationship, target_id)

                    if relationship == "UPDATES":
                        self.graph.nodes[target_id].isLatest = False
                        self.graph.add_edge(node.id, "TEMPORAL_CHAIN", target_id)

    def query(self, query: str):
        path = classify_tree_path(query)
        nodes = self.graph.get_subtree_nodes(path[:1])

        # Optional minimal embedding filtering
        relevant = sorted(
            nodes,
            key=lambda n: n.strength(),
            reverse=True
        )

        return [n.content for n in relevant[:5]]

    def consolidate(self):
        episodes = [n for n in self.graph.nodes.values() if n.type == "episode"]

        if len(episodes) < 3:
            return

        text = "\n".join([e.content for e in episodes])

        response = client.responses.create(
            model="gpt-4o-mini",
            input=f"""
Create a stable semantic summary from these episodes:
{text}
"""
        )

        summary = response.output[0].content[0].text

        node = MemoryNode(
            content=summary,
            node_type="fact",
            tree_path=["People"]
        )

        self.graph.add_node(node)

        for e in episodes:
            self.graph.add_edge(node.id, "CONSOLIDATES", e.id)

    def decay(self, threshold=0.1):
        to_archive = []
        for node in self.graph.nodes.values():
            if node.strength() < threshold:
                to_archive.append(node.id)

        for nid in to_archive:
            del self.graph.nodes[nid]

    def export(self, filename="memory_capsule.umc"):
        with zipfile.ZipFile(filename, "w") as z:
            z.writestr("graph.json", json.dumps(self.graph.to_dict()))
        print(f"Exported to {filename}")

    def load(self, filename="memory_capsule.umc"):
        with zipfile.ZipFile(filename, "r") as z:
            data = json.loads(z.read("graph.json"))
            self.graph = MemoryGraph.from_dict(data)
        print("Memory capsule loaded.")