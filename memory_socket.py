"""
memory_socket.py — Shared memory interface for multi-agent systems

The MemorySocket is the answer to: "how do multiple agents share one memory?"

Architecture:
─────────────
  MemorySocket   — owns one MemoryCapsule. Multiple agents connect to it.
  MemoryHandle   — what each agent holds. Enforces read/write permissions.
  CapsuleRegistry — manages a collection of named sockets (multi-domain memory).

Socket metaphor:
  Think of it like a USB socket:
    - The socket exposes a standard interface.
    - Any compatible agent can plug in.
    - The device (memory capsule) is the same for all connectors.
    - Access mode (r/rw) is enforced per connection.

Quick start:
────────────
  # Agent A builds up knowledge
  socket = MemorySocket(name="project-alpha")
  handle_a = socket.connect("agent_a")
  handle_a.ingest("We're building a payments API in Python with FastAPI.")
  handle_a.ingest("Deadline is end of Q2. Lead engineer is Priya.")

  # Save the capsule
  socket.export("project-alpha.umc")

  # Agent B loads the same knowledge (read-only, can't pollute it)
  socket2 = MemorySocket.from_file("project-alpha.umc")
  handle_b = socket2.connect("agent_b", mode=AccessMode.READ_ONLY)
  handle_b.query("What's the tech stack?")
  # → ["The project uses Python with FastAPI.", ...]

  # Multi-agent: all agents share live memory
  shared_socket = MemorySocket(name="team")
  handle_a = shared_socket.connect("research_agent")
  handle_b = shared_socket.connect("writing_agent", mode=AccessMode.READ_ONLY)
  handle_c = shared_socket.connect("qa_agent")
  # When handle_a ingests, handle_b and handle_c immediately see it.

Registry (multiple domains):
─────────────────────────────
  registry = CapsuleRegistry()
  registry.register("users",    "users.umc")
  registry.register("codebase", "codebase.umc")
  registry.register("docs",     "docs.umc")

  agent_handle = registry.connect("my_agent", "codebase")
  # Agent talks to the codebase memory only
"""

import os
import time
from typing import Dict, List, Optional, Any

from capsule import MemoryCapsule
from models import MemoryNode


# ---------------------------------------------------------------------------
# Access modes
# ---------------------------------------------------------------------------

class AccessMode:
    READ_WRITE = "rw"   # Can query and ingest
    READ_ONLY  = "r"    # Can only query — perfect for agents that should
                        # learn from existing memory but not modify it
    WRITE_ONLY = "w"    # Can only ingest — useful for data pipeline agents
                        # that feed knowledge but don't answer questions


# ---------------------------------------------------------------------------
# MemoryHandle — what an agent actually holds
# ---------------------------------------------------------------------------

class MemoryHandle:
    """
    A connection handle between an agent and a MemorySocket.

    This is the object agents work with. It gates all operations through
    the socket's access control layer.

    Don't create this directly — get it from socket.connect().
    """

    def __init__(self, agent_id: str, socket: "MemorySocket", mode: str):
        self.agent_id  = agent_id
        self._socket   = socket
        self.mode      = mode
        self._connected_at = time.time()

    # ── Memory operations ─────────────────────────────────────────────────

    def query(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant memories. Requires READ access."""
        if self.mode == AccessMode.WRITE_ONLY:
            raise PermissionError(
                f"Agent '{self.agent_id}' has WRITE_ONLY access — cannot query."
            )
        return self._socket._capsule.query(query, top_k=top_k)

    def ingest(self, text: str, hint: Optional[str] = None, verbose: bool = False) -> List[MemoryNode]:
        """Store new memories. Requires WRITE access."""
        if self.mode == AccessMode.READ_ONLY:
            raise PermissionError(
                f"Agent '{self.agent_id}' has READ_ONLY access — cannot ingest."
            )
        return self._socket._capsule.ingest(text, hint, verbose=verbose)

    def query_with_history(self, query: str, top_k: int = 5) -> Dict[str, List]:
        """Query current + superseded memory history."""
        if self.mode == AccessMode.WRITE_ONLY:
            raise PermissionError(
                f"Agent '{self.agent_id}' has WRITE_ONLY access — cannot query."
            )
        return self._socket._capsule.query_with_history(query, top_k=top_k)

    def query_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Rich query with metadata — for agent introspection."""
        if self.mode == AccessMode.WRITE_ONLY:
            raise PermissionError(
                f"Agent '{self.agent_id}' has WRITE_ONLY access — cannot query."
            )
        return self._socket._capsule.query_with_context(query, top_k=top_k)

    def stats(self) -> dict:
        s = self._socket.stats()
        s["my_agent_id"] = self.agent_id
        s["my_access"]   = self.mode
        return s

    def consolidate(self, min_episodes: int = 3, regenerate_kb: bool = True, verbose: bool = False) -> int:
        """Consolidate episodic memories. Requires WRITE access."""
        if self.mode == AccessMode.READ_ONLY:
            raise PermissionError(
                f"Agent '{self.agent_id}' has READ_ONLY access — cannot consolidate."
            )
        return self._socket._capsule.consolidate(
            min_episodes=min_episodes, regenerate_kb=regenerate_kb, verbose=verbose
        )

    def decay(self, threshold: float = 0.05, verbose: bool = False) -> int:
        """Prune weak memories. Requires WRITE access."""
        if self.mode == AccessMode.READ_ONLY:
            raise PermissionError(
                f"Agent '{self.agent_id}' has READ_ONLY access — cannot decay."
            )
        return self._socket._capsule.decay(threshold=threshold, verbose=verbose)

    def dump_branch(self, branch: str, include_superseded: bool = False):
        """Dump all nodes in a branch for inspection."""
        return self._socket._capsule.dump_branch(
            branch, include_superseded=include_superseded
        )

    @property
    def graph(self):
        """Direct access to the underlying MemoryGraph (read-only use)."""
        return self._socket._capsule.graph

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def disconnect(self):
        """Unplug this agent from the socket."""
        self._socket.disconnect(self.agent_id)

    def export_snapshot(self, path: Optional[str] = None) -> str:
        """
        Export the current shared memory to a .umc file.
        Any agent with any access level can snapshot — reading doesn't require write.
        """
        return self._socket.export(path)

    def __repr__(self) -> str:
        return (
            f"<MemoryHandle agent='{self.agent_id}' "
            f"socket='{self._socket.name}' "
            f"mode={self.mode}>"
        )


# ---------------------------------------------------------------------------
# MemorySocket — the shared memory endpoint
# ---------------------------------------------------------------------------

class MemorySocket:
    """
    A shared memory endpoint. One capsule, many agents.

    All connected agents read from and write to the same MemoryCapsule.
    When agent A ingests, agent B immediately sees the new memory.

    Access control is enforced per-connection via AccessMode.
    """

    def __init__(
        self,
        capsule: Optional[MemoryCapsule] = None,
        name: str = "shared",
    ):
        self._capsule     = capsule or MemoryCapsule(name=name)
        self.name         = self._capsule.name
        self._connections: Dict[str, str] = {}  # agent_id → AccessMode

    # ── Connection management ─────────────────────────────────────────────

    def connect(
        self,
        agent_id: str,
        mode: str = AccessMode.READ_WRITE,
    ) -> MemoryHandle:
        """
        Plug an agent into this socket.

        Returns a MemoryHandle the agent uses for all memory operations.
        Multiple agents can be connected simultaneously.

        agent_id: unique identifier for this agent (e.g. "research_agent", "alex")
        mode:     AccessMode.READ_WRITE | READ_ONLY | WRITE_ONLY
        """
        if agent_id in self._connections:
            # Already connected — return a new handle with updated mode
            self._connections[agent_id] = mode
        else:
            self._connections[agent_id] = mode

        handle = MemoryHandle(agent_id=agent_id, socket=self, mode=mode)
        print(
            f"[socket] '{agent_id}' connected to '{self.name}' "
            f"(mode={mode}, {len(self._capsule.graph.nodes)} memories available)"
        )
        return handle

    def disconnect(self, agent_id: str):
        """Unplug an agent."""
        if agent_id in self._connections:
            del self._connections[agent_id]
            print(f"[socket] '{agent_id}' disconnected from '{self.name}'")

    # ── Direct capsule access (internal use) ──────────────────────────────

    @property
    def capsule(self) -> MemoryCapsule:
        """Direct access to the underlying capsule (no access control)."""
        return self._capsule

    # ── Persistence ───────────────────────────────────────────────────────

    def export(self, path: Optional[str] = None) -> str:
        """Save the shared memory to a .umc file."""
        return self._capsule.export(path or f"{self.name}.umc")

    @classmethod
    def from_file(cls, path: str) -> "MemorySocket":
        """
        Create a socket backed by an existing .umc capsule file.
        This is the standard way to share a trained agent's memory.

          socket = MemorySocket.from_file("alex_assistant.umc")
          new_agent_handle = socket.connect("new_agent", mode=AccessMode.READ_ONLY)
        """
        capsule = MemoryCapsule.load(path)
        return cls(capsule=capsule, name=capsule.name)

    # ── Merge ─────────────────────────────────────────────────────────────

    def absorb(self, other_path: str, prefer: str = "self") -> int:
        """
        Load another capsule and merge it into this socket's memory in-place.

        Use this when you want the current socket to gain knowledge from
        another agent's capsule without creating a new socket.

        Returns the number of new nodes added.
        """
        before = len(self._capsule.graph.nodes)
        other  = MemoryCapsule.load(other_path)
        merged = self._capsule.merge(other, name=self.name, prefer=prefer)

        # Replace the internal capsule
        self._capsule = merged
        after = len(self._capsule.graph.nodes)
        added = after - before
        print(f"[socket] absorbed '{other_path}' → +{added} new nodes")
        return added

    # ── Introspection ─────────────────────────────────────────────────────

    def stats(self) -> dict:
        s = self._capsule.stats()
        s["socket_name"]       = self.name
        s["connected_agents"]  = list(self._connections.keys())
        s["connection_count"]  = len(self._connections)
        return s

    def who_is_connected(self) -> Dict[str, str]:
        """Returns {agent_id: access_mode} for all connected agents."""
        return dict(self._connections)

    def __repr__(self) -> str:
        n = len(self._capsule.graph.nodes)
        c = len(self._connections)
        return f"<MemorySocket '{self.name}' nodes={n} agents={c}>"


# ---------------------------------------------------------------------------
# CapsuleRegistry — manage multiple named memory domains
# ---------------------------------------------------------------------------

class CapsuleRegistry:
    """
    A registry of named sockets. Use this when you have multiple distinct
    domains of memory (per-user, per-project, per-topic).

    Agents can be connected to one or multiple domains.

    Example:
      registry = CapsuleRegistry()
      registry.register("users",    "memory/users.umc")
      registry.register("codebase", "memory/codebase.umc")
      registry.register("docs",     "memory/docs.umc")

      # Research agent reads codebase + docs, can't write
      h1 = registry.connect("research_agent", "codebase", AccessMode.READ_ONLY)
      h2 = registry.connect("research_agent", "docs",     AccessMode.READ_ONLY)

      # Ingestion agent writes to codebase only
      h3 = registry.connect("ingest_agent", "codebase", AccessMode.WRITE_ONLY)
    """

    def __init__(self):
        self._sockets: Dict[str, MemorySocket] = {}

    def register(
        self,
        name: str,
        capsule_path: Optional[str] = None,
        overwrite: bool = False,
    ) -> MemorySocket:
        """
        Register a named memory domain.

        name:          Identifier for this domain (e.g. "users", "codebase")
        capsule_path:  Optional path to a .umc file to load. If not provided or
                       file doesn't exist, starts with a fresh capsule.
        overwrite:     If True, replace existing registration with this name.
        """
        if name in self._sockets and not overwrite:
            return self._sockets[name]

        if capsule_path and os.path.exists(capsule_path):
            socket = MemorySocket.from_file(capsule_path)
        else:
            socket = MemorySocket(name=name)

        self._sockets[name] = socket
        return socket

    def get(self, name: str) -> Optional[MemorySocket]:
        """Get a socket by name, or None if not registered."""
        return self._sockets.get(name)

    def connect(
        self,
        agent_id: str,
        capsule_name: str,
        mode: str = AccessMode.READ_WRITE,
    ) -> MemoryHandle:
        """
        Connect an agent to a named memory domain.
        Raises KeyError if the capsule_name isn't registered.
        """
        socket = self._sockets.get(capsule_name)
        if not socket:
            raise KeyError(
                f"No memory domain registered as '{capsule_name}'. "
                f"Available: {list(self._sockets.keys())}"
            )
        return socket.connect(agent_id, mode)

    def save_all(self, directory: str = "."):
        """Export all registered capsules to a directory."""
        os.makedirs(directory, exist_ok=True)
        for name, socket in self._sockets.items():
            path = os.path.join(directory, f"{name}.umc")
            socket.export(path)
        print(f"[registry] saved {len(self._sockets)} capsule(s) to '{directory}/'")

    def load_all(self, directory: str = "."):
        """
        Load all .umc files from a directory into the registry.
        The capsule name is taken from the filename (without extension).
        """
        loaded = 0
        for filename in os.listdir(directory):
            if filename.endswith(".umc"):
                name = filename[:-4]
                path = os.path.join(directory, filename)
                self._sockets[name] = MemorySocket.from_file(path)
                loaded += 1
        print(f"[registry] loaded {loaded} capsule(s) from '{directory}/'")

    def merge_into(self, target_name: str, source_name: str) -> int:
        """
        Merge source capsule into target capsule in-place.
        Returns number of new nodes added to target.
        """
        target = self._sockets.get(target_name)
        source = self._sockets.get(source_name)
        if not target:
            raise KeyError(f"Target '{target_name}' not registered.")
        if not source:
            raise KeyError(f"Source '{source_name}' not registered.")

        before  = len(target.capsule.graph.nodes)
        merged  = target.capsule.merge(source.capsule, name=target_name)
        target._capsule = merged
        after   = len(target.capsule.graph.nodes)
        return after - before

    def list_domains(self) -> List[dict]:
        """List all registered domains with their stats."""
        return [
            {
                "name":   name,
                "nodes":  len(socket.capsule.graph.nodes),
                "agents": list(socket.who_is_connected().keys()),
            }
            for name, socket in self._sockets.items()
        ]

    def __repr__(self) -> str:
        names = list(self._sockets.keys())
        return f"<CapsuleRegistry domains={names}>"