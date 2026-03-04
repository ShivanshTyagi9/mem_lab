"""
agent.py — Agent integration layer

Wires MemorySocket into any agent. Supports any LLM backend.

AgentSession wraps a MemoryHandle (socket connection) and provides:
  - Automatic memory injection into system prompts
  - Ingestion of conversation turns
  - Session save/restore (plug/unplug)
  - Works with OpenAI OR Anthropic (or any LLMBackend)

Patterns:
─────────
  # 1. Private memory (one agent, one capsule)
  agent = AgentSession(name="alex")
  agent.chat("I'm Alex, VP of Product at Stripe in Seattle.")
  agent.save("alex.umc")

  # 2. Shared memory (multiple agents, one socket)
  from memory_socket import MemorySocket, AccessMode
  socket = MemorySocket(name="team_knowledge")
  agent_a = AgentSession.from_socket(socket.connect("agent_a"))
  agent_b = AgentSession.from_socket(socket.connect("agent_b", AccessMode.READ_ONLY))
  # agent_b reads what agent_a learned — live, no file round-trip

  # 3. Inherit memory from a trained agent
  socket = MemorySocket.from_file("expert_agent.umc")
  new_agent = AgentSession.from_socket(socket.connect("new_agent"))
  new_agent.chat("What do you know about payments?")
  # Already knows everything from expert_agent

  # 4. Choose your LLM backend
  from llm import AnthropicBackend, OpenAIBackend
  agent = AgentSession(name="claude_agent", backend=AnthropicBackend())
"""

import os
import time
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv

from capsule import MemoryCapsule
from memory_socket import MemorySocket, MemoryHandle, AccessMode
from llm import LLMBackend, auto_backend, AnthropicBackend, OpenAIBackend, route_message

load_dotenv()


class AgentSession:
    """
    A conversational agent with pluggable memory and swappable LLM backend.

    The agent holds a MemoryHandle (connection to a MemorySocket).
    It can share that socket with other agents — giving them access to the
    same memories — or keep it private.
    """

    def __init__(
        self,
        name: str = "agent",
        system_prompt: str = "You are a helpful assistant with memory.",
        backend: Optional[LLMBackend] = None,
        handle: Optional[MemoryHandle] = None,
    ):
        self.name          = name
        self.system_prompt = system_prompt
        self._backend      = backend or auto_backend()
        self.history: List[Dict] = []

        # If no handle provided, create a private socket for this agent
        if handle is None:
            socket       = MemorySocket(name=name)
            self._socket = socket  # own socket (can export later)
            self.memory  = socket.connect(name, mode=AccessMode.READ_WRITE)
        else:
            self._socket = None   # external socket — we don't own it
            self.memory  = handle

    # ──────────────────────────────────────────────────────────────────────
    # Factory constructors
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_socket(
        cls,
        handle: MemoryHandle,
        system_prompt: str = "You are a helpful assistant with memory.",
        backend: Optional[LLMBackend] = None,
    ) -> "AgentSession":
        """
        Create an agent connected to an existing shared socket.

          socket = MemorySocket.from_file("team.umc")
          agent  = AgentSession.from_socket(socket.connect("my_agent"))
        """
        return cls(
            name=handle.agent_id,
            system_prompt=system_prompt,
            backend=backend,
            handle=handle,
        )

    @classmethod
    def resume(
        cls,
        capsule_path: str,
        name: Optional[str] = None,
        system_prompt: str = "You are a helpful assistant with memory.",
        backend: Optional[LLMBackend] = None,
    ) -> "AgentSession":
        """
        Plug-in: restore an agent with its memories intact from a .umc file.
        Agent immediately knows everything from previous sessions.
        """
        socket      = MemorySocket.from_file(capsule_path)
        agent_name  = name or socket.capsule.name
        handle      = socket.connect(agent_name, mode=AccessMode.READ_WRITE)

        agent           = cls.__new__(cls)
        agent.name      = agent_name
        agent.system_prompt = system_prompt
        agent._backend  = backend or auto_backend()
        agent.history   = []
        agent._socket   = socket
        agent.memory    = handle
        return agent

    # ──────────────────────────────────────────────────────────────────────
    # Core chat loop
    # ──────────────────────────────────────────────────────────────────────

    def chat(self, user_message: str, verbose: bool = False) -> str:
        """
        Send a message, get a response.

        Pipeline:
          1. Retrieve relevant memories for this query
          2. Build memory-augmented system prompt
          3. Call LLM (OpenAI or Claude, transparently)
          4. Ingest the full turn into memory
        """
        # 1. Retrieve relevant memories
        memories = self.memory.query(user_message, top_k=6)

        # 2. Build system prompt with memory context
        system = self._build_system_prompt(memories)

        # 3. Append user message to history
        self.history.append({"role": "user", "content": user_message})

        # 4. Call LLM — handle OpenAI and Anthropic differently
        context = self.history[-20:]  # last 20 turns
        assistant_reply = self._llm_chat(system, context)

        # 5. Add reply to history
        self.history.append({"role": "assistant", "content": assistant_reply})

        # 6. Ingest the full turn into memory.
        # Both sides together give the extractor full context.
        # e.g. "my name is Shivansh" + "Nice to meet you, Shivansh!" confirms the fact.
        try:
            category = route_message(user_message)
        except Exception:
            category = None

        turn_text = f"User said: {user_message}\nAssistant replied: {assistant_reply}"
        # Pass hint into the capsule ingest path (backwards compatible: default hint=None)
        nodes = self.memory.ingest(turn_text, hint=category, verbose=verbose)

        if verbose:
            print(f"\n[memory injected] {self._format_memories(memories)}")
            print(f"[memory stored] {len(nodes)} node(s) created\n")

        return assistant_reply

    def _llm_chat(self, system: str, messages: List[Dict]) -> str:
        """
        Route chat to the right backend API.
        OpenAI and Anthropic have different system prompt conventions.
        """
        backend = self._backend

        if isinstance(backend, AnthropicBackend):
            # Anthropic: system is a separate parameter
            # Filter out any "system" role messages from history
            user_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
            return backend.chat(user_msgs, system=system)

        elif isinstance(backend, OpenAIBackend):
            # OpenAI: system is prepended as first message
            full_messages = [{"role": "system", "content": system}] + messages
            return backend.chat(full_messages)

        else:
            # Generic fallback: try OpenAI-style, then Anthropic-style
            try:
                full_messages = [{"role": "system", "content": system}] + messages
                return backend.chat(full_messages)  # type: ignore
            except (AttributeError, TypeError):
                # Last resort: single-turn prompt
                prompt = f"System: {system}\n\nUser: {messages[-1]['content']}"
                return backend.complete(prompt)

    def _build_system_prompt(self, memories: List[str]) -> str:
        base = self.system_prompt

        # Inject pending tasks as a separate block — always visible to agent
        task_block = self._format_pending_tasks()

        if not memories and not task_block:
            return base

        mem_block = ""
        if memories:
            mem_block = "\nYou have the following relevant memories about this user/context:\n"
            mem_block += "\n".join(f"  - {m}" for m in memories)
            mem_block += "\nUse these to personalise your responses naturally.\n"

        return f"""{base}
{mem_block}
{task_block}
Don't announce "I remember that..." — respond as if you already know.
"""

    def _format_pending_tasks(self) -> str:
        """
        Get pending tasks from memory and format them for the system prompt.
        This keeps the agent aware of what the user has scheduled or needs to do.
        """
        try:
            capsule = self.memory._socket._capsule
            tasks   = capsule.get_pending_tasks()
        except AttributeError:
            return ""

        if not tasks:
            return ""

        lines = ["You are aware of these pending tasks for the user:"]
        for task in tasks[:10]:  # cap at 10 to stay token-efficient
            title    = task.task_title or task.content[:60]
            due      = f" — due {task.due_date}" if task.due_date else ""
            lines.append(f"  • [{task.id[:6]}] {title}{due}")
        lines.append(
            "If the user asks about tasks or their schedule, refer to these. "
            "When a task is completed, note the task ID so it can be marked done."
        )
        return "\n".join(lines) + "\n"

    def _format_memories(self, memories: List[str]) -> str:
        if not memories:
            return "(none)"
        return "\n".join(f"  • {m}" for m in memories)

    # ──────────────────────────────────────────────────────────────────────
    # Session management
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Plug-out: save memory capsule to a .umc file.
        Conversation history is NOT saved — only extracted memories.
        """
        if path is None:
            path = f"{self.name}.umc"
        return self.memory.export_snapshot(path)

    def share_memory(
        self,
        other_agent: "AgentSession",
        mode: str = AccessMode.READ_ONLY,
    ) -> MemoryHandle:
        """
        Give another agent access to this agent's memory socket.

        By default the other agent gets READ_ONLY — they can learn from
        this agent's knowledge but can't modify it.

          expert.share_memory(new_agent)  # new_agent reads expert's memory

        Returns the handle given to the other agent.
        """
        # Figure out which socket to share
        socket = self._socket
        if socket is None:
            # This agent uses an external socket — get it from the handle
            socket = self.memory._socket

        handle = socket.connect(other_agent.name, mode=mode)
        other_agent.memory  = handle
        other_agent._socket = None  # doesn't own the shared socket
        print(
            f"[agent] '{self.name}' shared memory with '{other_agent.name}' "
            f"(mode={mode})"
        )
        return handle

    # ──────────────────────────────────────────────────────────────────────
    # Introspection
    # ──────────────────────────────────────────────────────────────────────

    def memory_stats(self) -> dict:
        return self.memory.stats()

    def what_do_you_know_about(self, topic: str) -> List[str]:
        """Convenience: retrieve memories about a specific topic."""
        return self.memory.query(topic, top_k=10)

    def __repr__(self) -> str:
        return (
            f"<AgentSession '{self.name}' "
            f"backend={self._backend} "
            f"memory={self.memory}>"
        )


# ---------------------------------------------------------------------------
# Demo — run directly to see it work
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("uMemory Agent Demo — multi-agent memory sharing")
    print("=" * 60)

    # ── Session 1: Expert agent builds up knowledge ──────────────────────
    print("\n[Session 1] Expert agent learning about Alex...")

    # Use whichever LLM key is available
    expert = AgentSession(name="expert_agent")

    exchanges = [
        "Hi! I'm Alex. I work at Stripe as a VP of Product, based in Seattle.",
        "I used to be a software engineer at Google before joining Stripe.",
        "I lead a team of 5 engineers focused on payments infrastructure.",
        "I really prefer morning meetings — I'm useless after 2pm honestly.",
        "Just started a new project: building a real-time fraud detection API.",
    ]

    for msg in exchanges:
        print(f"\nUser: {msg}")
        reply = expert.chat(msg)
        print(f"Expert: {reply}")

    print(f"\n{expert.memory.stats()}")

    # ── Save expert's knowledge ──────────────────────────────────────────
    capsule_path = expert.save("expert.umc")

    # ── Session 2: New agent inherits expert's knowledge ─────────────────
    print("\n" + "=" * 60)
    print("[Session 2] New agent inheriting expert's knowledge...")
    print("=" * 60)

    new_agent = AgentSession.resume(capsule_path, name="new_agent")

    queries = [
        "Where does Alex work?",
        "What's Alex's background?",
        "What project is Alex working on?",
    ]

    for q in queries:
        print(f"\nUser: {q}")
        reply = new_agent.chat(q)
        print(f"New agent: {reply}")

    # ── Session 3: Live memory sharing (both agents share one socket) ─────
    print("\n" + "=" * 60)
    print("[Session 3] Live memory sharing between two agents...")
    print("=" * 60)

    shared_socket = MemorySocket(name="shared_knowledge")
    agent_a = AgentSession.from_socket(
        shared_socket.connect("agent_a"),
        system_prompt="You are a research assistant.",
    )
    agent_b = AgentSession.from_socket(
        shared_socket.connect("agent_b", mode=AccessMode.READ_ONLY),
        system_prompt="You are a helpful summariser.",
    )

    # Agent A learns something
    agent_a.chat("The user is building a real-time payments API using FastAPI and Redis.")
    agent_a.chat("The architecture uses event-driven design with Kafka for message queues.")

    # Agent B immediately sees it (same socket)
    print("\nAgent B querying shared memory:")
    result = agent_b.memory.query("What is the tech stack?", top_k=5)
    for m in result:
        print(f"  · {m}")
