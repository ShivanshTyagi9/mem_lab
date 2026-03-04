"""
network_agent.py — Conversational agent backed by the uMemory HTTP server

This agent keeps its LLM chat local but delegates all memory to the server.
The server holds the capsule; the agent just queries and ingests through HTTP.

Quick start:
────────────
  # Terminal 1 — start the server
  uvicorn server:app --port 8000

  # Terminal 2 — run the agent interactively
  python network_agent.py

  # Or resume a saved session
  python network_agent.py --load ./alex.umc --name alex

  # Programmatic use
  from network_agent import NetworkAgent
  agent = NetworkAgent(name="alex", server="http://localhost:8000")
  agent.chat("My name is Alex and I work at Stripe.")
  agent.chat("What do you know about me?")
  agent.save("./alex.umc")
  agent.close()
"""

import argparse
import os
import sys
from typing import List, Optional, Dict

from dotenv import load_dotenv

from client import MemoryHandleProxy, connect_to_server, load_capsule_on_server
from llm import LLMBackend, auto_backend, AnthropicBackend, OpenAIBackend, route_message

load_dotenv()


class NetworkAgent:
    """
    A conversational agent that uses the uMemory HTTP server for memory.

    Memory lives on the server. LLM chat happens locally.
    This means you can run multiple agents against the same server socket,
    swap LLM backends without touching the server, and keep the server stateless
    with respect to conversation history.

    Lifecycle:
      agent = NetworkAgent(name="alex", server="http://localhost:8000")
      agent.chat("...")        # query memory → LLM → ingest turn
      agent.save("alex.umc")  # tell server to persist capsule
      agent.close()           # disconnect handle from server

    Context manager:
      with NetworkAgent(name="alex") as agent:
          agent.chat("...")
      # auto-saves and disconnects on exit
    """

    def __init__(
        self,
        name: str = "agent",
        server: str = "http://localhost:8000",
        socket_name: str = "default",
        mode: str = "rw",
        system_prompt: Optional[str] = None,
        backend: Optional[LLMBackend] = None,
        api_key: Optional[str] = None,
        auto_save_path: Optional[str] = None,
    ):
        self.name           = name
        self.server         = server
        self.auto_save_path = auto_save_path
        self._backend       = backend or auto_backend()
        self.history: List[Dict] = []

        # System prompt — used on every LLM call
        self.system_prompt = system_prompt or (
            f"You are a helpful, thoughtful assistant. "
            f"You have memory of past conversations with this user. "
            f"Use it naturally — never announce 'I remember that...', "
            f"just respond as if you already know."
        )

        # Connect to the memory server
        self.memory: MemoryHandleProxy = connect_to_server(
            base_url=server,
            agent_id=name,
            socket_name=socket_name,
            mode=mode,
            api_key=api_key,
        )

        print(f"[agent] '{name}' connected to {server} (socket='{socket_name}', mode={mode})")

    # ── Factory constructors ──────────────────────────────────────────────

    @classmethod
    def resume(
        cls,
        capsule_path: str,
        name: Optional[str] = None,
        server: str = "http://localhost:8000",
        socket_name: str = "default",
        system_prompt: Optional[str] = None,
        backend: Optional[LLMBackend] = None,
        api_key: Optional[str] = None,
    ) -> "NetworkAgent":
        """
        Resume from a saved .umc capsule.

        Tells the server to load the capsule, then connects normally.
        The agent immediately knows everything from the previous session.

          agent = NetworkAgent.resume("./alex.umc", name="alex")
          agent.chat("Where did we leave off?")
        """
        # Tell server to load the file into the target socket slot
        result = load_capsule_on_server(
            base_url=server,
            path=capsule_path,
            socket_name=socket_name,
            api_key=api_key,
        )
        agent_name = name or os.path.splitext(os.path.basename(capsule_path))[0]
        print(f"[agent] loaded capsule → {result.get('nodes', '?')} nodes in '{socket_name}'")

        return cls(
            name=name or agent_name,
            server=server,
            socket_name=socket_name,
            system_prompt=system_prompt,
            backend=backend,
            api_key=api_key,
            auto_save_path=capsule_path,
        )

    # ── Core chat loop ────────────────────────────────────────────────────

    def chat(self, user_message: str, verbose: bool = False) -> str:
        """
        Send a message, get a response with memory context.

        Pipeline:
          1. Query memory server for relevant context
          2. Build memory-augmented system prompt
          3. Call LLM (local, any backend)
          4. Ingest the full turn into memory server
          5. Return assistant reply
        """
        # 1. Retrieve relevant memories
        memories = self.memory.query(user_message, top_k=6)

        # 2. Build system prompt
        system = self._build_system_prompt(memories)

        # 3. Append to history and call LLM
        self.history.append({"role": "user", "content": user_message})
        reply = self._llm_chat(system, self.history[-20:])
        self.history.append({"role": "assistant", "content": reply})

        if verbose:
            print(f"\n[memory] {len(memories)} relevant memories injected")
            for m in memories:
                print(f"  · {m}")

        # 4. Ingest the full turn — both sides give the extractor full context
        turn = f"User said: {user_message}\nAssistant replied: {reply}"
        try:
            hint = route_message(user_message, backend=self._backend)
        except Exception:
            hint = None

        try:
            result = self.memory.ingest(turn, hint=hint, verbose=verbose)
            if verbose:
                print(f"[memory] {result.get('ingested', 0)} node(s) stored")
        except Exception as e:
            # Memory ingest failure should never crash the conversation
            if verbose:
                print(f"[memory] ingest warning: {e}")

        return reply

    def _llm_chat(self, system: str, messages: List[Dict]) -> str:
        """Route chat to the correct backend API convention."""
        backend = self._backend

        if isinstance(backend, AnthropicBackend):
            user_msgs = [m for m in messages if m["role"] in ("user", "assistant")]
            return backend.chat(user_msgs, system=system)

        elif isinstance(backend, OpenAIBackend):
            full = [{"role": "system", "content": system}] + messages
            return backend.chat(full)

        else:
            # Generic fallback: OpenAI-style
            try:
                full = [{"role": "system", "content": system}] + messages
                return backend.chat(full)  # type: ignore
            except (AttributeError, TypeError):
                prompt = f"System: {system}\n\nUser: {messages[-1]['content']}"
                return backend.complete(prompt)

    def _build_system_prompt(self, memories: List[str]) -> str:
        base = self.system_prompt

        # Pending tasks block
        task_block = self._format_tasks()

        if not memories and not task_block:
            return base

        mem_block = ""
        if memories:
            mem_block = "\nRelevant memories about this user:\n"
            mem_block += "\n".join(f"  - {m}" for m in memories)
            mem_block += "\n"

        return f"{base}\n{mem_block}\n{task_block}"

    def _format_tasks(self) -> str:
        try:
            tasks = self.memory.get_tasks()
        except Exception:
            return ""
        if not tasks:
            return ""
        lines = ["Pending tasks for this user:"]
        for t in tasks[:10]:
            title = t.get("title") or t.get("content", "")[:60]
            due   = f" — due {t['due_date']}" if t.get("due_date") else ""
            lines.append(f"  • [{t['id'][:6]}] {title}{due}")
        lines.append(
            "When a task is completed, note the task ID so it can be marked done."
        )
        return "\n".join(lines) + "\n"

    # ── Memory convenience methods ────────────────────────────────────────

    def what_do_you_know_about(self, topic: str, top_k: int = 10) -> List[str]:
        """Query memory about a specific topic."""
        return self.memory.query(topic, top_k=top_k)

    def pending_tasks(self) -> List[Dict]:
        """Return pending tasks from memory."""
        return self.memory.get_tasks()

    def complete_task(self, node_id: str) -> bool:
        """Mark a task done by its node_id."""
        try:
            self.memory.mark_task_done(node_id)
            return True
        except Exception:
            return False

    def consolidate(self) -> int:
        """Consolidate episodic memories into stable facts. Good to run end-of-session."""
        result = self.memory.consolidate()
        return result.get("groups_consolidated", 0)

    def stats(self) -> dict:
        return self.memory.stats()

    # ── Session management ────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """
        Save memory capsule on the server to a .umc file.
        Path must be accessible from the server's filesystem.
        """
        save_path = path or self.auto_save_path or f"{self.name}.umc"
        result = self.memory.save(save_path)
        print(f"[agent] saved → {result or save_path}")
        return result or save_path

    def close(self, save: bool = True) -> None:
        """Disconnect from the server. Optionally save first."""
        if save and self.auto_save_path:
            try:
                self.save(self.auto_save_path)
            except Exception as e:
                print(f"[agent] auto-save failed: {e}")
        try:
            self.memory.disconnect()
            print(f"[agent] '{self.name}' disconnected")
        except Exception:
            pass

    # ── Context manager ───────────────────────────────────────────────────

    def __enter__(self) -> "NetworkAgent":
        return self

    def __exit__(self, *_) -> None:
        self.close(save=bool(self.auto_save_path))

    def __repr__(self) -> str:
        return (
            f"<NetworkAgent '{self.name}' "
            f"server={self.server} "
            f"backend={self._backend}>"
        )


# ── Interactive CLI ───────────────────────────────────────────────────────────

def run_interactive(agent: NetworkAgent):
    """
    Simple REPL loop. Special commands:
      /stats      — show memory statistics
      /tasks      — list pending tasks
      /done <id>  — mark task done
      /save       — save capsule now
      /memory <q> — inspect what the agent knows about a topic
      /quit       — save and exit
    """
    print(f"\n{'─' * 60}")
    print(f"  uMemory Agent — '{agent.name}'")
    print(f"  Server: {agent.server}")
    try:
        s = agent.stats()
        print(f"  Memory: {s.get('latest_nodes', 0)} active nodes")
    except Exception:
        pass
    print(f"{'─' * 60}")
    print("  Commands: /stats  /tasks  /done <id>  /save  /memory <q>  /quit")
    print(f"{'─' * 60}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[agent] Interrupted. Saving and exiting...")
            break

        if not user_input:
            continue

        # ── Special commands ──────────────────────────────────────────
        if user_input.startswith("/"):
            parts = user_input.split(maxsplit=1)
            cmd   = parts[0].lower()
            arg   = parts[1].strip() if len(parts) > 1 else ""

            if cmd == "/quit":
                break

            elif cmd == "/stats":
                try:
                    s = agent.stats()
                    print(f"  Nodes: {s.get('latest_nodes', '?')} active / "
                          f"{s.get('total_nodes', '?')} total | "
                          f"Vectors: {s.get('vector_count', '?')}")
                except Exception as e:
                    print(f"  Error: {e}")

            elif cmd == "/tasks":
                tasks = agent.pending_tasks()
                if not tasks:
                    print("  No pending tasks.")
                else:
                    for t in tasks:
                        due = f"  due {t['due_date']}" if t.get("due_date") else ""
                        print(f"  [{t['id'][:8]}] {t.get('title', t.get('content',''))}{due}")

            elif cmd == "/done":
                if not arg:
                    print("  Usage: /done <node_id>")
                else:
                    ok = agent.complete_task(arg)
                    print(f"  {'✓ marked done' if ok else '✗ task not found'}: {arg[:8]}")

            elif cmd == "/save":
                try:
                    path = agent.save()
                    print(f"  Saved → {path}")
                except Exception as e:
                    print(f"  Save failed: {e}")

            elif cmd == "/memory":
                if not arg:
                    print("  Usage: /memory <topic>")
                else:
                    mems = agent.what_do_you_know_about(arg)
                    if not mems:
                        print("  (nothing found)")
                    else:
                        for m in mems:
                            print(f"  · {m}")

            else:
                print(f"  Unknown command: {cmd}")

            continue

        # ── Normal chat ───────────────────────────────────────────────
        try:
            reply = agent.chat(user_input)
            print(f"\nAssistant: {reply}\n")
        except Exception as e:
            print(f"\n[error] {e}\n")

    # End of loop — save and disconnect
    agent.close(save=True)
    print("Goodbye.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="uMemory Network Agent — conversational agent with server-backed memory"
    )
    parser.add_argument("--name",   default="agent",              help="Agent name / session ID")
    parser.add_argument("--server", default="http://localhost:8000", help="uMemory server URL")
    parser.add_argument("--socket", default="default",            help="Memory socket / domain name")
    parser.add_argument("--load",   default=None,                 help="Path to .umc file to resume from")
    parser.add_argument("--save",   default=None,                 help="Path to save capsule on exit")
    parser.add_argument("--key",    default=None,                 help="API key (if server requires one)")
    parser.add_argument("--model",  default=None,                 help="Override LLM model (e.g. gpt-4o)")
    args = parser.parse_args()

    # ── Backend selection ─────────────────────────────────────────────
    backend: Optional[LLMBackend] = None
    if args.model:
        if os.getenv("ANTHROPIC_API_KEY"):
            backend = AnthropicBackend(model=args.model)
        elif os.getenv("OPENAI_API_KEY"):
            backend = OpenAIBackend(model=args.model)

    # ── Create or resume agent ────────────────────────────────────────
    if args.load:
        agent = NetworkAgent.resume(
            capsule_path=args.load,
            name=args.name,
            server=args.server,
            socket_name=args.socket,
            backend=backend,
            api_key=args.key,
        )
    else:
        agent = NetworkAgent(
            name=args.name,
            server=args.server,
            socket_name=args.socket,
            backend=backend,
            api_key=args.key,
        )

    # Auto-save path
    agent.auto_save_path = args.save or args.load or f"{args.name}.umc"

    run_interactive(agent)
