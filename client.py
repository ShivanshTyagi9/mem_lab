# client.py — uMemory HTTP client
#
# Usage:
#   from client import connect_to_server
#   memory = connect_to_server("http://localhost:8000", agent_id="my_agent")
#   memory.ingest("Alex works at Stripe as VP of Product.")
#   memory.query("Where does Alex work?")
#   memory.save("./alex.umc")

import requests
from typing import Optional, List, Dict, Any


class MemoryHandleProxy:
    """
    Client-side proxy for a MemoryHandle living on the server.
    Mirrors the MemoryHandle API so network_agent.py can use it
    identically to a local handle.
    """

    def __init__(self, base_url: str, handle_id: str, api_key: Optional[str] = None):
        self.base_url  = base_url.rstrip("/")
        self.handle_id = handle_id
        self.api_key   = api_key

    # ── Internal helpers ──────────────────────────────────────────────────

    def _headers(self) -> dict:
        h = {}
        if self.api_key:
            h["X-API-KEY"] = self.api_key
        return h

    def _post(self, path: str, json: dict, timeout: int = 60) -> Any:
        res = requests.post(
            f"{self.base_url}{path}",
            json=json,
            headers=self._headers(),
            timeout=timeout,
        )
        res.raise_for_status()
        return res.json()

    def _get(self, path: str, params: dict = None, timeout: int = 10) -> Any:
        res = requests.get(
            f"{self.base_url}{path}",
            params=params or {},
            headers=self._headers(),
            timeout=timeout,
        )
        res.raise_for_status()
        return res.json()

    # ── Core memory operations ────────────────────────────────────────────

    def ingest(self, text: str, hint: Optional[str] = None, verbose: bool = False) -> dict:
        """Ingest text into memory. Returns {ingested: int, nodes: [...]}."""
        return self._post("/ingest", {
            "handle_id": self.handle_id,
            "text":      text,
            "hint":      hint,
            "verbose":   verbose,
        }, timeout=120)  # LLM extraction can take a moment

    def query(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve relevant memory strings. Returns a plain list."""
        data = self._post("/query", {
            "handle_id": self.handle_id,
            "query":     query,
            "top_k":     top_k,
        })
        return data.get("results", [])

    def query_with_context(self, query: str, top_k: int = 5) -> List[Dict]:
        """Rich retrieval with per-node metadata."""
        data = self._post("/query_with_context", {
            "handle_id": self.handle_id,
            "query":     query,
            "top_k":     top_k,
        })
        return data.get("results", [])

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def export_snapshot(self, path: Optional[str] = None) -> str:
        """Tell the server to save the capsule. Returns the saved path."""
        data = self._post("/export", {
            "handle_id": self.handle_id,
            "path":      path,
        }, timeout=30)
        return data.get("exported", "")

    def disconnect(self) -> dict:
        """Release this handle on the server."""
        return self._post("/disconnect", {"handle_id": self.handle_id}, timeout=10)

    def stats(self) -> dict:
        """Capsule + connection statistics."""
        return self._get("/stats", {"handle_id": self.handle_id})

    # ── Session management ────────────────────────────────────────────────

    def consolidate(self, min_episodes: int = 3, regenerate_kb: bool = True) -> dict:
        """
        Consolidate episodic memories into durable derived facts.
        Run at end of session or when memory feels noisy.
        """
        return self._post("/consolidate", {
            "handle_id":    self.handle_id,
            "min_episodes": min_episodes,
            "regenerate_kb": regenerate_kb,
        }, timeout=120)

    def get_tasks(self) -> List[Dict]:
        """Return pending tasks from memory, sorted by due date."""
        data = self._get("/tasks", {"handle_id": self.handle_id})
        return data.get("tasks", [])

    def mark_task_done(self, node_id: str) -> dict:
        """Mark a task as completed by its full node_id."""
        return self._post("/tasks/done", {
            "handle_id": self.handle_id,
            "node_id":   node_id,
        })

    # ── Convenience ───────────────────────────────────────────────────────

    def save(self, path: Optional[str] = None) -> str:
        """Alias for export_snapshot — more natural name for agent use."""
        return self.export_snapshot(path)

    def __repr__(self) -> str:
        return f"<MemoryHandleProxy handle={self.handle_id[:8]} url={self.base_url}>"


# ── Factory function ──────────────────────────────────────────────────────────

def connect_to_server(
    base_url: str,
    agent_id: str,
    socket_name: str = "default",
    mode: str = "rw",
    api_key: Optional[str] = None,
) -> MemoryHandleProxy:
    """
    Connect an agent to the uMemory server.

    Args:
      base_url:    Server URL, e.g. "http://localhost:8000"
      agent_id:    Unique name for this agent / session
      socket_name: Which memory domain to connect to (default: "default")
      mode:        "rw" (read+write), "r" (read-only), "w" (write-only)
      api_key:     Optional API key if server has MCP_API_KEY set

    Returns:
      MemoryHandleProxy — use like a local MemoryHandle
    """
    headers = {"X-API-KEY": api_key} if api_key else {}
    res = requests.post(
        f"{base_url.rstrip('/')}/connect",
        json={"socket_name": socket_name, "agent_id": agent_id, "mode": mode},
        headers=headers,
        timeout=10,
    )
    res.raise_for_status()
    data = res.json()
    return MemoryHandleProxy(base_url, data["handle_id"], api_key=api_key)


def load_capsule_on_server(
    base_url: str,
    path: str,
    socket_name: str = "default",
    api_key: Optional[str] = None,
) -> dict:
    """
    Tell the server to load a .umc file into a named socket slot.
    The path must be accessible from the server's filesystem.

    Call this before connect_to_server() when you want to resume a session.
    """
    headers = {"X-API-KEY": api_key} if api_key else {}
    res = requests.post(
        f"{base_url.rstrip('/')}/load",
        json={"socket_name": socket_name, "path": path},
        headers=headers,
        timeout=30,
    )
    res.raise_for_status()
    return res.json()
