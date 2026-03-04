# server.py — uMemory HTTP server
#
# Run:
#   uvicorn server:app --host 0.0.0.0 --port 8000
#
# Auto-load a capsule on startup:
#   DEFAULT_CAPSULE=./my.umc uvicorn server:app --port 8000
#
# Set an API key:
#   MCP_API_KEY=secret uvicorn server:app --port 8000

import asyncio
import os
import threading
import time
import uuid
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel

from memory_socket import MemorySocket, AccessMode

app = FastAPI(title="uMemory Server")

# ── Server state ──────────────────────────────────────────────────────────────
_SERVER_STATE: Dict = {
    "sockets": {},   # socket_name → MemorySocket
    "handles": {},   # handle_id   → {"handle": MemoryHandle, "socket_name": str}
}
_STATE_LOCK = threading.Lock()
MCP_API_KEY: Optional[str] = os.getenv("MCP_API_KEY")  # None = no auth


# ── Auth ──────────────────────────────────────────────────────────────────────
def _auth(key: Optional[str]):
    if MCP_API_KEY is not None and key != MCP_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def _get_handle(handle_id: str):
    rec = _SERVER_STATE["handles"].get(handle_id)
    if not rec:
        raise HTTPException(status_code=404, detail="handle not found")
    return rec["handle"]


# ── Startup: auto-load capsule from env ──────────────────────────────────────
@app.on_event("startup")
def _startup():
    path = os.getenv("DEFAULT_CAPSULE")
    if path and os.path.exists(path):
        from capsule import MemoryCapsule
        cap = MemoryCapsule.load(path)
        sock = MemorySocket(capsule=cap, name=cap.name)
        _SERVER_STATE["sockets"]["default"] = sock
        print(f"[server] auto-loaded capsule: {path} ({len(cap.graph.nodes)} nodes)")
    else:
        # Always ensure a default socket exists
        _SERVER_STATE["sockets"]["default"] = MemorySocket(name="default")
        print("[server] started with a fresh default socket")


# ── Request / response models ─────────────────────────────────────────────────
class ConnectRequest(BaseModel):
    socket_name: str = "default"
    agent_id: str
    mode: str = AccessMode.READ_WRITE


class DisconnectRequest(BaseModel):
    handle_id: str


class IngestRequest(BaseModel):
    handle_id: str
    text: str
    hint: Optional[str] = None
    verbose: bool = False


class QueryRequest(BaseModel):
    handle_id: str
    query: str
    top_k: int = 5


class ExportRequest(BaseModel):
    handle_id: str
    path: Optional[str] = None


class LoadRequest(BaseModel):
    socket_name: str = "default"
    path: str  # server-side path to a .umc file


class ConsolidateRequest(BaseModel):
    handle_id: str
    min_episodes: int = 3
    regenerate_kb: bool = True


class TaskDoneRequest(BaseModel):
    handle_id: str
    node_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "ok": True,
        "time": time.time(),
        "sockets": list(_SERVER_STATE["sockets"].keys()),
        "active_handles": len(_SERVER_STATE["handles"]),
    }


@app.post("/connect")
def connect(req: ConnectRequest, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    with _STATE_LOCK:
        sock = _SERVER_STATE["sockets"].get(req.socket_name)
        if sock is None:
            sock = MemorySocket(name=req.socket_name)
            _SERVER_STATE["sockets"][req.socket_name] = sock

        handle = sock.connect(req.agent_id, mode=req.mode)
        hid = str(uuid.uuid4())
        _SERVER_STATE["handles"][hid] = {
            "handle":      handle,
            "socket_name": req.socket_name,
        }

    return {"handle_id": hid, "agent_id": req.agent_id, "mode": req.mode}


@app.post("/disconnect")
def disconnect(req: DisconnectRequest, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    with _STATE_LOCK:
        rec = _SERVER_STATE["handles"].pop(req.handle_id, None)
    if not rec:
        raise HTTPException(status_code=404, detail="handle not found")
    try:
        rec["handle"].disconnect()
    except Exception:
        pass
    return {"ok": True}


@app.post("/ingest")
async def ingest(req: IngestRequest, x_api_key: Optional[str] = Header(None)):
    """
    Ingest is async — LLM calls (extract, classify, detect_relationship) are
    offloaded to a thread pool so they don't block the event loop.
    """
    _auth(x_api_key)
    handle = _get_handle(req.handle_id)
    try:
        nodes = await asyncio.to_thread(
            handle.ingest, req.text, req.hint, req.verbose
        )
        return {
            "ingested": len(nodes),
            "nodes": [
                {"id": n.id[:8], "content": n.content, "mem_type": n.mem_type}
                for n in nodes
            ],
        }
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query")
def query(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    handle = _get_handle(req.handle_id)
    try:
        results = handle.query(req.query, top_k=req.top_k)
        return {"results": results}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/query_with_context")
def query_with_context(req: QueryRequest, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    handle = _get_handle(req.handle_id)
    try:
        results = handle.query_with_context(req.query, top_k=req.top_k)
        return {"results": results}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))


@app.post("/export")
def export_capsule(req: ExportRequest, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    handle = _get_handle(req.handle_id)
    try:
        out_path = handle.export_snapshot(req.path)
        return {"ok": True, "exported": out_path}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
def load_capsule(req: LoadRequest, x_api_key: Optional[str] = Header(None)):
    """
    Load a .umc file from the server filesystem into a named socket slot.
    Replaces whatever was in that slot. All existing handles to that socket
    automatically see the new capsule — they share the same MemorySocket object.
    """
    _auth(x_api_key)
    if not os.path.exists(req.path):
        raise HTTPException(status_code=404, detail=f"File not found: {req.path}")
    try:
        from capsule import MemoryCapsule
        cap = MemoryCapsule.load(req.path)
        with _STATE_LOCK:
            existing = _SERVER_STATE["sockets"].get(req.socket_name)
            if existing:
                # Swap the capsule in-place so live handles still work
                existing._capsule = cap
                existing.name     = cap.name
            else:
                sock = MemorySocket(capsule=cap, name=cap.name)
                _SERVER_STATE["sockets"][req.socket_name] = sock
        return {
            "ok":     True,
            "socket": req.socket_name,
            "nodes":  len(cap.graph.nodes),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/consolidate")
async def consolidate(req: ConsolidateRequest, x_api_key: Optional[str] = Header(None)):
    """Consolidate episodic memories into stable derived facts. LLM-heavy, runs async."""
    _auth(x_api_key)
    handle = _get_handle(req.handle_id)
    try:
        count = await asyncio.to_thread(
            handle.consolidate, req.min_episodes, req.regenerate_kb
        )
        return {"ok": True, "groups_consolidated": count}
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tasks")
def get_tasks(handle_id: str, x_api_key: Optional[str] = Header(None)):
    """Return all pending tasks from memory, sorted by due date."""
    _auth(x_api_key)
    handle = _get_handle(handle_id)
    try:
        tasks = handle._socket.capsule.get_pending_tasks()
        return {
            "count": len(tasks),
            "tasks": [
                {
                    "id":       t.id,
                    "title":    t.task_title or t.content[:60],
                    "content":  t.content,
                    "due_date": t.due_date,
                    "status":   t.task_status,
                }
                for t in tasks
            ],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/tasks/done")
def mark_task_done(req: TaskDoneRequest, x_api_key: Optional[str] = Header(None)):
    """Mark a task as completed by node_id."""
    _auth(x_api_key)
    handle = _get_handle(req.handle_id)
    ok = handle._socket.capsule.mark_task_done(req.node_id)
    if not ok:
        raise HTTPException(status_code=404, detail=f"Task '{req.node_id}' not found")
    return {"ok": True, "node_id": req.node_id}


@app.get("/stats")
def stats(handle_id: str, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    handle = _get_handle(handle_id)
    return handle.stats()
