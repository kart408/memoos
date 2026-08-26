"""
MemoOS API — the terminal's memory, and the window onto it.

    uvicorn api:app --reload       (or: memoos serve)

Two audiences share these endpoints. The CLI writes through them
indirectly, by way of the same core; the dashboard at `/` reads through
them directly. Nothing here generates text — MemoOS extracts, connects
and retrieves. There is no chat endpoint because there is no chatbot.

Everything is scoped to a *container*: the git project the work happened
in. Projects never see each other's memory.
"""

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from memoos_core import config, connection, quick
from memoos_core.journal import Journal, container_for
from memoos_core.models import MemoryStatus
from memoos_core.terminal import TerminalMemory, import_claude_sessions

app = FastAPI(title="MemoOS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD = Path(__file__).parent / "demo.html"

# A container name becomes a filename, so it accepts only [A-Za-z0-9._-]
# and must end alphanumerically. Rejecting it here turns what would be an
# opaque failure deep in the store into an answer the caller can act on.
_VALID = re.compile(r"^[A-Za-z0-9._-]{0,502}[A-Za-z0-9]$")

_layers: Dict[str, TerminalMemory] = {}


def layer(container: str) -> TerminalMemory:
    """
    The terminal memory for one project, cached across requests.

    Cached because constructing one is cheap but *using* one loads the
    embedding model, and paying three and a half seconds per request
    would make the dashboard unusable.
    """
    if not _VALID.match(container):
        raise HTTPException(
            400,
            "container names may use letters, digits, dot, dash and underscore, "
            "and must end with a letter or digit",
        )
    # Normalised before it is cached, so "MyProj" and "myproj" are one
    # entry pointing at the one file, rather than two layers writing two
    # sets of rows into it under labels that never see each other.
    container = config.safe_container(container)
    if container not in _layers:
        _layers[container] = TerminalMemory(container=container)
    return _layers[container]


# --------------------------------------------------------------- projects


@app.get("/projects")
def projects() -> Dict[str, Any]:
    """Every project the journal has seen, and the one we're standing in."""
    journal = Journal()
    seen = journal.containers()
    here = container_for()
    if here not in {row["container"] for row in seen}:
        seen.insert(0, {"container": here, "events": 0, "last_seen": None})
    for row in seen:
        row.update(quick.counts(row["container"]))
    return {"here": here, "projects": seen}


@app.get("/projects/{container}/stats")
def stats(container: str) -> Dict[str, Any]:
    journal = Journal()
    data = quick.counts(container)
    data["sessions"] = len(journal.sessions(container, limit=10_000))
    data["pending"] = len(journal.events(container, undistilled_only=True, limit=10_000))
    data["container"] = container
    return data


# ---------------------------------------------------------------- memory


@app.get("/projects/{container}/memories")
def memories(container: str, limit: int = 200,
             status: Optional[str] = "active") -> Dict[str, Any]:
    """
    What this project's memory layer holds.

    `status=all` includes superseded rows — how the dashboard shows that
    a fact which stopped being true was retired rather than overwritten.
    """
    memo = layer(container).memo
    if status in (None, "", "all"):
        wanted = None
    else:
        try:
            wanted = MemoryStatus(status)
        except ValueError:
            raise HTTPException(400, f"unknown status {status!r}")

    rows = memo.all(limit=limit, status=wanted)
    return {"memories": rows,
            "strengths": {m.id: memo.strength_of(m.id) for m in rows}}


@app.delete("/projects/{container}/memories/{memory_id}")
def forget(container: str, memory_id: str) -> Dict[str, str]:
    memo = layer(container).memo
    if memo.get(memory_id) is None:
        raise HTTPException(404, "no such memory")
    memo.delete(memory_id)
    return {"deleted": memory_id}


@app.post("/projects/{container}/reset")
def reset(container: str) -> Dict[str, int]:
    """Empty one project's memory and journal. Other projects are untouched."""
    memo = layer(container).memo
    existing = memo.all(limit=10_000, status=None)
    for memory in existing:
        memo.delete(memory.id)
    events = Journal().forget(container)
    return {"memories": len(existing), "events": events}


# ----------------------------------------------------------------- graph


@app.get("/projects/{container}/graph")
def graph(container: str, limit: int = 60) -> Dict[str, Any]:
    """
    The memory graph, shaped for drawing.

    Two kinds of node and two kinds of edge, because the graph is only
    interesting when both are present. Entity-to-entity *relations* are
    the sparse, high-value edges — they say how two named things are
    connected. Memory-to-entity *mentions* are the dense ones, and they
    are what makes a project's shape visible: six memories hanging off
    api.py is a fact about the work, not about the schema.

    Ids are stable strings the client can key on; nothing here requires
    the view to resolve a database id itself.
    """
    entities = quick.top_entities(container, limit=limit)
    known = {e["name"] for e in entities}

    nodes: List[Dict[str, Any]] = [
        {"id": f"e:{e['name']}", "label": e["name"], "kind": "entity",
         "type": e["entity_type"], "weight": e["mention_count"]}
        for e in entities
    ]
    edges: List[Dict[str, Any]] = [
        {"source": f"e:{r['subject']}", "target": f"e:{r['object']}",
         "label": r["predicate"], "kind": "relation"}
        for r in quick.relations(container, limit=limit * 4)
        # An edge to a node that didn't make the cut would dangle.
        if r["subject"] in known and r["object"] in known
    ]

    seen_memories: Dict[str, bool] = {}
    for link in quick.memory_links(container, limit=limit * 8):
        if link["entity"] not in known:
            continue
        node_id = f"m:{link['memory_id']}"
        if node_id not in seen_memories:
            seen_memories[node_id] = True
            nodes.append({"id": node_id, "label": link["text"], "kind": "memory",
                          "type": link["memory_type"], "weight": 1})
        edges.append({"source": node_id, "target": f"e:{link['entity']}",
                      "label": "", "kind": "mentions"})

    return {"nodes": nodes, "edges": edges,
            "counts": {"entities": len(entities), "memories": len(seen_memories),
                       "relations": sum(1 for e in edges if e["kind"] == "relation")}}


@app.get("/projects/{container}/entities/{name}")
def about(container: str, name: str) -> Dict[str, Any]:
    return layer(container).memo.about(name)


# ------------------------------------------------------------- retrieval


@app.get("/projects/{container}/search")
def search(container: str, q: str = Query(..., min_length=1),
           top_k: int = 8) -> Dict[str, Any]:
    """
    Hybrid search, with the ranking explained.

    `touch=False`: inspecting the store must not reinforce what it
    returns, or looking at retrieval would quietly reshape it.
    """
    results = layer(container).memo.search(q, top_k=top_k, touch=False)
    return {"query": q, "results": results}


@app.get("/projects/{container}/context")
def context(container: str, q: str = Query(..., min_length=1),
            top_k: int = 5, include_stale: bool = False) -> Dict[str, Any]:
    """
    The handover endpoint: what an agent should know before this task.

    This is where MemoOS stops. It returns the context block, the
    memories behind it, and the expansion that found them — and nothing
    generated. The agent on the other end does the work and does the
    answering; it is the only thing that knows what the user actually
    asked for.
    """
    return layer(container).memo.context_for(
        q, top_k=top_k, include_stale=include_stale
    )


@app.get("/projects/{container}/signals")
def project_signals(container: str, limit: int = 25) -> Dict[str, Any]:
    """
    What has gone wrong here before, and what fixed it.

    Reported, never acted on. MemoOS does not decide what to do about a
    known failure — the agent holding the task does, because it is the
    only thing that can see what the task actually is.
    """
    episodes = layer(container).memo.signals(limit=limit)
    return {
        "container": container,
        "open": [e.as_dict() for e in episodes if not e.resolved],
        "resolved": [e.as_dict() for e in episodes if e.resolved],
    }


@app.get("/projects/{container}/memories/{memory_id}/source")
def memory_source(container: str, memory_id: str) -> Dict[str, Any]:
    """Trace a memory back to the passage and the events it came from."""
    trail = layer(container).memo.source(memory_id)
    if trail is None:
        raise HTTPException(404, "no such memory")
    return trail


# -------------------------------------------------------------- journal


@app.get("/projects/{container}/sessions")
def sessions(container: str, limit: int = 30) -> Dict[str, Any]:
    return {"sessions": Journal().sessions(container, limit=limit)}


@app.get("/projects/{container}/events")
def events(container: str, session: Optional[str] = None,
           limit: int = 300) -> Dict[str, Any]:
    return {"events": Journal().events(container, session_id=session, limit=limit)}


@app.post("/projects/{container}/note")
def note(container: str, payload: Dict[str, str]) -> Dict[str, str]:
    text = (payload.get("text") or "").strip()
    if not text:
        raise HTTPException(400, "note text is required")
    return {"id": Journal().record("note", text, container=container,
                                   session_id="dashboard")}


@app.post("/projects/{container}/distill")
def distill(container: str, session: Optional[str] = None) -> Dict[str, Any]:
    """
    Fold journalled events into memories and graph structure.

    Slow on purpose — this is the one place the extraction model runs.
    """
    result = layer(container).distill(session_id=session)
    return {
        "distilled": result["distilled"],
        "digest": result.get("digest", ""),
        "skipped": result.get("skipped"),
        "created": result.get("created", []),
        "entities": result.get("entities", []),
        "summary": result.get("summary", ""),
        # Reported, not hidden: a partial distil leaves its events in the
        # journal to be retried, and the caller should be told so rather
        # than shown a shorter list and left to assume that was all of it.
        "chunks_total": result.get("chunks_total", 0),
        "chunks_failed": result.get("chunks_failed", 0),
        "retained": result.get("retained", 0),
    }


@app.post("/projects/{container}/claude-import")
def claude_import(container: str, newest: int = 3) -> Dict[str, Any]:
    """Journal what was asked of Claude Code in this project's recent sessions."""
    return import_claude_sessions(layer(container), newest=newest)


# ------------------------------------------------------------------- kb


@app.post("/projects/{container}/ingest")
async def ingest(container: str, file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Upload a document so the layer knows about you or the project.

    Written to a temp file rather than decoded in memory so ingestion
    goes through exactly the same path as `memoos ingest`, rather than a
    second one that could drift from it.
    """
    raw = await file.read()
    if not raw.strip():
        raise HTTPException(400, "that file is empty")

    suffix = os.path.splitext(file.filename or "upload.txt")[1] or ".txt"
    handle, path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(raw)
        result = layer(container).ingest_kb(path)
    finally:
        os.unlink(path)

    return {"filename": file.filename, "created": result["created"],
            "entities": result.get("entities", []), "summary": result["summary"]}


# ------------------------------------------------------------ connection


@app.get("/connection")
def connection_status() -> Dict[str, Any]:
    """Whether terminals are currently feeding the memory layer."""
    return connection.status()


@app.post("/connection")
def set_connection(payload: Dict[str, bool]) -> Dict[str, Any]:
    """
    Attach or detach the terminal, for every shell at once.

    Only recording is switched. Everything else on this API keeps working
    while disconnected, because declining to record today says nothing
    about what was learned yesterday.
    """
    if "connected" not in payload:
        raise HTTPException(400, "expected {\"connected\": true|false}")
    if payload["connected"]:
        connection.connect()
    else:
        connection.disconnect()
    return connection.status()


# --------------------------------------------------------------- serving


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def dashboard() -> FileResponse:
    if not DASHBOARD.exists():
        raise HTTPException(404, "demo.html is missing")
    return FileResponse(DASHBOARD)
