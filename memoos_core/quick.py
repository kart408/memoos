"""
The read half of the fast path.

Memories, entities and relations all live in SQLite; only *semantic*
search needs the vector store. So the question a terminal asks when it
opens — "what was I doing here?" — is answerable with plain SQL, and
answering it that way costs about ten milliseconds instead of the three
and a half seconds it takes to load Chroma and an embedding model.

Same constraint as `journal`: stdlib only, `config` the sole import.
Anything here that starts needing embeddings belongs in `memory` instead.
"""

import os
import sqlite3
from typing import Any, Dict, List, Optional

from . import config


def _connect(container: str, path: Optional[str] = None) -> sqlite3.Connection:
    """
    Open the file this container's data lives in, read-only in spirit.

    Every function here already took a container; now it decides the file
    as well, because a container *is* a file. `sqlite3.connect` would
    happily create an empty database for a container that has never been
    seen, so callers get `_has_tables` as the guard against reading one.
    """
    target = path or config.db_path(container=container)
    if not os.path.exists(target):
        # `sqlite3.connect` creates whatever it is pointed at, and every
        # function here is a read. Left alone, asking what an unknown
        # project remembers would *create* that project — one empty file
        # per directory anyone ever ran `memoos status` in. An in-memory
        # database gives the same answer (nothing) and leaves no trace.
        target = ":memory:"
    conn = sqlite3.connect(target, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _has_tables(conn: sqlite3.Connection) -> bool:
    """True once the real schema exists — the journal may have got here first."""
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='memories'"
    ).fetchone()
    return row is not None


def recent_memories(container: str, limit: int = 8,
                    path: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect(container, path) as conn:
        if not _has_tables(conn):
            return []
        rows = conn.execute(
            """SELECT text, memory_type, importance, created_at, access_count
                 FROM memories
                WHERE container = ? AND status = 'active'
             ORDER BY created_at DESC LIMIT ?""",
            (container, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def top_entities(container: str, limit: int = 12,
                 path: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect(container, path) as conn:
        if not _has_tables(conn):
            return []
        rows = conn.execute(
            """SELECT name, entity_type, mention_count
                 FROM entities
                WHERE container = ? AND norm_name != '__user__'
             ORDER BY mention_count DESC, name LIMIT ?""",
            (container, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def relations(container: str, limit: int = 60,
              path: Optional[str] = None) -> List[Dict[str, Any]]:
    with _connect(container, path) as conn:
        if not _has_tables(conn):
            return []
        rows = conn.execute(
            """SELECT s.name AS subject, r.predicate, o.name AS object
                 FROM relations r
                 JOIN entities s ON s.id = r.subject_id
                 JOIN entities o ON o.id = r.object_id
                WHERE r.container = ? AND r.invalidated_at IS NULL
             ORDER BY r.created_at DESC LIMIT ?""",
            (container, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def counts(container: str, path: Optional[str] = None) -> Dict[str, int]:
    with _connect(container, path) as conn:
        if not _has_tables(conn):
            return {"memories": 0, "entities": 0, "relations": 0}
        one = lambda sql: conn.execute(sql, (container,)).fetchone()[0]
        return {
            "memories": one("SELECT COUNT(*) FROM memories WHERE container=? AND status='active'"),
            "entities": one("SELECT COUNT(*) FROM entities WHERE container=? AND norm_name!='__user__'"),
            "relations": one("SELECT COUNT(*) FROM relations WHERE container=? AND invalidated_at IS NULL"),
        }


def memory_links(container: str, limit: int = 400,
                 path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Which memory mentions which entity.

    Entity-to-entity relations are the sparse part of the graph: they
    only appear when one sentence names two things and states how they
    connect. Mentions are the dense part, and they are what makes the
    graph legible — every memory hangs off the things it talks about, so
    you can see at a glance that six memories all concern api.py.
    """
    with _connect(container, path) as conn:
        if not _has_tables(conn):
            return []
        rows = conn.execute(
            """SELECT me.memory_id, m.text, m.memory_type, e.name AS entity
                 FROM memory_entities me
                 JOIN memories m ON m.id = me.memory_id
                 JOIN entities e ON e.id = me.entity_id
                WHERE m.container = ? AND m.status = 'active'
                      AND e.norm_name != '__user__'
             ORDER BY m.created_at DESC LIMIT ?""",
            (container, limit),
        ).fetchall()
    return [dict(r) for r in rows]
