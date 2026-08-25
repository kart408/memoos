"""
SQLite: the source of truth.

Chroma holds vectors. SQLite holds everything you'd actually want to ask
a question about — the memory rows, their supersession history, the
entity graph, and an FTS5 index for keyword search. Splitting it this way
means a corrupted or rebuilt vector index costs a re-embed, not data.

It also buys the two things a pure vector store can't give you:
  - exact lookups and filters (by container, type, status, document)
  - BM25 keyword search, the other half of hybrid retrieval

Everything is scoped by `container` — the tenant boundary. There is no
query in this module that reads across containers.
"""

import json
import os
import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import (
    Document,
    DocumentStatus,
    Entity,
    EntityType,
    Memory,
    MemoryStatus,
    MemoryType,
    Relation,
)
from .text_utils import normalise_name, significant_tokens

SCHEMA_VERSION = 2

_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id           TEXT PRIMARY KEY,
    container    TEXT NOT NULL,
    title        TEXT,
    uri          TEXT,
    source       TEXT NOT NULL DEFAULT 'user',
    raw_text     TEXT NOT NULL DEFAULT '',
    status       TEXT NOT NULL DEFAULT 'queued',
    error        TEXT,
    chunk_count  INTEGER NOT NULL DEFAULT 0,
    memory_count INTEGER NOT NULL DEFAULT 0,
    created_at   TEXT NOT NULL,
    updated_at   TEXT NOT NULL,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_documents_container
    ON documents(container, created_at DESC);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    container   TEXT NOT NULL,
    ordinal     INTEGER NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id, ordinal);

CREATE TABLE IF NOT EXISTS memories (
    id               TEXT PRIMARY KEY,
    container        TEXT NOT NULL,
    text             TEXT NOT NULL,
    memory_type      TEXT NOT NULL DEFAULT 'fact',
    source           TEXT NOT NULL DEFAULT 'user',
    document_id      TEXT,
    chunk_id         TEXT,
    importance       REAL NOT NULL DEFAULT 0.5,
    confidence       REAL NOT NULL DEFAULT 0.8,
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL,
    last_accessed_at TEXT,
    access_count     INTEGER NOT NULL DEFAULT 0,
    status           TEXT NOT NULL DEFAULT 'active',
    superseded_by    TEXT,
    superseded_at    TEXT,
    supersede_reason TEXT,
    metadata         TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_memories_container_status
    ON memories(container, status, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_memories_document ON memories(document_id);
CREATE INDEX IF NOT EXISTS idx_memories_superseded_by ON memories(superseded_by);

-- External-content FTS index: the text lives in `memories`, this table
-- only holds the inverted index. Triggers keep the two in step.
CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(
    text,
    content='memories',
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS memories_fts_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text)
        VALUES ('delete', old.rowid, old.text);
END;
CREATE TRIGGER IF NOT EXISTS memories_fts_au AFTER UPDATE OF text ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, text)
        VALUES ('delete', old.rowid, old.text);
    INSERT INTO memories_fts(rowid, text) VALUES (new.rowid, new.text);
END;

CREATE TABLE IF NOT EXISTS entities (
    id            TEXT PRIMARY KEY,
    container     TEXT NOT NULL,
    name          TEXT NOT NULL,
    norm_name     TEXT NOT NULL,
    entity_type   TEXT NOT NULL DEFAULT 'other',
    mention_count INTEGER NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    UNIQUE (container, norm_name, entity_type)
);
CREATE INDEX IF NOT EXISTS idx_entities_container ON entities(container, norm_name);

CREATE TABLE IF NOT EXISTS memory_entities (
    memory_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    entity_id TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    PRIMARY KEY (memory_id, entity_id)
);
CREATE INDEX IF NOT EXISTS idx_memory_entities_entity ON memory_entities(entity_id);

CREATE TABLE IF NOT EXISTS relations (
    id             TEXT PRIMARY KEY,
    container      TEXT NOT NULL,
    subject_id     TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    predicate      TEXT NOT NULL,
    object_id      TEXT NOT NULL REFERENCES entities(id) ON DELETE CASCADE,
    memory_id      TEXT REFERENCES memories(id) ON DELETE SET NULL,
    confidence     REAL NOT NULL DEFAULT 0.8,
    created_at     TEXT NOT NULL,
    invalidated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(container, subject_id);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(container, object_id);
CREATE INDEX IF NOT EXISTS idx_relations_memory ON relations(memory_id);
"""

# Kept apart from `_SCHEMA` because `vectors.VectorIndex` opens its own
# connection to this same file and applies it there too. One definition,
# so the two can never drift into disagreeing about the table they share.
#
# The vector is a raw float32 buffer rather than JSON: 384 floats cost
# 1536 bytes packed against roughly 8KB of text, and unpacking is a
# memoryview rather than a parse. `model` is recorded so a change of
# embedding model is detectable — vectors from two different models are
# not comparable, and silently mixing them makes search quietly wrong
# rather than loudly broken.
EMBEDDINGS_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    memory_id  TEXT PRIMARY KEY REFERENCES memories(id) ON DELETE CASCADE,
    container  TEXT NOT NULL,
    dim        INTEGER NOT NULL,
    model      TEXT NOT NULL DEFAULT '',
    vector     BLOB NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_embeddings_container ON embeddings(container, dim);
"""

_SCHEMA = _SCHEMA + EMBEDDINGS_SCHEMA

_WORD = re.compile(r"[A-Za-z0-9]+")


# ------------------------------------------------------------- helpers

def _iso(value: Optional[datetime]) -> Optional[str]:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat()


def _dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _json_loads(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        loaded = json.loads(value)
        return loaded if isinstance(loaded, dict) else {}
    except json.JSONDecodeError:
        return {}


def normalise_entity_name(name: str) -> str:
    """The matching key for an entity: case- and punctuation-insensitive."""
    return normalise_name(name)


def build_fts_query(text: str) -> Optional[str]:
    """
    Turn free text into a safe, discriminating FTS5 MATCH expression.

    Two problems to solve at once:

    Safety — user text goes straight into MATCH, and FTS5 treats '*',
    '"', ':', '-' and bare AND/OR/NOT as query syntax, so an unescaped
    question is both a crash risk and a correctness bug. Quoting each
    token individually sidesteps all of it.

    Discrimination — stopwords must go. Every memory is a statement about
    the user, so ORing in tokens like "user", "the" or "where" makes
    every memory match every query. BM25 then ranks a set that contains
    everything, and the keyword half of hybrid search degenerates into
    noise that drags the vector half down with it during fusion.

    Returns None when nothing discriminating survives, which correctly
    leaves that query to vector search alone.
    """
    words = [t for t in significant_tokens(text, min_length=2)]
    if not words:
        return None
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(words))


class Database:
    """Synchronous SQLite access, safe to share across threads."""

    def __init__(self, path: str):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

        # check_same_thread=False + an explicit lock: FastAPI serves from a
        # thread pool, and one connection guarded by a lock is simpler and
        # faster here than a connection per thread.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._migrate()

    def _migrate(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            self._conn.execute(
                "INSERT OR REPLACE INTO schema_meta(key, value) VALUES ('version', ?)",
                (str(SCHEMA_VERSION),),
            )
            self._conn.commit()

    @contextmanager
    def transaction(self):
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --------------------------------------------------------- documents

    def insert_document(self, document: Document) -> Document:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO documents
                   (id, container, title, uri, source, raw_text, status, error,
                    chunk_count, memory_count, created_at, updated_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    document.id, document.container, document.title, document.uri,
                    document.source, document.raw_text, document.status.value,
                    document.error, document.chunk_count, document.memory_count,
                    _iso(document.created_at), _iso(document.updated_at),
                    json.dumps(document.metadata),
                ),
            )
        return document

    def update_document(self, document_id: str, *, status: Optional[DocumentStatus] = None,
                        error: Optional[str] = None, chunk_count: Optional[int] = None,
                        memory_count: Optional[int] = None) -> None:
        fields, values = [], []
        if status is not None:
            fields.append("status = ?")
            values.append(status.value)
        if error is not None:
            fields.append("error = ?")
            values.append(error)
        if chunk_count is not None:
            fields.append("chunk_count = ?")
            values.append(chunk_count)
        if memory_count is not None:
            fields.append("memory_count = ?")
            values.append(memory_count)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(_iso(datetime.now(timezone.utc)))
        values.append(document_id)
        with self.transaction() as conn:
            conn.execute(f"UPDATE documents SET {', '.join(fields)} WHERE id = ?", values)

    def get_document(self, document_id: str) -> Optional[Document]:
        row = self._conn.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def list_documents(self, container: str, limit: int = 50) -> List[Document]:
        rows = self._conn.execute(
            "SELECT * FROM documents WHERE container = ? ORDER BY created_at DESC LIMIT ?",
            (container, limit),
        ).fetchall()
        return [self._row_to_document(r) for r in rows]

    # ------------------------------------------------------------ chunks

    def insert_chunks(self, rows: Sequence[Tuple[str, str, str, int, str]]) -> None:
        """rows: (id, document_id, container, ordinal, text)"""
        now = _iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            conn.executemany(
                """INSERT INTO chunks (id, document_id, container, ordinal, text, created_at)
                   VALUES (?,?,?,?,?,?)""",
                [(*r, now) for r in rows],
            )

    # ---------------------------------------------------------- memories

    def insert_memory(self, memory: Memory) -> Memory:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO memories
                   (id, container, text, memory_type, source, document_id, chunk_id,
                    importance, confidence, created_at, updated_at, last_accessed_at,
                    access_count, status, superseded_by, superseded_at,
                    supersede_reason, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    memory.id, memory.container, memory.text, memory.memory_type.value,
                    memory.source, memory.document_id, memory.chunk_id,
                    memory.importance, memory.confidence,
                    _iso(memory.created_at), _iso(memory.updated_at),
                    _iso(memory.last_accessed_at), memory.access_count,
                    memory.status.value, memory.superseded_by,
                    _iso(memory.superseded_at), memory.supersede_reason,
                    json.dumps(memory.metadata),
                ),
            )
        return memory

    def get_memory(self, memory_id: str) -> Optional[Memory]:
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (memory_id,)
        ).fetchone()
        return self._row_to_memory(row) if row else None

    def get_memories(self, memory_ids: Sequence[str]) -> Dict[str, Memory]:
        """Bulk fetch, returned as a dict so callers can preserve their own order."""
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._conn.execute(
            f"SELECT * FROM memories WHERE id IN ({placeholders})", tuple(memory_ids)
        ).fetchall()
        return {r["id"]: self._row_to_memory(r) for r in rows}

    def list_memories(self, container: str, *, status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
                      memory_type: Optional[MemoryType] = None, limit: int = 100,
                      offset: int = 0) -> List[Memory]:
        clauses = ["container = ?"]
        values: List[Any] = [container]
        if status is not None:
            clauses.append("status = ?")
            values.append(status.value)
        if memory_type is not None:
            clauses.append("memory_type = ?")
            values.append(memory_type.value)
        values.extend([limit, offset])
        rows = self._conn.execute(
            f"""SELECT * FROM memories WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC LIMIT ? OFFSET ?""",
            values,
        ).fetchall()
        return [self._row_to_memory(r) for r in rows]

    def count_memories(self, container: str, status: Optional[MemoryStatus] = None) -> int:
        if status is None:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE container = ?", (container,)
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE container = ? AND status = ?",
                (container, status.value),
            ).fetchone()
        return int(row["n"])

    def set_status(self, memory_id: str, status: MemoryStatus) -> None:
        with self.transaction() as conn:
            conn.execute(
                "UPDATE memories SET status = ?, updated_at = ? WHERE id = ?",
                (status.value, _iso(datetime.now(timezone.utc)), memory_id),
            )

    def mark_superseded(self, old_id: str, new_id: str, reason: str = "",
                        status: MemoryStatus = MemoryStatus.SUPERSEDED) -> None:
        now = _iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            conn.execute(
                """UPDATE memories
                   SET status = ?, superseded_by = ?, superseded_at = ?,
                       supersede_reason = ?, updated_at = ?
                   WHERE id = ?""",
                (status.value, new_id, now, reason, now, old_id),
            )

    def touch(self, memory_ids: Sequence[str]) -> None:
        """
        Record that these memories were retrieved.

        This is the reinforcement half of decay: a memory that keeps
        proving useful gets its clock reset and its access count bumped,
        so it stops looking stale to the scorer.
        """
        if not memory_ids:
            return
        now = _iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            conn.executemany(
                """UPDATE memories
                   SET access_count = access_count + 1, last_accessed_at = ?
                   WHERE id = ?""",
                [(now, mid) for mid in memory_ids],
            )

    def reinforce(self, memory_id: str, importance_delta: float = 0.0) -> None:
        """
        Strengthen a memory that was just restated.

        Bumps importance (clamped at 1.0 in SQL so concurrent writers
        can't drift it past the valid range) and resets the decay clock,
        since a restatement is itself a form of access.
        """
        now = _iso(datetime.now(timezone.utc))
        with self.transaction() as conn:
            conn.execute(
                """UPDATE memories
                   SET importance = MIN(1.0, importance + ?),
                       access_count = access_count + 1,
                       last_accessed_at = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (importance_delta, now, now, memory_id),
            )

    def delete_memory(self, memory_id: str) -> None:
        """
        Remove a memory, and the graph data that existed only to describe it.

        Foreign keys cascade the memory_entities links, but that is not
        enough on its own: an entity row outlives its last mention, and a
        relation outlives the memory it was drawn from — leaving a graph
        that talks about memories the store no longer holds. Emptying a
        container would still report entities "discovered", which is how
        a demo came to list a city before anyone had mentioned it.

        Entities mentioned elsewhere are kept, with their count re-derived
        from the links that actually survive.
        """
        with self.transaction() as conn:
            entity_ids = [
                row["entity_id"] for row in conn.execute(
                    "SELECT entity_id FROM memory_entities WHERE memory_id = ?",
                    (memory_id,),
                ).fetchall()
            ]

            # An edge is only as good as the memory it was drawn from.
            conn.execute("DELETE FROM relations WHERE memory_id = ?", (memory_id,))
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))

            # The link rows are gone by cascade, so count what remains
            # rather than decrementing — a re-derived count self-heals,
            # a decremented one drifts.
            for entity_id in entity_ids:
                remaining = conn.execute(
                    "SELECT COUNT(*) AS n FROM memory_entities WHERE entity_id = ?",
                    (entity_id,),
                ).fetchone()["n"]
                if remaining:
                    conn.execute(
                        "UPDATE entities SET mention_count = ? WHERE id = ?",
                        (remaining, entity_id),
                    )
                else:
                    # Last mention gone. Deleting the entity cascades any
                    # relations still hanging off it as subject or object.
                    conn.execute("DELETE FROM entities WHERE id = ?", (entity_id,))

    def keyword_search(self, container: str, query: str,
                       limit: int = 30) -> List[Tuple[str, float]]:
        """
        BM25 keyword search. Returns (memory_id, score) with higher = better.

        FTS5's bm25() is negative-is-better, so it gets negated here. This
        catches what embeddings miss: exact names, IDs, rare terms — the
        cases where a query and its answer share a literal token but sit
        far apart in vector space.
        """
        match = build_fts_query(query)
        if not match:
            return []
        rows = self._conn.execute(
            """SELECT m.id AS id, bm25(memories_fts) AS rank
               FROM memories_fts
               JOIN memories m ON m.rowid = memories_fts.rowid
               WHERE memories_fts MATCH ?
                 AND m.container = ?
                 AND m.status = 'active'
               ORDER BY rank
               LIMIT ?""",
            (match, container, limit),
        ).fetchall()
        return [(r["id"], -float(r["rank"])) for r in rows]

    # ---------------------------------------------------------- entities

    def upsert_entity(self, container: str, name: str,
                      entity_type: EntityType = EntityType.OTHER) -> Optional[Entity]:
        """Find-or-create an entity, bumping its mention count either way."""
        norm = normalise_entity_name(name)
        if not norm:
            return None

        with self.transaction() as conn:
            # Matched on name alone, not name *and* type. The type is the
            # model's guess and it is not stable — the same "memoos" comes
            # back as org one call and other the next, and matching on both
            # turned one project into two nodes sitting side by side in the
            # graph with the mentions split between them. A thing is the
            # thing it is; the type is a label on it. Prefer the most-seen
            # row so the surviving label is the one guessed most often.
            row = conn.execute(
                """SELECT * FROM entities
                   WHERE container = ? AND norm_name = ?
                   ORDER BY mention_count DESC LIMIT 1""",
                (container, norm),
            ).fetchone()

            if row:
                conn.execute(
                    "UPDATE entities SET mention_count = mention_count + 1 WHERE id = ?",
                    (row["id"],),
                )
                entity = self._row_to_entity(row)
                entity.mention_count += 1
                return entity

            entity = Entity(
                container=container, name=name.strip(),
                norm_name=norm, entity_type=entity_type, mention_count=1,
            )
            conn.execute(
                """INSERT INTO entities
                   (id, container, name, norm_name, entity_type, mention_count, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    entity.id, entity.container, entity.name, entity.norm_name,
                    entity.entity_type.value, entity.mention_count,
                    _iso(entity.created_at),
                ),
            )
            return entity

    def find_entity(self, container: str, name: str) -> Optional[Entity]:
        """Look up by normalised name, ignoring type."""
        norm = normalise_entity_name(name)
        if not norm:
            return None
        row = self._conn.execute(
            """SELECT * FROM entities WHERE container = ? AND norm_name = ?
               ORDER BY mention_count DESC LIMIT 1""",
            (container, norm),
        ).fetchone()
        return self._row_to_entity(row) if row else None

    def link_memory_entities(self, memory_id: str, entity_ids: Iterable[str]) -> None:
        pairs = [(memory_id, eid) for eid in dict.fromkeys(entity_ids)]
        if not pairs:
            return
        with self.transaction() as conn:
            conn.executemany(
                """INSERT OR IGNORE INTO memory_entities (memory_id, entity_id)
                   VALUES (?,?)""",
                pairs,
            )

    def entity_ids_for_memories(self, memory_ids: Sequence[str]) -> Dict[str, List[str]]:
        if not memory_ids:
            return {}
        placeholders = ",".join("?" * len(memory_ids))
        rows = self._conn.execute(
            f"""SELECT memory_id, entity_id FROM memory_entities
                WHERE memory_id IN ({placeholders})""",
            tuple(memory_ids),
        ).fetchall()
        out: Dict[str, List[str]] = {}
        for r in rows:
            out.setdefault(r["memory_id"], []).append(r["entity_id"])
        return out

    def memories_for_entities(self, container: str, entity_ids: Sequence[str],
                              exclude: Sequence[str] = (),
                              limit: int = 20) -> List[Tuple[str, str]]:
        """
        Memories linked to any of these entities. Returns (memory_id, entity_id).

        This is the graph hop: given what the query already matched, find
        what else talks about the same things.
        """
        if not entity_ids:
            return []
        placeholders = ",".join("?" * len(entity_ids))
        values: List[Any] = [container, *entity_ids]
        exclude_sql = ""
        if exclude:
            exclude_sql = f" AND m.id NOT IN ({','.join('?' * len(exclude))})"
            values.extend(exclude)
        values.append(limit)
        rows = self._conn.execute(
            f"""SELECT me.memory_id AS memory_id, me.entity_id AS entity_id
                FROM memory_entities me
                JOIN memories m ON m.id = me.memory_id
                WHERE m.container = ? AND m.status = 'active'
                  AND me.entity_id IN ({placeholders}){exclude_sql}
                ORDER BY m.importance DESC, m.created_at DESC
                LIMIT ?""",
            values,
        ).fetchall()
        return [(r["memory_id"], r["entity_id"]) for r in rows]

    def get_entities(self, entity_ids: Sequence[str]) -> Dict[str, Entity]:
        if not entity_ids:
            return {}
        placeholders = ",".join("?" * len(entity_ids))
        rows = self._conn.execute(
            f"SELECT * FROM entities WHERE id IN ({placeholders})", tuple(entity_ids)
        ).fetchall()
        return {r["id"]: self._row_to_entity(r) for r in rows}

    def list_entities(self, container: str, limit: int = 100) -> List[Entity]:
        rows = self._conn.execute(
            """SELECT * FROM entities WHERE container = ?
               ORDER BY mention_count DESC, created_at DESC LIMIT ?""",
            (container, limit),
        ).fetchall()
        return [self._row_to_entity(r) for r in rows]

    # --------------------------------------------------------- relations

    def insert_relation(self, relation: Relation) -> Relation:
        with self.transaction() as conn:
            conn.execute(
                """INSERT INTO relations
                   (id, container, subject_id, predicate, object_id, memory_id,
                    confidence, created_at, invalidated_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    relation.id, relation.container, relation.subject_id,
                    relation.predicate, relation.object_id, relation.memory_id,
                    relation.confidence, _iso(relation.created_at),
                    _iso(relation.invalidated_at),
                ),
            )
        return relation

    def invalidate_relations_for_memory(self, memory_id: str) -> None:
        """When a memory is superseded, the edges it asserted stop holding."""
        with self.transaction() as conn:
            conn.execute(
                """UPDATE relations SET invalidated_at = ?
                   WHERE memory_id = ? AND invalidated_at IS NULL""",
                (_iso(datetime.now(timezone.utc)), memory_id),
            )

    def list_relations(self, container: str, *, entity_id: Optional[str] = None,
                       include_invalid: bool = False,
                       limit: int = 200) -> List[Relation]:
        clauses = ["container = ?"]
        values: List[Any] = [container]
        if entity_id:
            clauses.append("(subject_id = ? OR object_id = ?)")
            values.extend([entity_id, entity_id])
        if not include_invalid:
            clauses.append("invalidated_at IS NULL")
        values.append(limit)
        rows = self._conn.execute(
            f"""SELECT * FROM relations WHERE {' AND '.join(clauses)}
                ORDER BY created_at DESC LIMIT ?""",
            values,
        ).fetchall()
        return [self._row_to_relation(r) for r in rows]

    # ------------------------------------------------------ row mapping

    @staticmethod
    def _row_to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            container=row["container"],
            text=row["text"],
            memory_type=MemoryType(row["memory_type"]),
            source=row["source"],
            document_id=row["document_id"],
            chunk_id=row["chunk_id"],
            importance=row["importance"],
            confidence=row["confidence"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            last_accessed_at=_dt(row["last_accessed_at"]),
            access_count=row["access_count"],
            status=MemoryStatus(row["status"]),
            superseded_by=row["superseded_by"],
            superseded_at=_dt(row["superseded_at"]),
            supersede_reason=row["supersede_reason"],
            metadata=_json_loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> Document:
        return Document(
            id=row["id"],
            container=row["container"],
            title=row["title"],
            uri=row["uri"],
            source=row["source"],
            raw_text=row["raw_text"],
            status=DocumentStatus(row["status"]),
            error=row["error"],
            chunk_count=row["chunk_count"],
            memory_count=row["memory_count"],
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            metadata=_json_loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_entity(row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            container=row["container"],
            name=row["name"],
            norm_name=row["norm_name"],
            entity_type=EntityType(row["entity_type"]),
            mention_count=row["mention_count"],
            created_at=_dt(row["created_at"]),
        )

    @staticmethod
    def _row_to_relation(row: sqlite3.Row) -> Relation:
        return Relation(
            id=row["id"],
            container=row["container"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            memory_id=row["memory_id"],
            confidence=row["confidence"],
            created_at=_dt(row["created_at"]),
            invalidated_at=_dt(row["invalidated_at"]),
        )
