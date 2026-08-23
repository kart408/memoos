"""
Storage.

Business logic talks to `MemoryRepository`, never to SQLite directly.
That indirection is the whole point of this module: swapping to Postgres
later means writing one new subclass, not editing memory_service.py.

The isolation rule lives here, and only here. Every method that touches
a single memory takes a user_id *and* an id, and puts both in the WHERE
clause:

    WHERE id = ? AND user_id = ?

Not "fetch by id, then check the owner in Python". The difference
matters — a client-side check is one forgotten `if` away from leaking
another tenant's data, while a scoped query simply returns nothing. A
user guessing a valid UUID from another tenant gets a 404, because as
far as the database is concerned that row does not exist for them.
"""

import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .models import Memory

SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    content     TEXT NOT NULL,
    embedding   BLOB,
    dim         INTEGER,
    metadata    TEXT NOT NULL DEFAULT '{}',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Every read is scoped by user_id, so that is the index that matters.
CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
CREATE INDEX IF NOT EXISTS idx_memories_user_created
    ON memories(user_id, created_at DESC);
"""


def pack(vector: Optional[List[float]]) -> Optional[bytes]:
    """Store embeddings as raw float32 — 4 bytes a dimension, no parsing."""
    if vector is None:
        return None
    return np.asarray(vector, dtype=np.float32).tobytes()


def unpack(blob: Optional[bytes]) -> Optional[List[float]]:
    if blob is None:
        return None
    return np.frombuffer(blob, dtype=np.float32).tolist()


def _iso(value: datetime) -> str:
    return value.isoformat()


def _dt(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------- interface


class MemoryRepository(ABC):
    """
    The contract business logic depends on.

    Implementations must scope every single-memory operation by user_id.
    """

    @abstractmethod
    def add(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def get(self, user_id: str, memory_id: str) -> Optional[Memory]: ...

    @abstractmethod
    def list(self, user_id: str, limit: int, offset: int) -> List[Memory]: ...

    @abstractmethod
    def count(self, user_id: str) -> int: ...

    @abstractmethod
    def update(self, user_id: str, memory_id: str, *, content: Optional[str],
               embedding: Optional[List[float]],
               metadata: Optional[Dict[str, Any]]) -> Optional[Memory]: ...

    @abstractmethod
    def delete(self, user_id: str, memory_id: str) -> bool: ...

    @abstractmethod
    def embeddings_for_user(self, user_id: str) -> List[Tuple[str, np.ndarray]]: ...

    @abstractmethod
    def close(self) -> None: ...


# ------------------------------------------------------------------ sqlite


class SQLiteMemoryRepository(MemoryRepository):
    def __init__(self, path: str = "memoos.db"):
        self.path = path
        # FastAPI serves sync endpoints from a threadpool, so the
        # connection is shared across threads and guarded by a lock
        # rather than opened per request.
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- writes

    def add(self, memory: Memory) -> Memory:
        with self._lock:
            self._conn.execute(
                """INSERT INTO memories
                   (id, user_id, content, embedding, dim, metadata,
                    created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    memory.id,
                    memory.user_id,
                    memory.content,
                    pack(memory.embedding),
                    len(memory.embedding) if memory.embedding else None,
                    json.dumps(memory.metadata),
                    _iso(memory.created_at),
                    _iso(memory.updated_at),
                ),
            )
            self._conn.commit()
        return memory

    def update(self, user_id: str, memory_id: str, *, content: Optional[str],
               embedding: Optional[List[float]],
               metadata: Optional[Dict[str, Any]]) -> Optional[Memory]:
        """
        Apply a partial edit, scoped to the owner.

        Returns None when the memory does not exist *for this user* —
        which is the same answer a caller gets for someone else's id.
        """
        sets: List[str] = []
        values: List[Any] = []

        if content is not None:
            sets.append("content = ?")
            values.append(content)
            # Content and embedding move together or the index goes stale
            # and search starts ranking edited memories by their old text.
            sets.append("embedding = ?")
            values.append(pack(embedding))
            sets.append("dim = ?")
            values.append(len(embedding) if embedding else None)

        if metadata is not None:
            sets.append("metadata = ?")
            values.append(json.dumps(metadata))

        if not sets:
            return self.get(user_id, memory_id)

        sets.append("updated_at = ?")
        values.append(_iso(datetime.now(timezone.utc)))
        values.extend([memory_id, user_id])

        with self._lock:
            cursor = self._conn.execute(
                f"UPDATE memories SET {', '.join(sets)} WHERE id = ? AND user_id = ?",
                values,
            )
            self._conn.commit()
            if cursor.rowcount == 0:
                return None
        return self.get(user_id, memory_id)

    def delete(self, user_id: str, memory_id: str) -> bool:
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            )
            self._conn.commit()
            return cursor.rowcount > 0

    # -- reads

    def get(self, user_id: str, memory_id: str) -> Optional[Memory]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, user_id),
            ).fetchone()
        return self._to_memory(row) if row else None

    def list(self, user_id: str, limit: int = 50, offset: int = 0) -> List[Memory]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM memories WHERE user_id = ?
                   ORDER BY created_at DESC, id LIMIT ? OFFSET ?""",
                (user_id, limit, offset),
            ).fetchall()
        return [self._to_memory(r) for r in rows]

    def count(self, user_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM memories WHERE user_id = ?", (user_id,)
            ).fetchone()
        return int(row["n"])

    def embeddings_for_user(self, user_id: str) -> List[Tuple[str, np.ndarray]]:
        """
        Every stored vector for one user, for brute-force ranking.

        Loading a user's whole set per search is fine at the scale this
        is built for (tens to low thousands of memories — a few ms), and
        it keeps the vector index from being a second thing that can
        drift out of sync with the rows. It is also the one method that
        would change when this moves to sqlite-vec or pgvector: the
        ranking would go into the query instead of into numpy.
        """
        with self._lock:
            rows = self._conn.execute(
                """SELECT id, embedding FROM memories
                   WHERE user_id = ? AND embedding IS NOT NULL""",
                (user_id,),
            ).fetchall()
        return [
            (r["id"], np.frombuffer(r["embedding"], dtype=np.float32))
            for r in rows
        ]

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- mapping

    @staticmethod
    def _to_memory(row: sqlite3.Row) -> Memory:
        return Memory(
            id=row["id"],
            user_id=row["user_id"],
            content=row["content"],
            embedding=unpack(row["embedding"]),
            metadata=json.loads(row["metadata"] or "{}"),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
        )
