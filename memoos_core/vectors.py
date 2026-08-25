"""
Vector index: the semantic half of retrieval.

Vectors live in the `embeddings` table of the very same SQLite file as
the memories they describe. That is the whole point of this module's
shape: a container's entire state — journal, memories, entities,
relations, and the vectors that make them searchable — is one file you
can copy, replicate or hand to a cloud backend as a unit. A sidecar
index directory would mean two things to keep in sync, and any sync
protocol between them is a source of drift that a single file simply
does not have.

Search is exhaustive: every vector in the container is scored against
the query. There is no approximate-nearest-neighbour index, and for this
workload that is the right trade. A terminal's memory is thousands of
rows, not millions; 5,000 vectors of 384 dimensions is a 7MB matrix and
one BLAS call, which lands well under a millisecond — faster than the
HNSW graph it replaces, because there is no graph to load first. The
crossover where ANN starts to win is somewhere north of a hundred
thousand memories, and `search` degrades linearly and predictably up to
it rather than falling off a cliff.

Deliberately narrow, as before: this file knows about ids, text and
similarity, and nothing about memories, decay or the graph.
"""

import os
import sqlite3
import threading
from array import array
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import config
from .db import EMBEDDINGS_SCHEMA
from .embedding import embed_batch, embed_query


def _pack(vector: Sequence[float]) -> bytes:
    """A vector as a raw little-endian float32 buffer."""
    return array("f", [float(x) for x in vector]).tobytes()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class VectorIndex:
    def __init__(self, container: str = "default", path: Optional[str] = None,
                 model: str = ""):
        self.container = container
        self.path = path or config.db_path(container=container)
        self.model = model or config.EMBED_MODEL
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

        # check_same_thread=False + a lock, matching `Database`: the API
        # serves from a thread pool, and one guarded connection beats a
        # connection per thread.
        self._conn = sqlite3.connect(self.path, check_same_thread=False,
                                     timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(EMBEDDINGS_SCHEMA)
            self._conn.commit()

        self._ids: List[str] = []
        self._matrix: Optional[np.ndarray] = None
        self._version: Optional[int] = None

    # ------------------------------------------------------------ cache

    def _snapshot(self) -> Tuple[List[str], Optional[np.ndarray]]:
        """
        This container's vectors as (ids, matrix), read from cache.

        Decoding every BLOB on every query would dominate the search
        itself, so the matrix is held in memory. Keeping it honest is the
        interesting part: the CLI and the API server are separate
        processes writing to the same file, so a cache that only tracked
        its own writes would happily serve a stale index — a memory added
        in the terminal would be invisible to the dashboard until it
        restarted.

        `PRAGMA data_version` is SQLite's own answer to this. It changes
        whenever *another* connection commits, so it catches the other
        process; it deliberately does not change for our own writes,
        which is why those invalidate the cache directly.
        """
        with self._lock:
            version = self._conn.execute("PRAGMA data_version").fetchone()[0]
            if self._matrix is not None and version == self._version:
                return self._ids, self._matrix

            # Only vectors of the current width are comparable. If the
            # embedding model changed, rows from the old one are ignored
            # here rather than crashing the reshape — `stale()` reports
            # them, and re-indexing clears them out.
            rows = self._conn.execute(
                """SELECT memory_id, vector FROM embeddings
                    WHERE container = ? AND dim = (
                          SELECT dim FROM embeddings WHERE container = ?
                           ORDER BY rowid DESC LIMIT 1)
                 ORDER BY rowid""",
                (self.container, self.container),
            ).fetchall()

            self._ids = [row["memory_id"] for row in rows]
            if rows:
                buffer = b"".join(row["vector"] for row in rows)
                self._matrix = (np.frombuffer(buffer, dtype=np.float32)
                                  .reshape(len(rows), -1))
            else:
                self._matrix = None
            self._version = version
            return self._ids, self._matrix

    def _invalidate(self) -> None:
        self._matrix = None
        self._version = None

    # ------------------------------------------------------------ write

    def add(self, ids: Sequence[str], texts: Sequence[str],
            metadatas: Optional[Sequence[Dict]] = None) -> None:
        """
        Index texts under their memory ids.

        `metadatas` is accepted and ignored. Callers pass fields like
        memory_type and created_at, and every one of them is already a
        column on `memories` in this same file — storing a second copy
        here would buy nothing and give the two a way to disagree.
        """
        if not ids:
            return
        vectors = embed_batch(list(texts))
        stamp = _now()
        rows = [(memory_id, self.container, len(vector), self.model,
                 _pack(vector), stamp)
                for memory_id, vector in zip(ids, vectors)]
        with self._lock:
            self._conn.executemany(
                """INSERT OR REPLACE INTO embeddings
                   (memory_id, container, dim, model, vector, created_at)
                   VALUES (?,?,?,?,?,?)""",
                rows,
            )
            self._conn.commit()
            self._invalidate()

    def add_one(self, memory_id: str, text: str,
                metadata: Optional[Dict] = None) -> None:
        self.add([memory_id], [text], [metadata] if metadata else None)

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        with self._lock:
            self._conn.execute(
                "DELETE FROM embeddings WHERE memory_id IN "
                f"({','.join('?' * len(ids))})",
                list(ids),
            )
            self._conn.commit()
            self._invalidate()

    def reset(self) -> None:
        """Drop this container's vectors. The memories remain."""
        with self._lock:
            self._conn.execute("DELETE FROM embeddings WHERE container = ?",
                               (self.container,))
            self._conn.commit()
            self._invalidate()

    # ------------------------------------------------------------- read

    def search(self, query: str, top_k: int = 10,
               exclude: Sequence[str] = ()) -> List[Tuple[str, float]]:
        """Nearest neighbours as (memory_id, cosine_similarity), best first."""
        return self.search_vector(embed_query(query), top_k=top_k,
                                  exclude=exclude)

    def search_vector(self, vector: Sequence[float], top_k: int = 10,
                      exclude: Sequence[str] = ()) -> List[Tuple[str, float]]:
        """
        Same as `search`, for an already-computed embedding.

        Embeddings arrive L2-normalised from `embedding.embed_*`, so the
        dot product *is* the cosine similarity and no division is needed.
        Scores are clamped into [0, 1] because float error can put an
        identical pair a hair above 1.0, and because everything
        downstream — thresholds, fusion weights — assumes that range.
        """
        ids, matrix = self._snapshot()
        if matrix is None or not ids or top_k <= 0:
            return []

        query = np.asarray(vector, dtype=np.float32)
        if query.ndim != 1 or query.shape[0] != matrix.shape[1]:
            # A query from a different embedding model than the index.
            return []

        scores = matrix @ query

        excluded = set(exclude)
        want = min(top_k + len(excluded), len(ids))
        # argpartition finds the top `want` without ordering the rest,
        # which is what keeps this sub-linear in sort cost as the index
        # grows; only the shortlist is then sorted properly.
        shortlist = np.argpartition(-scores, want - 1)[:want]
        shortlist = shortlist[np.argsort(-scores[shortlist])]

        out: List[Tuple[str, float]] = []
        for index in shortlist:
            memory_id = ids[int(index)]
            if memory_id in excluded:
                continue
            out.append((memory_id, max(0.0, min(1.0, float(scores[index])))))
            if len(out) >= top_k:
                break
        return out

    def count(self) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE container = ?",
                (self.container,),
            ).fetchone()[0]

    def stale(self) -> int:
        """
        Vectors written by a different embedding model than the current one.

        Non-zero means part of the container is unsearchable until it is
        re-indexed, so `MemoOS.stats` surfaces it rather than leaving a
        silently half-working index.
        """
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM embeddings WHERE container = ? AND model != ?",
                (self.container, self.model),
            ).fetchone()[0]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
