"""
The raw journal: what actually happened in the terminal, written fast.

This is deliberately the dumb half of the memory layer. A shell hook runs
on *every* command you type, so the write path here has a hard budget of
a few milliseconds and a hard rule against importing anything heavy —
no Chroma, no embedding model, no LLM. It appends a row and returns.

The clever half happens later. `terminal.distill()` reads a finished
session out of this table and hands it to the extraction pipeline, which
turns a hundred shell lines into a handful of atomic memories and the
entities that connect them. Separating the two is what makes the layer
usable: you never wait for a model to finish before your next prompt
comes back, and you still end up with a real memory graph.

Nothing in this module imports from the rest of memoos_core beyond
`config`, and `config` imports only `os`. That is a constraint, not an
accident — see the note in `memoos_core/__init__.py`.
"""

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config

# Kinds of thing worth journalling. Kept as plain strings rather than an
# enum so the hook can pass one without importing anything.
COMMAND = "command"   # a shell command, its cwd and exit code
GIT = "git"           # a branch change, commit, or checkout
CLAUDE = "claude"     # a turn imported from a Claude Code transcript
NOTE = "note"         # something you told the journal yourself

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id           TEXT PRIMARY KEY,
    container    TEXT NOT NULL,
    session_id   TEXT NOT NULL,
    kind         TEXT NOT NULL,
    text         TEXT NOT NULL,
    cwd          TEXT,
    exit_code    INTEGER,
    created_at   TEXT NOT NULL,
    distilled_at TEXT,
    metadata     TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_events_session
    ON events(container, session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_events_undistilled
    ON events(container, distilled_at, created_at);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# Which files this process has already applied the schema to. The hook
# spawns a fresh process per command so it pays this once either way, but
# the API server constructs a `Journal` per request and should not re-run
# DDL on the read path every time.
_SCHEMA_APPLIED: set = set()


def _ensure_schema(conn: sqlite3.Connection, path: str) -> None:
    if path in _SCHEMA_APPLIED:
        return
    conn.executescript(_SCHEMA)
    _SCHEMA_APPLIED.add(path)


# ------------------------------------------------------------ containers

def sanitise_container(name: str) -> str:
    """
    Make a name safe to use as a container.

    A container name is also a filename now — the container *is* the file
    — so this is `config.safe_container` and nothing more. Sharing the one
    implementation is what guarantees a container can always be resolved
    back to the file holding it: two different normalisations would put
    the events in one file and then look for them in another.
    """
    return config.safe_container(name)


def project_root(start: Optional[str] = None) -> str:
    """
    The directory a session belongs to: the git repo root, or the cwd.

    Walked directly rather than shelled out to `git rev-parse`, because
    this runs inside the per-command hook and a subprocess there costs
    more than everything else in this module put together.
    """
    path = os.path.abspath(start or os.getcwd())
    current = path
    while True:
        if os.path.isdir(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return path
        current = parent


def container_for(start: Optional[str] = None) -> str:
    """The memory container a directory's work belongs to."""
    return sanitise_container(os.path.basename(project_root(start)))


# --------------------------------------------------------------- journal


class Journal:
    """
    Append-only event log, one row per thing that happened.

    Opens a connection per operation rather than holding one. The hook
    process is short-lived and single-shot, so a pool would be overhead
    with no upside, and not holding the file open keeps this out of the
    way of the long-running API server writing to the same database.
    """

    def __init__(self, path: Optional[str] = None,
                 data_dir: Optional[str] = None):
        # `path` pins every operation to one file regardless of container,
        # which is what tests and an older single-file store want. Left
        # unset, each container is routed to its own file.
        self.path = path
        self.data_dir = data_dir or config.DATA_DIR

    def path_for(self, container: str) -> str:
        """The file this container's events live in."""
        return self.path or config.db_path(self.data_dir, container)

    @staticmethod
    def _key(container: str) -> str:
        """
        The one spelling of a container name, used for both file and filter.

        The file is chosen by `safe_container` but the WHERE clause used
        whatever the caller passed, so a name that normalises — "MyProj",
        arriving from a hand-typed dashboard URL — wrote rows into
        myproj.db under a label nothing else ever queries. Two halves of
        one project, in one file, invisible to each other. Normalising
        here makes the two agree by construction.
        """
        return sanitise_container(container)

    def _connect(self, container: str, create: bool = True) -> sqlite3.Connection:
        """
        Open this container's file. `create` is the write/read distinction.

        `sqlite3.connect` creates whatever it is pointed at, so a read of
        an unknown container would bring that container into existence as
        an empty file. Reads pass `create=False` and get an empty
        in-memory database instead, which answers "nothing stored" without
        leaving one file behind per name anyone ever asked about.
        """
        path = self.path_for(container)
        exists = os.path.exists(path)
        if not exists:
            # The file can go away underneath a living process — `memoos
            # clear` deletes it. Whatever we knew about its schema died
            # with it, so forget that and let the DDL run again rather
            # than reconnecting to an empty file and reading tables that
            # are no longer there.
            _SCHEMA_APPLIED.discard(path)
        if not create and not exists:
            conn = sqlite3.connect(":memory:")
            conn.row_factory = sqlite3.Row
            conn.executescript(_SCHEMA)
            return conn

        # sqlite3 creates the file but not the directory holding it, and
        # a container's first ever event is also the first thing to need
        # `containers/` to exist.
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        conn = sqlite3.connect(path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        # WAL so journalling a command never blocks on the API server, or
        # the other way round. Both processes write to this same file.
        conn.execute("PRAGMA journal_mode=WAL")
        _ensure_schema(conn, path)
        return conn

    # ------------------------------------------------------------ write

    def record(self, kind: str, text: str, *, container: Optional[str] = None,
               session_id: str = "adhoc", cwd: Optional[str] = None,
               exit_code: Optional[int] = None,
               metadata: Optional[Dict[str, Any]] = None) -> str:
        """Append one event. This is the hot path — keep it boring."""
        text = (text or "").strip()
        if not text:
            return ""
        cwd = cwd or os.getcwd()
        container = self._key(container) if container else container_for(cwd)
        event_id = str(uuid.uuid4())
        with self._connect(container) as conn:
            conn.execute(
                """INSERT INTO events
                   (id, container, session_id, kind, text, cwd, exit_code,
                    created_at, distilled_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,NULL,?)""",
                (event_id, container, session_id, kind,
                 text, cwd, exit_code, _now(), json.dumps(metadata or {})),
            )
        return event_id

    def mark_distilled(self, event_ids: List[str], container: str) -> None:
        """Stamp events as folded into memory. `container` names their file."""
        if not event_ids:
            return
        container = self._key(container)
        stamp = _now()
        with self._connect(container) as conn:
            conn.executemany(
                "UPDATE events SET distilled_at = ? WHERE id = ?",
                [(stamp, eid) for eid in event_ids],
            )

    def pending_count(self, container: str) -> int:
        """How many events are waiting to be folded into memory."""
        container = self._key(container)
        with self._connect(container, create=False) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM events WHERE container = ? AND distilled_at IS NULL",
                (container,),
            ).fetchone()[0]

    def forget(self, container: str) -> int:
        container = self._key(container)
        with self._connect(container) as conn:
            cur = conn.execute("DELETE FROM events WHERE container = ?", (container,))
            return cur.rowcount

    # ------------------------------------------------------------- read

    def events(self, container: str, *, session_id: Optional[str] = None,
               kind: Optional[str] = None, undistilled_only: bool = False,
               limit: int = 500) -> List[Dict[str, Any]]:
        container = self._key(container)
        sql = "SELECT * FROM events WHERE container = ?"
        params: List[Any] = [container]
        if session_id:
            sql += " AND session_id = ?"
            params.append(session_id)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if undistilled_only:
            sql += " AND distilled_at IS NULL"
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        with self._connect(container, create=False) as conn:
            rows = conn.execute(sql, params).fetchall()
        # Read newest-first so LIMIT keeps the *recent* end of a long
        # session, then hand them back in the order they happened.
        return [self._row(r) for r in reversed(rows)]

    def sessions(self, container: str, limit: int = 20) -> List[Dict[str, Any]]:
        """One row per session: when it ran, how big, what it touched."""
        container = self._key(container)
        with self._connect(container, create=False) as conn:
            rows = conn.execute(
                """SELECT session_id,
                          COUNT(*)                                   AS events,
                          MIN(created_at)                            AS started_at,
                          MAX(created_at)                            AS ended_at,
                          SUM(CASE WHEN distilled_at IS NULL THEN 1 ELSE 0 END) AS pending,
                          SUM(CASE WHEN exit_code NOT IN (0) THEN 1 ELSE 0 END) AS failures
                     FROM events
                    WHERE container = ?
                 GROUP BY session_id
                 ORDER BY MAX(created_at) DESC
                    LIMIT ?""",
                (container, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def containers(self) -> List[Dict[str, Any]]:
        """
        Every project the journal has seen, most recent first.

        One file per container turns this from a GROUP BY into a
        directory listing. Pinned to a single file it groups as before,
        so an older store still enumerates correctly.
        """
        if self.path:
            with self._connect("") as conn:
                rows = conn.execute(
                    """SELECT container, COUNT(*) AS events, MAX(created_at) AS last_seen
                         FROM events GROUP BY container ORDER BY MAX(created_at) DESC"""
                ).fetchall()
            return [dict(r) for r in rows]

        out: List[Dict[str, Any]] = []
        for name in config.stored_containers(self.data_dir):
            with self._connect(name, create=False) as conn:
                row = conn.execute(
                    """SELECT COUNT(*) AS events, MAX(created_at) AS last_seen,
                              MIN(container) AS container FROM events"""
                ).fetchone()
            out.append({"container": row["container"] or name,
                        "events": row["events"],
                        "last_seen": row["last_seen"]})
        # A container with a file but no events yet sorts last, not first.
        out.sort(key=lambda r: r["last_seen"] or "", reverse=True)
        return out

    @staticmethod
    def _row(row: sqlite3.Row) -> Dict[str, Any]:
        out = dict(row)
        try:
            out["metadata"] = json.loads(out.get("metadata") or "{}")
        except json.JSONDecodeError:
            out["metadata"] = {}
        return out
