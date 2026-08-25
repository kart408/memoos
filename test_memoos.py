"""
End-to-end tests for memoos_core.

    python test_memoos.py

Runs against a scratch data directory, so it never touches your real
store. Needs no Ollama and no extraction model: everything here exercises
storage, routing, isolation and search, none of which involve the LLM.
The embedding model does load — it is what search *is* — which costs a
few seconds once.

What it covers, roughly in the order a command flows through the system:

  paths       a container name resolves to exactly one file, safely
  journal     events land in their own container's file and nowhere else
  reads       asking about an unknown project does not create it
  vectors     ranking, exclusion, deletion, and the cross-process cache
  cascade     deleting a memory takes its vector with it
  one file    everything about a container is in that container's file

No test framework, so there is nothing to install: the assertions are
plain, and a failure prints what it expected against what it got.
"""

import os
import shutil
import sys
import tempfile

SCRATCH = tempfile.mkdtemp(prefix="memoos-test-")
os.environ["MEMOOS_DATA_DIR"] = SCRATCH

from memoos_core import config, quick  # noqa: E402
from memoos_core.db import Database  # noqa: E402
from memoos_core.journal import Journal, sanitise_container  # noqa: E402
from memoos_core.models import Memory  # noqa: E402
from memoos_core.vectors import VectorIndex  # noqa: E402

PASSED = 0
FAILED = 0


def check(label: str, got, want) -> None:
    global PASSED, FAILED
    if got == want:
        PASSED += 1
    else:
        FAILED += 1
        print(f"  FAIL  {label}\n          expected {want!r}\n          got      {got!r}")


def ok(label: str, condition) -> None:
    check(label, bool(condition), True)


def section(title: str) -> None:
    print(f"\n{title}")


def files_on_disk() -> set:
    directory = config.containers_dir()
    if not os.path.isdir(directory):
        return set()
    return {name for name in os.listdir(directory) if name.endswith(".db")}


# ------------------------------------------------------------------ paths

def test_paths() -> None:
    section("paths")

    check("a name is lowercased and cleaned",
          sanitise_container("My Project (v2)"), "my-project-v2")
    check("an empty name falls back", sanitise_container("   "), "default")

    # The container name becomes a filename, so `..` and `/` must not
    # survive to walk out of the containers directory.
    traversal = config.db_path(container="../../etc/passwd")
    ok("path traversal cannot escape",
       os.path.dirname(os.path.abspath(traversal))
       == os.path.abspath(config.containers_dir()))

    ok("two containers are two files",
       config.db_path(container="a") != config.db_path(container="b"))


# ---------------------------------------------------------------- journal

def test_journal_isolation() -> None:
    section("journal")

    journal = Journal()
    journal.record("command", "pytest -q", container="alice",
                   session_id="s1", exit_code=0)
    journal.record("command", "git rebase", container="alice",
                   session_id="s1", exit_code=1)
    journal.record("command", "npm build", container="bob",
                   session_id="s2", exit_code=0)

    check("alice's events", len(journal.events("alice")), 2)
    check("bob's events", len(journal.events("bob")), 1)

    # Isolation is a property of the layout, not of a WHERE clause: the
    # two tenants are in different files, so a leak is not expressible.
    check("one file per container", files_on_disk(), {"alice.db", "bob.db"})
    check("bob cannot see alice's work",
          [e["text"] for e in journal.events("bob")], ["npm build"])

    check("exit codes survive",
          sorted(e["exit_code"] for e in journal.events("alice")), [0, 1])

    listed = {row["container"]: row["events"] for row in journal.containers()}
    check("containers() scans the directory", listed, {"alice": 2, "bob": 1})

    pending = journal.events("alice", undistilled_only=True)
    journal.mark_distilled([e["id"] for e in pending], "alice")
    check("mark_distilled routes by container",
          len(journal.events("alice", undistilled_only=True)), 0)
    check("and does not touch the other container",
          len(journal.events("bob", undistilled_only=True)), 1)


def test_reads_do_not_create_files() -> None:
    section("reads")

    # sqlite3 creates whatever it is pointed at, so a read of an unknown
    # container would bring that container into existence. Asking what a
    # project remembers must never be what makes the project exist.
    before = files_on_disk()

    check("journal.events on an unknown container",
          Journal().events("ghost"), [])
    check("journal.sessions on an unknown container",
          Journal().sessions("ghost"), [])
    check("quick.counts on an unknown container", quick.counts("ghost"),
          {"memories": 0, "entities": 0, "relations": 0})
    check("quick.recent_memories on an unknown container",
          quick.recent_memories("ghost"), [])

    check("no file was created", files_on_disk(), before)


# ---------------------------------------------------------------- vectors

def test_vector_search() -> None:
    section("vectors")

    index = VectorIndex("search", config.db_path(container="search"))
    index.add(
        ["ci", "coffee", "actions"],
        ["the deploy pipeline runs on GitHub Actions",
         "prefers dark roast coffee in the morning",
         "CI is configured through GitHub Actions workflows"],
    )
    check("everything indexed", index.count(), 3)
    check("nothing stale", index.stale(), 0)

    hits = index.search("how is CI set up?", top_k=3)
    ok("semantic search finds the CI memories first",
       {hits[0][0], hits[1][0]} == {"ci", "actions"})
    ok("scores are cosine, in range",
       all(0.0 <= score <= 1.0 for _, score in hits))
    ok("and ordered best-first",
       [s for _, s in hits] == sorted((s for _, s in hits), reverse=True))

    # A query sharing no words with the memory still finds it — that is
    # the whole point of ranking by meaning.
    drink = index.search("what do they drink?", top_k=1)
    check("meaning beats keyword overlap", drink[0][0], "coffee")

    excluded = index.search("how is CI set up?", top_k=2, exclude=["actions"])
    ok("exclusions are honoured",
       "actions" not in {memory_id for memory_id, _ in excluded})
    check("and the caller still gets top_k", len(excluded), 2)

    index.delete(["coffee"])
    check("delete removes one", index.count(), 2)
    index.reset()
    check("reset empties the container", index.count(), 0)
    check("search on an empty index is empty, not an error",
          index.search("anything"), [])


def test_cache_invalidation() -> None:
    section("cache")

    # The CLI and the dashboard are separate processes on one file. A
    # cache that only tracked its own writes would serve a stale index:
    # a memory added in the terminal would stay invisible to the
    # dashboard until it restarted.
    path = config.db_path(container="shared")
    server = VectorIndex("shared", path)
    client = VectorIndex("shared", path)

    client.add(["first"], ["the auth rewrite is blocked on token refresh"])
    server.search("auth", top_k=1)               # warm the server's cache

    client.add(["second"], ["token refresh design landed on Tuesday"])
    seen = {memory_id for memory_id, _ in server.search("token refresh", top_k=5)}
    check("a write elsewhere is visible immediately", seen, {"first", "second"})

    client.delete(["first"])
    seen = {memory_id for memory_id, _ in server.search("token refresh", top_k=5)}
    check("and so is a delete", seen, {"second"})


def test_model_change_is_survivable() -> None:
    section("model change")

    # Vectors from two different embedding models are not comparable.
    # Mixing them silently would make search quietly wrong, which is
    # worse than loudly broken.
    path = config.db_path(container="mixed")
    index = VectorIndex("mixed", path)
    index.add(["real"], ["a vector from the current model"])

    import sqlite3
    from array import array
    with sqlite3.connect(path) as conn:
        conn.execute(
            """INSERT INTO embeddings
               (memory_id, container, dim, model, vector, created_at)
               VALUES (?,?,?,?,?,?)""",
            ("old", "mixed", 8, "some-older-model",
             array("f", [0.1] * 8).tobytes(), "2020-01-01T00:00:00+00:00"),
        )

    check("the stale vector is reported", index.stale(), 1)
    hits = index.search("current model", top_k=5)
    ok("but it is not searched, and nothing crashes",
       "old" not in {memory_id for memory_id, _ in hits})


# ---------------------------------------------------------------- storage

def test_cascade_and_single_file() -> None:
    section("one file")

    container = "whole"
    path = config.db_path(container=container)

    # Each subsystem creates its own tables on first use, so a container
    # that has only ever been journalled has no `memories` table yet.
    # The claim under test is about a container in real use, so touch it
    # through all three.
    Journal().record("command", "make", container=container,
                     session_id="s", exit_code=0)
    db = Database(path)
    index = VectorIndex(container, path)

    memory = Memory(text="vectors live beside the memories they describe",
                    container=container)
    db.insert_memory(memory)
    index.add_one(memory.id, memory.text)

    check("the memory is stored", db.count_memories(container), 1)
    check("so is its vector", index.count(), 1)

    with __import__("sqlite3").connect(path) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
    for table in ("events", "memories", "embeddings", "documents",
                  "entities", "relations", "memory_entities"):
        ok(f"{table} is in the same file", table in tables)

    # The foreign key is what makes an orphaned vector unrepresentable,
    # rather than something the application has to remember to clean up.
    db.delete_memory(memory.id)
    check("deleting the memory drops it", db.count_memories(container), 0)
    check("and cascades to its vector", index.count(), 0)

    db.close()
    index.close()


def main() -> int:
    print(f"scratch store: {SCRATCH}")
    test_paths()
    test_journal_isolation()
    test_reads_do_not_create_files()
    test_vector_search()
    test_cache_invalidation()
    test_model_change_is_survivable()
    test_cascade_and_single_file()

    print(f"\n  {PASSED} passed, {FAILED} failed\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    sys.exit(code)
