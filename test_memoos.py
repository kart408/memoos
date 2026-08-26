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


def reports(test):
    """
    Make a test function fail its runner, not just print about it.

    `check` records a failure and keeps going, which is what lets one run
    report every broken assertion instead of only the first. But nothing
    raised, so under pytest — which calls these functions directly and
    never reaches `main()` — a suite with failures still reported "9
    passed". Every assertion in this file was decorative. Wrapping each
    test lets it print its whole tally and then still fail.
    """
    def run(*args, **kwargs):
        before = FAILED
        result = test(*args, **kwargs)
        broken = FAILED - before
        if broken:
            raise AssertionError(f"{test.__name__}: {broken} check(s) failed "
                                 f"(see the FAIL lines above)")
        return result
    run.__name__ = test.__name__
    run.__doc__ = test.__doc__
    return run


def section(title: str) -> None:
    print(f"\n{title}")


def files_on_disk() -> set:
    directory = config.containers_dir()
    if not os.path.isdir(directory):
        return set()
    return {name for name in os.listdir(directory) if name.endswith(".db")}


# ------------------------------------------------------------------ paths

@reports
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

@reports
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


@reports
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

@reports
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


@reports
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


@reports
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

@reports
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


# ------------------------------------------------------------ autodistil

@reports
def test_autodistil_lock() -> None:
    section("auto-distil")

    from memoos_core import autodistill

    # Keep the real ~/.memoos/locks out of it.
    autodistill.LOCK_DIR = os.path.join(SCRATCH, "locks")

    ok("nothing held to begin with", autodistill._held_by("demo") is None)

    path = autodistill._lock_path("demo")
    autodistill.claim(path)
    check("claiming records this process", autodistill._held_by("demo"), os.getpid())

    # Distillation takes tens of seconds and commands arrive far faster.
    # Without this, a busy session would spawn a distil per command.
    check("a second distil is refused while one runs",
          autodistill.spawn_if_idle("demo"), False)

    autodistill.release(path)
    ok("releasing frees it", autodistill._held_by("demo") is None)

    # A distil killed mid-run leaves its lock behind. That must read as
    # free, or the container would never distil again.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("999999")
    ok("a lock from a dead process is not honoured",
       autodistill._held_by("demo") is None)
    ok("and is cleaned up", not os.path.exists(path))

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("not-a-pid")
    ok("a corrupt lock is not honoured either",
       autodistill._held_by("demo") is None)

    ok("locks are per container",
       autodistill._lock_path("one") != autodistill._lock_path("two"))


@reports
def test_pending_count() -> None:
    section("pending")

    journal = Journal()
    check("an unknown container has nothing pending",
          journal.pending_count("nobody"), 0)

    journal.record("command", "make", container="counted",
                   session_id="s", exit_code=0)
    journal.record("command", "make test", container="counted",
                   session_id="s", exit_code=0)
    check("both events are pending", journal.pending_count("counted"), 2)

    events = journal.events("counted", undistilled_only=True)
    journal.mark_distilled([events[0]["id"]], "counted")
    check("distilling one leaves one", journal.pending_count("counted"), 1)


# ------------------------------------------------------------ regressions

@reports
def test_container_names_are_one_spelling() -> None:
    section("names")

    # The file is chosen by `safe_container`; the WHERE clause used to
    # use whatever the caller passed. A name that normalises therefore
    # wrote into the right file under a label nothing else queried —
    # two halves of one project, in one file, invisible to each other.
    journal = Journal()
    journal.record("command", "make", container="MixedCase", session_id="s")
    journal.record("command", "make test", container="mixedcase", session_id="s")

    check("both spellings reach the same events",
          len(journal.events("MixedCase")), 2)
    check("and it does not matter which one you ask with",
          len(journal.events("mixedcase")), 2)
    check("counts agree too",
          journal.pending_count("MixedCase"), journal.pending_count("mixedcase"))
    ok("one file, not two",
       "mixedcase.db" in files_on_disk() and "MixedCase.db" not in files_on_disk())


@reports
def test_schema_survives_the_file_being_deleted() -> None:
    section("clear")

    # `memoos clear` deletes a container's file. A long-lived process —
    # the dashboard — had already cached "schema applied" for that path,
    # so it reconnected to the fresh empty file, skipped the DDL, and
    # every read failed with `no such table: events`.
    journal = Journal()
    journal.record("command", "make", container="wiped", session_id="s")
    path = config.db_path(container="wiped")
    for suffix in ("", "-wal", "-shm"):
        if os.path.exists(path + suffix):
            os.remove(path + suffix)

    journal.record("command", "make again", container="wiped", session_id="s")
    check("journalling works again after a clear",
          [e["text"] for e in journal.events("wiped")], ["make again"])


@reports
def test_global_container_reaches_every_command() -> None:
    section("cli")

    # `--container` is a global option. `clear` and `doctor` each declared
    # one of their own, and argparse let the subparser default (None)
    # overwrite the global value — so `memoos --container other clear`
    # deleted the project you were standing in instead.
    import memoos_cli

    parser = memoos_cli.build_parser()
    for command in ("recall", "clear", "doctor", "distill", "graph", "stats"):
        check(f"--container survives `{command}`",
              parser.parse_args(["--container", "elsewhere", command]).container,
              "elsewhere")


# ------------------------------------------------------------- pipeline

@reports
def test_query_expansion() -> None:
    section("query understanding")

    # A question and the memory that answers it often share no token at
    # all: "how do I deploy?" against "deployed on Vercel". Vector search
    # bridges that; BM25 abstains. Expansion reads the concept back out
    # of the graph so the keyword half has something to match on.
    from memoos_core import MemoOS
    from memoos_core.models import (ExtractedEntity, ExtractedMemory,
                                    MemoryType)

    memo = MemoOS(container="expand")
    facts = [
        ("User's application is deployed on Vercel.", ["Vercel"]),
        ("User's project uses Next.js.", ["Next.js"]),
        ("User has a sister named Priya.", ["Priya"]),
    ]
    for text, names in facts:
        stored = memo.add(text)
        memo.graph.attach(stored, ExtractedMemory(
            text=text, memory_type=MemoryType.FACT,
            entities=[ExtractedEntity(name=n) for n in names]))

    plan = memo.retriever.plan_for("how should I deploy my application?")
    ok("the question's own words are the terms", "deploy" in plan.terms)
    ok("a concept is read back out of the store", "Vercel" in plan.concepts)

    # The probe floor is what stops this harvesting the whole container.
    # Without it a small store returns every memory as a "neighbour", and
    # BM25 then matches everything on somebody's sister.
    ok("an unrelated entity is not harvested", "Priya" not in plan.concepts)

    family = memo.retriever.plan_for("who is in my family?")
    ok("and the reverse holds too", "Priya" in family.concepts)
    ok("expansion stays bounded",
       len(plan.concepts) <= config.QUERY_MAX_CONCEPTS)

    # Expansion is a hint for the lexical side only. Embedding the padded
    # text would move the query vector off what was actually asked.
    ok("the concepts reach the keyword text",
       "Vercel" in plan.keyword_text())
    ok("and the original question survives in it",
       "deploy" in plan.keyword_text().lower())
    memo.close()


@reports
def test_validity_window() -> None:
    section("validity")

    # Status and validity answer different questions. A superseded memory
    # is not false — it stopped being current, and "what did I use
    # before?" is only answerable if both facts survive.
    from memoos_core import MemoOS

    memo = MemoOS(container="valid")
    old = memo.add("User's application uses MongoDB.")
    ok("a new memory is current", memo.get(old.id).is_current())
    check("and open-ended", memo.get(old.id).valid_until, None)

    new = memo.supersede(old.id, "User's application uses PostgreSQL.")
    retired = memo.get(old.id)
    ok("supersession closes the interval", retired.valid_until is not None)
    ok("so it is no longer current", not retired.is_current())
    ok("but the row is still there", retired.text.endswith("MongoDB."))
    ok("and the replacement is current", memo.get(new.id).is_current())

    # One statement, not two: a crash between a status write and a
    # validity write would leave a retired memory reading as current.
    ok("status and validity agree",
       retired.status.value == "superseded" and retired.valid_until is not None)

    current = [r.memory.text for r in
               memo.retriever.search("what database?", top_k=5, touch=False)]
    ok("retrieval filters the outdated one out",
       "User's application uses MongoDB." not in current)
    ok("and returns the one that replaced it",
       any("PostgreSQL" in t for t in current))

    # Supersession un-indexes as well as retires, so no search flag
    # brings it back — that is deliberate, and it is why the chain is
    # walked in SQLite instead, where the row still lives.
    chain = [m.text for m in memo.history(old.id)]
    ok("the history is still walkable", any("MongoDB" in t for t in chain))
    ok("and it names what replaced it",
       any("PostgreSQL" in t for t in chain))
    memo.close()


@reports
def test_migration_from_an_older_store() -> None:
    section("migration")

    # The only file that matters is the one that already exists. A schema
    # grown a column applies cleanly to a fresh database and not at all
    # to that one, so the upgrade has to be explicit — and the index over
    # the new column has to come after it, or the whole script fails at
    # open with `no such column`.
    import sqlite3

    path = os.path.join(config.containers_dir(), "legacy.db")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript("""
            CREATE TABLE memories (
                id TEXT PRIMARY KEY, container TEXT NOT NULL, text TEXT NOT NULL,
                memory_type TEXT NOT NULL DEFAULT 'fact',
                source TEXT NOT NULL DEFAULT 'user',
                document_id TEXT, chunk_id TEXT,
                importance REAL NOT NULL DEFAULT 0.5,
                confidence REAL NOT NULL DEFAULT 0.8,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                last_accessed_at TEXT, access_count INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'active',
                superseded_by TEXT, superseded_at TEXT, supersede_reason TEXT,
                metadata TEXT NOT NULL DEFAULT '{}');
        """)
        conn.execute(
            """INSERT INTO memories
               (id, container, text, created_at, updated_at, superseded_at, status)
               VALUES ('a','legacy','an older memory',
                       '2020-01-01T00:00:00+00:00','2020-01-01T00:00:00+00:00',
                       NULL,'active')""")
        conn.execute(
            """INSERT INTO memories
               (id, container, text, created_at, updated_at, superseded_at, status)
               VALUES ('b','legacy','one that was replaced',
                       '2020-01-01T00:00:00+00:00','2021-06-01T00:00:00+00:00',
                       '2021-06-01T00:00:00+00:00','superseded')""")

    db = Database(path)
    check("nothing was lost", db.count_memories("legacy", None), 2)

    kept = db.get_memory("a")
    ok("valid_from is back-filled from created_at",
       kept.valid_from is not None and kept.valid_from.year == 2020)
    ok("an un-superseded memory stays open", kept.valid_until is None)
    ok("and reads as current", kept.is_current())

    replaced = db.get_memory("b")
    ok("a superseded one gets its interval closed from superseded_at",
       replaced.valid_until is not None and replaced.valid_until.year == 2021)
    ok("so an upgraded store knows its own history",
       not replaced.is_current())
    db.close()


@reports
def test_context_block_is_the_handover() -> None:
    section("context")

    # This block goes into somebody else's prompt. It is read by a model
    # that has never seen the project, so it names what it describes.
    from memoos_core import MemoOS

    memo = MemoOS(container="handover")
    memo.add("The project uses Next.js.")
    memo.add("User prefers TypeScript.")

    result = memo.context_for("what stack is this?", top_k=5)
    block = result["context"]
    ok("it names the project", block.startswith("Project: handover"))
    ok("facts are bulleted", "\u2022 " in block)
    ok("and the memories behind it come back too", bool(result["results"]))
    ok("along with what it searched for", "terms" in result["plan"])

    # MemoOS stops here. Anything that looks like an answer would be this
    # layer guessing at a task only the calling agent can see.
    ok("nothing generated is returned", "answer" not in result)
    memo.close()


@reports
def test_memory_traces_back_to_its_source() -> None:
    section("provenance")

    # A memory store you cannot audit is one you cannot trust: the
    # failure that matters is a confidently-retrieved fact nobody ever
    # stated, and the only way to tell that from a real one is to follow
    # it back.
    from memoos_core import MemoOS
    from memoos_core.models import Document, Memory

    memo = MemoOS(container="trace")
    document = Document(container="trace", title="terminal session",
                        source="terminal",
                        raw_text="The user ran the tests, and it failed.",
                        metadata={"event_ids": ["evt-1"],
                                  "session_ids": ["sess-1"]})
    memo.db.insert_document(document)
    memo.db.insert_chunks([("chunk-1", document.id, "trace", 0,
                            "The user ran the tests, and it failed.")])
    stored = Memory(container="trace", text="The tests were failing.",
                    document_id=document.id, chunk_id="chunk-1")
    memo.db.insert_memory(stored)

    trail = memo.source(stored.id)
    ok("the memory itself", trail["memory"].id == stored.id)
    ok("the exact passage the model read",
       trail["chunk"] == "The user ran the tests, and it failed.")
    ok("the document it came from", trail["document"].id == document.id)
    ok("and the session it derives from", trail["sessions"] == ["sess-1"])
    ok("an unknown memory traces to nothing", memo.source("nope") is None)
    memo.close()


@reports
def test_writes_survive_an_unreachable_model() -> None:
    section("model down")

    # Distillation runs in the background when a terminal closes, which
    # is the worst place for a hard failure: nobody is watching. A
    # refused connection used to raise straight out through
    # `judge_conflict` — which documents the opposite — and take the
    # whole write with it.
    from memoos_core import config as live
    from memoos_core import llm

    was = live.OLLAMA_URL
    live.OLLAMA_URL = "http://127.0.0.1:59999"   # nothing listens here
    try:
        from memoos_core import MemoOS

        memo = MemoOS(container="offline")
        memo.add("User's application uses MongoDB.")
        memo.add("User's application uses PostgreSQL.")
        check("both writes land anyway", len(memo.all(limit=10)), 2)

        ok("a failed call parses as nothing, not an exception",
           llm.generate_json("anything") is None)

        # The conflict judge's whole bias is that losing a memory is
        # worse than keeping a redundant one, so an unreachable model
        # must read as "independent" rather than as a crash.
        from memoos_core.extraction import judge_conflict
        check("an unreachable judge abstains",
              judge_conflict("User lives in Delhi.",
                             "User lives in Mumbai.").decision.value,
              "independent")

        # An ingest degrades rather than exploding, and says so — which
        # is what lets the caller leave its events pending for a retry.
        result = memo.ingest_document("The user ran the tests, and it failed.")
        check("the chunk is counted as failed", result.chunks_failed, 1)
        ok("and the result knows it is incomplete", not result.complete)
        memo.close()
    finally:
        live.OLLAMA_URL = was


@reports
def test_entity_names_worth_having() -> None:
    section("entities")

    # A graph node has to be a thing the project is *about*. An absolute
    # path is a location on one machine — the same file under a name
    # nobody else will ever write — and it surfaces in query expansion as
    # a wall of text that says nothing.
    from memoos_core.extraction import _is_junk_entity

    for name in ("/Users/someone/proj/venv/bin/activate", "~/.zshrc",
                 "a/b/c/d.py", "tech", "user", "2026-08-26", "7", "x",
                 "# MemoOS \u2014 Build Instructions for Claude Code",
                 "- a bulleted line scraped out of a document"):
        ok(f"dropped: {name}", _is_junk_entity(name))

    for name in ("api.py", "test_auth.py", "PostgreSQL", "Next.js",
                 "src/auth/login.ts", "pyjwt"):
        ok(f"kept: {name}", not _is_junk_entity(name))


@reports
def test_relation_only_entities_are_linked() -> None:
    section("graph links")

    # An entity named only inside a relation triple was created and
    # counted, but never linked to the memory — so it claimed a mention
    # it could not show, and graph expansion could never reach it, since
    # expansion travels through memory_entities.
    from memoos_core import MemoOS
    from memoos_core.graph import USER_NORM_NAME
    from memoos_core.models import (ExtractedEntity, ExtractedMemory,
                                    ExtractedRelation, Memory)

    memo = MemoOS(container="links")
    stored = Memory(container="links", text="The master branch was renamed to main.")
    memo.db.insert_memory(stored)
    memo.graph.attach(stored, ExtractedMemory(
        text=stored.text,
        entities=[ExtractedEntity(name="master")],           # listed
        relations=[ExtractedRelation(subject="master",
                                     predicate="renamed_to",
                                     object="main"),         # `main` is not
                   ExtractedRelation(subject="User",
                                     predicate="renamed",
                                     object="master")]))

    linked = memo.db.entity_ids_for_memories([stored.id])[stored.id]
    names = {e.name for e in memo.db.get_entities(linked).values()}
    ok("the listed entity is linked", "master" in names)
    ok("and so is the one only a relation named", "main" in names)

    for entity in memo.db.list_entities("links", limit=50):
        if entity.norm_name == USER_NORM_NAME:
            # The reserved node may appear in relations but must never be
            # linked: linking it would attach it to every memory in the
            # store and collapse expansion into a star.
            ok("the reserved user node stays unlinked",
               entity.id not in linked)
            continue
        actual = len(memo.db.memories_for_entities("links", [entity.id], limit=99))
        check(f"{entity.name}: count matches its links",
              entity.mention_count, actual)
    memo.close()


@reports
def test_a_request_is_not_a_memory() -> None:
    section("dumping ground")

    # The single biggest source of junk in a real terminal store: every
    # imported Claude prompt is literally somebody asking for something,
    # and the digest used to assert it as fact. 38 of 51 memories in the
    # shipped store were records of a request — "User wanted to commit
    # the changes", true for thirty seconds, stored forever, and ranked
    # against facts that still hold.
    from memoos_core.extraction import is_transient_intent
    from memoos_core.terminal import build_digest, is_noise
    from memoos_core.journal import CLAUDE

    # Markdown swallowed into a memory's text: the model describing the
    # shape of a file it read, rather than anything the project is or does.
    from memoos_core.extraction import _is_scaffolding
    for text in ("The project uses # MemoOS — Build Instructions as its docs.",
                 "Files involved: api.py, db.py"):
        ok(f"scaffolding dropped: {text[:38]}", _is_scaffolding(text))
    for text in ("The C# codebase uses NuGet.", "Issue #42 was closed.",
                 "The project uses JWT for authentication."):
        ok(f"real text kept: {text[:38]}", not _is_scaffolding(text))

    for text in ("User wanted to commit the changes.",
                 "The user wanted to run the demo.",
                 "User wants to kill process 22742.",
                 "The user decided to rename 'master' branch to 'main'.",
                 "The user asked for cleaning up unnecessary files.",
                 "The user asked for killing process ID 22742.",
                 "User wants to fix the issue with ollama."):
        ok(f"dropped: {text[:44]}", is_transient_intent(text))

    # A lasting aim shares the frame but not the fate, and an outcome is
    # always welcome — that is the sentence worth keeping.
    for text in ("User wants to build a local-first memory layer.",
                 "The project uses JWT for authentication.",
                 "The master branch was renamed to main.",
                 "Editing api.py made the authentication tests pass.",
                 "User prefers TypeScript."):
        ok(f"kept: {text[:44]}", not is_transient_intent(text))

    # The digest is where this is really fixed: it no longer asserts that
    # the user wanted anything, so the extractor has nothing to record.
    digest = build_digest("proj", [{"kind": CLAUDE, "text": "Convert auth to JWT"}])
    ok("the digest marks a prompt as a request", "asked for" in digest)
    ok("and no longer asserts wanting", "wanted to" not in digest)

    # Operating a machine is not working on a project.
    for command in ("sleep 60", "kill 23643 && pkill ollama", "ps aux"):
        ok(f"noise: {command}", is_noise(command))
    for command in ("git commit -m x", "npm install pyjwt", "pytest"):
        ok(f"real work: {command}", not is_noise(command))


@reports
def test_past_signals_are_surfaced_not_acted_on() -> None:
    section("signals")

    # A session is mostly things breaking and then being made to work,
    # and that arc is the most valuable thing in the store. Filed as
    # loose facts the two halves sit in separate rows with nothing saying
    # they are one story.
    from memoos_core import MemoOS, signals
    from memoos_core.models import (Document, ExtractedEntity,
                                    ExtractedMemory, Memory, MemoryType)

    memo = MemoOS(container="episodes")
    document = Document(container="episodes", source="terminal")
    memo.db.insert_document(document)

    def remember(text, kind, entity):
        stored = Memory(container="episodes", text=text, memory_type=kind,
                        document_id=document.id)
        memo.db.insert_memory(stored)
        memo.vectors.add_one(stored.id, stored.text)
        memo.graph.attach(stored, ExtractedMemory(
            text=text, memory_type=kind,
            entities=[ExtractedEntity(name=entity)]))
        return stored

    # The fix lives somewhere other than the thing that broke, which is
    # the ordinary case and the one an entity-overlap rule alone misses.
    problem = remember("The tests in test_auth.py were failing.",
                       MemoryType.PROBLEM, "test_auth.py")
    remember("Editing api.py made the authentication tests pass.",
             MemoryType.SOLUTION, "api.py")
    orphan = remember("The deploy step times out on Vercel.",
                      MemoryType.PROBLEM, "Vercel")

    found = {e.problem.id: e for e in memo.signals()}
    check("both problems are reported", len(found), 2)
    ok("the pair is matched across different files",
       found[problem.id].resolved)
    check("with the fix attached",
          found[problem.id].solutions[0].text,
          "Editing api.py made the authentication tests pass.")
    ok("a problem with no fix reads as open", not found[orphan.id].resolved)

    # Open problems first: the thing least likely to be written down
    # anywhere else is the thing worth leading with.
    ordered = memo.signals()
    ok("unresolved leads", not ordered[0].resolved)

    # Surfaced, never acted on. Nothing here may retire a memory or
    # reorder a ranking — deciding what to do about a known failure needs
    # to know the task, and a memory layer cannot see it.
    before = [m.id for m in memo.all(limit=50)]
    ranked_before = [r.memory.id for r in
                     memo.retriever.search("auth", top_k=5, touch=False)]
    memo.signals()
    check("no memory was retired", [m.id for m in memo.all(limit=50)], before)
    check("no ranking was changed",
          [r.memory.id for r in memo.retriever.search("auth", top_k=5, touch=False)],
          ranked_before)

    # And they reach the agent through the handover.
    block = memo.context_for("touching the auth code", top_k=5)["context"]
    ok("the block carries the signals", "what fixed them" in block
       or "Open problems" in block)
    ok("under a heading of their own, not mixed in with the facts",
       "Project: episodes" in block)

    rendered = signals.render(memo.signals(), unresolved_only=True)
    ok("unresolved_only drops the solved ones", "\u21b3" not in rendered)
    memo.close()


@reports
def test_out_of_scope_questions_return_nothing() -> None:
    section("scope gate")

    # Search returns its nearest neighbours however distant — that is
    # what nearest-neighbour means, and for `search()` it is the right
    # contract, since the caller can see the scores. A context block
    # cannot: it goes into somebody else's prompt with every number
    # stripped off. Asked who won the world cup, a real store offered
    # three memories about itself at cosine 0.49.
    from memoos_core import config
    from memoos_core.models import Memory, MemoryQueryResult
    from memoos_core.retrieval import in_scope

    CORR = config.RECALL_MIN_SIMILARITY_CORROBORATED   # 0.40
    VEC = config.RECALL_MIN_SIMILARITY_VECTOR_ONLY     # 0.48

    def result(text, similarity, matched_by):
        return MemoryQueryResult(
            memory=Memory(container="gate", text=text),
            score=0.016, vector_score=similarity, matched_by=list(matched_by))

    # --- borderline: the calibrated bars, exactly ---
    at_bar = result("the deploy runs on Vercel", VEC, ["vector"])
    under = result("the deploy runs on Vercel", VEC - 0.01, ["vector"])
    check("a vector-only hit exactly at the bar is kept",
          len(in_scope([at_bar])), 1)
    check("one hair under it is not", len(in_scope([under])), 0)

    # Agreement from a second retriever is evidence, and buys the lower
    # bar — but only when it rests on the user's own words.
    agreed = result("the deploy runs on Vercel", CORR, ["vector", "keyword"])
    check("corroborated, at the lower bar, sharing a word with the query",
          len(in_scope([agreed], query="how does the deploy work")), 1)
    check("the same hit is held to the strict bar when the query shares nothing",
          len(in_scope([agreed], query="who is my sister")), 0)

    # The reason that distinction exists: expansion can manufacture
    # agreement. Asked "who is my sister", the probe harvested `main` and
    # `master` off the nearest memories and searched for "who is my
    # sister main master"; the keyword retriever matched on two words the
    # user never typed and the vector search had just invented. That is
    # echoing, not agreeing, and it let cosine 0.428 through a 0.48 bar.
    echoed = result("The master branch was renamed to main.", 0.428,
                    ["vector", "keyword"])
    check("manufactured corroboration does not buy the lower bar",
          len(in_scope([echoed], query="who is my sister")), 0)

    # A hit with no vector opinion arrived on a literal token or a shared
    # entity. That is concrete evidence, not a distance, and judging it
    # by a cosine it does not have would throw away the half of hybrid
    # search that exists for rare names and IDs.
    lexical = result("deploy id 8fa21c failed", None, ["keyword"])
    check("a keyword-only hit is kept", len(in_scope([lexical])), 1)

    # --- in scope vs out of scope, end to end ---
    from memoos_core import MemoOS

    memo = MemoOS(container="scope")
    for text in ("The project uses PostgreSQL as its database.",
                 "The project is deployed on Vercel.",
                 "The project uses Next.js for the frontend."):
        memo.add(text)

    asked = memo.context_for("what database does the project use", top_k=3)
    ok("an in-scope question still returns memories", bool(asked["results"]))
    ok("and the right one leads",
       "PostgreSQL" in asked["results"][0].memory.text)
    ok("with a context block to hand over", "PostgreSQL" in asked["context"])

    for question in ("what is 2 plus 2",
                     "what is the capital of France"):
        empty = memo.context_for(question, top_k=3)
        check(f"out of scope returns nothing: {question!r}",
              len(empty["results"]), 0)
        check("and an empty block rather than a confident wrong one",
              empty["context"], "")
        # Nothing relevant means nothing to scope signals to. Reporting
        # them anyway would answer an out-of-scope question with the
        # project's entire failure history.
        check("and no signals", len(empty["signals"]), 0)

    # The gate is on the handover, not on search: a caller that can see
    # the scores still gets everything, which is the existing contract.
    ungated = memo.context_for("what is 2 plus 2", top_k=3, gate=False)
    ok("search itself is unchanged", bool(ungated["results"]))
    memo.close()


def main() -> int:
    print(f"scratch store: {SCRATCH}")
    for test in (test_paths, test_journal_isolation,
                 test_reads_do_not_create_files, test_vector_search,
                 test_cache_invalidation, test_model_change_is_survivable,
                 test_cascade_and_single_file, test_pending_count,
                 test_autodistil_lock, test_container_names_are_one_spelling,
                 test_schema_survives_the_file_being_deleted,
                 test_global_container_reaches_every_command,
                 test_query_expansion, test_validity_window,
                 test_migration_from_an_older_store,
                 test_context_block_is_the_handover,
                 test_memory_traces_back_to_its_source,
                 test_writes_survive_an_unreachable_model,
                 test_entity_names_worth_having,
                 test_a_request_is_not_a_memory,
                 test_past_signals_are_surfaced_not_acted_on,
                 test_out_of_scope_questions_return_nothing,
                 test_relation_only_entities_are_linked):
        try:
            test()
        except AssertionError:
            # Already printed by `check`. Keep going so one run reports
            # everything that is broken, not just the first thing.
            pass

    print(f"\n  {PASSED} passed, {FAILED} failed\n")
    return 1 if FAILED else 0


if __name__ == "__main__":
    try:
        code = main()
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)
    sys.exit(code)
