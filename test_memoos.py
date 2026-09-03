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
  dashboard   the server stops when the terminal that started it does

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

    # Memory types were cut to four characters to line the column up,
    # which prints `[even]` and `[prob]` — not words, and `[pref]` and
    # `[prob]` differ by one letter at a glance. The column still has to
    # line up, so the label is padded rather than truncated.
    from memoos_core.models import MemoryType

    # --- the handover has to publish the number that means similarity ---
    #
    # `--json` exposed `score` alone, which is RRF: it encodes rank and
    # tops out near 1/(RRF_K + 1), about 0.016. An agent thresholding on
    # it makes the category error `in_scope` documents, and one reading
    # 0.0153 as near-zero confidence throws away a good memory. The
    # cosine is the number the bars are calibrated against, so it ships
    # alongside.
    import io
    import json as jsonlib
    from contextlib import redirect_stdout

    from memoos_core import MemoOS
    from memoos_core.models import Memory, MemoryQueryResult

    ctx = MemoOS(container="ctxjson")
    ctx.add("The project uses PostgreSQL for storage.")
    ctx.add("The project runs its tests with pytest.")
    ctx.close()

    ctx_args = memoos_cli.build_parser().parse_args(
        ["--container", "ctxjson", "context", "--json", "what database is used"])
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        memoos_cli.cmd_context(ctx_args)
    payload = jsonlib.loads(buffer.getvalue())

    ok("the block is still valid json", bool(payload["memories"]))
    for entry in payload["memories"]:
        for field in ("vector_score", "keyword_score", "strength"):
            ok(f"every result carries {field}", field in entry)
            # Null is a real answer, and the key stays present to say so.
            # A hit that arrived on distance has no BM25 opinion and one
            # that arrived on a literal token has no cosine; a zero would
            # read as "measured, and far", which is a different claim and
            # a false one.
            value = entry[field]
            ok(f"{field} is a number or an honest null",
               value is None or isinstance(value, float))

    scored = [e for e in payload["memories"] if e["vector_score"] is not None]
    ok("the cosine is not the rank score wearing a different name",
       all(abs(e["score"] - e["vector_score"]) > 0.1 for e in scored))
    ok("the rank score still tops out where RRF does",
       all(e["score"] < 0.1 for e in payload["memories"]))

    # None of them may be coerced on the way out. Serialising a missing
    # number as 0.0 would be indistinguishable from a measured zero.
    absent = MemoryQueryResult(
        memory=Memory(container="ctxjson", text="The API key is AKIA7Q."),
        score=0.0163, matched_by=["keyword"])
    encoded = jsonlib.dumps({"vector_score": absent.vector_score,
                             "keyword_score": absent.keyword_score,
                             "strength": absent.strength})
    check("an unmeasured number serialises as null, never as zero",
          encoded, '{"vector_score": null, "keyword_score": null, '
                   '"strength": null}')

    labels = memoos_cli.type_labels([t.value for t in MemoryType])
    for memory_type, label in zip(MemoryType, labels):
        ok(f"{memory_type.value} is spelled out", memory_type.value in label)
    check("every label is the same width", len(set(len(t) for t in labels)), 1)
    check("an empty column does not blow up",
          memoos_cli.type_labels([]), [])

    # `distill` reports the nodes it touched, and one entry arrives per
    # *attachment* — so a node mentioned by two memories printed as
    # `start_demo.sh, start_demo.sh`, which reads as two nodes sitting
    # side by side when the graph holds one.
    from memoos_core.models import Entity

    node = Entity(container="cli", name="start_demo.sh", norm_name="start demo sh")
    other = Entity(container="cli", name="api.py", norm_name="api py")
    check("a node attached twice is named once",
          memoos_cli.entity_names([node, node, other]),
          ["start_demo.sh", "api.py"])

    # Deduped by id, not by label: `upsert_entity` matches on the
    # normalised name, so two spellings are one node arriving twice.
    spelled = Entity(container="cli", name="Start_Demo.sh",
                     norm_name="start demo sh")
    spelled.id = node.id
    check("and two spellings of one node are still one",
          memoos_cli.entity_names([node, spelled]), ["start_demo.sh"])
    check("nothing touched, nothing named", memoos_cli.entity_names([]), [])


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

    # A `./` prefix is how a shell writes a file in the current
    # directory, not part of its name. Stripping punctuation and the
    # leading dot in one pass turned `./api.py` into `/api.py`, which the
    # absolute-path rule above then dropped — so the same file was a node
    # when the model wrote `api.py` and vanished when it wrote `./api.py`.
    # A distil whose only named thing was a `./script.sh` left an empty
    # graph, and nothing said why.
    from memoos_core.extraction import _parse_entities

    for name in ("./api.py", "./start_demo.sh", "`./api.py`",
                 "../lib/util.py", ".env"):
        ok(f"a relative path is kept: {name}", not _is_junk_entity(name))

    ok("an absolute path still is not", _is_junk_entity("/Users/someone"))

    # And the two spellings have to land on one node, not two.
    check("the prefix is dropped from the stored name",
          [e.name for e in _parse_entities([{"name": "./start_demo.sh"},
                                            {"name": "start_demo.sh"}])],
          ["start_demo.sh", "start_demo.sh"])


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

    # --- the prompt and its own validator have to agree ---
    #
    # The extraction prompt names two subjects, "User" for the person and
    # "The project" for the codebase, and the guard under it checked for
    # the literal word "user" — so four of the prompt's own worked
    # examples failed it. Silently: an empty extraction looked exactly
    # like a session with nothing in it.
    from memoos_core.extraction import names_known_subject

    for text in ("User moved to Delhi.",
                 "User switched from PyTorch to JAX.",
                 "The project uses JWT for authentication.",
                 "The project uses the pyjwt library."):
        ok(f"a documented subject is kept: {text[:34]}", names_known_subject(text))

    for text in ("The message mentions a script.",
                 "This text describes an installation."):
        ok(f"narration is dropped: {text[:34]}", not names_known_subject(text))

    # A known, deliberate gap: these name neither subject, so they still
    # fail. Asserted so the gap is a decision on the record rather than a
    # surprise the next time somebody reads the prompt.
    for text in ("The authentication tests in test_auth.py were failing.",
                 "Editing api.py made the authentication tests pass."):
        ok(f"still dropped, knowingly: {text[:34]}", not names_known_subject(text))

    # The word being present is not the sentence being about it. Searched
    # across the whole text, `\buser\b` passed "A user guide was written"
    # — a fact about documentation — through a guard whose only job is
    # deciding what the sentence is about.
    for text in ("A user guide was written.",
                 "Documentation for the user was updated.",
                 "Zomato hired a new user researcher."):
        ok(f"the word is not the subject: {text[:34]}",
           not names_known_subject(text))

    # Every form the prompt mandates still passes, article and possessive
    # included: the anchor moved, the vocabulary did not.
    for text in ("User moved to Delhi.",
                 "User is doing an internship at Zomato.",
                 "The user ran `./start_demo.sh`.",
                 "User's application is deployed on Vercel.",
                 "The user's manager is Anjali.",
                 "The project uses JWT for authentication.",
                 "The project's tests run with pytest."):
        ok(f"a mandated subject form is kept: {text[:34]}",
           names_known_subject(text))

    # --- an empty extraction has to say which kind of empty it is ---
    from memoos_core import extraction

    real = extraction.generate_json
    extraction.generate_json = lambda *a, **k: {"memories": [
        {"text": "The project uses JWT for authentication.", "type": "fact"},
        {"text": "The message mentions a script.", "type": "fact"},
        {"text": "User wanted to commit the changes.", "type": "event"},
    ]}
    try:
        report = {}
        kept = extraction.extract_memories(
            "The project uses JWT for authentication and commits were made.",
            subject_scoped=True, check_grounding=False, report=report)
        check("only the grounded, subject-named one survives", len(kept), 1)
        check("every candidate is counted", report["candidates"], 3)
        check("and the count that survived", report["kept"], 1)
        check("narration is attributed to the subject guard",
              report.get("subject"), 1)
        check("a request is attributed to the intent guard",
              report.get("intent"), 1)

        # Nothing rejected must stay quiet, or the note becomes noise on
        # every healthy distil.
        extraction.generate_json = lambda *a, **k: {"memories": []}
        quiet = {}
        extraction.extract_memories("nothing here", subject_scoped=True,
                                    report=quiet)
        check("a genuinely silent model reports no candidates",
              quiet["candidates"], 0)
    finally:
        extraction.generate_json = real

    # The CLI turns that into a line only when there is something to say.
    import memoos_cli

    check("silence stays silent",
          memoos_cli.rejection_note({"candidates": 0, "rejected_summary": "",
                                     "created": []}), "")
    ok("all-rejected is a warning",
       "none kept" in memoos_cli.rejection_note(
           {"candidates": 2, "rejected_summary": "subject \u00d72", "created": []}))
    ok("partly-rejected still says so",
       "1 kept" in memoos_cli.rejection_note(
           {"candidates": 2, "rejected_summary": "subject \u00d71", "created": [1]}))


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

    # --- and `recall` has to use the bars too ---
    #
    # They were calibrated and then wired into `context` alone, so
    # `memoos recall "why did the auth tests fail"` still answered a
    # question the store knew nothing about with its three least-bad
    # guesses. The contract above is unchanged — `search()` still returns
    # its neighbours and the scores are still printed — but a person
    # reading three lines in a terminal is not a caller who can act on a
    # cosine, and the state that means "I don't know about that" has to
    # reach them.
    import io
    from contextlib import redirect_stdout

    import memoos_cli

    cli = MemoOS(container="gatecli")
    cli.add("The project uses PostgreSQL for storage.")
    cli.close()

    def recall(*argv):
        args = memoos_cli.build_parser().parse_args(
            ["--container", "gatecli", "recall", *argv])
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            memoos_cli.cmd_recall(args)
        return buffer.getvalue()

    off_topic = recall("who won the world cup in 1998")
    ok("an out-of-scope question answers with nothing",
       "nothing relevant remembered yet" in off_topic)
    ok("and never with the memory it was not about",
       "PostgreSQL" not in off_topic)

    # Suppressed is not the same as absent. This is the command you reach
    # for when you are asking why the store said what it said, so what
    # the bar removed stays reachable.
    ok("but it says something was suppressed",
       "below the confidence bar" in off_topic)
    ok("--all brings it back",
       "PostgreSQL" in recall("--all", "who won the world cup in 1998"))

    # The gate must not swallow a question the store genuinely answers.
    on_topic = recall("what database does the project use")
    ok("a question it does know is still answered",
       "PostgreSQL" in on_topic)
    ok("with no suppression note when nothing was suppressed",
       "below the confidence bar" not in on_topic)

    # The note itself, all three branches. The partial case — some shown,
    # some below the bar — is the one you meet most often and the one a
    # live container cannot be relied on to produce, since which side of
    # the bar a result falls on depends on real cosines. It stayed
    # untested while the message was inline in `cmd_recall`.
    check("nothing suppressed says nothing",
          memoos_cli.suppression_note(0, any_shown=True), "")
    check("and stays quiet with no results either",
          memoos_cli.suppression_note(0, any_shown=False), "")

    partial = memoos_cli.suppression_note(1, any_shown=True)
    ok("some shown, some below the bar, reads as a footnote",
       "1 more below the confidence bar" in partial and "`--all`" in partial)

    total = memoos_cli.suppression_note(2, any_shown=False)
    ok("nothing shown says how to get at what was dropped",
       "2 below the confidence bar" in total and "to see them" in total)
    ok("and does not call them 'more' when none were shown",
       "more" not in total)

    # --- `--limit` counts what is shown, not what is fetched ---
    #
    # Fetching exactly the limit and then gating meant the bar ate slots
    # that results further down were ready to fill: a store with plenty
    # to say answered `--limit 8` with 7, or 2. The fetch widens so the
    # page can be filled; the pool is the ceiling because past it there
    # is nothing left to ask for.
    check("the window widens to leave room for the bar",
          memoos_cli.search_window(8, 60), 32)
    check("but never past the candidate pool",
          memoos_cli.search_window(40, 60), 60)
    check("and never below what was asked for",
          memoos_cli.search_window(80, 60), 80)

    def hit(name, similarity):
        return MemoryQueryResult(
            memory=Memory(container="page", text=name),
            score=0.016, vector_score=similarity, matched_by=["vector"])

    # Ranked best-first; the bar takes the 2nd and 3rd. A page of 3 has
    # to come back full, from further down, rather than one short.
    ranked = [hit("a", 0.90), hit("b", 0.20), hit("c", 0.20),
              hit("d", 0.80), hit("e", 0.70), hit("f", 0.60)]
    survivors = [r for r in ranked if r.vector_score > 0.48]
    page, suppressed = memoos_cli.gated_page(ranked, survivors, 3)
    check("the page is filled from further down", len(page), 3)
    check("with the best survivors, in order",
          [r.memory.text for r in page], ["a", "d", "e"])

    # The count is over the page asked for, not over everything fetched
    # to fill it: two of the top three were removed, and the three weak
    # ones deeper in the pool are not the user's business.
    check("suppressed counts this page, not the whole fetch", suppressed, 2)

    check("nothing suppressed when the bar took nothing",
          memoos_cli.gated_page(ranked, ranked, 3)[1], 0)
    check("and an empty store pages to nothing",
          memoos_cli.gated_page([], [], 3), ([], 0))


@reports
def test_api_validates_every_container_name() -> None:
    section("api guard")

    # Five endpoints took a container and never checked it. The split
    # showed as an inconsistency — the same name got 400 from /memories
    # and 200 from /stats — and `POST /note`, which writes, accepted
    # anything and brought a container file into existence named after
    # it. Traversal was never possible (safe_container collapses it),
    # but a write path with no guard on its tenant key is a bug whether
    # or not it is exploitable.
    import api

    # Rejected: characters that cannot be in a filename, and lengths that
    # cannot be one.
    for name in ("evil!!name", "has space", "with/slash", "", "a" * 600):
        try:
            api.valid_container(name)
            ok(f"rejected: {name[:18]!r}", False)
        except api.HTTPException as error:
            check(f"rejected with 400: {name[:18]!r}", error.status_code, 400)

    # Accepted and normalised: the name that picks the file has to be the
    # name the queries filter on, or a container writes rows into its own
    # file under a label nothing else ever reads.
    check("case is folded", api.valid_container("MyProj"), "myproj")
    check("leading punctuation is stripped", api.valid_container("-leading"), "leading")
    check("an already-safe name is unchanged", api.valid_container("memoos"), "memoos")

    # Every route taking a container must go through the guard, so this
    # cannot drift back apart one endpoint at a time.
    import inspect, re
    source = inspect.getsource(api)
    unguarded = []
    for block in re.split(r"\n@app\.", source)[1:]:
        route = block.split("\n")[0]
        name = re.search(r"def (\w+)", block)
        if "{container}" not in route or not name:
            continue
        body = block.split("\n@app.")[0]
        if "valid_container(" not in body and "layer(container)" not in body:
            unguarded.append(name.group(1))
    check("no route takes a container without validating it", unguarded, [])


@reports
def test_looking_around_is_not_work() -> None:
    section("noise filter")

    from memoos_core.journal import COMMAND
    from memoos_core.terminal import build_digest, is_noise

    # A command line is usually several commands, and the old rule gave
    # up on the first pipe — so `ps aux | grep -E "uvicorn|api.py"`
    # counted as real work and became a permanent memory, with `ps aux`
    # as a graph node on top. A pipeline is noise exactly when all of its
    # stages are.
    for command in ('ps aux | grep -E "uvicorn|api.py" | grep -v grep',
                    "find . -type d -name memoos",
                    "grep -rn TODO .",
                    "kill 5938 && pkill ollama",
                    "cat api.py", "cd memoos", "ls -la", "git status"):
        ok(f"noise: {command[:40]}", is_noise(command))

    for command in ("./start_demo.sh", "git commit -m x", "npm install pyjwt",
                    "pytest", "cat data.json | python load.py"):
        ok(f"real work: {command[:40]}", not is_noise(command))

    # zsh's AUTO_CD: a bare directory path *is* the command, so there is
    # no command word for NOISE to match. `cd /Users/me` was filtered as
    # navigation and `/Users/me` was not, and the gap became the memory
    # "The user ran `/Users/karthikreddy`" plus a graph node for a
    # directory on one machine.
    for command in ("/Users/karthikreddy", "~/projects/memoos", "..",
                    "../memoos", "/usr/local/bin"):
        ok(f"auto_cd is navigation: {command}", is_noise(command))

    # An extension is what separates going somewhere from running
    # something, and an argument settles it whatever the path looks like.
    for command in ("./start_demo.sh", "~/bin/build.sh",
                    "/usr/bin/python3 script.py"):
        ok(f"but running a path is work: {command}", not is_noise(command))

    # A redirect is what actually says something was written, and it
    # settles the line on its own.
    ok("a redirect makes it work", not is_noise("cat api.py > backup.py"))
    ok("even from a noise command", not is_noise("echo x > file"))

    # Separators inside quotes are arguments, not structure. Splitting on
    # them produced stages called `"uvicorn` and `api.py`, neither of
    # which is a command.
    ok("a pipe inside a quoted string is not a separator",
       not is_noise('git commit -m "fix a|b thing"'))

    # Arguments can turn an inspection into a write.
    ok("find that looks is noise", is_noise("find . -name '*.py'"))
    ok("find that deletes is not", not is_noise("find . -name '*.tmp' -delete"))

    # --- Ctrl-C is not a failure ---
    #
    # A shell reports a signalled process as 128 + signal. Stopping a
    # server you started on purpose was being written down as "the demo
    # script failed with exit code 130", which reads as a bug in the demo.
    stopped = build_digest("p", [{"kind": COMMAND, "text": "./start_demo.sh",
                                  "exit_code": 130}])
    ok("an interrupted command is not reported as failed",
       "failed" not in stopped)
    ok("but it is still remembered", "start_demo.sh" in stopped)

    for code in (1, 2, 139):   # 139 is SIGSEGV — a genuine crash
        broke = build_digest("p", [{"kind": COMMAND, "text": "./build.sh",
                                    "exit_code": code}])
        ok(f"exit {code} still reads as a failure", "failed" in broke)

    # --- and the shell verbs do not become graph nodes ---
    from memoos_core.extraction import _is_junk_entity

    for name in ("ps aux", "find", "grep", "kill", "cat", "sleep"):
        ok(f"not a node: {name}", _is_junk_entity(name))
    # Tool names that are genuine project facts must survive.
    for name in ("Docker", "pytest", "git", "npm", "api.py", "PostgreSQL"):
        ok(f"still a node: {name}", not _is_junk_entity(name))


# ------------------------------------------------------------- dashboard

@reports
def test_dashboard_dies_with_its_terminal() -> None:
    """
    The whole point of the parent watchdog.

    A dashboard that outlives its terminal is invisible and still bound
    to the port, so the next `memoos start` fails against a server the
    user has no window for. SIGKILLing the parent here reproduces the
    case a signal handler cannot: the shell goes without ever getting to
    hang anything up, and the only notice is the reparenting.

    The child exits 7 from inside the stop callback, which is also the
    assertion that the callback ran — rather than that something else
    happened to kill it.
    """
    section("dashboard")
    import subprocess
    import time

    root = os.path.dirname(os.path.abspath(__file__))
    child = (f"import os, sys, time; sys.path.insert(0, {root!r}); "
             "from memoos_core import dashboard; "
             "dashboard.stop_when_terminal_closes(lambda: os._exit(7), 0.1); "
             "time.sleep(30)")

    # An intermediate shell, so there is a parent to kill that is not the
    # test runner. It also makes the child a background job, which is
    # exactly where a SIGINT-based stop would be silently ignored.
    holder = subprocess.Popen(
        ["/bin/sh", "-c", f'{sys.executable} -c "$0" & echo $!; sleep 30', child],
        stdout=subprocess.PIPE, text=True)
    watched = int(holder.stdout.readline().strip())

    def alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True

    time.sleep(0.5)
    ok("a live parent is left alone", alive(watched))

    holder.kill()
    holder.wait(timeout=5)

    for _ in range(100):
        if not alive(watched):
            break
        time.sleep(0.1)
    orphaned = alive(watched)
    if orphaned:
        os.kill(watched, 9)
    ok("the server stops when the shell goes", not orphaned)


@reports
def test_hangup_is_a_shutdown_not_a_kill() -> None:
    """
    A closing terminal has to reach the teardown, not skip it.

    SIGHUP's default action ends the process where it stands, which is
    why an orphaned dashboard used to leave its browser tab behind. With
    the handler installed the exit code below is the child's own — proof
    that the stop callback ran — rather than a signalled death, which
    would report as -1.
    """
    import signal as _signal
    import subprocess

    root = os.path.dirname(os.path.abspath(__file__))
    child = (f"import os, sys, time; sys.path.insert(0, {root!r}); "
             "from memoos_core import dashboard; "
             "dashboard.stop_when_terminal_closes(lambda: os._exit(7)); "
             "print('ready', flush=True); time.sleep(30)")

    for name in ("SIGHUP", "SIGTERM"):
        process = subprocess.Popen([sys.executable, "-c", child],
                                   stdout=subprocess.PIPE, text=True)
        process.stdout.readline()      # wait until the handlers are installed
        process.send_signal(getattr(_signal, name))
        try:
            code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            code = None
        check(f"{name} runs the teardown instead of killing", code, 7)


@reports
def test_every_way_of_stopping_reaches_the_teardown() -> None:
    """
    The same test, against the command rather than the module.

    Worth its seconds because the bug it is here for lived in the
    *interaction*, where a unit test could not see it. uvicorn catches
    SIGINT and SIGTERM itself, and once it has shut down gracefully it
    restores the handler that was there before it started and re-raises
    the signal, so the exit status reports what stopped it. SIGHUP it
    never touches; SIGINT lands on Python's default and becomes a
    KeyboardInterrupt that unwinds normally. SIGTERM was the one with
    nothing of ours to restore, so the re-raise was the default action
    and the process died inside uvicorn's own shutdown, teardown and
    browser tab still ahead of it — while every log line up to that
    point said the shutdown had gone perfectly.

    `--no-open` keeps a browser out of it. What is being checked is that
    the teardown is reached at all, which is what the last line says.
    """
    import signal as _signal
    import socket as _socket
    import subprocess
    import time

    root = os.path.dirname(os.path.abspath(__file__))
    with _socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    def serving() -> bool:
        with _socket.socket() as client:
            client.settimeout(0.5)
            return client.connect_ex(("127.0.0.1", port)) == 0

    for name in ("SIGHUP", "SIGTERM", "SIGINT"):
        process = subprocess.Popen(
            [sys.executable, "memoos_cli.py", "serve",
             "--port", str(port), "--no-open"],
            cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True)
        for _ in range(200):
            if serving():
                break
            time.sleep(0.1)
        else:
            process.kill()
            ok(f"{name}: server came up", False)
            continue

        process.send_signal(getattr(_signal, name))
        try:
            output = process.communicate(timeout=30)[0]
        except subprocess.TimeoutExpired:
            process.kill()
            output = ""
        ok(f"{name} reaches the teardown", "dashboard stopped" in output)
        ok(f"{name} releases the port", not serving())


@reports
def test_a_detached_dashboard_still_belongs_to_its_shell() -> None:
    """
    `memoos start` puts the dashboard in its own session so the prompt
    can come back, and that throws away the parent link the watchdog
    reads: getppid() is init from the first moment and never changes
    again. The shell is named explicitly instead, and watched by whether
    it is still there — which is the same question, asked of a pid we
    were told rather than one we can look up.
    """
    import subprocess
    import time

    root = os.path.dirname(os.path.abspath(__file__))
    # A stand-in for the shell: something with a pid, that we can end.
    shell = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])

    child = (f"import os, sys, time; sys.path.insert(0, {root!r}); "
             "from memoos_core import dashboard; "
             f"dashboard.stop_when_terminal_closes(lambda: os._exit(7), 0.1, {shell.pid}); "
             "print('ready', flush=True); time.sleep(30)")
    detached = subprocess.Popen([sys.executable, "-c", child],
                                stdout=subprocess.PIPE, text=True,
                                start_new_session=True)
    detached.stdout.readline()

    time.sleep(0.5)
    ok("a live shell keeps it running", detached.poll() is None)

    shell.kill()
    shell.wait(timeout=5)
    try:
        code = detached.wait(timeout=10)
    except subprocess.TimeoutExpired:
        detached.kill()
        code = None
    check("the named shell going takes it with it", code, 7)


@reports
def test_only_this_dashboard_is_closed() -> None:
    """
    Which tabs the shutdown is allowed to touch.

    A loopback bind answers to three spellings and the browser records
    whichever one was typed, so all three have to match — but a
    different port is a different server, and closing someone's other
    localhost tab because it shares a hostname would be unforgivable.
    """
    from memoos_core import dashboard

    matched = dashboard.origins("127.0.0.1", 8000)
    ok("localhost is the same dashboard", "http://localhost:8000" in matched)
    ok("0.0.0.0 is the same dashboard", "http://0.0.0.0:8000" in matched)
    ok("another port is not", "http://127.0.0.1:8001" not in matched)

    # A bind to a real interface is one address, with no aliasing.
    external = dashboard.origins("192.168.1.10", 8000)
    check("a routable host gets no aliases", external, ["http://192.168.1.10:8000"])

    # The origin is interpolated into AppleScript, so a hostile hostname
    # must not be able to close the quote and run something else.
    quoted = dashboard._applescript_string('a"b\\c')
    check("applescript strings are escaped", quoted, '"a\\"b\\\\c"')


@reports
def test_port_conflict_is_seen_before_binding() -> None:
    """
    The preflight that turns a bind traceback into a fix.

    Reported the way uvicorn would find it, SO_REUSEADDR included, so a
    socket in TIME_WAIT is not mistaken for a server still sitting there.
    """
    import socket as _socket

    from memoos_core import dashboard

    with _socket.socket() as holder:
        holder.setsockopt(_socket.SOL_SOCKET, _socket.SO_REUSEADDR, 1)
        holder.bind(("127.0.0.1", 0))
        holder.listen(1)
        taken = holder.getsockname()[1]
        ok("a bound port is reported taken", dashboard.port_in_use("127.0.0.1", taken))
        found = dashboard.listening_pid(taken)
        # lsof may be absent or restricted; only the answer it gives has
        # to be right.
        ok("the holder is named, if it can be", found in (None, os.getpid()))

    ok("a free port is not", not dashboard.port_in_use("127.0.0.1", taken))



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
                 test_looking_around_is_not_work,
                 test_past_signals_are_surfaced_not_acted_on,
                 test_out_of_scope_questions_return_nothing,
                 test_api_validates_every_container_name,
                 test_relation_only_entities_are_linked,
                 test_dashboard_dies_with_its_terminal,
                 test_hangup_is_a_shutdown_not_a_kill,
                 test_every_way_of_stopping_reaches_the_teardown,
                 test_a_detached_dashboard_still_belongs_to_its_shell,
                 test_only_this_dashboard_is_closed,
                 test_port_conflict_is_seen_before_binding):
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
