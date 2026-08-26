# MemoOS

**Memory for your terminal.** A shell forgets everything the moment you close
it. MemoOS puts a memory layer underneath it: every command you run is
journalled as it happens, and folded into structured memories when the
terminal closes.

Everything runs on your machine. One SQLite file per project holds the rows
and the vectors, a local [Ollama](https://ollama.com) instance does the
extraction, and no request ever leaves localhost.

```bash
$ memoos recall
  User switched the auth from sessions to JWT in api.py.
  Tests failed with exit code 4 on tests/test_auth.py.
  User installed Python package pyjwt.
```

Nothing there was typed by hand. It is four shell commands, read back as
facts.

---

## Memory for your terminal

A shell forgets everything the moment you close it. Scrollback is not memory —
it is a transcript with no index, no structure, and no idea what mattered.
Reopen the window tomorrow and the work you were three hours into is gone.

`memoos` puts a memory layer underneath the terminal:

```bash
python memoos_cli.py install     # writes the zsh hook to ~/.memoos/memoos.zsh
echo 'source ~/.memoos/memoos.zsh' >> ~/.zshrc
```

Then work normally. Every command, its exit code, your Claude Code prompts and
your own notes are journalled as they happen. Later, they are folded into
atomic memories and an entity graph:

```bash
memoos recall            # what was I doing in this project?
memoos recall "why did the auth tests fail"    # semantic search
memoos distill           # fold the journal into memory (runs the model)
memoos graph             # the entities, and how they connect
memoos ingest NOTES.md   # teach it about you or the project
memoos serve             # the dashboard, with the graph drawn
```

Memory is scoped to the **git project you are standing in**. Projects never see
each other's memory.

### Connecting and disconnecting

Attaching memory to a shell is not a commitment. Some work is worth
remembering and some is not.

```bash
memoos disconnect          # stop recording, everywhere, immediately
memoos connect             # resume
memoos disconnect --here   # this terminal only
memoos status              # which state you are in, and what is stored
```

Terminals already open respect the switch on their next command — the hook
tests for it rather than caching it, so disconnecting never means "open a new
window first". There is a toggle in the dashboard that does the same thing.

**Only recording is switched.** `recall`, `search` and the graph keep working
while disconnected, because declining to record today says nothing about what
you learned yesterday.

### Two speeds, on purpose

The split between journalling and distilling is the whole design.

| | writes | cost | when |
| --- | --- | --- | --- |
| **Journal** | raw events, verbatim | ~0.3 ms | every command |
| **Distil** | atomic memories + graph | ~seconds/minutes | session end, or on demand |

A shell hook runs before *every* prompt you see, so the write path imports no
embedding model and no LLM — it appends a row and returns.
Interpretation is expensive and belongs nowhere near the prompt you are
waiting on. Recall, meanwhile, needs structure, and structure is exactly what
raw scrollback lacks.

Reading is kept fast the same way: memories live in SQLite, so `memoos recall`
answers from plain SQL in about a tenth of a second including Python startup.
Only *semantic* search loads the embedding model.

---
## Quickstart

**1. Install Ollama and pull the extraction model.**

```bash
# https://ollama.com/download
ollama pull mistral        # used only to turn shell activity into facts
```

**2. Install the Python dependencies.**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

**3. Attach it to your shell.**

```bash
python memoos_cli.py install
echo 'source ~/.memoos/memoos.zsh' >> ~/.zshrc
```

**4. Check every part of the chain.**

```bash
memoos doctor
```

```
attach
  ✓ shell hook installed       ~/.memoos/memoos.zsh
  ✓ sourced from ~/.zshrc
  ✓ attached to this shell     1787636502-9744
  ✓ recording
store
  ✓ MEMOOS_DATA_DIR pinned     ~/memoos/memoos_data
  ✓ writable
  ✓ container                  memoos → …/containers/memoos.db
distillation   runs in the background when a terminal closes
  ✓ ollama reachable           http://localhost:11434
  ✓ extraction model           mistral:latest
  ✓ events awaiting distill    none
```

That last section is the one worth reading. Journalling is SQLite and can
barely fail. Distillation needs Ollama and runs *in the background as your
terminal dies* — the worst possible place for a failure, because nobody is
watching. A failed distil loses nothing (its events stay pending and the next
run picks them up), but without this you would never know to look.

---

## Commands

```
memoos install        wire it into your shell
memoos recall         what was I doing here?
memoos recall "why did the auth tests fail"    semantic search
memoos context TASK   what an agent should know before starting (--json)
memoos signals        what has gone wrong here before, and what fixed it
memoos trace ID       follow a memory back to the commands behind it
memoos distill        fold this session into memory
memoos ingest FILE    teach it about you or the project
memoos claude         import Claude Code prompts for this project
memoos graph          the entities, and how they connect
memoos sessions       what has been journalled, per session
memoos stats          counts, as JSON
memoos files          where each project's data lives, and how big
memoos clear          forget a project, or every project
memoos doctor         check every part of the chain
memoos connect        start recording
memoos disconnect     stop recording (reading still works)
memoos status         connected or not, and what is stored
memoos serve          the dashboard, with the graph drawn
```

---

## The handover

`memoos context` is the point of the whole thing. An agent is about to work on
your project; this is what it was missing.

```console
$ memoos context "why is the auth test failing?"

searching for auth, test, failing + test_auth.py

  Project: my-app

  • The authentication tests in test_auth.py were failing.
  • Editing api.py made the authentication tests pass.
  • User wanted to convert auth to JWT.
  • User installed the npm package pyjwt.

  hand this to your agent — memoos stops here
```

The `+ test_auth.py` is stage 9, and it is doing real work. "auth", "test" and
"failing" are your words; `test_auth.py` was read back out of the graph,
because a vector probe found the memories the question is *about* and harvested
the entities hanging off them. BM25 cannot match "how do I deploy?" against
"deployed on Vercel" — no shared token — so without expansion the keyword half
of hybrid search abstains on exactly the queries that need it.

Expansion is grounded rather than invented. A thesaurus would add words the
project has never heard of, and every one is a chance for BM25 to match
something irrelevant with confidence. Reading concepts out of the store means
expansion can only ever add terms this project already knows.

`--json` gives an agent the block plus each memory's score, type, validity and
why it matched.

An out-of-scope question comes back empty rather than with three confident
irrelevancies:

```console
$ memoos context "what is the capital of France"

  nothing remembered about that in memoos
```

That needs a gate, because search has no "I don't know" state — it returns its
nearest neighbours however distant, which is the right contract for a caller
that can see the scores and the wrong one for a block that goes into a prompt
with every number stripped off. BGE rates *completely unrelated* text at ~0.46
cosine, so asked who won the world cup this store offered three memories about
itself. The bars are `RECALL_MIN_SIMILARITY_*`, calibrated against measured
pairs and orphaned since the chat layer that used them was deleted; the gate
only wires them back up.

Corroboration from a second retriever buys a lower bar — but only when it rests
on your own words. Query expansion can manufacture agreement: asked "who is my
sister" the probe harvested `main` and `master` off the nearest memories and
searched for *"who is my sister main master"*, so the keyword retriever matched
on two words nobody typed and the vector search had just invented. That is
echoing, not agreeing, and it let cosine 0.428 through a 0.48 bar.

**MemoOS stops here.** It does not answer. The agent that asked is the only
thing that knows what you are actually trying to do, and a memory layer that
also wrote the reply would be guessing at that.

---

## Not a dumping ground

A memory has to still be worth knowing next session, and most of what a
terminal produces is not. The store this was built against had 51 memories,
**38 of them records of a request** — "User wanted to commit the changes",
true for thirty seconds, stored forever, ranking against facts that still hold.

The cause was one line. Every imported Claude Code prompt was written into the
digest as `The user wanted to "X"`, and the extractor faithfully recorded the
wanting. A request is context for reading the commands that follow; it is never
a memory. What it *produced* might be, and that is a different sentence.

Re-distilling the same 67 events after the fix:

| | before | after |
|---|---:|---:|
| memories | 51 | 26 |
| records of a request | 38 (74%) | 0 |
| entities | 22 | 7 |

Fewer memories, and the survivors are state: `The project uses a multi-tenant
architecture`, `The master branch was renamed to main`, `Editing api.py made the
authentication tests pass`.

The fix is at the source — the digest no longer asserts intent, and the
extraction prompt asks for what became true. A deterministic gate backs it up,
and earned its place immediately: the moment the digest stopped saying "wanted
to", the model started writing "asked for" instead. Request phrasing is now
rejected on principle rather than by pattern, while a lasting aim survives,
because `build` is not a chore and `commit` is.

---

## Past signals

`memoos signals` pairs a `problem` back up with whatever fixed it.

```console
$ memoos signals

my-app · open problems
  ! The deploy step times out on Vercel.

known failures, and what fixed them
  ✗ The authentication tests in test_auth.py were failing.
    ↳ Editing api.py made the authentication tests pass.
```

Pairing is by evidence, and the evidence is layered because no single signal
works. A shared entity is strongest, but requiring it misses the ordinary case —
the test that broke and the file that fixed it are usually *different* files.
Coming out of the same session is necessary and nowhere near sufficient: on its
own it pairs everything with everything, and an afternoon that broke the deploy
and separately fixed the auth tests would report the auth fix as the answer to
the deploy timeout. So proximity has to be corroborated by shared wording. A
solution describes undoing its problem, so the two discuss the same things even
when they name different files.

These ride along in the context block under a heading of their own, because a
known failure is a different kind of thing from "the project uses Next.js" — it
is a warning, and a reader skimming a prompt should see that at a glance.

**Surfaced, never acted on.** Nothing in `signals.py` changes a ranking, retires
a memory or adjusts a threshold. Deciding what to do about a known failure
requires knowing what you are trying to do, and that is exactly what a memory
layer cannot see.

---

## Where a memory came from

Every memory traces back to the commands that produced it. A store you cannot
audit is a store you cannot trust: the failure that matters is a confidently
retrieved fact nobody ever stated, and following it back is the only way to
tell that from a real one.

```console
$ memoos trace c0a04613-e9e5-4fa2-bf39-0889c9ea80b5

[problem] The authentication tests in test_auth.py were failing.
  valid        26 Aug 20:41 → current
  confidence   0.80   importance 0.70
  from         terminal session — my-app

passage the model read
  The user wanted to "Convert auth to JWT". The user installed the npm
  package pyjwt. The user ran the tests in test_auth.py, and it failed
  with exit code 1. The user edited api.py...

6 journalled event(s) behind it
  ✓ 26 Aug 20:41  Convert auth to JWT
  ✓ 26 Aug 20:41  npm install pyjwt
  ✗ 26 Aug 20:41  pytest test_auth.py
  ✓ 26 Aug 20:41  vim api.py
  ✓ 26 Aug 20:41  pytest test_auth.py
  ✓ 26 Aug 20:41  git commit -m 'switch auth to JWT'
```

---

## Facts have a lifetime

A memory carries `valid_from` / `valid_until`, which is not the same as when
its row was written. "The project uses MongoDB" did not become *false* when you
migrated — it stopped being *current*. Deleting it loses the history; leaving it
active answers "what database?" with two databases.

Supersession closes the interval in the same statement that retires the row, so
a crash cannot leave a retired memory still reading as current. Retrieval drops
what is no longer current; `MemoOS.history()` walks the chain backwards when you
want to know what a thing used to be.

Distilling a session produces three types a chat log never does — `decision`,
`problem` and `solution` — because a terminal is mostly a record of things
going wrong and then going right, and that arc is the most valuable thing in
it. Filed as `event`, "the tests failed on the import path and moving the
fixture fixed it" decayed on an event's short half-life and read like trivia.

---

## One project, one file

Everything MemoOS knows about a project lives in a single SQLite database
named after it:

```
memoos_data/containers/
├── memoos.db
└── some-other-project.db
```

That file holds the journal, the memories, the documents they came from, the
entity graph, and the embeddings — nothing about a project is stored anywhere
else.

```bash
$ memoos files
stored data  ~/memoos/memoos_data/containers

  memoos                         248.0KB   14 memories · 9 entities
  dotfiles                        52.0KB    2 memories · 1 entities
```

It is a deliberate constraint, and it exists so the store can be handed to
something bigger later. A single file is what replication understands:
Litestream streams it to S3, Turso hosts it, `rsync` moves it, `cp` backs it
up. The moment a project's state is split across a database and a sidecar
index directory, none of that works without a custom protocol to keep the
halves consistent.

Per-project rather than one shared file follows from the same logic: exporting,
migrating or forgetting one project is a filesystem operation, and projects
never see each other's memory because isolation is a property of the layout
rather than a `WHERE` clause someone can forget.

> **While a terminal is open there is a `-wal` file beside the database.**
> To copy a live store, use `sqlite3 <file> ".backup out.db"` rather than
> `cp`, or let Litestream handle it.

---

## Architecture

```
memoos_core/
├── journal.py       raw events, written fast — stdlib only, no heavy imports
├── quick.py         the read half of the fast path — plain SQL, no model
├── terminal.py      sessions, distillation, Claude Code transcript import
├── extraction.py    text → atomic memories, entities and relations
├── pipeline.py      ingestion: chunk, extract, store, index, connect
├── db.py            SQLite: memories, documents, entities, relations, FTS5
├── vectors.py       embeddings in that same file, searched with numpy
├── query.py         question → the terms worth searching for, grounded
├── signals.py       problems paired with their fixes — reported, not acted on
├── retrieval.py     hybrid search: vector + BM25 + graph expansion, fused
├── consolidation.py duplicates, contradictions, supersession, decay
├── graph.py         the entity graph
├── hook.py          the zsh hook, generated and bound at install time
└── connection.py    the recording switch — stdlib, read on every command
```

A few decisions worth knowing about:

**Embeddings are stored inline, as `float32` BLOBs**, in the same file as the
memories. There is no second index that can drift out of sync — and
`embeddings.memory_id` is a foreign key with `ON DELETE CASCADE`, so deleting
a memory drops its vector in the same transaction.

**Vectors are normalised once, at write time.** Cosine similarity is then just
a dot product, so a search is one BLAS call rather than N norm computations.

**Ranking is brute-force numpy** over one project's vectors. At the scale this
is built for — thousands of memories, not millions — that is well under a
millisecond, and faster than the ANN index it replaced because there is no
graph to load first. `VectorIndex._snapshot()` in `vectors.py` is the single
seam where `sqlite-vec` would take over.

**The index cache is invalidated by `PRAGMA data_version`.** The CLI and the
dashboard are separate processes writing to one file, and a cache that only
tracked its own writes would serve a stale index — a memory added in the
terminal would stay invisible to the dashboard until it restarted.

**The journal is the source of truth.** Distillation only marks events as done
if the model actually read them. If a chunk times out, leaving its events
pending costs a re-run; marking them done would cost the work itself, silently
and permanently. That extends to the model being *unreachable*: a refused
connection reads as "this chunk was not processed", never as an exception that
takes the write with it. Distillation runs in the background when a terminal
closes, which is the worst possible place for a hard failure — nobody is
watching.

**Schema changes are applied in place.** `CREATE TABLE IF NOT EXISTS` does
nothing to a table that already exists, so a schema grown a column applies
cleanly to a fresh file and not at all to the one holding a year of memories.
`Database._add_missing_columns` runs the `ALTER`s and back-fills; indexes over
those columns run afterwards, because naming a column that is not there yet
fails the whole script at open.

---

## Configuration

All optional, all environment variables. These are the ones worth knowing;
`memoos_core/config.py` documents the rest, including retrieval and decay
tuning.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMOOS_DATA_DIR` | `./memoos_data` | Where the per-project files live |
| `MEMOOS_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `MEMOOS_EXTRACT_MODEL` | `mistral:latest` | Turns shell activity into facts |
| `MEMOOS_EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model |
| `MEMOOS_EMBED_DEVICE` | `cpu` | Where embeddings run |
| `MEMOOS_LLM_TIMEOUT` | `180` | Extraction request timeout (seconds) |
| `MEMOOS_OFF` | unset | Set in a shell to stop recording there |

`MEMOOS_DATA_DIR` is pinned to an absolute path in the generated hook. It has
to be: the default is relative and the hook runs from whatever directory you
are standing in, so left unpinned the first `cd` into another project would
grow a second, empty store there and silently split the journal in half.

---

## Scope

**This is a memory layer, and only that.** No chatbot, no auth system, and no
cloud model calls of any kind. A local model is used, but only as a *parser* —
it turns text into structured facts. Nothing here generates a reply, and
nothing talks back.

---

## Accepted behaviour

Two things look like bugs on inspection and are not. Both are recorded here
because the tempting fix for each is worse than the thing it fixes.

**Near-duplicates just below the threshold stay separate.** The store holds

```
"The user encountered an issue with running './start_demo.sh' in the Mac terminal."
"The user encountered an error with './start_demo.sh'."
```

at cosine 0.9224, against a `DUPLICATE_THRESHOLD` of 0.93. That reads as one
thing said twice, and merging it means moving a *global* threshold that judges
every write in every container. A wrong merge destroys information silently; a
redundant memory is merely untidy. That asymmetry is the whole argument, and it
does not justify 0.008. To merge a specific pair, supersede one explicitly —
local decision, local blast radius.

**`documents.memory_count` is history, not a live tally.** A document reading
24 alongside 23 active memories is correct: 24 is what that ingest produced,
and one was deleted afterwards. Recomputing it would lose the distinction
between "this source yielded little" and "this source yielded plenty and most
was later retired" — which is exactly the question worth asking of a source
that turned out to be low quality. Count `memories` directly when you want the
live number.

---

## Testing

```bash
python test_memoos.py
```

176 assertions against a scratch data directory — it never touches your real
store. No Ollama and no extraction model are needed, and that is enforced
rather than assumed: one of the tests points the client at a dead port and
checks the write path still completes.

```
  176 passed, 0 failed
```

It also runs under `pytest`, and now actually fails there. The assertions
record failures and keep going, so one run reports everything that is broken
rather than only the first thing — but nothing raised, so pytest reported
"passed" no matter what. Each test is wrapped to print its whole tally and then
still fail.

It checks that a container name resolves to exactly one file and cannot escape
the containers directory, that events land in their own tenant's file and
nowhere else, that *asking* what an unknown project remembers does not create
it, that search ranks by meaning rather than keyword overlap, that a write from
one process is visible to another with a warm cache, that a vector from a
different embedding model is reported and skipped rather than crashing search,
that deleting a memory takes its vector with it, that a question is expanded
with concepts the project actually knows and not with somebody's sister, that
supersession closes a memory's validity window in the same statement that
retires it, that a store written before those columns existed upgrades in place
without losing a row, and that a memory can be traced back to the passage and
the session it came from.

`memoos doctor` is the complement: it checks the running system — hook, store,
Ollama, model — rather than the code.
