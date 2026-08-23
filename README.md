# MemoOS

A self-hosted, multi-tenant **memory layer for AI applications**.

Give it a `user_id` and a piece of text, and it remembers. Give it a `user_id`
and a question, and it returns the most relevant things it knows, ranked by
meaning rather than by keyword.

Everything runs on your machine. SQLite holds the rows, a local
[Ollama](https://ollama.com) instance produces the embeddings, and no request
ever leaves localhost.

```bash
curl -s -X POST localhost:8000/users/alice/memories \
  -H 'Content-Type: application/json' \
  -d '{"content":"Alice has a golden retriever named Rex."}'

curl -s -G localhost:8000/users/alice/memories/search \
  --data-urlencode 'q=does she have any pets'
#  -> "Alice has a golden retriever named Rex."   score 0.651
```

Note that the query and the stored memory share no words at all. That is the
point: retrieval is semantic, so *"does she have any pets"* finds a golden
retriever, and *"what food should she avoid"* finds a peanut allergy.

---

## Quickstart

**1. Install Ollama and pull an embedding model.**

```bash
# https://ollama.com/download
ollama pull nomic-embed-text          # 274 MB, 768 dimensions
ollama pull mistral                   # optional — only for fact extraction
```

**2. Install the Python dependencies.**

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements-app.txt
```

**3. Run it.**

```bash
uvicorn app.main:app --reload
```

**4. Confirm it is healthy.** This checks Ollama too, not just the process:

```bash
curl -s localhost:8000/health
# {"status":"ok","database":"ok","ollama":"ok",
#  "embed_model":"nomic-embed-text","detail":null}
```

Interactive API docs are at <http://127.0.0.1:8000/docs>.

---

## API

Every endpoint except `/health` is scoped to a user.

| Method | Path | Purpose |
| --- | --- | --- |
| `POST` | `/users/{user_id}/memories` | Remember something |
| `GET` | `/users/{user_id}/memories` | List memories, paginated |
| `GET` | `/users/{user_id}/memories/search?q=…` | Semantic search, ranked |
| `GET` | `/users/{user_id}/memories/{id}` | Fetch one memory |
| `PATCH` | `/users/{user_id}/memories/{id}` | Edit (re-embeds on content change) |
| `DELETE` | `/users/{user_id}/memories/{id}` | Forget it |
| `GET` | `/health` | Liveness, including Ollama reachability |

There is **no provisioning step**. A user's memory space comes into existence
the first time you write under a new `user_id`.

`test_requests.http` holds ready-to-send requests for VS Code / JetBrains /
Zed REST clients, with `curl` equivalents at the bottom.

### Adding a memory

```jsonc
POST /users/alice/memories
{
  "content": "Alice is allergic to peanuts.",
  "metadata": {"source": "onboarding"},   // optional, free-form
  "extract": false                        // optional, see below
}
```

The response is a **list**, because fact extraction can turn one input into
several memories:

```json
{"created": [{"id": "7041f36c-…", "content": "Alice is allergic to peanuts.", …}],
 "extracted": false}
```

### Searching

```
GET /users/alice/memories/search?q=where does she work&top_k=3&min_score=0.4
```

```json
{"query": "where does she work",
 "hits": [{"memory": {…}, "score": 0.712}]}
```

`score` is cosine similarity in `[-1, 1]`. `min_score` sets a relevance floor —
useful because nearest-neighbour search always returns *something*, however
far away it is.

### Optional fact extraction

Raw conversational text makes poor memories. Pass `"extract": true` and the
input goes through a local chat model first, which distils it into standalone
facts before anything is stored:

```jsonc
POST /users/alice/memories
{"content": "hey so I finally moved to Pune last week, joined Zomato as a data engineer",
 "extract": true}
```

```
->  "The user moved to Pune last week."
->  "The user joined Zomato as a data engineer."
```

Each becomes its own memory, separately retrievable, and each keeps the
original text in `metadata.source_text` so a bad extraction is diagnosable
rather than merely wrong.

This costs a model call and is off by default. If the model is unreachable or
returns nothing usable, **the raw text is stored instead** — an optional
enhancement must never be the reason a write disappears.

---

## Tenant isolation

Isolation is the property this service exists to guarantee, so it is enforced
in SQL rather than in Python. Every operation on a single memory scopes on the
owner *and* the id:

```sql
WHERE id = ? AND user_id = ?
```

Not "fetch by id, then check the owner in application code". The difference
matters: a client-side check is one forgotten `if` away from leaking another
tenant's data, whereas a scoped query simply matches no row. A user who guesses
a genuine UUID belonging to someone else gets a `404` — the same answer they
get for an id that never existed, which means an id guess cannot even confirm
that a memory exists.

Search works the same way. The candidate set is filtered by `user_id` in the
database before any ranking happens, so another tenant's vectors are never in
scope to be scored, let alone returned.

`test_memoos.py` proves this rather than asserting it: two users, and Bob is
denied `GET`, `PATCH` and `DELETE` on a real id of Alice's, with her data
verified intact afterwards.

> **Auth is not implemented.** `user_id` is read from the URL and trusted,
> which is fine for local development and wrong for a deployment — any caller
> can type any `user_id`. In production this would come from a verified token
> and the path segment would be checked against it. The layers underneath are
> already written for that: they take `user_id` as an argument and scope every
> query by it, so only *where it comes from* would change.

---

## Architecture

```
app/
├── main.py             FastAPI assembly, lifespan-managed repository
├── routes.py           the six HTTP endpoints
├── memory_service.py   business logic + optional fact extraction
├── db.py               MemoryRepository (ABC) + SQLite implementation
├── embeddings.py       Ollama client, LRU cache, cosine helpers
└── models.py           Pydantic shapes
```

Requests flow **routes → service → repository**. The service holds a
`MemoryRepository` and never sees SQLite, which is what keeps the storage
engine swappable: moving to Postgres means writing one new subclass, not
editing business logic.

A few decisions worth knowing about:

**Embeddings are stored inline with the row**, as `float32` BLOBs, rather than
in a separate vector store. There is no second index that can drift out of sync
with the data.

**Vectors are normalised once, at write time.** Cosine similarity is then just
a dot product, so a search does N multiply-adds instead of N norm computations.

**Ranking is brute-force numpy** over one user's vectors. At the scale this is
built for — tens to low thousands of memories per user — that is a few
milliseconds, and it keeps the design simple. `embeddings_for_user()` in
`db.py` is the single seam where `sqlite-vec` or `pgvector` would take over,
moving the ranking into the query.

**Editing content re-embeds in the same `UPDATE`.** Content and vector move
together or they don't move at all, so search can never rank a memory by text
it no longer has.

**Stored vs returned shapes are separate.** `Memory` carries the embedding;
`MemoryOut` does not. Returning 768 floats by default would make every
response an order of magnitude larger than the content anyone asked for, and
useless to a caller with nothing to compare it against.

---

## Testing

```bash
python test_memoos.py
```

42 assertions against a scratch database — it never touches `memoos.db`. It
creates two users, checks that search ranks sensibly, that edits re-embed,
that deletes stick, and that neither user can reach the other's memories by
any route.

```
  42 passed, 0 failed
```

Ollama must be running; the suite tells you if it isn't.

---

## Configuration

All optional, all environment variables.

| Variable | Default | Meaning |
| --- | --- | --- |
| `MEMOOS_DB_PATH` | `memoos.db` | SQLite file |
| `MEMOOS_OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `MEMOOS_EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `MEMOOS_CHAT_MODEL` | `mistral:latest` | Model used for fact extraction |
| `MEMOOS_CACHE_SIZE` | `512` | Embedding cache entries |
| `MEMOOS_TIMEOUT` | `120` | Embedding request timeout (seconds) |
| `MEMOOS_EXTRACT_TIMEOUT` | `180` | Extraction request timeout (seconds) |

---

## Scope

**This is a memory layer, and only that.** No chatbot UI, no
retrieval-augmented generation over documents, no auth system, and no cloud
model calls of any kind.

### Also in this repository

`memoos_core/` is an earlier, more experimental engine exploring what a memory
system can do beyond storage and recall: LLM-based fact extraction into atomic
memories, an entity graph, hybrid retrieval fusing vector + BM25 + graph
expansion, contradiction detection with supersession history, and
time-based decay. It has its own dependencies (`requirements.txt` — chromadb
and sentence-transformers) and its own demo entry point:

```bash
pip install -r requirements.txt
python main.py
```

The two are independent. `app/` is the tight, tested service; `memoos_core/`
is the research sketch some of its ideas came from.
