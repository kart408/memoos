"""
Configuration: every knob MemoOS exposes, in one place.

Defaults are local-first — nothing here requires an API key or a
running service other than Ollama. Everything is overridable by
environment variable so the same code runs in a demo, a test, and a
deployment without edits.
"""

import os
import re


def _env_str(name: str, default: str) -> str:
    return os.getenv(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------- models

OLLAMA_URL = _env_str("MEMOOS_OLLAMA_URL", "http://localhost:11434")

# Two model *roles*, deliberately separate. Extraction is the quality
# bottleneck of the whole system — a bad extraction poisons every future
# retrieval — so it gets the strongest local model available. Chat replies
# are more forgiving and can use the smaller, faster fine-tune.
CHAT_MODEL = _env_str("MEMOOS_CHAT_MODEL", "memoos-model")
EXTRACT_MODEL = _env_str("MEMOOS_EXTRACT_MODEL", "mistral:latest")

# Retrieval here is *asymmetric*: short questions on one side ("what
# framework does the user use?"), declarative statements on the other
# ("User switched from PyTorch to JAX"). MiniLM is trained for symmetric
# similarity and shows it — it scored an unrelated memory about
# competitive programming above the correct answer, and rated the right
# answer to "tell me about my family" at 0.245, low enough to be filtered
# out as noise. BGE is trained for exactly this query/passage shape, at
# the same 384 dimensions and the same speed, and lifts that case to
# 0.554.
EMBED_MODEL = _env_str("MEMOOS_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# BGE expects queries (not stored passages) to carry an instruction
# prefix; omitting it forfeits most of the asymmetric-search benefit.
# Auto-selected from the model name so overriding EMBED_MODEL back to a
# symmetric model doesn't silently apply a prefix it was never trained on.
_DEFAULT_QUERY_PREFIX = (
    "Represent this sentence for searching relevant passages: "
    if "bge" in EMBED_MODEL.lower() else ""
)
EMBED_QUERY_PREFIX = _env_str("MEMOOS_EMBED_QUERY_PREFIX", _DEFAULT_QUERY_PREFIX)

# Embeddings run on CPU by default, and that is not a performance
# oversight. Ollama holds several GB of GPU memory while a model is
# loaded, which starves Apple's MPS backend — and MPS does not raise
# when it runs out. It returns *silently corrupted tensors*: the same
# sentence embedded twice came back with a self-similarity of -0.03
# instead of 1.0, and unrelated sentences scored 1.0. Every downstream
# behaviour (dedup, contradiction detection, ranking) was quietly wrong.
# MiniLM on CPU is a few milliseconds per sentence, so the GPU buys
# almost nothing here and costs correctness. Override at your own risk;
# `embedding.self_check()` verifies whichever device you pick.
EMBED_DEVICE = _env_str("MEMOOS_EMBED_DEVICE", "cpu")

LLM_TIMEOUT = _env_int("MEMOOS_LLM_TIMEOUT", 180)
LLM_JSON_RETRIES = _env_int("MEMOOS_LLM_JSON_RETRIES", 2)


# --------------------------------------------------------------- storage
#
# One container, one file. Everything a user has — their journal, their
# memories, the entities and relations between them, and the embeddings
# that make them searchable — lives in a single SQLite database named
# after them. Nothing about a user is stored anywhere else.
#
# That is a deliberate constraint, and it exists so the store can be
# handed to something bigger later. A single file is what replication
# understands: Litestream streams it to S3, Turso hosts it, rsync and
# Dropbox move it, and `cp` backs it up. The moment user state is split
# across a database and a sidecar index directory, none of that works
# without a custom sync protocol to keep the halves consistent.
#
# Per-user rather than one shared file follows from the same logic:
# exporting, migrating or deleting one user is a filesystem operation,
# and a tenant can be moved to the cloud without dragging the others
# along.

DATA_DIR = _env_str("MEMOOS_DATA_DIR", "./memoos_data")
CONTAINERS_DIRNAME = "containers"
DB_SUFFIX = ".db"

# Kept only so an older single-file store can still be found and
# migrated. Nothing writes here any more.
DB_FILENAME = "memoos.db"

_UNSAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def safe_container(name: str) -> str:
    """
    A container name reduced to something safe to put in a filename.

    Container names come from directory basenames, so they arrive with
    spaces, slashes and the occasional emoji. Everything outside
    [A-Za-z0-9._-] collapses to a dash, and leading dots and dashes are
    stripped — which also means `..` and `/` cannot survive to become a
    path traversal out of the containers directory.
    """
    cleaned = _UNSAFE_NAME.sub("-", (name or "").strip().lower()).strip("-._")
    return cleaned or "default"


def containers_dir(data_dir: str | None = None) -> str:
    return os.path.join(data_dir or DATA_DIR, CONTAINERS_DIRNAME)


def db_path(data_dir: str | None = None, container: str | None = None) -> str:
    """
    The one file holding everything about `container`.

    With no container this returns the legacy shared database, which is
    what `migrate_legacy_store` reads and nothing writes.
    """
    if container is None:
        return os.path.join(data_dir or DATA_DIR, DB_FILENAME)
    return os.path.join(containers_dir(data_dir),
                        safe_container(container) + DB_SUFFIX)


def stored_containers(data_dir: str | None = None) -> list[str]:
    """Every container with a file on disk, alphabetically."""
    directory = containers_dir(data_dir)
    if not os.path.isdir(directory):
        return []
    return sorted(name[:-len(DB_SUFFIX)] for name in os.listdir(directory)
                  if name.endswith(DB_SUFFIX))


# ------------------------------------------------------------- retrieval

# How many candidates each retriever contributes before fusion. Both are
# over-fetched relative to top_k so that fusion has enough overlap to
# actually rank on — with only 5 each, the two lists often disjoint and
# RRF degenerates into "whatever vector search said".
VECTOR_CANDIDATES = _env_int("MEMOOS_VECTOR_CANDIDATES", 30)
KEYWORD_CANDIDATES = _env_int("MEMOOS_KEYWORD_CANDIDATES", 30)

# Reciprocal Rank Fusion constant. 60 is the value from the original RRF
# paper; it damps the influence of the very top ranks just enough that a
# single retriever can't unilaterally decide the final order.
RRF_K = _env_int("MEMOOS_RRF_K", 60)

# Graph expansion: after fusion, pull in memories that share an entity
# with the top hits. Damped, because "mentions the same entity" is a
# weaker signal than "matches the query".
GRAPH_EXPANSION_SEEDS = _env_int("MEMOOS_GRAPH_EXPANSION_SEEDS", 5)
GRAPH_EXPANSION_LIMIT = _env_int("MEMOOS_GRAPH_EXPANSION_LIMIT", 10)
GRAPH_DAMPING = _env_float("MEMOOS_GRAPH_DAMPING", 0.4)

# Minimum fused score for a result to be worth returning at all.
MIN_RESULT_SCORE = _env_float("MEMOOS_MIN_RESULT_SCORE", 0.0)

# Relevance floor for vector-only matches.
#
# Vector search always returns its nearest neighbours, however far away
# they are. In a small store that means *every* memory gets a rank and
# therefore a score, so a question about frameworks happily returns
# "User has a sister named Priya" simply because nothing else was left.
# A memory that matched only weakly, and matched on no keyword and no
# entity, is not an answer — it's the least-bad noise available.
MIN_VECTOR_SIMILARITY = _env_float("MEMOOS_MIN_VECTOR_SIMILARITY", 0.25)


# ---------------------------------------------------------------- recall

# When the assistant answers a personal question straight from memory
# instead of admitting it doesn't know.
#
# These are compared against raw cosine similarity, NOT against the fused
# score `search()` returns. That distinction is the whole point: RRF
# scores encode *rank*, not similarity, and top out near 1/(RRF_K + 1) —
# about 0.016 here. Comparing one to a constant like 0.2 is a category
# error, and one that fails silently: the gate simply never opens and the
# assistant denies knowing things it was just told.
#
# Two tiers, calibrated against measured pairs rather than guessed,
# because a single cutoff cannot separate these cases — the best wrong
# answer scores within 0.01 of the worst right one:
#
#   corroborated — vector and keyword both matched
#       "where do I live?"  -> "User lives in Hyderabad."        0.438
#       "what do I prefer?" -> "User prefers PyTorch..."         0.578
#     No wrong answer in testing ever earned keyword agreement: an
#     unrelated question shares no significant token with a memory, so
#     the second retriever abstains rather than concurring.
#
#   vector only — no lexical overlap to corroborate with
#       "what did I say about running?" -> marathon memory       0.509  right
#       "who am I related to?"          -> sister memory         0.591  right
#       "what is 2 plus 2?"             -> sister memory         0.462  wrong
#       "who won the world cup?"        -> sister memory         0.430  wrong
#     Right answers bottom out at 0.509, wrong ones top out at 0.462, so
#     the bar sits between them.
#
# Agreement from a second, independent retriever is evidence, so it buys
# a lower similarity requirement.
RECALL_MIN_SIMILARITY_CORROBORATED = _env_float(
    "MEMOOS_RECALL_MIN_SIMILARITY_CORROBORATED", 0.40)
RECALL_MIN_SIMILARITY_VECTOR_ONLY = _env_float(
    "MEMOOS_RECALL_MIN_SIMILARITY_VECTOR_ONLY", 0.48)

# How many memories get folded into a general reply as context. Kept
# small on purpose: this is a prompt budget, not a recall target, and
# padding it with marginal hits gives the model more to be distracted by
# rather than more to be right about.
CONTEXT_TOP_K = _env_int("MEMOOS_CONTEXT_TOP_K", 3)


# --------------------------------------------------------- consolidation

# Cosine similarity above which two memories are treated as the same fact
# restated, and merged without asking the LLM.
DUPLICATE_THRESHOLD = _env_float("MEMOOS_DUPLICATE_THRESHOLD", 0.93)

# Cosine similarity above which two memories are *related enough* that a
# contradiction is plausible and worth spending an LLM call to judge.
# Below this we assume independence — cheaper, and almost always right.
#
# Calibrated against measured pairs rather than guessed, because the
# useful range is narrower than intuition suggests:
#     "lives in Mumbai"  vs "lives in Bengaluru"   0.71  (contradiction)
#     "moved to Delhi"   vs "lives in Bengaluru"   0.59  (contradiction)
#     "has a dog Rex"    vs "lives in Bengaluru"   0.25  (independent)
# Real contradictions bottom out near 0.59, so the threshold sits well
# below it. Set it at 0.55 and the Delhi case is missed by 0.04.
RELATED_THRESHOLD = _env_float("MEMOOS_RELATED_THRESHOLD", 0.45)

# Cap on how many existing memories get compared against one new memory.
# Without this, a dense memory store makes every write O(n).
MAX_CONFLICT_CANDIDATES = _env_int("MEMOOS_MAX_CONFLICT_CANDIDATES", 5)

# Of those, how many may actually cost an LLM call. Candidates arrive in
# descending similarity order, so this keeps the most plausible conflicts
# and bounds a write at ~3 model calls instead of ~5.
MAX_JUDGE_CALLS = _env_int("MEMOOS_MAX_JUDGE_CALLS", 3)


# --------------------------------------------------------------- decay

# Memory strength halves after this many days without access. Facts and
# preferences are durable; events are tied to a moment and matter less as
# that moment recedes.
HALF_LIFE_DAYS = {
    "fact": 540.0,
    "preference": 540.0,
    "skill": 540.0,
    "goal": 180.0,
    "relationship": 540.0,
    "event": 120.0,
}
DEFAULT_HALF_LIFE_DAYS = _env_float("MEMOOS_DEFAULT_HALF_LIFE_DAYS", 365.0)

# Below this strength a memory is a candidate for forgetting. Nothing is
# deleted automatically — forget_weak() has to be called explicitly.
FORGET_THRESHOLD = _env_float("MEMOOS_FORGET_THRESHOLD", 0.05)

# How much repeated access reinforces a memory, log-scaled so that the
# 100th recall matters far less than the 2nd.
REINFORCEMENT_WEIGHT = _env_float("MEMOOS_REINFORCEMENT_WEIGHT", 0.30)
REINFORCEMENT_CAP = _env_float("MEMOOS_REINFORCEMENT_CAP", 2.0)

# How many of a search's results earn that reinforcement.
#
# Reinforcement is supposed to record "this memory was useful". But a
# search returns its k nearest candidates whether or not any of them
# answered the question — vector search always returns *something*, however
# far away. Crediting the whole result list conflates "appeared in a list"
# with "was the answer".
#
# That distinction matters because strength feeds back into ranking: a
# memory surfaced once as filler ranks higher next time, which makes it
# more likely to be surfaced again. Rich-get-richer, on noise. It is how a
# memory about building a research assistant climbed above the correct
# answer to "where does the user live?" — it had been returned three times
# as an also-ran and out-reinforced the real answer.
REINFORCE_TOP_N = _env_int("MEMOOS_REINFORCE_TOP_N", 1)

# How far memory strength may reorder results.
#
# This has a correct upper bound, and it is smaller than intuition
# suggests. RRF scores adjacent ranks 1/(K+r) and 1/(K+r+1), a relative
# gap of about 1/(K+r) — roughly 1.6% at the top of the list with K=60.
# The strength multiplier spans `weight` (strength runs 0..1), so any
# weight above that gap lets strength flip genuinely better matches
# rather than merely separate equal ones.
#
# That bound was learned twice. At 0.5 the swing was 2x, and a memory
# about building a research assistant beat "User moved to Mumbai" for
# "Where does the user live?" despite scoring 0.564 against 0.600. At
# 0.15 the same inversion came back as soon as the demo was run a second
# time: restating a fact reinforces it as a duplicate, which was worth a
# 4.1% strength edge — still comfortably above 1.6%.
#
# At 0.01 the multiplier spans 1%, below one rank step, so strength can
# separate results that fusion rates equally and cannot outvote results
# it rates differently. Decay's real job — deciding what to forget — runs
# through forget_weak() and is unaffected by this number.
STRENGTH_WEIGHT = _env_float("MEMOOS_STRENGTH_WEIGHT", 0.01)


# ------------------------------------------------------------- chunking

CHUNK_SIZE_WORDS = _env_int("MEMOOS_CHUNK_SIZE_WORDS", 220)
CHUNK_OVERLAP_WORDS = _env_int("MEMOOS_CHUNK_OVERLAP_WORDS", 40)
