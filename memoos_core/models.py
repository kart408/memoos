"""
Core data models for MemoOS.

The unit that matters is the Memory: one atomic, standalone statement
plus the metadata needed to reason about it later — when it was learned,
how sure we are, whether something newer has replaced it, and how much
it has decayed since it was last useful.

Around it sit three supporting shapes:
  Document  — the raw thing that was ingested (a note, a page, a file)
  Entity    — a named thing memories talk about (a person, place, tool)
  Relation  — a typed edge between two entities, drawn from a memory

Documents give provenance, entities and relations give the memory graph
its structure. A memory that mentions "Zomato" becomes reachable from
every other memory that mentions Zomato, even when the wording shares
no vocabulary at all.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid4())


class MemoryType(str, Enum):
    FACT = "fact"                  # durable info ("lives in Bengaluru")
    EVENT = "event"                # something that happened, tied to a time
    PREFERENCE = "preference"      # a stated like/dislike/style choice
    SKILL = "skill"                # something the user can do
    GOAL = "goal"                  # something the user intends to do
    RELATIONSHIP = "relationship"  # a link to another person/org

    # The three a terminal produces that a chat log does not.
    #
    # A session is mostly a record of things going wrong and then going
    # right, and that shape is the most valuable thing in it — "the tests
    # failed on the import path and moving the fixture fixed it" is worth
    # more six months later than any command in the transcript. Filed as
    # `event`, it decayed on an event's short half-life and read like
    # trivia. These three separate the arc:
    DECISION = "decision"          # a choice made, and what it was over
    PROBLEM = "problem"            # something that broke or blocked
    SOLUTION = "solution"          # what actually fixed it


class MemoryStatus(str, Enum):
    ACTIVE = "active"          # live and retrievable
    SUPERSEDED = "superseded"  # a newer memory contradicts/updates it
    DUPLICATE = "duplicate"    # merged into an existing memory
    FORGOTTEN = "forgotten"    # decayed below the retention threshold


class EntityType(str, Enum):
    PERSON = "person"
    PLACE = "place"
    ORG = "org"
    TECH = "tech"
    EVENT = "event"
    OTHER = "other"


class DocumentStatus(str, Enum):
    QUEUED = "queued"
    CHUNKING = "chunking"
    EXTRACTING = "extracting"
    EMBEDDING = "embedding"
    DONE = "done"
    FAILED = "failed"


class Memory(BaseModel):
    id: str = Field(default_factory=_new_id)
    text: str
    memory_type: MemoryType = MemoryType.FACT
    source: str = "user"

    # Scoping. `container` is the tenant boundary — one user, one project,
    # one workspace. Every read and write is filtered by it.
    container: str = "default"

    # Provenance: which raw input this memory was distilled from.
    document_id: Optional[str] = None
    chunk_id: Optional[str] = None

    # How much this matters (set at extraction time) and how sure we are
    # it was actually stated (lowered when extraction looks shaky).
    importance: float = 0.5
    confidence: float = 0.8

    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)

    # When the *fact* held, as distinct from when the row was written.
    #
    # The two come apart constantly and the difference is the whole point:
    # "User used MongoDB" is not false, it stopped being current. Deleting
    # it loses the history; leaving it active answers "what database?"
    # with two databases. A closed interval says both things at once —
    # the memory is still there, still traceable, and no longer current.
    #
    # valid_from defaults to created_at. valid_until stays None while the
    # fact holds and is stamped the moment something supersedes it.
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None

    # Reinforcement signals — a memory that keeps getting recalled is a
    # memory that keeps mattering, and should resist decay.
    last_accessed_at: Optional[datetime] = None
    access_count: int = 0

    status: MemoryStatus = MemoryStatus.ACTIVE
    superseded_by: Optional[str] = None  # id of the memory that replaced it
    superseded_at: Optional[datetime] = None
    supersede_reason: Optional[str] = None

    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return self.status == MemoryStatus.ACTIVE

    def is_current(self, now: Optional[datetime] = None) -> bool:
        """
        Does this memory describe how things are *now*?

        Status and validity answer different questions. Status is about
        the row (active, merged away, retired); validity is about the
        world (this held until March). A memory can be perfectly active
        and no longer current, which is what makes "what did I use
        before?" answerable at all.
        """
        if self.status != MemoryStatus.ACTIVE:
            return False
        if self.valid_until is None:
            return True
        moment = now or _now()
        until = self.valid_until
        if until.tzinfo is None:
            until = until.replace(tzinfo=timezone.utc)
        return until > moment


class Entity(BaseModel):
    id: str = Field(default_factory=_new_id)
    container: str = "default"
    name: str                      # as written, e.g. "Zomato"
    norm_name: str                 # matching key, e.g. "zomato"
    entity_type: EntityType = EntityType.OTHER
    mention_count: int = 0
    created_at: datetime = Field(default_factory=_now)


class Relation(BaseModel):
    id: str = Field(default_factory=_new_id)
    container: str = "default"
    subject_id: str                # Entity.id
    predicate: str                 # normalised, e.g. "works_at"
    object_id: str                 # Entity.id
    memory_id: Optional[str] = None  # the memory this edge was drawn from
    confidence: float = 0.8
    created_at: datetime = Field(default_factory=_now)
    invalidated_at: Optional[datetime] = None


class Document(BaseModel):
    id: str = Field(default_factory=_new_id)
    container: str = "default"
    title: Optional[str] = None
    uri: Optional[str] = None       # file path or URL, when there was one
    source: str = "user"
    raw_text: str = ""
    status: DocumentStatus = DocumentStatus.QUEUED
    error: Optional[str] = None
    chunk_count: int = 0

    # How many memories this ingest *produced*, not how many are still
    # active. The two diverge the moment one is deleted or superseded,
    # and that divergence is correct: this column is provenance, and
    # provenance records what happened. A real store reads 24 here with
    # 23 active, because one was removed afterwards.
    #
    # Recomputing it to match the live count would make the record lie
    # about the ingest — you would lose the ability to tell "this
    # document yielded little" from "this document yielded plenty and
    # most of it was later retired", which is exactly the question you
    # ask when a source turns out to be low quality. Count active
    # memories with a query against `memories` when that is what you
    # want; leave this one alone.
    memory_count: int = 0
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ------------------------------------------------------- extraction shapes
# What the LLM is asked to produce. Kept separate from the stored models
# so a sloppy extraction can be validated and repaired before anything
# touches the database.


class ExtractedEntity(BaseModel):
    name: str
    entity_type: EntityType = EntityType.OTHER


class ExtractedRelation(BaseModel):
    subject: str
    predicate: str
    object: str


class ExtractedMemory(BaseModel):
    text: str
    memory_type: MemoryType = MemoryType.FACT
    importance: float = 0.5
    confidence: float = 0.8
    entities: List[ExtractedEntity] = Field(default_factory=list)
    relations: List[ExtractedRelation] = Field(default_factory=list)


# --------------------------------------------------------- retrieval shapes


class MemoryQueryResult(BaseModel):
    """
    One retrieved memory plus the reasoning behind its position.

    `score` is the final fused, decay-adjusted number used for ordering.
    The rest is kept for explainability — being able to answer "why did
    this surface?" is what makes a memory system debuggable instead of
    a black box.
    """
    memory: Memory
    score: float
    vector_score: Optional[float] = None   # raw cosine similarity
    keyword_score: Optional[float] = None  # normalised BM25
    strength: Optional[float] = None       # decay/reinforcement multiplier
    matched_by: List[str] = Field(default_factory=list)  # vector|keyword|graph
    via_entities: List[str] = Field(default_factory=list)  # graph hop path


class ConflictDecision(str, Enum):
    """What a newly extracted memory does to an existing, similar one."""
    INDEPENDENT = "independent"  # unrelated, keep both
    DUPLICATE = "duplicate"      # same fact restated, merge
    UPDATE = "update"            # newer version of the same fact
    CONTRADICTION = "contradiction"  # directly incompatible, newer wins


class ConflictJudgement(BaseModel):
    decision: ConflictDecision = ConflictDecision.INDEPENDENT
    reason: str = ""


class IngestResult(BaseModel):
    """What actually happened when something was ingested."""
    document: Document
    created: List[Memory] = Field(default_factory=list)
    superseded: List[Memory] = Field(default_factory=list)
    duplicates: List[Memory] = Field(default_factory=list)
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

    # How much of the input actually made it through extraction. A caller
    # that is about to mark its source as processed needs to know whether
    # any of it was silently skipped.
    chunks_total: int = 0
    chunks_failed: int = 0

    @property
    def complete(self) -> bool:
        """Did every chunk reach the model successfully?"""
        return self.chunks_failed == 0

    @property
    def summary(self) -> str:
        text = (
            f"{len(self.created)} new, {len(self.superseded)} superseded, "
            f"{len(self.duplicates)} duplicate, {len(self.entities)} entities, "
            f"{len(self.relations)} relations"
        )
        if self.chunks_failed:
            text += f" ({self.chunks_failed} of {self.chunks_total} chunks failed)"
        return text
