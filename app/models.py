"""
Data shapes for MemoOS.

Two layers, kept deliberately separate:

  Memory      what the storage layer moves around, embedding included.
  MemoryOut   what the API returns, embedding excluded.

The split is not ceremony. An embedding is 768 floats, so returning it
by default would make every list response ~20x larger than the content
anyone actually asked for, and it is useless to a caller who cannot
compare it against anything.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


def new_id() -> str:
    return str(uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ------------------------------------------------------------- stored shape


class Memory(BaseModel):
    """One remembered thing, scoped to exactly one user."""

    id: str = Field(default_factory=new_id)

    # The tenant boundary. There is no separate provisioning step — the
    # first memory written under a user_id is that user's memory space
    # coming into existence. Real auth would derive this from a verified
    # token rather than trusting the caller; see routes.py.
    user_id: str

    content: str
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)


# ------------------------------------------------------------ API requests


class MemoryCreate(BaseModel):
    content: str = Field(..., min_length=1, description="Raw text to remember.")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    extract: bool = Field(
        False,
        description=(
            "Run the text through the local chat model first and store the "
            "extracted standalone fact(s) instead of the raw input. Costs a "
            "model call, and one input can yield several memories."
        ),
    )

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("content cannot be blank")
        return v.strip()


class MemoryUpdate(BaseModel):
    """
    A partial edit. `content` is optional so metadata can be changed on
    its own — which matters because changing content forces a re-embed
    and changing metadata does not.
    """

    content: Optional[str] = Field(None, min_length=1)
    metadata: Optional[Dict[str, Any]] = None

    @field_validator("content")
    @classmethod
    def not_blank(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not v.strip():
            raise ValueError("content cannot be blank")
        return v.strip() if v else v


# ----------------------------------------------------------- API responses


class MemoryOut(BaseModel):
    id: str
    user_id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime

    @classmethod
    def of(cls, memory: Memory) -> "MemoryOut":
        return cls(
            id=memory.id,
            user_id=memory.user_id,
            content=memory.content,
            metadata=memory.metadata,
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )


class MemoryPage(BaseModel):
    """A page of memories, with enough context to request the next one."""

    memories: List[MemoryOut]
    total: int
    limit: int
    offset: int


class CreateResult(BaseModel):
    """
    What a write produced.

    A list rather than a single memory because fact extraction can turn
    one sentence into several standalone facts, and silently keeping only
    the first would lose the rest.
    """

    created: List[MemoryOut]
    extracted: bool = False


class SearchHit(BaseModel):
    memory: MemoryOut
    score: float = Field(..., description="Cosine similarity, -1.0 to 1.0.")


class SearchResponse(BaseModel):
    query: str
    hits: List[SearchHit]


class HealthResponse(BaseModel):
    status: str
    database: str
    ollama: str
    embed_model: str
    detail: Optional[str] = None
