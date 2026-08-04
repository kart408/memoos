"""
Core data models for MemoOS.

A Memory is the atomic unit MemoOS stores: a piece of text plus
metadata that lets us reason about it later (when it was said,
where it came from, how important it is).
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class MemoryType(str, Enum):
    FACT = "fact"          # durable info about the user ("I live in Bengaluru")
    EVENT = "event"        # something that happened ("had an interview on Aug 3")
    PREFERENCE = "preference"  # a stated like/dislike/style preference


class Memory(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    memory_type: MemoryType = MemoryType.FACT
    source: str = "user"                 # e.g. "chat", "upload", "note"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5              # 0-1, used to weight retrieval later
    superseded_by: Optional[str] = None  # id of a newer memory that overrides this one


class MemoryQueryResult(BaseModel):
    memory: Memory
    score: float  # similarity score from the vector store