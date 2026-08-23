"""
MemoOS — a memory layer for AI applications.

    from memoos_core import MemoOS

    memo = MemoOS(container="karthik")
    memo.remember("I moved to Delhi for an internship at Zomato.")
    memo.search("where does the user work?")
"""

from .memory import MemoOS
from .models import (
    ConflictDecision,
    Document,
    DocumentStatus,
    Entity,
    EntityType,
    IngestResult,
    Memory,
    MemoryQueryResult,
    MemoryStatus,
    MemoryType,
    Relation,
)

__all__ = [
    "MemoOS",
    "Memory",
    "MemoryType",
    "MemoryStatus",
    "MemoryQueryResult",
    "Entity",
    "EntityType",
    "Relation",
    "Document",
    "DocumentStatus",
    "IngestResult",
    "ConflictDecision",
]
