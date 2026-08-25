"""
MemoOS — a memory layer for AI applications and for the terminal.

    from memoos_core import MemoOS

    memo = MemoOS(container="karthik")
    memo.remember("I moved to Delhi for an internship at Zomato.")
    memo.search("where does the user work?")

Names are resolved lazily. Importing MemoOS pulls in Chroma and the
embedding model, which costs about four seconds — fine once per process,
ruinous for a shell hook that runs on every command you type. Lazy
binding lets `memoos_core.journal` and `memoos_core.config` be imported
for their own sake in milliseconds, while `from memoos_core import
MemoOS` keeps working exactly as before.
"""

from typing import TYPE_CHECKING

# Attribute name -> the submodule that defines it.
_EXPORTS = {
    "MemoOS": "memory",
    "ConflictDecision": "models",
    "Document": "models",
    "DocumentStatus": "models",
    "Entity": "models",
    "EntityType": "models",
    "IngestResult": "models",
    "Memory": "models",
    "MemoryQueryResult": "models",
    "MemoryStatus": "models",
    "MemoryType": "models",
    "Relation": "models",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """PEP 562: resolve an export on first use, not at import time."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(f".{module_name}", __name__), name)
    globals()[name] = value  # cache, so the next lookup skips all this
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:  # static analysers need the real thing
    from .memory import MemoOS
    from .models import (
        ConflictDecision, Document, DocumentStatus, Entity, EntityType,
        IngestResult, Memory, MemoryQueryResult, MemoryStatus, MemoryType,
        Relation,
    )
