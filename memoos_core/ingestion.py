"""
Ingestion helpers, kept for backwards compatibility.

The real work moved out: chunking lives in `chunking.py`, and turning
raw input into structured memories is now the pipeline's job. This
module stays so existing imports (`chunk_document`, `ingest_text`)
keep resolving.
"""

from typing import List

from .chunking import chunk_document, chunk_text, split_sentences  # noqa: F401
from .models import Memory, MemoryType


def ingest_text(text: str, memory_type: MemoryType = MemoryType.FACT,
                source: str = "user", container: str = "default") -> Memory:
    """Wrap a single piece of text as a Memory, ready to embed and store."""
    return Memory(
        text=text.strip(),
        memory_type=memory_type,
        source=source,
        container=container,
    )


__all__ = ["ingest_text", "chunk_document", "chunk_text", "split_sentences"]
