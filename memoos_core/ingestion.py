"""
Ingestion: turn raw text into Memory objects.

Phase 1 keeps this intentionally simple — one memory per input string.
Later you can extend this to split long documents into overlapping
chunks before embedding.
"""

from typing import List

from .models import Memory, MemoryType


def ingest_text(text: str, memory_type: MemoryType = MemoryType.FACT,
                 source: str = "user") -> Memory:
    """Wrap a single piece of text as a Memory, ready to embed and store."""
    cleaned = text.strip()
    return Memory(text=cleaned, memory_type=memory_type, source=source)


def chunk_document(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    """
    Naive word-based chunking with overlap, for longer documents.
    chunk_size / overlap are word counts, not tokens — good enough for v1.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text.strip()]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap  # step back to create overlap
    return chunks