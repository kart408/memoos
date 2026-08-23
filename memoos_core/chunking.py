"""
Chunking: split long input into pieces small enough to extract from.

The old word-window approach cut mid-sentence, which matters more here
than in ordinary RAG: a chunk is the unit an LLM extracts *facts* from,
and a sentence severed in half either yields nothing or yields something
wrong. So we pack whole sentences up to a word budget instead, and carry
whole sentences across the boundary as overlap.
"""

import re
from typing import List

from . import config

# Split on sentence-ending punctuation followed by whitespace, but don't
# break on common abbreviations or decimals ("Dr. Rao", "version 3.11").
_ABBREVIATIONS = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bProf)(?<!\bSr)(?<!\bJr)(?<!\bvs)(?<!\betc)(?<!\be\.g)(?<!\bi\.e)"
_SENTENCE_BREAK = re.compile(rf"{_ABBREVIATIONS}(?<=[.!?])(?<!\d\.)\s+")

# Blank line = a hard boundary. Paragraphs and markdown blocks shouldn't
# be glued together just because they fit in the same word budget.
_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


def split_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting paragraph boundaries."""
    sentences: List[str] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BREAK.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def chunk_text(text: str, chunk_size: int | None = None,
               overlap: int | None = None) -> List[str]:
    """
    Pack sentences into chunks of at most `chunk_size` words, overlapping
    consecutive chunks by roughly `overlap` words of trailing sentences.

    Sizes are word counts, not tokens — close enough at this scale, and
    it keeps the module dependency-free.
    """
    size = chunk_size if chunk_size is not None else config.CHUNK_SIZE_WORDS
    lap = overlap if overlap is not None else config.CHUNK_OVERLAP_WORDS

    # Overlap must stay strictly under the chunk size, or the carry-over
    # consumes the whole next chunk and the loop never advances.
    lap = max(0, min(lap, size - 1))

    cleaned = text.strip()
    if not cleaned:
        return []

    sentences = split_sentences(cleaned)
    if not sentences:
        return [cleaned]

    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for sentence in sentences:
        words = len(sentence.split())

        # A single sentence longer than the budget can't be packed; give
        # it a chunk of its own rather than dropping or truncating it.
        if words >= size:
            if current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            chunks.append(sentence)
            continue

        if current_words + words > size and current:
            chunks.append(" ".join(current))
            current, current_words = _carry_over(current, lap)

        current.append(sentence)
        current_words += words

    if current:
        chunks.append(" ".join(current))

    return chunks


def _carry_over(sentences: List[str], overlap_words: int) -> tuple[List[str], int]:
    """Take whole trailing sentences worth up to `overlap_words` words."""
    if overlap_words <= 0:
        return [], 0

    carried: List[str] = []
    total = 0
    for sentence in reversed(sentences):
        words = len(sentence.split())
        if total + words > overlap_words:
            break
        carried.insert(0, sentence)
        total += words
    return carried, total


# Kept under its original name so the knowledge base and any existing
# callers keep working unchanged.
def chunk_document(text: str, chunk_size: int = 300, overlap: int = 50) -> List[str]:
    return chunk_text(text, chunk_size=chunk_size, overlap=overlap)
