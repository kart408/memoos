"""
Embedding: turn text into vectors.

Swapping models later (e.g. to an API-based embedding model) only
means changing this file — nothing else in the pipeline should care
how the vector was produced.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, good enough for a v1 demo


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the (relatively slow) model load only happens once per process.
    return SentenceTransformer(MODEL_NAME)


def embed_text(text: str) -> List[float]:
    model = _get_model()
    return model.encode(text, normalize_embeddings=True).tolist()


def embed_batch(texts: List[str]) -> List[List[float]]:
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()