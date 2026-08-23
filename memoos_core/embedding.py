"""
Embedding: turn text into vectors.

Swapping models later (e.g. to an API-based embedder) only means
changing this file — nothing else in the pipeline should care how the
vector was produced.

Vectors are L2-normalised at encode time, which lets every downstream
consumer treat a dot product as cosine similarity and lets Chroma's
cosine distance map cleanly onto `1 - distance`.
"""

import warnings
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from . import config


def _self_check(model: SentenceTransformer) -> bool:
    """
    Verify the model returns sane vectors on its current device.

    This exists because a broken accelerator backend does not fail loudly.
    Apple's MPS, when starved of memory by a co-resident Ollama model,
    returns corrupted tensors instead of raising — the same sentence
    embedded twice scored -0.03 against itself, and unrelated sentences
    scored 1.0. Nothing downstream can detect that; retrieval just
    quietly becomes random. Two cheap invariants catch it:

      identical text  must be ~1.0 similar to itself
      unrelated text  must not be

    Cheaper than one search, and it runs once per process.
    """
    try:
        vectors = model.encode(
            ["memory system self check", "memory system self check",
             "quantum chromodynamics lattice gauge theory"],
            normalize_embeddings=True,
        )
    except Exception:
        return False

    identical = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    return identical > 0.99 and unrelated < 0.95


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the (relatively slow) model load happens once per process.
    model = SentenceTransformer(config.EMBED_MODEL, device=config.EMBED_DEVICE)

    if _self_check(model):
        return model

    if config.EMBED_DEVICE != "cpu":
        warnings.warn(
            f"Embedding self-check failed on device {config.EMBED_DEVICE!r} — "
            f"vectors were corrupt, falling back to CPU. This usually means the "
            f"GPU is out of memory (a local LLM may be holding it).",
            RuntimeWarning,
            stacklevel=2,
        )
        model = SentenceTransformer(config.EMBED_MODEL, device="cpu")
        if _self_check(model):
            return model

    # Refusing to continue is correct here: silently bad embeddings poison
    # every memory written from now on, and the damage is invisible until
    # someone notices retrieval has become nonsense.
    raise RuntimeError(
        f"Embedding model {config.EMBED_MODEL!r} is producing invalid vectors "
        f"on CPU. Refusing to continue — every stored memory would be corrupt."
    )


@lru_cache(maxsize=2048)
def _embed_cached(text: str) -> tuple:
    # Consolidation re-embeds the same text several times per write (once
    # to store, again to find conflict candidates), and queries repeat
    # constantly in a chat loop. A tuple is returned because lru_cache
    # keys and values must be hashable/immutable.
    model = _get_model()
    return tuple(model.encode(text, normalize_embeddings=True).tolist())


def embed_text(text: str) -> List[float]:
    """Embed a stored memory or passage."""
    return list(_embed_cached(text))


def embed_query(text: str) -> List[float]:
    """
    Embed a search query.

    Distinct from `embed_text` because retrieval here is asymmetric: a
    question and the statement that answers it are different kinds of
    text, and retrieval-tuned models expect the query side to carry an
    instruction prefix. Passages must be embedded *without* it — applying
    it to both sides collapses the distinction the model was trained on.
    """
    return list(_embed_cached(config.EMBED_QUERY_PREFIX + text))


def embed_batch(texts: List[str]) -> List[List[float]]:
    if not texts:
        return []
    model = _get_model()
    return model.encode(texts, normalize_embeddings=True).tolist()


def cosine(a: List[float], b: List[float]) -> float:
    """Similarity between two already-normalised vectors."""
    return sum(x * y for x, y in zip(a, b))


def dimension() -> int:
    return _get_model().get_sentence_embedding_dimension()


def self_check() -> bool:
    """Public health check: are embeddings currently trustworthy?"""
    return _self_check(_get_model())


def active_device() -> str:
    return str(_get_model().device)
