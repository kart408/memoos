"""
Embeddings, from the local Ollama instance.

Nothing here leaves the machine. The model is pulled once
(`ollama pull nomic-embed-text`) and served on localhost.

Vectors are L2-normalised on the way out, which makes cosine similarity
a plain dot product downstream. Doing it once at embed time rather than
per comparison means a search over N memories does N multiply-adds
instead of N norm computations.
"""

import os
import threading
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests

OLLAMA_URL = os.getenv("MEMOOS_OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = os.getenv("MEMOOS_EMBED_MODEL", "nomic-embed-text")
CHAT_MODEL = os.getenv("MEMOOS_CHAT_MODEL", "mistral:latest")
TIMEOUT = int(os.getenv("MEMOOS_TIMEOUT", "120"))
CACHE_SIZE = int(os.getenv("MEMOOS_CACHE_SIZE", "512"))


class EmbeddingError(RuntimeError):
    """Ollama was unreachable, or returned something unusable."""


class _LRUCache:
    """
    A small bounded cache of text -> vector.

    Testing re-embeds the same handful of strings constantly (the same
    query run against different users, the same fixture reloaded), and
    each miss is a real model call. Bounded so a long-running server
    cannot grow it without limit.
    """

    def __init__(self, capacity: int = CACHE_SIZE):
        self.capacity = capacity
        self._data: "OrderedDict[Tuple[str, str], List[float]]" = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: Tuple[str, str]) -> Optional[List[float]]:
        with self._lock:
            if key in self._data:
                self._data.move_to_end(key)
                self.hits += 1
                return self._data[key]
            self.misses += 1
            return None

    def put(self, key: Tuple[str, str], value: List[float]) -> None:
        with self._lock:
            self._data[key] = value
            self._data.move_to_end(key)
            while len(self._data) > self.capacity:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self.hits = self.misses = 0

    def stats(self) -> Dict[str, int]:
        with self._lock:
            return {"size": len(self._data), "capacity": self.capacity,
                    "hits": self.hits, "misses": self.misses}


_cache = _LRUCache()


def normalise(vector: List[float]) -> List[float]:
    """Scale to unit length so cosine similarity becomes a dot product."""
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        # A zero vector has no direction; leave it alone rather than
        # dividing by zero and poisoning the index with NaNs.
        return array.tolist()
    return (array / norm).tolist()


def embed(text: str, *, model: str = EMBED_MODEL, use_cache: bool = True) -> List[float]:
    """Embed one piece of text. Returns a unit-length vector."""
    key = (model, text)
    if use_cache:
        cached = _cache.get(key)
        if cached is not None:
            return cached

    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=TIMEOUT,
        )
    except requests.RequestException as exc:
        raise EmbeddingError(
            f"cannot reach Ollama at {OLLAMA_URL} — is it running? ({exc})"
        ) from exc

    if response.status_code != 200:
        raise EmbeddingError(
            f"Ollama returned {response.status_code} for model '{model}': "
            f"{response.text[:200]}"
        )

    vector = response.json().get("embedding")
    if not vector:
        raise EmbeddingError(
            f"Ollama returned no embedding for model '{model}'. "
            f"Is it an embedding model? Try: ollama pull {EMBED_MODEL}"
        )

    unit = normalise(vector)
    if use_cache:
        _cache.put(key, unit)
    return unit


def embed_many(texts: List[str], *, model: str = EMBED_MODEL) -> List[List[float]]:
    """
    Embed several texts.

    Ollama's embeddings endpoint takes one prompt at a time, so this is a
    loop rather than a batch — but it still cuts request setup and lets
    the cache absorb repeats within the batch.
    """
    return [embed(t, model=model) for t in texts]


def cosine_scores(query: List[float], matrix: np.ndarray) -> np.ndarray:
    """
    Similarity of one query vector against a stack of stored vectors.

    Both sides are already unit length, so the dot product *is* the
    cosine. The stored side is re-normalised defensively: a vector that
    predates normalisation, or arrived through a direct DB write, would
    otherwise score above 1.0 and outrank everything legitimately.
    """
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)

    q = np.asarray(query, dtype=np.float32)
    q_norm = np.linalg.norm(q)
    if q_norm:
        q = q / q_norm

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms) @ q


def ping() -> Tuple[bool, str]:
    """Is Ollama up, and is the embedding model actually loadable?"""
    try:
        response = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if response.status_code != 200:
            return False, f"Ollama returned {response.status_code}"
        names = [m.get("name", "") for m in response.json().get("models", [])]
        if not any(n.split(":")[0] == EMBED_MODEL.split(":")[0] for n in names):
            return False, (f"model '{EMBED_MODEL}' not pulled — "
                           f"run: ollama pull {EMBED_MODEL}")
        return True, "ok"
    except requests.RequestException as exc:
        return False, f"cannot reach Ollama at {OLLAMA_URL} ({exc})"


def cache_stats() -> Dict[str, int]:
    return _cache.stats()


def clear_cache() -> None:
    _cache.clear()
