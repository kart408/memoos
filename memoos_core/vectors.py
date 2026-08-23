"""
Vector index: the semantic half of retrieval.

A thin wrapper over Chroma that keeps one collection per container, so
tenant isolation is a property of the storage layout rather than a
filter someone can forget to apply.

Deliberately narrow: this file knows about ids, text and similarity, and
nothing about memories, decay or the graph. Chroma stores no state that
SQLite doesn't also hold, so the index can always be rebuilt from the
database if it's lost or the embedding model changes.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import chromadb

from . import config
from .embedding import embed_batch, embed_query


class VectorIndex:
    def __init__(self, container: str = "default", path: Optional[str] = None):
        self.container = container
        self._client = chromadb.PersistentClient(path=path or config.vector_path())
        self._collection = self._client.get_or_create_collection(
            f"memories_{container}", metadata={"hnsw:space": "cosine"}
        )

    def add(self, ids: Sequence[str], texts: Sequence[str],
            metadatas: Optional[Sequence[Dict]] = None) -> None:
        if not ids:
            return
        self._collection.add(
            ids=list(ids),
            embeddings=embed_batch(list(texts)),
            documents=list(texts),
            metadatas=list(metadatas) if metadatas else None,
        )

    def add_one(self, memory_id: str, text: str,
                metadata: Optional[Dict] = None) -> None:
        self.add([memory_id], [text], [metadata] if metadata else None)

    def search(self, query: str, top_k: int = 10,
               exclude: Sequence[str] = ()) -> List[Tuple[str, float]]:
        """
        Nearest neighbours as (memory_id, cosine_similarity), best first.

        Over-fetches when ids are excluded so the caller still gets top_k
        usable results instead of a short list.
        """
        total = self.count()
        if total == 0:
            return []

        want = top_k + len(exclude)
        # Chroma errors rather than truncating if n_results exceeds the
        # collection size, so clamp it.
        n_results = min(max(want, 1), total)

        results = self._collection.query(
            query_embeddings=[embed_query(query)], n_results=n_results
        )
        if not results.get("ids") or not results["ids"][0]:
            return []

        excluded = set(exclude)
        out: List[Tuple[str, float]] = []
        for memory_id, distance in zip(results["ids"][0], results["distances"][0]):
            if memory_id in excluded:
                continue
            # Chroma's cosine "distance" is 1 - similarity for normalised
            # vectors; clamp because float error can push it just past 1.
            out.append((memory_id, max(0.0, min(1.0, 1.0 - float(distance)))))
            if len(out) >= top_k:
                break
        return out

    def search_vector(self, vector: List[float], top_k: int = 10,
                      exclude: Sequence[str] = ()) -> List[Tuple[str, float]]:
        """Same as `search` but for an already-computed embedding."""
        total = self.count()
        if total == 0:
            return []
        n_results = min(max(top_k + len(exclude), 1), total)
        results = self._collection.query(query_embeddings=[vector], n_results=n_results)
        if not results.get("ids") or not results["ids"][0]:
            return []

        excluded = set(exclude)
        out: List[Tuple[str, float]] = []
        for memory_id, distance in zip(results["ids"][0], results["distances"][0]):
            if memory_id in excluded:
                continue
            out.append((memory_id, max(0.0, min(1.0, 1.0 - float(distance)))))
            if len(out) >= top_k:
                break
        return out

    def delete(self, ids: Sequence[str]) -> None:
        if not ids:
            return
        self._collection.delete(ids=list(ids))

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Drop the whole collection. SQLite remains the source of truth."""
        self._client.delete_collection(f"memories_{self.container}")
        self._collection = self._client.get_or_create_collection(
            f"memories_{self.container}", metadata={"hnsw:space": "cosine"}
        )
