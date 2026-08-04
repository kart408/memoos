"""
Storage: persist memories + their vectors, and query by similarity.

Uses ChromaDB in local persistent mode — zero external services to
run, which keeps the demo frictionless. Swap for Qdrant/pgvector
later if you outgrow this; the MemoryStore interface below is the
seam where that swap happens.
"""

from typing import List, Optional

import chromadb

from .embedding import embed_batch, embed_text
from .models import Memory, MemoryQueryResult


class MemoryStore:
    def __init__(self, persist_path: str = "./memoos_data"):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
    "memories", metadata={"hnsw:space": "cosine"}
)
    def add(self, memory: Memory) -> None:
        vector = embed_text(memory.text)
        self._collection.add(
            ids=[memory.id],
            embeddings=[vector],
            documents=[memory.text],
            metadatas=[{
                "memory_type": memory.memory_type.value,
                "source": memory.source,
                "created_at": memory.created_at.isoformat(),
                "importance": memory.importance,
                "superseded_by": memory.superseded_by or "",
            }],
        )

    def add_many(self, memories: List[Memory]) -> None:
        vectors = embed_batch([m.text for m in memories])
        self._collection.add(
            ids=[m.id for m in memories],
            embeddings=vectors,
            documents=[m.text for m in memories],
            metadatas=[{
                "memory_type": m.memory_type.value,
                "source": m.source,
                "created_at": m.created_at.isoformat(),
                "importance": m.importance,
                "superseded_by": m.superseded_by or "",
            } for m in memories],
        )

    def search(self, query: str, top_k: int = 5) -> List[MemoryQueryResult]:
        query_vector = embed_text(query)
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)

        output = []
        for i in range(len(results["ids"][0])):
            meta = results["metadatas"][0][i]
            # Skip memories that have been superseded by a newer one.
            if meta.get("superseded_by"):
                continue
            memory = Memory(
                id=results["ids"][0][i],
                text=results["documents"][0][i],
                memory_type=meta["memory_type"],
                source=meta["source"],
                created_at=meta["created_at"],
                importance=meta["importance"],
            )
            # Chroma returns a distance; convert to a similarity-style score.
            distance = results["distances"][0][i]
            score = 1 - distance
            output.append(MemoryQueryResult(memory=memory, score=score))
        return output

    def mark_superseded(self, old_id: str, new_id: str) -> None:
        """Mark an old memory as replaced by a newer, contradicting one."""
        self._collection.update(ids=[old_id], metadatas=[{"superseded_by": new_id}])