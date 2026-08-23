"""
Knowledge base: a client's static documents (FAQs, policies, course
lists, product info) — separate from user memory. Every client gets
their own collection, keyed by client_id.
"""

from typing import List

import chromadb

from .embedding import embed_batch, embed_text
from .ingestion import chunk_document


class KnowledgeBase:
    def __init__(self, client_id: str, persist_path: str = "./kb_data"):
        self._client = chromadb.PersistentClient(path=persist_path)
        self._collection = self._client.get_or_create_collection(
            f"kb_{client_id}", metadata={"hnsw:space": "cosine"}
        )

    def add_document(self, text: str, doc_name: str = "document") -> int:
        chunks = chunk_document(text)
        vectors = embed_batch(chunks)
        ids = [f"{doc_name}_{i}" for i in range(len(chunks))]
        self._collection.add(
            ids=ids,
            embeddings=vectors,
            documents=chunks,
            metadatas=[{"source": doc_name} for _ in chunks],
        )
        return len(chunks)

    def search(self, query: str, top_k: int = 3, min_score: float = 0.35) -> List[str]:
        """
        Returns only chunks whose similarity to the query is above
        min_score - so unrelated statements don't drag in irrelevant
        facts just because they're the 'closest' available match.
        """
        query_vector = embed_text(query)
        results = self._collection.query(query_embeddings=[query_vector], n_results=top_k)

        if not results["documents"] or not results["documents"][0]:
            return []

        relevant = []
        for doc, distance in zip(results["documents"][0], results["distances"][0]):
            score = 1 - distance
            if score >= min_score:
                relevant.append(doc)
        return relevant
