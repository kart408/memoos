"""
MemoOS: the public interface.

This is what an application imports. Everything else in memoos_core is
an implementation detail behind this class — the database, the vector
index, the graph, the extraction pipeline and the consolidation rules
are all assembled here and hidden.

Two ways in, deliberately distinct:

    add(text)       stores exactly what you give it, as one memory.
                    Fast and predictable — you already know what it says.

    remember(text)  runs the full pipeline: chunk, extract atomic facts,
                    build the entity graph, resolve contradictions.
                    Slower (it calls the model) and far more capable.

Both consolidate against what's already stored, so neither can silently
duplicate or contradict existing memory.
"""

from typing import Dict, List, Optional

from . import config
from .consolidation import Consolidator, strength
from .db import Database
from .graph import MemoryGraph
from .models import (
    Entity,
    IngestResult,
    Memory,
    MemoryQueryResult,
    MemoryStatus,
    MemoryType,
)
from .pipeline import IngestionPipeline
from .retrieval import Retriever
from .vectors import VectorIndex


class MemoOS:
    def __init__(self, client_id: str = "default",
                 persist_path: Optional[str] = None,
                 container: Optional[str] = None):
        # `container` is the name this concept goes by everywhere else;
        # `client_id` is kept as the first positional argument so existing
        # callers keep working.
        self.container = container or client_id
        self.persist_path = persist_path or config.DATA_DIR

        self.db = Database(config.db_path(self.persist_path))
        self.vectors = VectorIndex(self.container, config.vector_path(self.persist_path))
        self.graph = MemoryGraph(self.db, self.container)
        self.consolidator = Consolidator(self.db, self.vectors, self.graph, self.container)
        self.retriever = Retriever(self.db, self.vectors, self.graph, self.container)
        self.pipeline = IngestionPipeline(
            self.db, self.vectors, self.graph, self.consolidator, self.container
        )

    # ------------------------------------------------------------ write

    def add(self, text: str, memory_type: MemoryType = MemoryType.FACT,
            source: str = "user", importance: float = 0.5) -> Memory:
        """
        Store text verbatim as a single memory.

        Returns the stored memory — or, if this restated something
        already known, the existing memory it was merged into.
        """
        memory, _ = self.pipeline.store_verbatim(
            text, memory_type=memory_type, source=source, importance=importance
        )
        return memory

    def remember(self, text: str, *, source: str = "user",
                 title: Optional[str] = None) -> IngestResult:
        """
        Extract and store everything worth remembering from `text`.

        This is the interesting path: one message can yield several
        atomic memories, populate the entity graph, and supersede facts
        that are no longer true.
        """
        return self.pipeline.ingest(text, source=source, title=title,
                                    subject_scoped=True)

    def ingest_document(self, text: str, *, title: Optional[str] = None,
                        uri: Optional[str] = None,
                        source: str = "document") -> IngestResult:
        """
        Ingest reference material rather than something the user said.

        Turns off the conversational guards — a policy document contains
        no statements about the user, and applying those checks here
        would reject every fact in it.
        """
        return self.pipeline.ingest(text, source=source, title=title, uri=uri,
                                    subject_scoped=False)

    def supersede(self, old_memory_id: str, new_text: str,
                  memory_type: MemoryType = MemoryType.FACT,
                  reason: str = "manual supersede") -> Memory:
        """Explicitly replace a memory with a newer one."""
        new_memory = self.add(new_text, memory_type=memory_type)
        self.db.mark_superseded(old_memory_id, new_memory.id, reason)
        self.graph.invalidate_for_memory(old_memory_id)
        self.vectors.delete([old_memory_id])
        return new_memory

    def delete(self, memory_id: str) -> None:
        """Remove a memory permanently, from both stores."""
        self.vectors.delete([memory_id])
        self.db.delete_memory(memory_id)

    # ------------------------------------------------------------- read

    def search(self, query: str, top_k: int = 5, *, use_graph: bool = True,
               memory_type: Optional[MemoryType] = None,
               touch: bool = True) -> List[MemoryQueryResult]:
        """Hybrid search: vector + keyword + graph, decay-weighted."""
        return self.retriever.search(query, top_k=top_k, use_graph=use_graph,
                                     memory_type=memory_type, touch=touch)

    def recall_as_context(self, query: str, top_k: int = 3,
                          include_types: bool = False) -> str:
        """
        Retrieved memories formatted for a system prompt.

        The wording matters: memories are presented as things the user
        told you, not as ground truth, so a model that finds them
        irrelevant is free to ignore them rather than forced to
        rationalise them into the answer.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""

        seen: set[str] = set()
        lines: List[str] = []
        for result in results:
            text = result.memory.text.strip()
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            label = f" [{result.memory.memory_type.value}]" if include_types else ""
            lines.append(f"-{label} {text}")

        return "\n".join(lines)

    def get(self, memory_id: str) -> Optional[Memory]:
        return self.db.get_memory(memory_id)

    def all(self, limit: int = 100, offset: int = 0,
            status: Optional[MemoryStatus] = MemoryStatus.ACTIVE,
            memory_type: Optional[MemoryType] = None) -> List[Memory]:
        return self.db.list_memories(self.container, status=status,
                                     memory_type=memory_type,
                                     limit=limit, offset=offset)

    def history(self, memory_id: str) -> List[Memory]:
        """
        Follow the supersession chain forward from a memory.

        Answers "what does the system believe now, and what did it
        believe before?" — the audit trail that makes contradiction
        handling trustworthy instead of merely convenient.
        """
        chain: List[Memory] = []
        seen: set[str] = set()
        current = self.db.get_memory(memory_id)
        while current and current.id not in seen:
            seen.add(current.id)
            chain.append(current)
            if not current.superseded_by:
                break
            current = self.db.get_memory(current.superseded_by)
        return chain

    # ------------------------------------------------------------ graph

    def entities(self, limit: int = 20) -> List[Entity]:
        """The things this container's memories are most about."""
        return self.graph.summary(limit=limit)

    def about(self, entity_name: str) -> Dict:
        """Everything the graph knows about one named thing."""
        return self.graph.neighbourhood(entity_name)

    # ------------------------------------------------------ maintenance

    def strength_of(self, memory_id: str) -> Optional[float]:
        memory = self.db.get_memory(memory_id)
        return strength(memory) if memory else None

    def weak(self, threshold: Optional[float] = None) -> List[tuple[Memory, float]]:
        """Memories that have decayed below the retention threshold."""
        return self.consolidator.weak_memories(threshold)

    def forget_weak(self, threshold: Optional[float] = None) -> List[Memory]:
        """Retire decayed memories. Reversible — rows are kept, not deleted."""
        return self.consolidator.forget_weak(threshold)

    def reindex(self) -> int:
        """
        Rebuild the vector index from SQLite.

        The escape hatch for a corrupted index or a changed embedding
        model. Possible only because SQLite, not Chroma, is the source
        of truth.
        """
        self.vectors.reset()
        total = 0
        offset = 0
        while True:
            batch = self.db.list_memories(self.container, limit=500, offset=offset)
            if not batch:
                break
            self.vectors.add(
                [m.id for m in batch],
                [m.text for m in batch],
                [{"memory_type": m.memory_type.value,
                  "created_at": m.created_at.isoformat()} for m in batch],
            )
            total += len(batch)
            offset += len(batch)
        return total

    def stats(self) -> Dict:
        return {
            "container": self.container,
            "active": self.db.count_memories(self.container, MemoryStatus.ACTIVE),
            "superseded": self.db.count_memories(self.container, MemoryStatus.SUPERSEDED),
            "forgotten": self.db.count_memories(self.container, MemoryStatus.FORGOTTEN),
            "total": self.db.count_memories(self.container),
            "entities": len(self.db.list_entities(self.container, limit=10_000)),
            "relations": len(self.db.list_relations(self.container, limit=10_000)),
            "indexed_vectors": self.vectors.count(),
        }

    def close(self) -> None:
        self.db.close()
