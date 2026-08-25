"""
The write path: raw input in, consolidated memories out.

    document -> chunks -> extracted memories -> consolidation -> storage

Every stage records its progress on the Document row, so a long ingest
is observable while it runs and diagnosable after it fails. That matters
here more than usual: extraction is the slow step (seconds per chunk on
a local model), and "is it stuck or is it working?" is otherwise
unanswerable.

Failures are contained at the chunk level. One chunk the model chokes on
costs that chunk's memories, not the whole document.
"""

from typing import Dict, List, Optional, Sequence
from uuid import uuid4

from .chunking import chunk_text
from .consolidation import Consolidator
from .db import Database
from .extraction import extract_memories
from .graph import MemoryGraph
from .models import (
    Document,
    DocumentStatus,
    Entity,
    ExtractedMemory,
    IngestResult,
    Memory,
    MemoryType,
    Relation,
)
from .vectors import VectorIndex


class IngestionPipeline:
    def __init__(self, db: Database, vectors: VectorIndex, graph: MemoryGraph,
                 consolidator: Consolidator, container: str = "default"):
        self.db = db
        self.vectors = vectors
        self.graph = graph
        self.consolidator = consolidator
        self.container = container

    def ingest(self, text: str, *, source: str = "user", title: Optional[str] = None,
               uri: Optional[str] = None, subject_scoped: bool = True,
               consolidate: bool = True,
               metadata: Optional[Dict] = None) -> IngestResult:
        """
        Run raw text through the full pipeline.

        `subject_scoped` marks this as the user talking about themselves,
        which turns on the question and user-subject guards in extraction.
        Set it False for reference documents, where nothing is about the
        user and those guards would reject everything.
        """
        document = Document(
            container=self.container,
            title=title,
            uri=uri,
            source=source,
            raw_text=text.strip(),
            status=DocumentStatus.QUEUED,
            metadata=metadata or {},
        )
        self.db.insert_document(document)
        result = IngestResult(document=document)

        try:
            self.db.update_document(document.id, status=DocumentStatus.CHUNKING)
            chunks = chunk_text(document.raw_text)
            if not chunks:
                self.db.update_document(document.id, status=DocumentStatus.DONE)
                return result

            chunk_ids = [str(uuid4()) for _ in chunks]
            self.db.insert_chunks([
                (chunk_ids[i], document.id, self.container, i, chunks[i])
                for i in range(len(chunks))
            ])
            self.db.update_document(document.id, chunk_count=len(chunks),
                                    status=DocumentStatus.EXTRACTING)

            extracted: List[tuple[str, ExtractedMemory]] = []
            result.chunks_total = len(chunks)
            for chunk_id, chunk in zip(chunk_ids, chunks):
                try:
                    for item in extract_memories(chunk, subject_scoped=subject_scoped,
                                                 strict=True):
                        extracted.append((chunk_id, item))
                except Exception:
                    # One unreadable chunk shouldn't sink the document — but
                    # it must be counted. Swallowing it made a timed-out
                    # chunk indistinguishable from an uneventful one, and a
                    # caller that then marked its source as processed threw
                    # the content away without ever knowing it existed.
                    result.chunks_failed += 1
                    continue

            self.db.update_document(document.id, status=DocumentStatus.EMBEDDING)

            # Memories from this same document can't supersede each other
            # — they were stated together, so they complement rather than
            # contradict. They can still merge as duplicates.
            batch_ids: List[str] = []
            for chunk_id, item in extracted:
                self._store(item, result, document_id=document.id,
                            chunk_id=chunk_id, source=source,
                            consolidate=consolidate, protected=tuple(batch_ids))
                batch_ids = [m.id for m in result.created]

            self.db.update_document(
                document.id,
                status=DocumentStatus.DONE,
                memory_count=len(result.created),
            )
        except Exception as exc:
            self.db.update_document(document.id, status=DocumentStatus.FAILED,
                                    error=str(exc)[:500])
            raise

        refreshed = self.db.get_document(document.id)
        if refreshed:
            result.document = refreshed
        return result

    def _store(self, item: ExtractedMemory, result: IngestResult, *,
               document_id: Optional[str], chunk_id: Optional[str],
               source: str, consolidate: bool,
               protected: Sequence[str] = ()) -> None:
        """Consolidate one extracted memory against the store, then write it."""
        if consolidate:
            # Entity names come from this memory's own extraction, but
            # are looked up against entities *already* in the graph — so
            # a new memory about PyTorch finds the older ones about
            # PyTorch, however differently they're worded.
            plan = self.consolidator.plan(
                item.text,
                entity_names=[e.name for e in item.entities],
                protected=protected,
            )
            if plan.is_duplicate:
                # Don't store it again — strengthen what's already there.
                reinforced = self.consolidator.reinforce(plan.duplicate_of)
                result.duplicates.append(reinforced)
                return
        else:
            plan = None

        memory = Memory(
            container=self.container,
            text=item.text,
            memory_type=item.memory_type,
            source=source,
            document_id=document_id,
            chunk_id=chunk_id,
            importance=item.importance,
            confidence=item.confidence,
        )

        # SQLite first: it's the source of truth, so if the vector index
        # write fails the memory still exists and can be re-indexed. The
        # reverse order would leave an orphaned vector pointing at nothing.
        self.db.insert_memory(memory)
        self.vectors.add_one(memory.id, memory.text, {
            "memory_type": memory.memory_type.value,
            "created_at": memory.created_at.isoformat(),
        })

        entities, relations = self.graph.attach(memory, item)
        result.entities.extend(entities)
        result.relations.extend(relations)

        if plan is not None and plan.supersedes:
            result.superseded.extend(self.consolidator.apply(plan, memory))

        result.created.append(memory)

    def store_verbatim(self, text: str, *, memory_type: MemoryType = MemoryType.FACT,
                       source: str = "user", importance: float = 0.5,
                       consolidate: bool = True) -> tuple[Memory, IngestResult]:
        """
        Store a piece of text as a single memory, exactly as written.

        Skips extraction entirely — the caller has already decided what
        the memory says. Consolidation still runs, so verbatim writes
        can't silently duplicate or contradict what's already stored.
        """
        document = Document(
            container=self.container, source=source, raw_text=text.strip(),
            status=DocumentStatus.DONE, chunk_count=0,
        )
        self.db.insert_document(document)
        result = IngestResult(document=document)

        item = ExtractedMemory(
            text=text.strip(), memory_type=memory_type, importance=importance,
            confidence=1.0,  # no extraction step, so nothing to be unsure about
        )
        self._store(item, result, document_id=document.id, chunk_id=None,
                    source=source, consolidate=consolidate)

        self.db.update_document(document.id, memory_count=len(result.created))

        # A duplicate write returns the memory it merged into, so callers
        # always get back a real, stored memory to hold onto.
        memory = result.created[0] if result.created else result.duplicates[0]
        return memory, result
