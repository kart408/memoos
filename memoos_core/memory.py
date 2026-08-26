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
from .retrieval import Retriever, in_scope
from .vectors import VectorIndex
from . import signals




def format_as_context(results: List[MemoryQueryResult],
                      include_types: bool = False,
                      project: Optional[str] = None,
                      episodes: Optional[List] = None) -> str:
    """
    Render retrieved memories as prompt-ready lines.

    This is the artefact MemoOS exists to produce — the last stage, and
    the handover point. It goes into somebody else's prompt, so it is
    written to be read by a model that has never seen this project: named
    subjects, one fact per line, no scores and no ids.

    The project header earns its line. Dropped into a prompt alongside
    the user's task, an unlabelled list of facts is ambiguous about what
    it describes; `Project: my-app` says these are facts about the thing
    you are being asked to work on.

    Split out from `recall_as_context` so a caller holding results from a
    search it already ran can format them without running that search a
    second time — which would not just cost another embedding round trip
    but also reinforce the same memories twice for a single message.
    """
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
        lines.append(f"•{label} {text}")

    block = "\n".join(lines)

    # Past signals get their own heading rather than being mixed in with
    # the facts. A known failure is a different kind of thing from "the
    # project uses Next.js" — it is a warning, and a reader skimming a
    # prompt should be able to see that at a glance.
    if episodes:
        rendered = signals.render(episodes)
        if rendered:
            block = f"{block}\n\n{rendered}"

    if project:
        return f"Project: {project}\n\n{block}"
    return block


class MemoOS:
    def __init__(self, client_id: str = "default",
                 persist_path: Optional[str] = None,
                 container: Optional[str] = None):
        # `container` is the name this concept goes by everywhere else;
        # `client_id` is kept as the first positional argument so existing
        # callers keep working.
        #
        # Normalised on the way in, because `db_path` normalises too: an
        # un-normalised name would open one container's file and write
        # rows into it under a label nothing else queries.
        self.container = config.safe_container(container or client_id)
        self.persist_path = persist_path or config.DATA_DIR

        # One file, this container's own: `Database` and `VectorIndex`
        # open separate connections to it, which WAL makes safe.
        self.path = config.db_path(self.persist_path, self.container)
        self.db = Database(self.path)
        self.vectors = VectorIndex(self.container, self.path)
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
                        source: str = "document",
                        metadata: Optional[Dict] = None) -> IngestResult:
        """
        Ingest reference material rather than something the user said.

        Turns off the conversational guards — a policy document contains
        no statements about the user, and applying those checks here
        would reject every fact in it.

        `metadata` rides along on the document row. It is what lets a
        memory be traced back past its own text to the raw events it was
        distilled from — see `source()`.
        """
        return self.pipeline.ingest(text, source=source, title=title, uri=uri,
                                    subject_scoped=False, metadata=metadata)

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

    def recall_as_context(self, query: str, top_k: Optional[int] = None,
                          include_types: bool = False) -> str:
        """
        Retrieved memories formatted for a system prompt.

        The wording matters: memories are presented as things the user
        told you, not as ground truth, so a model that finds them
        irrelevant is free to ignore them rather than forced to
        rationalise them into the answer.
        """
        results = self.search(
            query, top_k=top_k if top_k is not None else config.CONTEXT_TOP_K
        )
        return format_as_context(results, include_types=include_types,
                                 project=self.container)

    def context_for(self, task: str, *, top_k: Optional[int] = None,
                    include_stale: bool = False, touch: bool = False,
                    with_signals: bool = True, gate: bool = True) -> Dict:
        """
        The whole read path, and the last thing MemoOS does.

        Understand the query, retrieve from both sides, drop what is no
        longer current, rerank, and render what survives as a context
        block. Then stop. The agent that asked — Claude Code, or anything
        else — is what answers, because it is the only thing that knows
        what the user is actually trying to do. A memory layer that also
        wrote the reply would be guessing at that.

        Returns the block *and* the memories behind it, so a caller can
        show its working rather than pasting an opaque paragraph into a
        prompt.

        `touch=False` by default: fetching context is an inspection, and
        reinforcing whatever came back would let the act of looking
        reshape the ranking.

        `include_stale` keeps active memories whose validity window has
        closed. Superseded ones are not reachable here by design — see
        `history()` for those.

        `gate` applies the calibrated relevance bars, so a question this
        container knows nothing about comes back empty instead of coming
        back with its three least-bad guesses. Turn it off to see what
        search would have said.
        """
        wanted = top_k if top_k is not None else config.CONTEXT_TOP_K
        plan = self.retriever.plan_for(task)
        results = self.retriever.search(task, top_k=wanted, touch=touch,
                                        include_stale=include_stale, plan=plan)

        # The handover needs an "I don't know about that" state, and
        # search does not have one — it returns its nearest neighbours
        # however distant. Fine for a caller that can see the scores;
        # not fine for a block that goes into a prompt with the numbers
        # stripped off. Asked who won the world cup, this store offered
        # three memories about itself at cosine 0.49.
        if gate:
            results = in_scope(results, query=task)

        # Past signals, scoped to what this task touches. Reported, never
        # acted on: nothing here reorders the results or retires anything.
        # What to do about a known failure is the calling agent's call,
        # because it is the only thing that can see the actual task.
        #
        # Nothing relevant means nothing to scope them to. Reporting them
        # anyway would answer an out-of-scope question with the project's
        # entire failure history — `episodes_for` with no focus reports
        # everything, which is right for a fresh terminal and wrong here.
        episodes = signals.episodes_for(
            self.db, self.container,
            memory_ids=[r.memory.id for r in results],
        ) if (with_signals and results) else []

        return {
            "container": self.container,
            "query": task,
            "plan": plan.as_dict(),
            "results": results,
            "signals": episodes,
            "context": format_as_context(results, project=self.container,
                                         episodes=episodes),
        }

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

    def signals(self, limit: int = 25) -> List:
        """
        Every problem/solution episode this project knows about.

        The whole-store view, for a terminal that has just opened and has
        no task yet. `context_for` gives the same thing scoped to what a
        specific task touches.
        """
        return signals.episodes_for(self.db, self.container, limit=limit)

    def source(self, memory_id: str) -> Optional[Dict]:
        """
        Where a memory came from, all the way back to the raw input.

        A memory store you cannot audit is a memory store you cannot
        trust: the one failure that matters is a confidently-retrieved
        fact nobody ever stated, and the only way to tell that from a
        real one is to follow it back. The chain is already in the
        columns — memory -> chunk -> document — and this walks it.

        For a distilled terminal session the document also carries the
        ids of the journalled events it was built from, so the trail ends
        at the actual commands you ran.
        """
        memory = self.db.get_memory(memory_id)
        if memory is None:
            return None

        chunk = self.db.get_chunk(memory.chunk_id) if memory.chunk_id else None
        document = (self.db.get_document(memory.document_id)
                    if memory.document_id else None)

        trail: Dict = {
            "memory": memory,
            # The exact passage the model read. Narrower than the whole
            # document and usually the only part worth reading back.
            "chunk": chunk["text"] if chunk else None,
            "document": document,
            # Memory -> derived_from -> Session. Not a graph edge, because
            # a session is not a thing memories are *about* — making it an
            # entity would link every memory from one afternoon to every
            # other and swamp graph expansion, which is the same mistake
            # the reserved User node exists to avoid. It is provenance,
            # and provenance belongs on the trail.
            "sessions": [],
            "events": [],
        }

        if document is None:
            return trail

        trail["sessions"] = document.metadata.get("session_ids") or []

        event_ids = document.metadata.get("event_ids") or []
        if event_ids:
            from .journal import Journal
            wanted = set(event_ids)
            journal = Journal(data_dir=self.persist_path)
            trail["events"] = [
                event for event in journal.events(self.container,
                                                  limit=max(len(wanted) * 20, 500))
                if event["id"] in wanted
            ]
            if not trail["sessions"]:
                trail["sessions"] = sorted(
                    {e["session_id"] for e in trail["events"]})
        return trail

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
        # Two connections to the one file, so closing one is closing half
        # of it — the vector index's stayed open until the process died.
        self.db.close()
        self.vectors.close()
