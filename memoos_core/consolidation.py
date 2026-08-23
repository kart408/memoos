"""
Consolidation: keeping the store honest as it grows.

An append-only memory store degrades. The same fact gets restated ten
ways, old facts stay retrievable long after they stop being true, and
stale trivia crowds out what matters. Three mechanisms push back:

  Deduplication  — near-identical restatements merge into the original
                   and *reinforce* it. Saying something twice is evidence
                   it matters, not a reason to store it twice.

  Supersession   — when a new memory updates or contradicts an old one,
                   the old one is marked superseded rather than deleted.
                   History stays inspectable; retrieval stops seeing it.

  Decay          — every memory has a strength that falls off with time
                   and climbs with use. Nothing is deleted automatically;
                   weak memories simply stop outranking strong ones.

The expensive part is deciding *whether* two memories conflict, which
needs the LLM. Two thresholds keep that cost bounded: near-identical
pairs are merged without asking, and unrelated pairs are skipped without
asking. Only the genuinely ambiguous middle band costs a model call.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional, Sequence, Tuple

from . import config
from .db import Database
from .embedding import cosine, embed_text
from .extraction import judge_conflict
from .graph import MemoryGraph
from .models import ConflictDecision, ConflictJudgement, Memory, MemoryStatus
from .vectors import VectorIndex


# ---------------------------------------------------------------- decay


def half_life_for(memory: Memory) -> float:
    return config.HALF_LIFE_DAYS.get(
        memory.memory_type.value, config.DEFAULT_HALF_LIFE_DAYS
    )


def strength(memory: Memory, now: Optional[datetime] = None) -> float:
    """
    How much this memory still counts, in [0, 1].

    Combines what it was worth when learned (importance x confidence),
    how long since it last mattered (exponential decay on a type-specific
    half-life), and how often it's proved useful (log-scaled, so the
    hundredth recall counts far less than the second).

    Decay is measured from last *access*, not creation — a fact recalled
    yesterday is live regardless of when it was learned.
    """
    now = now or datetime.now(timezone.utc)
    reference = memory.last_accessed_at or memory.created_at
    if reference.tzinfo is None:
        reference = reference.replace(tzinfo=timezone.utc)

    days = max(0.0, (now - reference).total_seconds() / 86400.0)
    recency = 0.5 ** (days / max(half_life_for(memory), 1.0))

    reinforcement = min(
        config.REINFORCEMENT_CAP,
        1.0 + config.REINFORCEMENT_WEIGHT * math.log1p(memory.access_count),
    )

    base = memory.importance * memory.confidence
    return max(0.0, min(1.0, base * recency * reinforcement))


# ------------------------------------------------------------- planning


@dataclass
class ConsolidationPlan:
    """What should happen to a candidate memory, decided before writing."""
    duplicate_of: Optional[Memory] = None
    supersedes: List[Tuple[Memory, ConflictJudgement]] = field(default_factory=list)
    considered: int = 0
    judged: int = 0

    @property
    def is_duplicate(self) -> bool:
        return self.duplicate_of is not None


class Consolidator:
    def __init__(self, db: Database, vectors: VectorIndex, graph: MemoryGraph,
                 container: str = "default"):
        self.db = db
        self.vectors = vectors
        self.graph = graph
        self.container = container

    def _candidates(self, text: str, entity_names: Sequence[str],
                    exclude: Sequence[str]) -> List[Tuple[str, float, bool]]:
        """
        Assemble memories worth comparing against, from two angles.

        Vector similarity alone misses real conflicts. "User switched to
        JAX" and "User prefers PyTorch for deep learning work" describe
        the same attribute and score 0.06 together — they share almost no
        wording, so no similarity threshold low enough to catch them is
        also high enough to be useful.

        The graph closes that gap. Both memories mention PyTorch, and
        two statements about the same named thing are worth a look
        regardless of how differently they're phrased.

        Returns (memory_id, similarity, found_via_entity), best first.
        """
        found: dict[str, Tuple[float, bool]] = {}

        # Passage-to-passage comparison, so the candidate memory is
        # embedded *without* the query prefix. This is a symmetric
        # "are these the same fact?" question, not a search — using the
        # query encoding here would compare two different kinds of text.
        text_vector = embed_text(text)

        for memory_id, similarity in self.vectors.search_vector(
            text_vector, top_k=config.MAX_CONFLICT_CANDIDATES, exclude=exclude
        ):
            found[memory_id] = (similarity, False)

        entity_ids = []
        for name in entity_names:
            entity = self.db.find_entity(self.container, name)
            if entity is not None:
                entity_ids.append(entity.id)

        if entity_ids:
            # Deliberately not excluding ids already found by vector
            # search. A memory can be reachable both ways, and it's the
            # *entity link* that earns it a judgement — so one already in
            # the vector list must still be upgraded to via_entity, not
            # skipped. Missing this silently dropped a real contradiction
            # sitting at 0.448 similarity, just under the 0.45 floor.
            linked = self.db.memories_for_entities(
                self.container, entity_ids,
                exclude=list(exclude),
                limit=config.MAX_CONFLICT_CANDIDATES,
            )
            if linked:
                stored = self.db.get_memories([mid for mid, _ in linked])
                for memory_id, _ in linked:
                    existing = stored.get(memory_id)
                    if existing is None:
                        continue
                    similarity, _ = found.get(memory_id, (None, False))
                    if similarity is None:
                        similarity = cosine(text_vector, embed_text(existing.text))
                    found[memory_id] = (similarity, True)

        ranked = [(mid, sim, via) for mid, (sim, via) in found.items()]
        ranked.sort(key=lambda row: row[1], reverse=True)
        return ranked

    def plan(self, text: str, *, entity_names: Sequence[str] = (),
             exclude: Sequence[str] = (),
             protected: Sequence[str] = ()) -> ConsolidationPlan:
        """
        Work out how a new memory relates to what's already stored.

        `protected` ids may be merged into as duplicates but can never be
        superseded. The pipeline passes the memories extracted from this
        same message: "User prefers PyTorch" and "User does not prefer
        TensorFlow" come from one sentence and describe the same
        attribute, so a conflict judge reads them as an update and
        deletes half of what the user just said. Facts stated in one
        breath complement each other by construction.

        Read-only — nothing is written until `apply` runs, so a failure
        mid-planning can't leave the store half-updated.
        """
        plan = ConsolidationPlan()

        candidates = self._candidates(text, entity_names, exclude)
        if not candidates:
            return plan

        protected_ids = set(protected)
        stored = self.db.get_memories([mid for mid, _, _ in candidates])

        for memory_id, similarity, via_entity in candidates:
            existing = stored.get(memory_id)
            # The vector index can lag the database (a memory superseded
            # since it was indexed), so re-check status against SQLite.
            if existing is None or existing.status != MemoryStatus.ACTIVE:
                continue

            plan.considered += 1

            if similarity >= config.DUPLICATE_THRESHOLD:
                # Near-identical: merge without spending a model call.
                if plan.duplicate_of is None:
                    plan.duplicate_of = existing
                continue

            # Below the duplicate bar and from the same message: keep it.
            if memory_id in protected_ids:
                continue

            # A shared entity is its own reason to look, so those bypass
            # the similarity floor entirely.
            if similarity < config.RELATED_THRESHOLD and not via_entity:
                # Too far apart to plausibly conflict. Skipping these is
                # what keeps writes from costing O(store size) LLM calls.
                continue

            if plan.judged >= config.MAX_JUDGE_CALLS:
                # Candidates arrive most-similar-first, so what's left
                # here is the least likely to conflict. Bounding the
                # spend beats judging a long tail of weak matches.
                break

            judgement = judge_conflict(text, existing.text)
            plan.judged += 1

            if judgement.decision == ConflictDecision.DUPLICATE:
                if plan.duplicate_of is None:
                    plan.duplicate_of = existing
            elif judgement.decision in (ConflictDecision.UPDATE,
                                        ConflictDecision.CONTRADICTION):
                plan.supersedes.append((existing, judgement))

        return plan

    def apply(self, plan: ConsolidationPlan, new_memory: Memory) -> List[Memory]:
        """
        Commit a plan. Assumes `new_memory` has already been stored.

        Only called when the memory was actually written — a duplicate is
        handled by `reinforce` instead, and never reaches this path.

        Returns the superseded memories re-read from the database, so
        callers see the recorded status and reason rather than the stale
        pre-update copies the plan was built from.
        """
        updated: List[Memory] = []

        for existing, judgement in plan.supersedes:
            reason = f"{judgement.decision.value}: {judgement.reason}".strip(": ")
            self.db.mark_superseded(existing.id, new_memory.id, reason)
            # A superseded memory's assertions stop holding, so its graph
            # edges shouldn't keep contributing structure.
            self.graph.invalidate_for_memory(existing.id)
            # Drop it from the vector index too. SQLite keeps the row for
            # history; leaving it indexed would let it keep surfacing.
            self.vectors.delete([existing.id])
            updated.append(self.db.get_memory(existing.id) or existing)

        return updated

    def reinforce(self, memory: Memory) -> Memory:
        """
        Strengthen an existing memory instead of storing a duplicate.

        Restating something is evidence it matters, so importance rises —
        with diminishing returns, so a chatty user can't drive everything
        to 1.0 by repeating themselves.
        """
        boost = (1.0 - memory.importance) * 0.25
        self.db.reinforce(memory.id, importance_delta=boost)
        refreshed = self.db.get_memory(memory.id)
        return refreshed or memory

    # ------------------------------------------------------- forgetting

    def weak_memories(self, threshold: Optional[float] = None,
                      limit: int = 1000) -> List[Tuple[Memory, float]]:
        """Active memories whose strength has fallen below the threshold."""
        cutoff = threshold if threshold is not None else config.FORGET_THRESHOLD
        now = datetime.now(timezone.utc)
        out = []
        for memory in self.db.list_memories(self.container, limit=limit):
            score = strength(memory, now)
            if score < cutoff:
                out.append((memory, score))
        return out

    def forget_weak(self, threshold: Optional[float] = None,
                    limit: int = 1000) -> List[Memory]:
        """
        Retire decayed memories.

        Marks them FORGOTTEN and unindexes them — the SQLite row survives,
        so this is reversible and auditable. Never runs on its own; a
        memory system that silently deletes things is not trustworthy.
        """
        forgotten = []
        for memory, _ in self.weak_memories(threshold, limit):
            self.db.set_status(memory.id, MemoryStatus.FORGOTTEN)
            self.vectors.delete([memory.id])
            forgotten.append(memory)
        return forgotten
