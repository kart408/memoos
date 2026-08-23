"""
Retrieval: finding the right memories, from three angles at once.

Each retriever fails differently, and the failures don't overlap:

  Vector search  understands meaning but is blind to rare literal tokens.
                 "Zomato" and "Swiggy" sit close together in embedding
                 space; a query for one happily returns the other.

  BM25 keyword   nails exact names, IDs and jargon, and is helpless when
                 the query and the answer share no vocabulary.

  Graph hops     find memories connected to what already matched, even
                 when they match the query on neither wording nor meaning.

Their outputs are fused with Reciprocal Rank Fusion, which combines by
*rank* rather than score. That matters because the scores aren't
comparable — cosine similarity lives in [0,1] while BM25 is unbounded and
corpus-dependent — so any weighted sum of the raw numbers would be
dominated by whichever scale happened to be larger.

Finally each fused score is modulated by memory strength, so a fact the
user relies on weekly outranks an equally-relevant one they mentioned
once a year ago.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .consolidation import strength
from .db import Database
from .graph import MemoryGraph
from .models import Memory, MemoryQueryResult, MemoryStatus, MemoryType
from .vectors import VectorIndex


def reciprocal_rank_fusion(ranked_lists: Sequence[Sequence[str]],
                           k: Optional[int] = None) -> Dict[str, float]:
    """
    Fuse ranked id lists into one score map.

    Each list contributes 1/(k + rank) for every id it ranks. `k` damps
    the top of each list so a single retriever can't unilaterally decide
    the final order — the standard value of 60 comes from the original
    RRF paper.
    """
    constant = k if k is not None else config.RRF_K
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (constant + rank)
    return scores


class Retriever:
    def __init__(self, db: Database, vectors: VectorIndex, graph: MemoryGraph,
                 container: str = "default"):
        self.db = db
        self.vectors = vectors
        self.graph = graph
        self.container = container

    def search(self, query: str, top_k: int = 5, *, use_graph: bool = True,
               memory_type: Optional[MemoryType] = None,
               touch: bool = True,
               min_score: Optional[float] = None) -> List[MemoryQueryResult]:
        """
        Hybrid search over this container's active memories.

        `touch` records the retrieval as an access, which is what feeds
        reinforcement. Turn it off for introspection or evaluation, so
        looking at the store doesn't change it.
        """
        cleaned = query.strip()
        if not cleaned:
            return []

        vector_hits = self.vectors.search(cleaned, top_k=config.VECTOR_CANDIDATES)
        keyword_hits = self.db.keyword_search(
            self.container, cleaned, limit=config.KEYWORD_CANDIDATES
        )

        vector_scores = dict(vector_hits)
        keyword_scores = dict(keyword_hits)

        fused = reciprocal_rank_fusion([
            [mid for mid, _ in vector_hits],
            [mid for mid, _ in keyword_hits],
        ])

        graph_entities: Dict[str, List[str]] = {}
        if use_graph and fused:
            for memory_id, (bonus, names) in self._expand(fused, top_k).items():
                fused[memory_id] = fused.get(memory_id, 0.0) + bonus
                graph_entities[memory_id] = names

        if not fused:
            return []

        return self._rank(
            fused, vector_scores, keyword_scores, graph_entities,
            top_k=top_k, memory_type=memory_type, touch=touch,
            min_score=min_score,
        )

    def _expand(self, fused: Dict[str, float],
                top_k: int) -> Dict[str, Tuple[float, List[str]]]:
        """
        One graph hop out from the strongest current matches.

        An expanded memory inherits a damped share of the score of the
        best seed it shares an entity with — so neighbours of a strong
        match rank above neighbours of a marginal one, and neither can
        outrank a direct hit.
        """
        seeds = sorted(fused, key=fused.get, reverse=True)[:config.GRAPH_EXPANSION_SEEDS]
        if not seeds:
            return {}

        seed_entities = self.db.entity_ids_for_memories(seeds)

        # Best fused score per entity, across the seeds that mention it.
        entity_weight: Dict[str, float] = {}
        for seed_id, entity_ids in seed_entities.items():
            for entity_id in entity_ids:
                entity_weight[entity_id] = max(
                    entity_weight.get(entity_id, 0.0), fused[seed_id]
                )
        if not entity_weight:
            return {}

        expanded = self.graph.expand(
            seeds,
            exclude=list(fused.keys()),
            limit=max(config.GRAPH_EXPANSION_LIMIT, top_k),
        )
        if not expanded:
            return {}

        entity_names = self.db.get_entities(list(entity_weight.keys()))

        out: Dict[str, Tuple[float, List[str]]] = {}
        for memory_id, entity_ids in expanded.items():
            relevant = [e for e in entity_ids if e in entity_weight]
            if not relevant:
                continue
            best = max(entity_weight[e] for e in relevant)
            names = [entity_names[e].name for e in relevant if e in entity_names]
            out[memory_id] = (config.GRAPH_DAMPING * best, names)
        return out

    def _rank(self, fused: Dict[str, float], vector_scores: Dict[str, float],
              keyword_scores: Dict[str, float], graph_entities: Dict[str, List[str]],
              *, top_k: int, memory_type: Optional[MemoryType],
              touch: bool, min_score: Optional[float]) -> List[MemoryQueryResult]:
        stored = self.db.get_memories(list(fused.keys()))
        now = datetime.now(timezone.utc)

        # BM25 is unbounded and corpus-relative, so the raw number means
        # nothing on its own. Normalising against the best hit in this
        # result set makes it a readable 0-1 signal for explainability.
        # It plays no part in ordering — that's RRF's job.
        best_keyword = max(keyword_scores.values(), default=0.0) or 1.0

        results: List[MemoryQueryResult] = []
        for memory_id, fusion_score in fused.items():
            memory = stored.get(memory_id)
            if memory is None or memory.status != MemoryStatus.ACTIVE:
                continue
            if memory_type is not None and memory.memory_type != memory_type:
                continue

            matched_keyword = memory_id in keyword_scores
            matched_graph = memory_id in graph_entities
            vector_score = vector_scores.get(memory_id)

            # Drop distant vector-only neighbours. A keyword hit or a
            # shared entity is concrete evidence of relevance and earns a
            # place regardless of embedding distance; a weak cosine on its
            # own does not.
            if (not matched_keyword and not matched_graph
                    and (vector_score is None
                         or vector_score < config.MIN_VECTOR_SIMILARITY)):
                continue

            memory_strength = strength(memory, now)

            # Strength modulates rather than multiplies outright: even a
            # fully decayed memory keeps half its fused score, so decay
            # reorders results but never hides a direct, obvious match.
            final = fusion_score * (0.5 + 0.5 * memory_strength)

            matched_by = []
            if vector_score is not None:
                matched_by.append("vector")
            if matched_keyword:
                matched_by.append("keyword")
            if matched_graph:
                matched_by.append("graph")

            results.append(MemoryQueryResult(
                memory=memory,
                score=final,
                vector_score=vector_score,
                keyword_score=(keyword_scores[memory_id] / best_keyword
                               if memory_id in keyword_scores else None),
                strength=memory_strength,
                matched_by=matched_by,
                via_entities=graph_entities.get(memory_id, []),
            ))

        results.sort(key=lambda r: r.score, reverse=True)

        floor = min_score if min_score is not None else config.MIN_RESULT_SCORE
        if floor > 0:
            results = [r for r in results if r.score >= floor]

        results = results[:top_k]

        if touch and results:
            self.db.touch([r.memory.id for r in results])

        return results

    def similar_to(self, memory_id: str, top_k: int = 5) -> List[MemoryQueryResult]:
        """Memories closest to a given one — useful for reviewing near-duplicates."""
        memory = self.db.get_memory(memory_id)
        if memory is None:
            return []
        # Searching by a memory's own text ranks that memory first, every
        # time. Over-fetch by one and drop it, so top_k means k *other*
        # memories rather than k-1 plus the one you already had.
        results = self.search(memory.text, top_k=top_k + 1, touch=False)
        return [r for r in results if r.memory.id != memory_id][:top_k]
