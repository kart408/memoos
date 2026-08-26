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
from .models import MemoryQueryResult, MemoryStatus, MemoryType
from .text_utils import significant_set
from .query import QueryPlan, understand
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


def in_scope(results: Sequence[MemoryQueryResult],
             query: str = "",
             corroborated: Optional[float] = None,
             vector_only: Optional[float] = None) -> List[MemoryQueryResult]:
    """
    Drop results the store is not actually confident about.

    Search always returns its best candidates, however poor they are —
    that is what nearest-neighbour means, and for `search()` it is the
    right contract, because the caller can see the scores and decide.
    A context block cannot: it goes into somebody else's prompt stripped
    of every number, and an agent handed three memories about MemoOS in
    reply to "who won the world cup" will try to use them. Somewhere
    between retrieval and the handover there has to be a state that means
    "I don't know about that", and this is it.

    The bars are `config.RECALL_MIN_SIMILARITY_*`, which were calibrated
    against measured pairs for exactly this question and have been
    orphaned since the chat layer that used them was deleted. Nothing is
    retuned here; this only wires them back up.

    Two of them, because a second retriever agreeing is evidence and buys
    a lower bar. Compared against raw cosine, never against `score` —
    RRF encodes *rank*, not similarity, and tops out near 1/(RRF_K + 1),
    about 0.016. Comparing that to 0.48 is a category error that fails
    silently: the gate simply never opens.
    """
    corroborated_bar = (corroborated if corroborated is not None
                        else config.RECALL_MIN_SIMILARITY_CORROBORATED)
    vector_bar = (vector_only if vector_only is not None
                  else config.RECALL_MIN_SIMILARITY_VECTOR_ONLY)

    kept: List[MemoryQueryResult] = []
    for result in results:
        similarity = result.vector_score
        if similarity is None:
            # No vector opinion at all: this arrived on a literal token
            # or a shared entity, which is concrete evidence rather than
            # a distance. There is no cosine to judge, and judging it by
            # one it does not have would throw away the half of hybrid
            # search that exists for rare names and IDs.
            kept.append(result)
            continue
        if similarity >= (_bar_for(result, query, corroborated_bar, vector_bar)):
            kept.append(result)
    return kept


def _bar_for(result: MemoryQueryResult, query: str,
             corroborated_bar: float, vector_bar: float) -> float:
    """
    Which bar this result has to clear, and why it might get the easier one.

    A second retriever agreeing is evidence, and evidence buys a lower
    bar. But it only counts when the agreement is *independent*, and
    query expansion can manufacture agreement that is not.

    Asked "who is my sister", expansion probed the store, harvested
    `main` and `master` off the nearest memories, and searched for
    "who is my sister main master". The keyword retriever duly matched
    "The master branch was renamed to main" — on two words the user never
    typed and the vector search had just invented. That counted as
    corroboration, halved the bar from 0.48 to 0.40, and let a memory
    through at cosine 0.428. The retriever was not agreeing; it was
    echoing.

    So corroboration has to rest on the user's own words: the result must
    share a content word with the query as asked. With no query to check
    against, agreement is taken at face value, which is the old behaviour.
    """
    if not any(source != "vector" for source in result.matched_by):
        return vector_bar
    asked = significant_set(query)
    if not asked:
        return corroborated_bar
    if asked & significant_set(result.memory.text):
        return corroborated_bar
    return vector_bar


class Retriever:
    def __init__(self, db: Database, vectors: VectorIndex, graph: MemoryGraph,
                 container: str = "default"):
        self.db = db
        self.vectors = vectors
        self.graph = graph
        self.container = container

    def plan_for(self, query: str, *, expand: Optional[bool] = None) -> QueryPlan:
        """Stage 9 on its own, for callers that want to show their working."""
        return understand(query, db=self.db, vectors=self.vectors,
                          container=self.container, enabled=expand)

    def search(self, query: str, top_k: int = 5, *, use_graph: bool = True,
               memory_type: Optional[MemoryType] = None,
               touch: bool = True,
               min_score: Optional[float] = None,
               include_stale: bool = False,
               expand: Optional[bool] = None,
               plan: Optional[QueryPlan] = None) -> List[MemoryQueryResult]:
        """
        Hybrid search over this container's active memories.

        `touch` records the retrieval as an access, which is what feeds
        reinforcement. Turn it off for introspection or evaluation, so
        looking at the store doesn't change it.

        `include_stale` keeps memories whose validity window has closed
        but which are still active. It does *not* resurrect superseded
        ones: supersession un-indexes them, deliberately, so that a fact
        which stopped being true stops competing for candidate slots.
        Their history is reachable through `MemoOS.history()`, which
        walks the supersession chain in SQLite where the rows still live.

        `plan` accepts an already-computed expansion, so a caller that
        showed the user what it was about to search for does not pay for
        the probe twice.
        """
        cleaned = query.strip()
        if not cleaned:
            return []

        if plan is None:
            plan = self.plan_for(cleaned, expand=expand)

        # The question goes to the vector retriever exactly as asked;
        # only the lexical half gets the expansion. Embedding the padded
        # text would move the query vector off what was actually asked.
        vector_hits = self.vectors.search(cleaned, top_k=config.VECTOR_CANDIDATES)
        keyword_hits = self.db.keyword_search(
            self.container, plan.keyword_text(), limit=config.KEYWORD_CANDIDATES
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
            min_score=min_score, include_stale=include_stale,
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
              touch: bool, min_score: Optional[float],
              include_stale: bool = False) -> List[MemoryQueryResult]:
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
            # Current validity, the last of the reranking signals. A
            # memory whose window has closed is history, not an answer:
            # "User's project uses MongoDB" is still true *of last year*,
            # and returning it for "what database do I use?" is how a
            # memory system confidently tells you something outdated.
            if not include_stale and not memory.is_current(now):
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

            # Strength nudges rather than decides. Fusion has already
            # judged relevance; decay and reinforcement only break ties
            # between results fusion rates as comparable. Weight it much
            # higher and it stops being a tiebreaker — RRF's adjacent
            # ranks sit ~1.6% apart, so a wide multiplier simply outvotes
            # the relevance signal it was meant to refine.
            weight = config.STRENGTH_WEIGHT
            final = fusion_score * ((1.0 - weight) + weight * memory_strength)

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
            # Only the best hits earn reinforcement. Crediting every
            # returned candidate would reward memories for merely being
            # the least-bad thing available, and that credit compounds
            # into future rankings — see config.REINFORCE_TOP_N.
            self.db.touch([r.memory.id for r in results[:config.REINFORCE_TOP_N]])

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
