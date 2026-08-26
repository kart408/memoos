"""
Query understanding: turning a question into the terms worth searching for.

A question and the memory that answers it routinely share no vocabulary
at all:

    "how should I deploy my application?"
    "User deploys the project on Vercel."

Vector search bridges that, because the two mean the same thing. BM25
cannot, because they have no token in common — so on exactly the queries
that need both retrievers, one of them abstains and hybrid search
degenerates into vector search with extra steps.

So the query is expanded before either retriever sees it. The expansion
is *grounded*: a cheap vector probe finds the handful of memories that
are about the question, and the entity names hanging off those memories
become the extra search terms. `deploy` finds the Vercel memory, and
`Vercel` becomes a term the keyword half can actually match on.

Grounding is what makes this safe. A thesaurus would invent terms the
container has never heard of, and every invented term is a chance for
BM25 to match something irrelevant with great confidence. Reading the
concepts back out of the store means expansion can only ever add words
this project already knows.

The original query still goes to the vector retriever unchanged.
Stuffing concepts into the text that gets embedded moves the query
vector away from what was actually asked — the expansion is a hint for
the lexical side, not a rewrite of the question.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Sequence

from . import config
from .text_utils import significant_tokens

if TYPE_CHECKING:  # avoid importing the heavy half at module load
    from .db import Database
    from .vectors import VectorIndex


@dataclass
class QueryPlan:
    """
    What to search for, and why — the record of stage 9's reasoning.

    Kept as a value rather than applied in place so a caller can show it.
    "Why did this surface?" is answerable at every other stage of
    retrieval; expansion should not be the one step that happens
    invisibly.
    """

    query: str
    terms: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)

    @property
    def expanded(self) -> bool:
        return bool(self.concepts)

    def keyword_text(self) -> str:
        """
        The text handed to the keyword retriever.

        The original query is kept in full rather than reduced to its
        significant tokens: `build_fts_query` drops the stopwords itself,
        and doing it twice would mean two places to keep in step.
        """
        if not self.concepts:
            return self.query
        return f"{self.query} {' '.join(self.concepts)}"

    def as_dict(self) -> dict:
        return {"query": self.query, "terms": self.terms,
                "concepts": self.concepts}


def understand(query: str, *, db: "Database", vectors: "VectorIndex",
               container: str, enabled: Optional[bool] = None) -> QueryPlan:
    """
    Work out what a question is really asking for.

    Costs one embedding and one entity lookup — the same embedding the
    vector retriever is about to compute anyway, so in practice this is
    a lookup and a dictionary. Nothing here calls a language model.
    """
    cleaned = query.strip()
    plan = QueryPlan(query=cleaned, terms=significant_tokens(cleaned))
    if not cleaned:
        return plan

    if enabled is None:
        enabled = config.QUERY_EXPANSION
    if not enabled:
        return plan

    plan.concepts = _grounded_concepts(cleaned, db=db, vectors=vectors,
                                       container=container,
                                       already=set(plan.terms))
    return plan


def _grounded_concepts(query: str, *, db: "Database", vectors: "VectorIndex",
                       container: str, already: set) -> List[str]:
    """
    Entity names belonging to the memories this question is about.

    Ranked by how many probe hits mention them, so a name that turns up
    across several relevant memories outranks one that appears in the
    weakest single hit. Ties break toward the better-ranked memory, which
    `dict` insertion order gives for free.
    """
    probe = vectors.search(query, top_k=config.QUERY_PROBE_CANDIDATES)
    # Vector search always hands back its nearest neighbours, however far
    # away they are. Unfiltered, a small store returns everything and
    # expansion harvests the whole container — which hands BM25 a term
    # from every memory and leaves it voting for all of them.
    memory_ids = [memory_id for memory_id, score in probe
                  if score >= config.QUERY_PROBE_MIN_SIMILARITY]
    if not memory_ids:
        return []
    entity_ids = db.entity_ids_for_memories(memory_ids)
    if not entity_ids:
        return []

    # Count in probe order, so the dict is already ranked before sorting.
    tally: dict = {}
    for memory_id in memory_ids:
        for entity_id in entity_ids.get(memory_id, ()):
            tally[entity_id] = tally.get(entity_id, 0) + 1

    best_hit = memory_ids[0]
    from_best = set(entity_ids.get(best_hit, ()))

    entities = db.get_entities(list(tally))
    scored = []
    for entity_id, hits in tally.items():
        entity = entities.get(entity_id)
        if entity is None:
            continue
        # One mention in one middling memory is coincidence. Either the
        # concept recurs across hits, or it belongs to the memory the
        # question matched best.
        if hits < config.QUERY_MIN_CONCEPT_HITS and entity_id not in from_best:
            continue
        scored.append((hits, entity.name))

    scored.sort(key=lambda row: row[0], reverse=True)

    concepts: List[str] = []
    seen = set(already)
    for _, name in scored:
        key = name.lower()
        # A concept already present in the question adds nothing — the
        # keyword retriever was going to search for it regardless.
        if key in seen or key in {t.lower() for t in already}:
            continue
        seen.add(key)
        concepts.append(name)
        if len(concepts) >= config.QUERY_MAX_CONCEPTS:
            break
    return concepts


def as_context_terms(plans: Sequence[QueryPlan]) -> List[str]:
    """Every distinct term and concept across a set of plans, in order."""
    out: List[str] = []
    seen = set()
    for plan in plans:
        for word in [*plan.terms, *plan.concepts]:
            key = word.lower()
            if key not in seen:
                seen.add(key)
                out.append(word)
    return out
