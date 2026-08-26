"""
The memory graph: entities and the edges between them.

Vector search finds memories that *sound* like the query. The graph
finds memories that are *about the same things* — which is a different
and often better signal. Ask "what do you know about my job?" and
embeddings will surface the memory containing the word "job"; the graph
will also surface "User is doing an internship at Zomato" and "User's
manager is Anjali", because all three hang off the same entities.

Two deliberate choices here:

  - The user themself is a *reserved* entity. Relations like
    User -> interns_at -> Zomato are worth storing, but if the user were
    an ordinary node, every memory would link to it and graph expansion
    would return the entire store on every query. So the reserved node
    can appear in relations but is never linked to memories.

  - Superseding a memory invalidates the edges it asserted. A fact that
    is no longer true shouldn't keep contributing structure.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from . import config
from .db import Database, normalise_entity_name
from .models import (
    Entity,
    EntityType,
    ExtractedMemory,
    Memory,
    Relation,
)

# Names that mean "the person this container belongs to".
USER_ALIASES = {"user", "the user", "me", "i", "myself", "my"}
USER_NORM_NAME = "__user__"


class MemoryGraph:
    """Entity/relation operations for one container."""

    def __init__(self, db: Database, container: str = "default"):
        self.db = db
        self.container = container

    # ---------------------------------------------------------- writing

    def _user_entity(self) -> Optional[Entity]:
        """
        The reserved node representing the container's owner.

        Written directly rather than through upsert_entity so its
        norm_name stays the sentinel `__user__` and can never collide
        with a real person who happens to be called "User".
        """
        with self.db.transaction() as conn:
            row = conn.execute(
                """SELECT * FROM entities
                   WHERE container = ? AND norm_name = ?""",
                (self.container, USER_NORM_NAME),
            ).fetchone()
            if row:
                return Database._row_to_entity(row)

        entity = Entity(
            container=self.container, name="User",
            norm_name=USER_NORM_NAME, entity_type=EntityType.PERSON,
        )
        with self.db.transaction() as conn:
            conn.execute(
                """INSERT INTO entities
                   (id, container, name, norm_name, entity_type, mention_count, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (entity.id, entity.container, entity.name, entity.norm_name,
                 entity.entity_type.value, 0, entity.created_at.isoformat()),
            )
        return entity

    def attach(self, memory: Memory,
               extracted: ExtractedMemory) -> Tuple[List[Entity], List[Relation]]:
        """
        Persist a memory's entities and relations, and link the entities
        to the memory so graph expansion can find it later.
        """
        entities: List[Entity] = []
        by_norm: Dict[str, Entity] = {}

        for candidate in extracted.entities:
            entity = self.db.upsert_entity(
                self.container, candidate.name, candidate.entity_type
            )
            if entity is None:
                continue
            entities.append(entity)
            by_norm[entity.norm_name] = entity

        # Relations are attached *before* the memory is linked, because
        # resolving a triple can bring a node into existence — the model
        # names something in a relation that it never listed as an entity.
        relations, resolved = self._attach_relations(memory, extracted, by_norm)

        # Those nodes are mentioned by this memory just as much as the
        # listed ones are. Linking happened first and only covered the
        # listed ones, so a relation-only entity got a mention count of 1
        # and no mentions to back it up — inflating the count, and leaving
        # a node graph expansion could never reach, since expansion
        # travels through `memory_entities`.
        known = {e.id for e in entities}
        for entity in resolved:
            # The reserved user node can appear in relations but is never
            # linked to a memory: linking it would attach it to every
            # memory in the store and collapse expansion into a star.
            if entity.norm_name == USER_NORM_NAME or entity.id in known:
                continue
            known.add(entity.id)
            entities.append(entity)

        if entities:
            self.db.link_memory_entities(memory.id, [e.id for e in entities])

        return entities, relations

    def _attach_relations(self, memory: Memory, extracted: ExtractedMemory,
                          by_norm: Dict[str, Entity]
                          ) -> Tuple[List[Relation], List[Entity]]:
        """Persist a memory's relations, and report every node they touched."""
        relations: List[Relation] = []
        touched: Dict[str, Entity] = {}

        for candidate in extracted.relations:
            subject = self._resolve(candidate.subject, by_norm)
            obj = self._resolve(candidate.object, by_norm)
            if subject is None or obj is None or subject.id == obj.id:
                continue
            touched[subject.id] = subject
            touched[obj.id] = obj
            relations.append(self.db.insert_relation(Relation(
                container=self.container,
                subject_id=subject.id,
                predicate=candidate.predicate,
                object_id=obj.id,
                memory_id=memory.id,
                confidence=extracted.confidence,
            )))

        return relations, list(touched.values())

    def _resolve(self, name: str, by_norm: Dict[str, Entity]) -> Optional[Entity]:
        """
        Map a name from a relation triple onto a stored entity.

        Prefers entities extracted alongside this same memory, so a
        relation always binds to the node the model meant rather than a
        same-named one from an unrelated memory.
        """
        cleaned = name.strip()
        if not cleaned:
            return None
        if cleaned.lower() in USER_ALIASES:
            return self._user_entity()

        norm = normalise_entity_name(cleaned)
        if not norm:
            return None
        if norm in by_norm:
            return by_norm[norm]

        existing = self.db.find_entity(self.container, cleaned)
        if existing:
            return existing

        # The model named something in a relation that it didn't list as
        # an entity. It's still a real node — record it.
        return self.db.upsert_entity(self.container, cleaned, EntityType.OTHER)

    def invalidate_for_memory(self, memory_id: str) -> None:
        """Drop the edges a memory asserted, once it stops being true."""
        self.db.invalidate_relations_for_memory(memory_id)

    # ---------------------------------------------------------- reading

    def expand(self, seed_memory_ids: Sequence[str], *, exclude: Sequence[str] = (),
               limit: Optional[int] = None) -> Dict[str, List[str]]:
        """
        One hop out from the seed memories.

        Returns {memory_id: [entity_id, ...]} for memories that share an
        entity with the seeds and aren't already in the result set.
        """
        if not seed_memory_ids:
            return {}

        entity_map = self.db.entity_ids_for_memories(list(seed_memory_ids))
        entity_ids = list({eid for ids in entity_map.values() for eid in ids})
        if not entity_ids:
            return {}

        hits = self.db.memories_for_entities(
            self.container,
            entity_ids,
            exclude=list(exclude),
            limit=limit if limit is not None else config.GRAPH_EXPANSION_LIMIT,
        )

        out: Dict[str, List[str]] = {}
        for memory_id, entity_id in hits:
            out.setdefault(memory_id, []).append(entity_id)
        return out

    def entities_for(self, memory_ids: Sequence[str]) -> Dict[str, List[Entity]]:
        """Hydrated entities per memory, for explaining a result."""
        id_map = self.db.entity_ids_for_memories(list(memory_ids))
        all_ids = list({eid for ids in id_map.values() for eid in ids})
        lookup = self.db.get_entities(all_ids)
        return {
            memory_id: [lookup[eid] for eid in ids if eid in lookup]
            for memory_id, ids in id_map.items()
        }

    def neighbourhood(self, entity_name: str, limit: int = 50) -> Dict:
        """
        Everything the graph knows about one named thing: the entity, the
        memories mentioning it, and its relation edges.
        """
        entity = self.db.find_entity(self.container, entity_name)
        if entity is None:
            return {"entity": None, "memories": [], "relations": []}

        memory_ids = [mid for mid, _ in self.db.memories_for_entities(
            self.container, [entity.id], limit=limit
        )]
        memories = self.db.get_memories(memory_ids)
        relations = self.db.list_relations(self.container, entity_id=entity.id)
        entity_lookup = self.db.get_entities(
            list({r.subject_id for r in relations} | {r.object_id for r in relations})
        )

        return {
            "entity": entity,
            "memories": [memories[mid] for mid in memory_ids if mid in memories],
            "relations": [
                {
                    "subject": entity_lookup[r.subject_id].name if r.subject_id in entity_lookup else "?",
                    "predicate": r.predicate,
                    "object": entity_lookup[r.object_id].name if r.object_id in entity_lookup else "?",
                }
                for r in relations
            ],
        }

    def summary(self, limit: int = 20) -> List[Entity]:
        """Most-mentioned entities — a quick view of what this store is about."""
        return [
            e for e in self.db.list_entities(self.container, limit=limit + 1)
            if e.norm_name != USER_NORM_NAME
        ][:limit]
