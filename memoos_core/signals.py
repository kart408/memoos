"""
Past signals: what went wrong before, and what fixed it.

A terminal session is mostly a record of things breaking and then being
made to work. That arc is the most valuable thing in the store and the
easiest to lose — filed as loose facts, "the tests failed on the import
path" and "moving the fixture fixed it" sit in separate rows with nothing
saying they are the same story.

This pairs them back up. A `problem` and a `solution` that talk about the
same named things are two halves of one episode, and an agent about to
touch that code wants both: here is what broke, here is what fixed it.
A problem with no matching solution is the more urgent signal — it is
still open, and nobody has written down an answer.

Two rules govern this module, and the second one is a constraint rather
than an implementation detail.

  Pairing is by evidence, not by guess. Two memories are the same
  episode when they share a named thing and came out of the same session.
  Entities are what the graph already knows; the session is what the
  document already records. Nothing here re-reads the text or asks a
  model — a pairing that needed an LLM call per query would not survive
  contact with the hot path.

  Signals are surfaced, never acted on. Nothing in this module changes a
  ranking, retires a memory, or adjusts a threshold. It reports what the
  store already knows and hands it over. MemoOS does not decide what to
  do about a known failure — the agent holding the task does, because it
  is the only thing that can see what the task actually is.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Sequence

from .models import Memory, MemoryType
from .text_utils import significant_set

if TYPE_CHECKING:
    from .db import Database

PROBLEM_TYPES = {MemoryType.PROBLEM}
SOLUTION_TYPES = {MemoryType.SOLUTION}

# How many candidate fixes to report against one problem. Past this it
# stops reading as "here is what fixed it" and starts reading as a list
# to be triaged, which is work rather than a signal.
MAX_SOLUTIONS_PER_PROBLEM = 3


@dataclass
class Episode:
    """One thing that went wrong, and whatever is known about fixing it."""

    problem: Memory
    solutions: List[Memory] = field(default_factory=list)
    shared: List[str] = field(default_factory=list)   # entity names in common

    @property
    def resolved(self) -> bool:
        return bool(self.solutions)

    def as_dict(self) -> Dict:
        return {
            "problem": self.problem.text,
            "problem_id": self.problem.id,
            "resolved": self.resolved,
            "solutions": [s.text for s in self.solutions],
            "shared_entities": self.shared,
            "last_seen": (self.problem.created_at.isoformat()
                          if self.problem.created_at else None),
        }


def episodes_for(db: "Database", container: str,
                 memory_ids: Sequence[str] = (),
                 limit: int = 25) -> List[Episode]:
    """
    The problem/solution episodes touching a set of memories.

    Pass the ids a search just returned and this reports the failures
    those memories are entangled with — the past signal for whatever the
    agent is about to do. Pass nothing and it reports the container's
    episodes at large, which is what a fresh terminal wants.

    Unresolved episodes come first. An open problem is the thing most
    worth knowing and the thing least likely to be written down anywhere
    else; a solved one is reference material.
    """
    problems = [m for m in _by_type(db, container, MemoryType.PROBLEM, limit)]
    if not problems:
        return []

    solutions = _by_type(db, container, MemoryType.SOLUTION, limit * 2)

    # Entity ids per memory, fetched once for everything in play.
    everything = [m.id for m in problems] + [m.id for m in solutions]
    entity_map = db.entity_ids_for_memories(everything)
    all_entities = {eid for ids in entity_map.values() for eid in ids}
    names = {eid: e.name for eid, e in db.get_entities(list(all_entities)).items()}

    focus = set(memory_ids)
    focus_entities: set = set()
    for memory_id in focus:
        focus_entities.update(entity_map.get(memory_id, ()))

    out: List[Episode] = []
    for problem in problems:
        problem_entities = set(entity_map.get(problem.id, ()))

        # When the caller named a focus, only report episodes that touch
        # it. Reporting every past failure on every query would be the
        # dumping ground again, one level up.
        if focus and not (problem_entities & focus_entities) \
                and problem.id not in focus:
            continue

        matched: List[Memory] = []
        shared: set = set()
        for solution in solutions:
            overlap = problem_entities & set(entity_map.get(solution.id, ()))
            same_session = (problem.document_id is not None
                            and problem.document_id == solution.document_id)
            # Three signals, in descending order of how much they prove.
            #
            # A shared *name* is the strongest, but requiring it misses
            # the ordinary case: the test that broke and the file that
            # fixed it are usually different files. "The tests in
            # test_auth.py were failing" and "editing api.py made them
            # pass" share no entity at all, and that pair is the whole
            # point of this module.
            #
            # Same *session* is necessary but nowhere near sufficient. On
            # its own it pairs everything with everything: an afternoon
            # that broke the deploy and separately fixed the auth tests
            # would report the auth fix as the answer to the deploy
            # timeout, which is worse than reporting nothing.
            #
            # So proximity has to be corroborated by shared *wording*. A
            # solution describes undoing its problem, so the two talk
            # about the same things even when they name different files —
            # "tests" and "authentication" here. Cheap, deterministic,
            # and it is the difference between an episode and a coincidence.
            corroborated = same_session and bool(
                _significant(problem.text) & _significant(solution.text))
            if overlap or corroborated:
                matched.append(solution)
                shared.update(overlap)
        # A wall of candidate fixes is not a signal. Ones sharing a name
        # with the problem come first — that is the stronger evidence.
        matched.sort(key=lambda s: not (problem_entities
                                        & set(entity_map.get(s.id, ()))))
        matched = matched[:MAX_SOLUTIONS_PER_PROBLEM]

        out.append(Episode(
            problem=problem,
            solutions=matched,
            shared=sorted(names[eid] for eid in shared if eid in names),
        ))

    # Unresolved first, then most recent. Both halves of that ordering
    # answer "what should I be told before I start?".
    out.sort(key=lambda e: (e.resolved,
                            -(e.problem.created_at.timestamp()
                              if e.problem.created_at else 0)))
    return out[:limit]


def _significant(text: str) -> set:
    """Content words, for judging whether two memories discuss one thing."""
    return significant_set(text, min_length=4)


def _by_type(db: "Database", container: str, kind: MemoryType,
             limit: int) -> List[Memory]:
    return [m for m in db.list_memories(container, memory_type=kind,
                                        limit=limit)
            if m.is_current()]


def render(episodes: Sequence[Episode], *, unresolved_only: bool = False) -> str:
    """
    Episodes as lines for the context block.

    Kept plain and subject-first, like the rest of the block: this goes
    into a prompt that another model reads, and the useful shape is a
    statement followed by what is known about it.
    """
    lines: List[str] = []
    open_ones = [e for e in episodes if not e.resolved]
    solved = [] if unresolved_only else [e for e in episodes if e.resolved]

    if open_ones:
        lines.append("Open problems (no fix recorded):")
        for episode in open_ones:
            lines.append(f"• {episode.problem.text}")

    if solved:
        if lines:
            lines.append("")
        lines.append("Known failures, and what fixed them:")
        for episode in solved:
            lines.append(f"• {episode.problem.text}")
            for solution in episode.solutions:
                lines.append(f"  ↳ {solution.text}")

    return "\n".join(lines)
