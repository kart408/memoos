"""
Extraction: turn raw text into structured, standalone memories.

This is the quality bottleneck of the entire system. A memory is only
ever as good as the sentence extracted here — retrieval can reorder and
filter, but it can never recover information that extraction mangled.
So three things happen on every extraction:

  1. The model is asked for *atomic, third-person, self-contained*
     statements. "I moved there last year" is useless in six months;
     "User moved to Delhi in 2025" is still true and still findable.

  2. Output is validated field by field. A local model will happily
     invent an enum value or return importance as the string "high".

  3. Every candidate is checked back against its source text. Small
     models confabulate, and a fabricated memory is worse than a missing
     one — it will be retrieved confidently and forever.
"""

import re
from typing import List, Optional

from . import config, text_utils
from .llm import generate_json
from .models import (
    ConflictDecision,
    ConflictJudgement,
    EntityType,
    ExtractedEntity,
    ExtractedMemory,
    ExtractedRelation,
    MemoryType,
)

EXTRACTION_SYSTEM = (
    "You extract durable memories from text for a long-term memory system. "
    "You output only JSON. You never invent information."
)

EXTRACTION_PROMPT = """Extract every durable fact worth remembering from the message below.

Rules:
- Write each memory as a standalone third-person sentence whose subject is named: "User" for something about the person, "The project" for something about the codebase. It must still make sense years from now with no other context. Never write "I" or "me". Never write "there", "then" or "it" - name the thing.
- One idea per memory. Split compound statements into separate memories.
- Use ONLY information stated in the message. Never infer, expand, or add.
- Ignore questions, greetings, and small talk. If nothing is worth remembering, return {{"memories": []}}.
- Record what is TRUE, not what was ASKED FOR. A line starting "The user asked for:" is the request that started the work - it is context for reading what follows, never a memory of its own. "User wanted to commit the changes" is worthless a day later; "The project uses JWT for authentication" is not.
- Never write a memory of the form "User wanted to X" or "User decided to X" for a one-off action (commit, push, delete, rename, install, run, fix, clean up). Write what became true instead, or write nothing.
- A long-term aim IS worth keeping: "User is building a local-first memory layer" is a goal that still holds next month. A task from this afternoon is not.
- "type" is one of: fact, event, preference, skill, goal, relationship, decision, problem, solution
- Use "problem" for something that broke or blocked, "solution" for what fixed it, and "decision" for a choice that was made. A failure and its fix are two memories, not one.
- "importance" is 0.0-1.0. Identity, location, work and long-term commitments are 0.8+. Passing detail is 0.3.
- "entities" are named things actually mentioned. "type" is one of: person, place, org, tech, event, other
- "relations" are triples between two named entities from this message. Use snake_case predicates. Omit if there are none.

Output this JSON shape and nothing else:
{{"memories": [{{"text": "...", "type": "fact", "importance": 0.7, "entities": [{{"name": "...", "type": "org"}}], "relations": [{{"subject": "...", "predicate": "...", "object": "..."}}]}}]}}

Example message: "I moved to Delhi last month for an internship at Zomato, and I've switched from PyTorch to JAX."
Example output:
{{"memories": [
  {{"text": "User moved to Delhi.", "type": "event", "importance": 0.9, "entities": [{{"name": "Delhi", "type": "place"}}], "relations": []}},
  {{"text": "User is doing an internship at Zomato.", "type": "event", "importance": 0.9, "entities": [{{"name": "Zomato", "type": "org"}}], "relations": [{{"subject": "User", "predicate": "interns_at", "object": "Zomato"}}]}},
  {{"text": "User switched from PyTorch to JAX.", "type": "preference", "importance": 0.7, "entities": [{{"name": "PyTorch", "type": "tech"}}, {{"name": "JAX", "type": "tech"}}], "relations": []}}
]}}

Example message: "The user asked for: Convert auth to JWT. The user installed the Python package pyjwt. The user ran the tests in test_auth.py, and it failed with exit code 1. The user edited api.py. The user ran the tests in test_auth.py. The user committed \"switch auth to JWT\"."
Example output (note: the request itself is NOT a memory - what it produced is):
{{"memories": [
  {{"text": "The project uses JWT for authentication.", "type": "fact", "importance": 0.9, "entities": [{{"name": "JWT", "type": "tech"}}], "relations": []}},
  {{"text": "The project uses the pyjwt library.", "type": "fact", "importance": 0.8, "entities": [{{"name": "pyjwt", "type": "tech"}}], "relations": []}},
  {{"text": "The authentication tests in test_auth.py were failing.", "type": "problem", "importance": 0.7, "entities": [{{"name": "test_auth.py", "type": "tech"}}], "relations": []}},
  {{"text": "Editing api.py made the authentication tests pass.", "type": "solution", "importance": 0.8, "entities": [{{"name": "api.py", "type": "tech"}}], "relations": []}}
]}}

Example message: "The user asked for: commit the changes and open a pull request. The user made a git commit. The user pushed the branch to the remote."
Example output (a one-off chore that leaves nothing true afterwards):
{{"memories": []}}

Example message: "what's the weather like today?"
Example output: {{"memories": []}}

Now extract from this message.
Message: "{message}"
Output:"""


CONFLICT_SYSTEM = (
    "You compare statements about a user and classify their relationship. "
    "You output only JSON."
)

CONFLICT_PROMPT = """Compare two statements about the same person.

Work in steps:
1. Name the single attribute each statement is about - for example "city of residence", "employer", "preferred framework", "sibling's name".
2. Decide whether those two attributes are the SAME attribute.
3. If they are the same attribute, compare the values. If they are different attributes, the answer is "independent".

Decisions:
- "duplicate"      - same attribute, same value, only reworded. Nothing changed.
- "update"         - same attribute, new value, where change over time is normal (city, employer, tools, goals).
- "contradiction"  - same attribute, values that cannot both hold at once.
- "independent"    - genuinely different attributes. Both stay true.

Key rule: a person has ONE city of residence, ONE current employer, ONE current answer for any given attribute. "Moved to X" and "lives in Y" are the SAME attribute - the city the person lives in. If X and Y differ, that is an update, never independent.

Examples:
OLD: "User lives in Bengaluru."  NEW: "User moved to Delhi."
{{"old_attribute": "city of residence", "new_attribute": "city of residence", "same_attribute": true, "decision": "update", "reason": "Moving to Delhi replaces Bengaluru as the city the user lives in."}}

OLD: "User lives in Bengaluru."  NEW: "User lives in Mumbai."
{{"old_attribute": "city of residence", "new_attribute": "city of residence", "same_attribute": true, "decision": "contradiction", "reason": "A person cannot live in two cities at once."}}

OLD: "User lives in Bengaluru."  NEW: "User has a sister named Priya."
{{"old_attribute": "city of residence", "new_attribute": "sibling", "same_attribute": false, "decision": "independent", "reason": "Different attributes; both remain true."}}

OLD: "User prefers PyTorch for deep learning."  NEW: "User switched to JAX."
{{"old_attribute": "preferred framework", "new_attribute": "preferred framework", "same_attribute": true, "decision": "update", "reason": "The preferred framework changed to JAX."}}

OLD: "User works at Zomato."  NEW: "User is doing an internship at Zomato."
{{"old_attribute": "employer", "new_attribute": "employer", "same_attribute": true, "decision": "duplicate", "reason": "Same employer stated a second way."}}

Now classify this pair.
OLD: "{old}"
NEW: "{new}"
Output JSON only, same shape as the examples:"""


# ------------------------------------------------------- grounding check

_WORD = text_utils.WORD


def _significant(text: str) -> set[str]:
    return text_utils.significant_set(text, min_length=3)


def _token_matches(token: str, source_tokens: set[str]) -> bool:
    """
    Match a token against the source, tolerating inflection.

    Extraction rewrites "I'm studying AI" as "User studies AI", so exact
    set intersection under-counts badly. Treating one token as matching
    when it's a prefix of the other (from 4 characters up) covers
    study/studies/studying without pulling in unrelated short words.
    """
    if token in source_tokens:
        return True
    for source_token in source_tokens:
        shortest = min(len(token), len(source_token))
        if shortest >= 4 and (token.startswith(source_token[:shortest])
                              or source_token.startswith(token[:shortest])):
            return True
    return False


_QUESTION_STARTERS = {
    "what", "whats", "when", "where", "who", "whom", "whose", "which", "why",
    "how", "is", "are", "was", "were", "am", "do", "does", "did", "can",
    "could", "should", "would", "will", "shall", "have", "has", "had",
    "tell", "explain", "describe", "list", "show", "give",
}


def is_pure_question(text: str) -> bool:
    """
    Is this text asking rather than telling?

    A question states nothing durable, so extracting from one produces
    junk like "The weather is being inquired about for today" — which
    then lives forever and surfaces on unrelated queries. The model is
    told to skip these and mostly does, but "mostly" isn't good enough
    on the write path, so this decides deterministically.

    Mixed input ("I moved to Delhi. What should I see?") is not a pure
    question — one declarative sentence is enough to be worth extracting.
    """
    sentences = [s for s in re.split(r"(?<=[.!?])\s+|\n+", text.strip()) if s.strip()]
    if not sentences:
        return False

    for sentence in sentences:
        stripped = sentence.strip()
        if stripped.endswith("?"):
            continue
        first = _WORD.findall(stripped.lower())
        # A sentence with no words at all (pure punctuation/emoji) carries
        # no assertion either, so it doesn't rescue the text.
        if not first:
            continue
        if first[0] in _QUESTION_STARTERS:
            continue
        return False  # found something declarative
    return True


def mentions_user(text: str) -> bool:
    """
    Does this memory actually say something about the user?

    Extraction is instructed to write every memory as a third-person
    statement about "User". Anything that comes back without a user
    reference is the model describing the *message* rather than
    recording a fact from it.
    """
    return re.search(r"\buser\b", text, re.IGNORECASE) is not None


def is_grounded(memory_text: str, source_text: str, min_ratio: float = 0.34) -> bool:
    """
    Does this extracted memory actually come from the source text?

    A cheap, deterministic guard against confabulation. Requires a decent
    share of the memory's content words to trace back to the source, so a
    model that invents "User is a software engineer" from "I like coffee"
    gets caught before it reaches storage.
    """
    memory_tokens = _significant(memory_text)
    if not memory_tokens:
        return False
    source_tokens = _significant(source_text)
    if not source_tokens:
        return False

    matched = sum(1 for t in memory_tokens if _token_matches(t, source_tokens))
    return (matched / len(memory_tokens)) >= min_ratio


# ------------------------------------------------------------ coercion


def _as_float(value, default: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        return max(low, min(high, float(value)))
    except (TypeError, ValueError):
        return default


def _as_memory_type(value) -> MemoryType:
    try:
        return MemoryType(str(value).strip().lower())
    except ValueError:
        return MemoryType.FACT


# Sentence shapes that describe the *input format* rather than the work.
_SCAFFOLDING = re.compile(
    r"""^(?:the\s+)?(?:
          (?:files?|commands?|entities|memories|notes?|sessions?)\s+
              (?:involved|listed|mentioned|run|executed|that\s+failed|the\s+user\s+ran)
        | \d+\s+\w+\s+(?:are|were|is|was)\s+(?:involved|listed|mentioned)
        | (?:work\s+)?session\s+(?:in|on|for)\b
        )""",
    re.IGNORECASE | re.VERBOSE,
)


# A markdown heading, list bullet or fence swallowed into the middle of
# a sentence. `_is_junk_entity` already refuses these as node names, but
# the same string reached the memory *text* untouched and produced "The
# project uses # MemoOS — Build Instructions for Claude Code as its build
# instructions." The model is describing the shape of a file it read
# rather than anything the project is or does.
_MARKUP_IN_TEXT = re.compile(r"(?:^|\s)(?:#{1,6}\s|```|\|\s*-{2,})")


def _is_scaffolding(text: str) -> bool:
    stripped = text.strip()
    if _SCAFFOLDING.match(stripped):
        return True
    return bool(_MARKUP_IN_TEXT.search(stripped))


# Verbs naming an action that is finished by the time anyone reads the
# memory back. Recording that someone *wanted* one of these records
# nothing that is still true: "User wanted to commit the changes" held
# for about thirty seconds, and then sat in the store forever, ranking
# against facts that still hold.
EPHEMERAL_ACTIONS = frozenset({
    "commit", "push", "pull", "merge", "rebase", "checkout", "clone",
    "delete", "remove", "drop", "clear", "clean", "cleanup", "prune",
    "rename", "publish", "deploy", "revert", "undo", "restore",
    "kill", "stop", "start", "restart", "run", "rerun", "retry",
    "open", "close", "continue", "proceed", "finish", "complete",
    "install", "uninstall", "reinstall", "update", "upgrade",
    "add", "move", "copy", "replace", "split", "combine", "rerunning",
    "fix", "debug", "check", "verify", "confirm", "look", "see",
    "show", "print", "list", "read", "know", "understand", "explain",
})

# "The user asked for X" — a record of the request itself, whatever X is.
# Rejected unconditionally, because a request is never a memory: what it
# produced might be, and that is a different sentence. This pattern is
# what the model reached for the moment the digest stopped saying
# "wanted to", which is a good reminder that the prompt is guidance and
# this is the part that actually holds.
_REQUEST = re.compile(
    r"^(?:the\s+)?user\s+"
    r"(?:ask(?:s|ed)?|request(?:s|ed)?|instruct(?:s|ed)?|told)\b",
    re.IGNORECASE)

# "The user wanted to <chore>" — transient only when the verb names a
# chore. The same frame around a lasting aim is worth keeping.
_INTENT = re.compile(
    r"^(?:the\s+)?user\s+"
    r"(?:want(?:s|ed)?|would\s+like|wish(?:es|ed)?|"
    r"decid(?:es|ed)|intend(?:s|ed)?|plan(?:s|ned)?|tri(?:es|ed)|attempted)"
    r"\s+(?:to\s+)?(?P<verb>[a-z]+)",
    re.IGNORECASE)


def is_transient_intent(text: str) -> bool:
    """
    Is this a record of somebody asking for something, rather than a fact?

    The backstop for the prompt rule above. The prompt is where this is
    really solved — a digest that stops asserting intent stops producing
    intent — but a small local model drifts, and one drift costs a
    permanent memory. This is deterministic and cheap.

    Two shapes, judged differently:

      A *request* is rejected outright. "The user asked for X" records
      that somebody asked, which stops being true the moment it is
      answered. Whatever the request produced is a separate sentence, and
      that one is welcome.

      An *intention* is judged on its verb, because the same frame covers
      both a chore and a direction. "User wanted to commit the changes"
      was true for thirty seconds; "User wants to build a local-first
      memory layer" still holds next month, and `build` is not a chore.

    Deliberately not keyed on memory type: the model labels these
    inconsistently, and "User wants to fix the issue with ollama" arrived
    typed as a `problem` while being pure intent. What makes it junk is
    that the action it names is over.
    """
    stripped = text.strip()
    if _REQUEST.match(stripped):
        return True
    match = _INTENT.match(stripped)
    if not match:
        return False
    return match.group("verb").lower() in EPHEMERAL_ACTIONS


def _as_entity_type(value) -> EntityType:
    try:
        return EntityType(str(value).strip().lower())
    except ValueError:
        return EntityType.OTHER


# Names that carry no information. The first group is the model echoing
# the *type* field back as if it were a thing ("tech", "org"); the second
# is scaffolding language from the prompt. Both produce graph nodes you
# cannot learn anything from — a node labelled "tech" tells you nothing
# about what the project is.
JUNK_ENTITY_NAMES = {
    "tech", "org", "person", "place", "event", "other", "thing", "entity",
    "name", "type", "user", "the user", "me", "i", "it", "they", "we",
    "project", "file", "files", "command", "commands", "session", "code",
    "text", "data", "none", "null", "n/a", "unknown", "example",
}


def _is_junk_entity(name: str) -> bool:
    stripped = name.strip().strip("'\"`.,").lower()
    if not stripped or stripped in JUNK_ENTITY_NAMES:
        return True
    # A bare number is a count, not a thing; a single character is noise.
    if len(stripped) < 2 or stripped.replace(".", "").isdigit():
        return True
    # An absolute path is a location on one machine, not a thing the
    # project is about. `api.py` is worth a node and recurs across
    # sessions; `/Users/someone/proj/venv/bin/activate` is the same file
    # under a name nobody else will ever write, and it turns up in
    # expansions as a wall of text that says nothing.
    if stripped.startswith(("/", "~/")) or stripped.count("/") >= 3:
        return True
    # A markdown heading or a sentence fragment scraped out of a document.
    # A node has a name; "# MemoOS — Build Instructions for Claude Code"
    # is a line of a file, and it is unmatchable and unreadable as a node.
    if stripped.startswith(("#", "-", "*", "`")) or len(stripped) > 48:
        return True
    if len(stripped.split()) > 5:
        return True
    # A date is when something happened, not a thing it happened to. As a
    # node it connects every memory made that day to every other one.
    return bool(re.fullmatch(r"[\d]{1,4}[-/][\d]{1,2}[-/][\d]{1,4}", stripped))


def _parse_entities(raw) -> List[ExtractedEntity]:
    if not isinstance(raw, list):
        return []
    entities: List[ExtractedEntity] = []
    for item in raw:
        # Models drift between [{"name": "X"}] and ["X"] — accept both.
        if isinstance(item, str):
            name = item.strip()
            entity_type = EntityType.OTHER
        elif isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            entity_type = _as_entity_type(item.get("type") or item.get("entity_type"))
        else:
            continue
        # "User" is the implicit subject of every memory; storing it as an
        # entity would link every memory to every other one and make graph
        # expansion useless.
        if not name or _is_junk_entity(name):
            continue
        # Quoting drifts between calls — 'memoos' and memoos are the same
        # project, and should not become two nodes.
        name = name.strip().strip("'\"`")
        entities.append(ExtractedEntity(name=name, entity_type=entity_type))
    return entities


def _parse_relations(raw) -> List[ExtractedRelation]:
    if not isinstance(raw, list):
        return []
    relations: List[ExtractedRelation] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        subject = str(item.get("subject", "")).strip()
        predicate = str(item.get("predicate", "")).strip()
        obj = str(item.get("object", "")).strip()
        if not (subject and predicate and obj):
            continue
        predicate = re.sub(r"[^a-z0-9]+", "_", predicate.lower()).strip("_")
        if not predicate:
            continue
        relations.append(ExtractedRelation(subject=subject, predicate=predicate, object=obj))
    return relations


# ------------------------------------------------------------ public API


class ExtractionUnavailable(RuntimeError):
    """
    The model could not be reached, or never returned usable JSON.

    Distinct from "this text held nothing worth remembering", which is an
    empty list and a perfectly good outcome. Collapsing the two is how a
    timeout comes to look like a quiet afternoon: the caller sees no
    memories, assumes there were none, marks the source as processed, and
    the content is gone for good.
    """


def extract_memories(text: str, *, model: Optional[str] = None,
                     check_grounding: bool = True,
                     subject_scoped: bool = True,
                     strict: bool = False) -> List[ExtractedMemory]:
    """
    Extract structured memories from a piece of text.

    `subject_scoped` says whether this text is the user talking about
    themselves (chat, notes) or arbitrary reference material (a PDF, a
    policy doc). When True, questions are skipped and every memory must
    actually be about the user. Those guards are right for conversation
    and wrong for documents, where nothing is about the user at all.

    Returns an empty list when there is nothing worth remembering, or
    when the model fails to produce usable output — never raises, because
    one bad extraction should cost one memory, not the whole ingest.

    Pass `strict=True` to raise `ExtractionUnavailable` instead when the
    *call itself* failed. Callers that will mark their source as
    processed afterwards want this: without it they cannot tell a chunk
    that said nothing from a chunk they never managed to read.
    """
    cleaned = text.strip()
    if not cleaned:
        return []

    # Cheap deterministic reject, before spending ~10s of model time.
    if subject_scoped and is_pure_question(cleaned):
        return []

    payload = generate_json(
        EXTRACTION_PROMPT.format(message=cleaned),
        system=EXTRACTION_SYSTEM,
        temperature=0.0,
        model=model or config.EXTRACT_MODEL,
    )
    if payload is None:
        if strict:
            raise ExtractionUnavailable(
                f"extraction model {model or config.EXTRACT_MODEL!r} returned "
                f"nothing usable for {len(cleaned.split())} words"
            )
        return []

    # Accept both {"memories": [...]} and a bare [...] — models emit both.
    if isinstance(payload, list):
        raw_memories = payload
    elif isinstance(payload, dict):
        raw_memories = payload.get("memories", [])
        if not isinstance(raw_memories, list):
            raw_memories = []
    else:
        return []

    extracted: List[ExtractedMemory] = []
    seen: set[str] = set()

    for item in raw_memories:
        if isinstance(item, str):
            item = {"text": item}
        if not isinstance(item, dict):
            continue

        memory_text = str(item.get("text", "")).strip()
        if len(memory_text) < 4:
            continue

        # A model that echoes the prompt's placeholder is a failed call.
        if memory_text.strip(". ") in {"...", "..", "text"}:
            continue

        # A memory has to say something you could act on or be reminded
        # by. These shapes never do: they are the digest's own scaffolding
        # read back as fact ("Files involved: a.py, b.py"), or a count of
        # something nobody asked about ("22 files are involved"). They also
        # crowd out real memories, because they are recent and they rank.
        if _is_scaffolding(memory_text):
            continue

        # A request is not a memory. See `is_transient_intent`: this is
        # the single biggest source of junk in a terminal store, because
        # every imported prompt is literally somebody asking for
        # something.
        if is_transient_intent(memory_text):
            continue

        key = memory_text.lower()
        if key in seen:
            continue
        seen.add(key)

        # Grounded in the source, and actually about the user. Both guards
        # catch different failures: grounding catches invention, the user
        # check catches the model narrating the message instead of
        # recording a fact from it.
        if check_grounding and not is_grounded(memory_text, cleaned):
            continue
        if subject_scoped and not mentions_user(memory_text):
            continue

        extracted.append(ExtractedMemory(
            text=memory_text,
            memory_type=_as_memory_type(item.get("type") or item.get("memory_type")),
            importance=_as_float(item.get("importance"), 0.5),
            confidence=_as_float(item.get("confidence"), 0.8),
            entities=_parse_entities(item.get("entities")),
            relations=_parse_relations(item.get("relations")),
        ))

    return extracted


def judge_conflict(new_text: str, old_text: str,
                   model: Optional[str] = None) -> ConflictJudgement:
    """
    Decide what a new memory does to an existing, similar one.

    Defaults to INDEPENDENT on any failure. That bias is intentional:
    wrongly keeping two memories is a retrieval nuisance, while wrongly
    superseding one silently destroys information.
    """
    payload = generate_json(
        CONFLICT_PROMPT.format(old=old_text, new=new_text),
        system=CONFLICT_SYSTEM,
        temperature=0.0,
        model=model or config.EXTRACT_MODEL,
    )
    if not isinstance(payload, dict):
        return ConflictJudgement()

    raw = str(payload.get("decision", "")).strip().lower()
    try:
        decision = ConflictDecision(raw)
    except ValueError:
        return ConflictJudgement(reason=f"unparseable decision: {raw!r}")

    return ConflictJudgement(decision=decision, reason=str(payload.get("reason", "")).strip())


def extract_fact(user_message: str) -> Optional[str]:
    """
    Backwards-compatible single-fact extraction.

    Returns the first extracted memory's text, or None. New code should
    call `extract_memories` and keep the structure.
    """
    memories = extract_memories(user_message)
    return memories[0].text if memories else None
