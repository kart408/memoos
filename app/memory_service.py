"""
Business logic: what it means to remember, recall, revise and forget.

This layer knows nothing about SQLite and nothing about HTTP. It holds a
`MemoryRepository` and calls the embedding model — which is what lets
the storage engine change underneath it without a single edit here.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests

from . import embeddings
from .db import MemoryRepository
from .models import Memory

CHAT_MODEL = embeddings.CHAT_MODEL
OLLAMA_URL = embeddings.OLLAMA_URL
EXTRACT_TIMEOUT = int(os.getenv("MEMOOS_EXTRACT_TIMEOUT", "180"))

EXTRACT_PROMPT = """Extract the durable facts from the text below.

Rules:
- Each fact must be a short, standalone sentence that still makes sense
  months later, with no pronouns left dangling ("He works there" is
  useless without the surrounding text; "Arjun works at Swiggy" is not).
- Use the third person and name the subject. If the text says "I", the
  subject is the speaker — write it as "The user".
- Keep only what is worth remembering. Pleasantries, questions and
  filler are not facts.
- If there is nothing worth remembering, return an empty list.

Return JSON only, in exactly this form:
{"facts": ["...", "..."]}

Text:
\"\"\"%s\"\"\"
"""


def _parse_facts(raw: str) -> List[str]:
    """
    Pull a fact list out of whatever the model actually said.

    Local models wrap JSON in prose and fences more often than not, so
    the happy path is tried first and a brace-scan is the fallback. A
    failure here returns nothing rather than raising: extraction is an
    enhancement, and it should never be the reason a write fails.
    """
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()

    candidates = [text]
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            facts = data.get("facts", [])
        elif isinstance(data, list):
            facts = data
        else:
            continue
        cleaned = [str(f).strip() for f in facts
                   if isinstance(f, (str, int, float)) and str(f).strip()]
        if cleaned:
            return cleaned
    return []


def extract_facts(text: str, *, model: str = CHAT_MODEL) -> List[str]:
    """
    Turn raw conversational input into standalone facts.

    Returns [] if the model is unavailable or produces nothing usable;
    the caller falls back to storing the raw text.
    """
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={
                "model": model,
                "prompt": EXTRACT_PROMPT % text,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.0},
            },
            timeout=EXTRACT_TIMEOUT,
        )
        if response.status_code != 200:
            return []
        return _parse_facts(response.json().get("response", ""))
    except (requests.RequestException, ValueError):
        return []


class MemoryService:
    def __init__(self, repository: MemoryRepository):
        self.repo = repository

    # ---------------------------------------------------------------- write

    def add(self, user_id: str, content: str, *,
            metadata: Optional[Dict[str, Any]] = None,
            extract: bool = False) -> Tuple[List[Memory], bool]:
        """
        Remember something.

        With `extract`, the text is distilled into standalone facts and
        each one is stored separately — so a rambling sentence becomes
        several precisely retrievable memories. If extraction yields
        nothing (model down, or nothing worth keeping), the raw text is
        stored instead: a write must not silently vanish because an
        optional enhancement failed.
        """
        metadata = dict(metadata or {})
        texts: List[str] = []
        extracted = False

        if extract:
            facts = extract_facts(content)
            if facts:
                texts, extracted = facts, True

        if not texts:
            texts = [content]

        created: List[Memory] = []
        for text in texts:
            entry = dict(metadata)
            if extracted:
                # Keep the provenance — being able to see what a memory
                # was distilled from is what makes a bad extraction
                # diagnosable instead of merely wrong.
                entry.setdefault("source_text", content)
            memory = Memory(
                user_id=user_id,
                content=text,
                embedding=embeddings.embed(text),
                metadata=entry,
            )
            created.append(self.repo.add(memory))

        return created, extracted

    def update(self, user_id: str, memory_id: str, *,
               content: Optional[str] = None,
               metadata: Optional[Dict[str, Any]] = None) -> Optional[Memory]:
        """
        Edit a memory. Changing content re-embeds it.

        Returns None when the memory is not this user's — the repository
        scopes the UPDATE, so another tenant's id simply matches nothing.
        """
        vector = embeddings.embed(content) if content is not None else None
        return self.repo.update(user_id, memory_id, content=content,
                                embedding=vector, metadata=metadata)

    def delete(self, user_id: str, memory_id: str) -> bool:
        return self.repo.delete(user_id, memory_id)

    # ----------------------------------------------------------------- read

    def get(self, user_id: str, memory_id: str) -> Optional[Memory]:
        return self.repo.get(user_id, memory_id)

    def list(self, user_id: str, *, limit: int = 50,
             offset: int = 0) -> Tuple[List[Memory], int]:
        return self.repo.list(user_id, limit, offset), self.repo.count(user_id)

    def search(self, user_id: str, query: str, *, top_k: int = 5,
               min_score: float = 0.0) -> List[Tuple[Memory, float]]:
        """
        Semantic search over one user's memories.

        The candidate set comes from the repository already filtered by
        user_id, so there is no point at which another tenant's vectors
        are in scope to be ranked, let alone returned.
        """
        rows = self.repo.embeddings_for_user(user_id)
        if not rows:
            return []

        ids = [r[0] for r in rows]
        matrix = np.vstack([r[1] for r in rows])
        scores = embeddings.cosine_scores(embeddings.embed(query), matrix)

        # argpartition beats a full sort once a user has many memories,
        # and costs nothing when they have few.
        k = min(top_k, len(ids))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]

        hits: List[Tuple[Memory, float]] = []
        for index in top:
            score = float(scores[index])
            if score < min_score:
                continue
            memory = self.repo.get(user_id, ids[index])
            if memory:
                hits.append((memory, score))
        return hits
