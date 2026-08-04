"""
MemoOS: the public interface. This is what an LLM app would import
and call — everything else in memoos_core is an implementation detail
behind this class.
"""

from typing import List, Optional

from .ingestion import ingest_text
from .models import Memory, MemoryQueryResult, MemoryType
from .storage import MemoryStore


class MemoOS:
    def __init__(self, persist_path: str = "./memoos_data"):
        self.store = MemoryStore(persist_path=persist_path)

    def add(self, text: str, memory_type: MemoryType = MemoryType.FACT,
            source: str = "user") -> Memory:
        memory = ingest_text(text, memory_type=memory_type, source=source)
        self.store.add(memory)
        return memory

    def search(self, query: str, top_k: int = 5) -> List[MemoryQueryResult]:
        return self.store.search(query, top_k=top_k)

    def recall_as_context(self, query: str, top_k: int = 3) -> str:
        """
        Format retrieved memories as a block you can drop straight into
        an LLM system prompt — this is the "memory layer" moment.
        """
        results = self.search(query, top_k=top_k)
        if not results:
            return ""
        lines = [f"- {r.memory.text} (relevance: {r.score:.2f})" for r in results]
        return "Relevant things you know about the user:\n" + "\n".join(lines)

    def supersede(self, old_memory_id: str, new_text: str,
                  memory_type: MemoryType = MemoryType.FACT) -> Memory:
        """Add a new memory that overrides/contradicts an older one."""
        new_memory = self.add(new_text, memory_type=memory_type)
        self.store.mark_superseded(old_memory_id, new_memory.id)
        return new_memory