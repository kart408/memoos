"""
MemoryAssistant: a general-purpose AI assistant with persistent memory
about the user - the core MemoOS product, not tied to any one domain.

Statements get remembered automatically. Personal questions are
answered deterministically from stored memory (reliable, no model
guesswork). Everything else gets a normal AI-generated reply, with
any relevant remembered facts folded in as context.
"""

import re
from .memory import MemoOS
from .extraction import extract_fact
from .llm import generate_reply

GENERAL_PROMPT = """You are a helpful AI assistant with memory of past conversations with this user. Use anything in "What you remember about this user" naturally in your answer if it's relevant. If nothing is relevant, just answer normally.

Be concise and direct."""

ACKNOWLEDGE_PROMPT = """The user just told you something about themselves - they did not ask a question. Write one short, natural sentence acknowledging what they said. Do not add unrelated information. Just acknowledge warmly and briefly."""

STATEMENT_STARTERS = (
    "i am ", "i'm ", "im ", "i live", "i prefer", "i like", "i enjoy",
    "i have", "i study", "i studying", "i am studying", "my name is",
    "my favorite", "my favourite", "i want to", "i plan to", "i love",
    "i work", "i was born", "i graduated", "i moved",
)

PERSONAL_QUESTION_MARKERS = (
    "what do i", "did i say", "what's my", "what is my",
    "do i prefer", "who am i", "where do i", "what did i", "am i",
    "do you remember", "what do you know about me",
)


def _is_personal_statement(message: str) -> bool:
    stripped = message.strip().lower()
    if stripped.endswith("?"):
        return False
    return any(stripped.startswith(s) for s in STATEMENT_STARTERS)


def _is_personal_question(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in PERSONAL_QUESTION_MARKERS)


def build_general_prompt(user_message: str, memory_context: str) -> str:
    parts = [GENERAL_PROMPT]
    if memory_context:
        parts.append(f"\nWhat you remember about this user:\n{memory_context}")
    parts.append(f"\nUser: {user_message}\nAssistant:")
    return "\n".join(parts)


def build_acknowledge_prompt(user_message: str) -> str:
    return f"{ACKNOWLEDGE_PROMPT}\n\nUser said: \"{user_message}\"\nAssistant:"


class MemoryAssistant:
    """A single user's AI assistant, backed by their own persistent memory."""

    def __init__(self, user_id: str, persist_path: str = "./memoos_data"):
        self.user_id = user_id
        self.memo = MemoOS(client_id=user_id, persist_path=persist_path)

    def chat(self, user_message: str) -> str:
        # 1. Personal statement -> remember it, acknowledge only
        if _is_personal_statement(user_message):
            fact = extract_fact(user_message)
            if fact:
                self.memo.add(fact)
            prompt = build_acknowledge_prompt(user_message)
            return generate_reply(prompt, temperature=0.3)

        # 2. Direct personal recall question -> answer deterministically
        #    from stored memory, no model guesswork
        if _is_personal_question(user_message):
            results = self.memo.search(user_message, top_k=1)
            if results and results[0].score >= 0.2:
                return f"You mentioned: {results[0].memory.text}"
            return "I don't think you've told me that yet - feel free to share, and I'll remember it."

        # 3. Everything else -> normal AI reply, with relevant memory folded in
        memory_context = self.memo.recall_as_context(user_message)
        prompt = build_general_prompt(user_message, memory_context)
        return generate_reply(prompt, temperature=0.5)
