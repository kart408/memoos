"""
MemoOS demo — run this to see the pipeline work end to end.

    python main.py

What this demonstrates (in order):
1. Extraction: one sentence in, several atomic facts out
2. Semantic retrieval by meaning, not keyword match
3. Contradiction handling: a later fact superseding an earlier one
4. Formatting memories as LLM-ready context — the actual "memory layer" moment

Note this uses `remember()` rather than `add()`. The difference is not
cosmetic. `add()` stores a sentence verbatim, which means one embedding
has to represent every fact in it at once, nothing populates the entity
graph, and retrieval is left leaning on vector search alone. `remember()`
splits "I moved to Mumbai for an internship" into separate facts and
indexes each one, which is what makes the query below land on the right
memory instead of a near-tie.
"""

from memoos_core import MemoOS


def show(memo: MemoOS, query: str) -> None:
    print(f"\nQuery: {query!r}")
    results = memo.search(query, top_k=3)
    if not results:
        print("  (nothing relevant)")
        return
    for r in results:
        similarity = f"{r.vector_score:.2f}" if r.vector_score is not None else "  - "
        print(f"  [sim {similarity}] {r.memory.text}")


def main():
    memo = MemoOS(persist_path="./memoos_data", container="demo")

    print("Remembering what the user said...")
    for statement in [
        "I live in Bengaluru and I'm studying AI/ML engineering.",
        "I prefer PyTorch over TensorFlow for deep learning work.",
        "I built an agentic research assistant using LangChain and Pydantic.",
    ]:
        result = memo.remember(statement)
        print(f"\n  {statement}")
        for m in result.created:
            print(f"    -> [{m.memory_type.value}] {m.text}")

    print(f"\n  entities discovered: {', '.join(e.name for e in memo.entities())}")

    show(memo, "Where does the user live?")

    print("\n" + "-" * 60)
    print("User now says something that contradicts an earlier memory.")
    print("Nothing here calls supersede() — the pipeline works it out.")
    result = memo.remember("I moved to Mumbai for an internship.")
    for m in result.superseded:
        print(f"  superseded: {m.text}")

    show(memo, "Where does the user live?")

    print("\nFormatted as LLM context, ready to inject into a system prompt:\n")
    print(memo.recall_as_context("What does the user prefer for deep learning?"))
    print()


if __name__ == "__main__":
    main()
