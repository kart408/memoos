"""
MemoOS demo — run this to see the pipeline work end to end.

    python main.py

What this demonstrates (in order):
1. Adding memories (ingestion -> embedding -> storage)
2. Semantic retrieval by meaning, not keyword match
3. Contradiction handling: a later fact overriding an earlier one
4. Formatting memories as LLM-ready context — the actual "memory layer" moment
"""

from memoos_core import MemoOS, MemoryType


def main():
    memo = MemoOS(persist_path="./memoos_data")

    print("Adding memories...\n")
    m1 = memo.add("I live in Bengaluru and I'm studying AI/ML engineering.")
    memo.add("I prefer PyTorch over TensorFlow for deep learning work.", memory_type=MemoryType.PREFERENCE)
    memo.add("I built an agentic research assistant using LangChain and Pydantic.", memory_type=MemoryType.EVENT)

    print("Query: 'Where does the user live?'")
    for r in memo.search("Where does the user live?", top_k=2):
        print(f"  [{r.score:.2f}] {r.memory.text}")

    print("\nUser says something that contradicts an earlier memory...")
    memo.supersede(m1.id, "I moved to Mumbai for an internship.")

    print("\nQuery again: 'Where does the user live?'")
    for r in memo.search("Where does the user live?", top_k=2):
        print(f"  [{r.score:.2f}] {r.memory.text}")
    print("  (notice the Bengaluru memory no longer surfaces — it was superseded)")

    print("\nFormatted as LLM context, ready to inject into a system prompt:\n")
    print(memo.recall_as_context("What does the user prefer for deep learning?"))


if __name__ == "__main__":
    main()