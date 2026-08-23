from api import get_assistant

assistant = get_assistant("karthik")

print("Stored memories for 'karthik':")
results = assistant.memo.search("Karthik MemoOS", top_k=5)
for r in results:
    print(f"  [{r.score:.2f}] {r.memory.text}")

print("\nSearch score for 'What do you know about me?':")
results2 = assistant.memo.search("What do you know about me?", top_k=3)
for r in results2:
    print(f"  [{r.score:.2f}] {r.memory.text}")
