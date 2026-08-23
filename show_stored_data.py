import chromadb

print("=" * 60)
print("ALL MEMORY COLLECTIONS (one per student/session)")
print("=" * 60)

client = chromadb.PersistentClient(path="./memoos_data")
collections = client.list_collections()

if not collections:
    print("(no memory collections found yet)")

for col in collections:
    collection = client.get_collection(col.name)
    data = collection.get()
    print(f"\nCollection: {col.name}")
    print(f"Number of stored memories: {len(data['ids'])}")
    for i in range(len(data['ids'])):
        print(f"  - \"{data['documents'][i]}\"")

print("\n" + "=" * 60)
print("KNOWLEDGE BASE COLLECTIONS (college FAQ data)")
print("=" * 60)

kb_client = chromadb.PersistentClient(path="./kb_data")
kb_collections = kb_client.list_collections()

for col in kb_collections:
    collection = kb_client.get_collection(col.name)
    data = collection.get()
    print(f"\nCollection: {col.name}")
    print(f"Number of stored chunks: {len(data['ids'])}")
    for i in range(len(data['ids'])):
        print(f"  - \"{data['documents'][i][:80]}...\"")
