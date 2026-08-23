import chromadb

client = chromadb.PersistentClient(path="./memoos_data")
for col in client.list_collections():
    client.delete_collection(col.name)
    print(f"Deleted: {col.name}")

print("\nAll test memory cleared. Knowledge base (college data) left untouched.")
