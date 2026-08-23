import chromadb

client = chromadb.PersistentClient(path="./memoos_data")
client.delete_collection("memories_demo_college")
print("Cleared demo_college's memory")
