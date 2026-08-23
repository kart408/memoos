from api import get_assistant

assistant = get_assistant("karthik")

print("1:", assistant.chat("My name is Karthik and I'm building an AI memory layer called MemoOS"))
print()
print("2:", assistant.chat("What's a good way to explain vector embeddings to a beginner?"))
print()
print("3:", assistant.chat("What do you know about me?"))
