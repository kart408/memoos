"""
A minimal chat loop: automatically extracts and stores facts from
what the user says, retrieves relevant memories, and generates a
memory-aware reply — no manual memo.add() calls needed.

    python chat.py
"""

from memoos_core import MemoOS
from memoos_core.llm import generate_reply
from memoos_core.extraction import extract_fact


def build_prompt(user_message: str, memory_context: str) -> str:
    system = "You are a helpful assistant. Use the context below if relevant."
    if memory_context:
        return f"{system}\n\n{memory_context}\n\nUser: {user_message}\nAssistant:"
    return f"{system}\n\nUser: {user_message}\nAssistant:"


def main():
    memo = MemoOS(client_id="demo_user", persist_path="./memoos_data")

    print("Chat with your local model (auto-memory enabled). Type 'quit' to exit.\n")
    while True:
        user_message = input("You: ")
        if user_message.strip().lower() == "quit":
            break

        # Automatically decide if this message contains something worth remembering
        fact = extract_fact(user_message)
        if fact:
            memo.add(fact)
            print(f"  [remembered: {fact}]")

        # Automatically recall anything relevant before replying
        memory_context = memo.recall_as_context(user_message)
        prompt = build_prompt(user_message, memory_context)
        reply = generate_reply(prompt)

        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()