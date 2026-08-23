"""
MemoOS API - a memory layer for AI. Any app can call /chat and get
a reply from an AI that remembers this specific user across every
conversation.

    uvicorn api:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from memoos_core.assistant import MemoryAssistant

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_assistants: dict[str, MemoryAssistant] = {}


def get_assistant(user_id: str) -> MemoryAssistant:
    if user_id not in _assistants:
        _assistants[user_id] = MemoryAssistant(user_id=user_id)
    return _assistants[user_id]


class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    assistant = get_assistant(req.user_id)
    reply = assistant.chat(req.message)
    return ChatResponse(reply=reply)


@app.get("/health")
def health():
    return {"status": "ok"}
