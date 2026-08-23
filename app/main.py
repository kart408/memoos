"""
MemoOS — a multi-tenant memory layer for AI applications.

    uvicorn app.main:app --reload

Give it a user_id and a piece of text and it remembers. Give it a
user_id and a query and it returns the most relevant things it knows,
ranked. Everything runs locally: SQLite for rows, Ollama for embeddings.

Interactive API docs: http://127.0.0.1:8000/docs
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from . import routes
from .db import SQLiteMemoryRepository
from .memory_service import MemoryService

DB_PATH = os.getenv("MEMOOS_DB_PATH", "memoos.db")

_repository: SQLiteMemoryRepository | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _repository
    _repository = SQLiteMemoryRepository(DB_PATH)
    routes.configure(MemoryService(_repository))
    yield
    _repository.close()


app = FastAPI(
    title="MemoOS",
    description=__doc__,
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router=routes.router)
