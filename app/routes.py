"""
HTTP surface.

Six endpoints, all but /health scoped to a user_id in the path.

On that user_id: it is taken from the URL and trusted, which is fine for
local testing and wrong for anything real — any caller can type any
user_id and read that user's memories. In a deployment this would come
from a verified token (the authenticated subject), and the path segment
would either be dropped or checked against it. The service and
repository layers below are already written for that: they take user_id
as an argument and scope every query by it, so swapping where it comes
from does not change how isolation works.
"""

from fastapi import APIRouter, Depends, HTTPException, Path, Query

from . import embeddings
from .memory_service import MemoryService
from .models import (
    CreateResult,
    HealthResponse,
    MemoryCreate,
    MemoryOut,
    MemoryPage,
    MemoryUpdate,
    SearchHit,
    SearchResponse,
)

router = APIRouter()

# Set by main.py at startup. A function indirection rather than a bare
# global so tests can point the routes at a scratch database.
_service: MemoryService | None = None


def configure(service: MemoryService) -> None:
    global _service
    _service = service


def get_service() -> MemoryService:
    if _service is None:
        raise HTTPException(500, "service not configured")
    return _service


UserId = Path(..., min_length=1, max_length=128, description="Tenant identifier")


@router.post("/users/{user_id}/memories", response_model=CreateResult,
             status_code=201, tags=["memories"])
def add_memory(req: MemoryCreate, user_id: str = UserId,
               service: MemoryService = Depends(get_service)):
    """
    Remember something for this user.

    The user's memory space is created implicitly by this call — there is
    no provisioning step, just the first write under a new user_id.
    """
    try:
        created, extracted = service.add(user_id, req.content,
                                         metadata=req.metadata,
                                         extract=req.extract)
    except embeddings.EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc
    return CreateResult(created=[MemoryOut.of(m) for m in created],
                        extracted=extracted)


@router.get("/users/{user_id}/memories", response_model=MemoryPage,
            tags=["memories"])
def list_memories(user_id: str = UserId,
                  limit: int = Query(50, ge=1, le=500),
                  offset: int = Query(0, ge=0),
                  service: MemoryService = Depends(get_service)):
    """All of this user's memories, newest first."""
    memories, total = service.list(user_id, limit=limit, offset=offset)
    return MemoryPage(memories=[MemoryOut.of(m) for m in memories],
                      total=total, limit=limit, offset=offset)


@router.get("/users/{user_id}/memories/search", response_model=SearchResponse,
            tags=["memories"])
def search_memories(user_id: str = UserId,
                    q: str = Query(..., min_length=1, description="Search query"),
                    top_k: int = Query(5, ge=1, le=100),
                    min_score: float = Query(0.0, ge=-1.0, le=1.0),
                    service: MemoryService = Depends(get_service)):
    """
    Semantic search, ranked by cosine similarity.

    Declared before the `/{memory_id}` route below so that "search" is
    matched as a literal path and never mistaken for a memory id.
    """
    try:
        hits = service.search(user_id, q, top_k=top_k, min_score=min_score)
    except embeddings.EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc
    return SearchResponse(
        query=q,
        hits=[SearchHit(memory=MemoryOut.of(m), score=s) for m, s in hits],
    )


@router.get("/users/{user_id}/memories/{memory_id}", response_model=MemoryOut,
            tags=["memories"])
def get_memory(user_id: str = UserId, memory_id: str = Path(...),
               service: MemoryService = Depends(get_service)):
    memory = service.get(user_id, memory_id)
    if memory is None:
        raise HTTPException(404, "memory not found")
    return MemoryOut.of(memory)


@router.patch("/users/{user_id}/memories/{memory_id}", response_model=MemoryOut,
              tags=["memories"])
def update_memory(req: MemoryUpdate, user_id: str = UserId,
                  memory_id: str = Path(...),
                  service: MemoryService = Depends(get_service)):
    """
    Edit a memory. New content is re-embedded so search stays truthful.

    Another user's memory id gets the same 404 as a nonexistent one —
    the scoped UPDATE matches no row either way, which is what stops an
    id guess from confirming that a memory exists at all.
    """
    try:
        memory = service.update(user_id, memory_id, content=req.content,
                                metadata=req.metadata)
    except embeddings.EmbeddingError as exc:
        raise HTTPException(503, str(exc)) from exc
    if memory is None:
        raise HTTPException(404, "memory not found")
    return MemoryOut.of(memory)


@router.delete("/users/{user_id}/memories/{memory_id}", tags=["memories"])
def delete_memory(user_id: str = UserId, memory_id: str = Path(...),
                  service: MemoryService = Depends(get_service)):
    if not service.delete(user_id, memory_id):
        raise HTTPException(404, "memory not found")
    return {"deleted": memory_id}


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(service: MemoryService = Depends(get_service)):
    """
    Liveness, including whether Ollama is actually usable.

    A plain {"status": "ok"} would report healthy while every write was
    failing on an unreachable model, so the dependency is checked too.
    """
    reachable, detail = embeddings.ping()
    try:
        service.repo.count("__health__")
        database = "ok"
    except Exception as exc:
        database = "error"
        detail = f"{detail}; database: {exc}"

    healthy = reachable and database == "ok"
    return HealthResponse(
        status="ok" if healthy else "degraded",
        database=database,
        ollama="ok" if reachable else "unavailable",
        embed_model=embeddings.EMBED_MODEL,
        detail=None if healthy else detail,
    )
